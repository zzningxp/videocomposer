import logging
from PIL import Image
import os
import sys
import time
from copy import deepcopy, copy

import torch
from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from functools import partial
from einops import rearrange
import torch.cuda.amp as amp

from accelerate import Accelerator
from tqdm.auto import tqdm
from importlib import reload

from .datasets import VideoDataset
from .inference_single import random_resize, beta_schedule, make_masked_images, get_first_stage_encoding, prepare_model_kwargs, CenterCrop
from .inference_single import FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder
from .config import cfg
# from utils.config import Config

import artist.data as data
import artist.models as models
import artist.ops as ops

from artist import DOWNLOAD_TO_CACHE
from torch.utils.data import DataLoader
from .autoencoder import  AutoencoderKL, DiagonalGaussianDistribution
from tools.annotator.canny import CannyDetector
from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan
from .unet_sd import UNetSD_temporal


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def train(cfg_update, **kwargs):
    cfg.update(**kwargs)
    
    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.read_image = getattr(cfg, 'read_image', False)
    cfg.read_sketch = getattr(cfg, 'read_sketch', False)
    cfg.read_style = getattr(cfg, 'read_style', False)
    cfg.save_origin_video = getattr(cfg, 'save_origin_video', True)

    gpu = 0
    cfg.world_size = 1
    cfg.gpu = gpu
    cfg.rank = 0
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    # if not cfg.debug:
    #     dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # logging
    log_dir = ops.generalized_all_gather(cfg.log_dir)[0]
    exp_name = os.path.basename(cfg.cfg_file).split('.')[0] + '-S%05d' % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    if cfg.rank == 0:
        name = os.path.basename(cfg.log_dir)
        cfg.log_file = os.path.join(cfg.log_dir, '{}_rank{}.log'.format(name, cfg.rank))
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
    
    # rank-wise params
    l1 = len(cfg.frame_lens)
    l2 = len(cfg.feature_framerates)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (l1*l2)// l2]
    # print(cfg.batch_sizes)
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    
    print("num_epochs : ", cfg.num_epochs)
    print("batch_size : ", cfg.batch_size)
    print("max_frames : ", cfg.max_frames)

    infer_trans = data.Compose([
        data.CenterCropV2(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])

    misc_transforms = data.Compose([
        T.Lambda(partial(random_resize, size=cfg.misc_size)),
        data.CenterCropV2(cfg.misc_size),
        data.ToTensor()])

    mv_transforms = data.Compose([
        T.Resize(size=cfg.resolution),
        T.CenterCrop(cfg.resolution)])

    dataset = VideoDataset(
        cfg=cfg,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_trans,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=T.Compose([
            CenterCrop(cfg.vit_image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)]),
        vit_image_size= cfg.vit_image_size,
        misc_size=cfg.misc_size)

    train_dataloader = DataLoader(
        dataset=dataset,
        num_workers=0,
        pin_memory=True)
    
    clip_encoder = FrozenOpenCLIPEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder.model.to(gpu)
    zero_y = clip_encoder("").detach() # [1, 77, 1024]
    
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder_visual.model.to(gpu)
    black_image_feature = clip_encoder_visual(clip_encoder_visual.black_image).unsqueeze(1) # [1, 1, 1024]
    black_image_feature = torch.zeros_like(black_image_feature) # for old

    frame_in = None
    if cfg.read_image:
        image_key = cfg.image_path # 
        frame = Image.open(open(image_key, mode='rb')).convert('RGB')
        frame_in = misc_transforms([frame]) 
    
    frame_sketch = None
    if cfg.read_sketch:
        sketch_key = cfg.sketch_path # 
        frame_sketch = Image.open(open(sketch_key, mode='rb')).convert('RGB')
        frame_sketch = misc_transforms([frame_sketch]) # 

    frame_style = None
    if cfg.read_style:
        frame_style = Image.open(open(cfg.style_image, mode='rb')).convert('RGB')
    
    # [Contions] Generators for various conditions
    if 'depthmap' in cfg.video_compositions:
        midas = models.midas_v3(pretrained=True).eval().requires_grad_(False).to(
            memory_format=torch.channels_last).half().to(gpu)
    if 'canny' in cfg.video_compositions:
        canny_detector = CannyDetector()
    if 'sketch' in cfg.video_compositions:
        pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
        cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
        pidi_mean = torch.tensor(cfg.sketch_mean).view(1, -1, 1, 1).to(gpu)
        pidi_std = torch.tensor(cfg.sketch_std).view(1, -1, 1, 1).to(gpu)
    # Placeholder for color inference
    palette = None

    # [model] auotoencoder
    ddconfig = {'double_z': True, 'z_channels': 4, \
                'resolution': 256, 'in_channels': 3, \
                'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], \
                'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=DOWNLOAD_TO_CACHE(cfg.sd_checkpoint))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim= cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention = cfg.temporal_attention,
            temporal_attn_times = cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            zero_y = zero_y,
            black_image_feature = black_image_feature,
            ).to(gpu)
    else:
        logging.info("Other model type not implement, exist")
        raise NotImplementedError(f"The model {cfg.network_name} not implement")
        return 

    # Load checkpoint
    resume_step = 1
    if not cfg.resume:
        pass
    elif cfg.resume and cfg.resume_checkpoint:
        if hasattr(cfg, "text_to_video_pretrain") and cfg.text_to_video_pretrain:
            print("text_to_video_pretrain", )
            ss = torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint))
            ss = {key:p for key,p in ss.items() if 'input_blocks.0.0' not in key}
            model.load_state_dict(ss,strict=False)
        else:
            print("load_state_dict", cfg.resume_checkpoint)
            # resume_checkpoint = DOWNLOAD_TO_CACHE()
            model.load_state_dict(torch.load(cfg.resume_checkpoint, map_location='cpu'),strict=False)
        if cfg.resume_step:
            resume_step = cfg.resume_step
        
        logging.info(f'Successfully load step model from {cfg.resume_checkpoint}')
        torch.cuda.empty_cache()
    else:
        logging.error(f'The checkpoint file {cfg.resume_checkpoint} is wrong')
        raise ValueError(f'The checkpoint file {cfg.resume_checkpoint} is wrong ')
        return
    
    # mark model size
    if cfg.rank == 0:
        logging.info(f'Created a model with {int(sum(p.numel() for p in model.parameters()) / (1024 ** 2))}M parameters')

    # diffusion
    betas = beta_schedule('linear_sd', cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    diffusion = ops.GaussianDiffusion(
        betas=betas,
        mean_type=cfg.mean_type,
        var_type=cfg.var_type,
        loss_type=cfg.loss_type,
        rescale_timesteps=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    pathstr = os.path.splitext(os.path.basename(cfg.DATAPATH))[0] + "-" + time.strftime('%y%m%d-%H%M', time.localtime(time.time()))
    cfg.output_dir = os.path.join("./work_dirs/", pathstr)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(cfg.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything

    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    viz_num = cfg.batch_size

    # Now you train the model
    
    for epoch in range(cfg.num_epochs):
        # tqdm derives from the Arabic word taqaddum (تقدّم) which can mean "progress," and is an abbreviation for "I love you so much" in Spanish (te quiero demasiado).
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            caps = batch[1]
            del batch[1]
            print("caps: ", caps)
            batch = ops.to_device(batch, gpu, non_blocking=True)
            if cfg.max_frames == 1 and cfg.use_image_dataset:
                ref_imgs, video_data, misc_data, mask, mv_data = batch
                fps =  torch.tensor([cfg.feature_framerate]*cfg.batch_size,dtype=torch.long, device=gpu)
            else:
                ref_imgs, video_data, misc_data, fps, mask, mv_data = batch
            
            ## save for visualization
            misc_backups = copy(misc_data)
            misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
            mv_data_video = []
            if 'motion' in cfg.video_compositions:
                mv_data_video = rearrange(mv_data, 'b f c h w -> b c f h w')

            ### mask images
            masked_video = []
            if 'mask' in cfg.video_compositions:
                masked_video = make_masked_images(misc_data.sub(0.5).div_(0.5), mask)
                masked_video = rearrange(masked_video, 'b f c h w -> b c f h w')
        
            ### Single Image
            image_local = []
            if 'local_image' in cfg.video_compositions:
                frames_num = misc_data.shape[1]
                bs_vd_local = misc_data.shape[0]
                if cfg.read_image:
                    image_local = frame_in.unsqueeze(0).repeat(bs_vd_local,frames_num,1,1,1).cuda()
                else:
                    image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
                image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
        
            ### encode the video_data
            bs_vd = video_data.shape[0]
            video_data_origin = video_data.clone()
            video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
            misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')
            # video_data_origin = video_data.clone() 

            video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
            misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)

            s1 = time.time()
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior).detach()
                decode_data.append(tmp)
            video_data = torch.cat(decode_data, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)
            s2 = time.time()
            print(f"video input data autoencoder: {s2 - s1} s")

            depth_data = []
            if 'depthmap' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    depth = midas(misc_imgs.sub(0.5).div_(0.5).to(memory_format=torch.channels_last).half())
                    depth = (depth / cfg.depth_std).clamp_(0, cfg.depth_clamp)
                    depth_data.append(depth)
                    # print("misc_imgs ", torch.sum(misc_imgs), torch.sum(depth))
                depth_data = torch.cat(depth_data, dim = 0)
                depth_data = rearrange(depth_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            canny_data = []
            if 'canny' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    # print(misc_imgs.shape)
                    misc_imgs = rearrange(misc_imgs.clone(), 'k c h w -> k h w c') # 'k' means 'chunk'.
                    canny_condition = torch.stack([canny_detector(misc_img) for misc_img in misc_imgs])
                    canny_condition = rearrange(canny_condition, 'k h w c-> k c h w')
                    canny_data.append(canny_condition)
                canny_data = torch.cat(canny_data, dim = 0)
                canny_data = rearrange(canny_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            sketch_data = []
            if 'sketch' in cfg.video_compositions:
                sketch_list = misc_data_list
                if cfg.read_sketch:
                    sketch_repeat = frame_sketch.repeat(frames_num, 1, 1, 1).cuda()
                    sketch_list = [sketch_repeat]

                for misc_imgs in sketch_list:
                    sketch = pidinet(misc_imgs.sub(pidi_mean).div_(pidi_std))
                    sketch = 1.0 - cleaner(1.0 - sketch)
                    sketch_data.append(sketch)
                sketch_data = torch.cat(sketch_data, dim = 0)
                sketch_data = rearrange(sketch_data, '(b f) c h w -> b c f h w', b = bs_vd)

            single_sketch_data = []
            if 'single_sketch' in cfg.video_compositions:
                single_sketch_data = sketch_data.clone()[:, :, :1].repeat(1, 1, frames_num, 1, 1)

            s1 = time.time()
            # preprocess for input text descripts
            y = clip_encoder(caps).detach()  # [1, 77, 1024]
            y0 = y.clone()
        
            y_visual = []
            if 'image' in cfg.video_compositions:
                with torch.no_grad():
                    if cfg.read_style:
                        y_visual = clip_encoder_visual(clip_encoder_visual.preprocess(frame_style).unsqueeze(0).cuda()).unsqueeze(0)
                        y_visual0 = y_visual.clone()
                    else:
                        ref_imgs = ref_imgs.squeeze(1)
                        y_visual = clip_encoder_visual(ref_imgs).unsqueeze(1) # [1, 1, 1024]
                        y_visual0 = y_visual.clone()

            s2 = time.time()

            with amp.autocast(enabled=cfg.use_fp16):
                if cfg.share_noise:
                    b, c, f, h, w = video_data.shape
                    noise = torch.randn((viz_num, c, h, w), device=gpu)
                    noise = noise.repeat_interleave(repeats=f, dim=0) 
                    noise = rearrange(noise, '(b f) c h w->b c f h w', b = viz_num) 
                    noise = noise.contiguous()
                else:
                    noise = torch.randn_like(video_data[:viz_num])

                full_model_kwargs=[
                    {'y': y0[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else y_visual0[:viz_num],
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}, 
                    {'y': zero_y.repeat(viz_num,1,1) if not cfg.use_fps_condition else torch.zeros_like(y0)[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else torch.zeros_like(y_visual0[:viz_num]),
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}
                ]
                
                partial_keys = cfg.guidances
                noise_motion = noise.clone()
                model_kwargs = prepare_model_kwargs(partial_keys = partial_keys,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                                    
                x0 = diffusion.ddim_sample_loop(
                    noise=noise_motion,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)

            timesteps = torch.LongTensor([50])
            noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
            # noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (viz_num,), device=gpu
            ).long()
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                # loss(self, x0, t, model, model_kwargs={}, noise=None, weight = None, use_div_loss= False):

                # loss = diffusion.loss(x0=noise, t=timesteps, model=model.eval(), model_kwargs=model_kwargs, noise=noise_motion, guide_scale=9.0)
                loss = diffusion.loss(x0=x0, t=timesteps, model=model.eval(), model_kwargs=model_kwargs[0])
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
                save_path = os.path.join(cfg.output_dir, f"ckpt_{global_step}_e{epoch}.pth")
                torch.save(model.state_dict(), save_path)
                logging.info(f'Successfully save step {global_step} model to {save_path}')

