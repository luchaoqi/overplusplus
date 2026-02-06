from absl import app
from absl import flags
from ml_collections import config_flags
from loguru import logger
import json
import os
import sys
import glob
import cv2
import pprint
import numpy as np
import mediapy as media
import torch
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                              WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanFunInpaintPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_video_mask_input, save_inout_row, save_videos_grid)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/default_wan.py", "Path to the python config file"
)

def load_pipeline(cfg):
    model_name = cfg.video_model.model_name
    weight_dtype = cfg.system.weight_dtype
    config_model = OmegaConf.load(cfg.video_model.config_path)
    device = set_multi_gpus_devices(cfg.system.ulysses_degree, cfg.system.ring_degree)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config_model['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config_model['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if cfg.video_model.transformer_path:
        logger.info(f"Load transformer from checkpoint: {cfg.video_model.transformer_path}")
        if cfg.video_model.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(cfg.video_model.transformer_path)
        else:
            state_dict = torch.load(cfg.video_model.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"Transformer missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config_model['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config_model['vae_kwargs']),
    ).to(weight_dtype)


    if cfg.video_model.vae_path:
        logger.info(f"Load VAE from checkpoint: {cfg.video_model.vae_path}")
        if cfg.video_model.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(cfg.video_model.vae_path)
        else:
            state_dict = torch.load(cfg.video_model.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        logger.info(f"VAE missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config_model['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config_model['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config_model['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config_model['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[cfg.video_model.sampler_name]
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config_model['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )
    if cfg.system.ulysses_degree > 1 or cfg.system.ring_degree > 1:
        transformer.enable_multi_gpus_inference()

    if cfg.system.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif cfg.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif cfg.system.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if cfg.system.enable_teacache else None
    if coefficients is not None:
        logger.info(f"Enable TeaCache with threshold {cfg.system.teacache_threshold} and skip the first {cfg.system.num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, cfg.video_model.num_inference_steps, cfg.system.teacache_threshold, num_skip_start_steps=cfg.system.num_skip_start_steps, offload=cfg.system.teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(cfg.system.seed)

    if cfg.video_model.lora_path:
        pipeline = merge_lora(pipeline, cfg.video_model.lora_path, cfg.video_model.lora_weight)

    return pipeline, vae, generator

@torch.no_grad()
def run_inference(config, pipeline, vae, generator, input_video_name, keep_fg_ids=[-1]):
    save_video_name = f'{input_video_name}-fg=' + '_'.join([f'{i:02d}' for i in keep_fg_ids])
    if (config.experiment.skip_if_exists and
        sorted(list(glob.glob(os.path.join(config.experiment.save_path, f"{save_video_name}*.mp4"))))):
        logger.debug(f"Skipping {save_video_name} as it already exists")
        return

    video_length = config.data.max_video_length
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    if config.system.enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
    logger.debug(f'Video length: {video_length}')

    sample_size = config.data.sample_size
    sample_size = (int(sample_size.split('x')[0]), int(sample_size.split('x')[1]))

    if not config.experiment.validation:
        input_video, input_video_mask, prompt, clip_image = get_video_mask_input(
            input_video_name,
            sample_size=sample_size,
            keep_fg_ids=keep_fg_ids,
            max_video_length=video_length,
            temporal_window_size=config.video_model.temporal_window_size,
            data_rootdir=config.data.data_rootdir,
            use_trimask=config.video_model.use_trimask,
            dilate_width=config.data.dilate_width,
        )
    else:
        input_video, input_video_mask, prompt = get_video_mask_validation(
            input_video_name,
            sample_size=sample_size,
            max_video_length=video_length,
            temporal_window_size=config.video_model.temporal_window_size,
            data_rootdir=config.data.data_rootdir,
            use_trimask=config.video_model.use_trimask,
            dilate_width=config.data.dilate_width,
        )

    # vae experiment
    if config.experiment.skip_unet:
        if config.experiment.mask_to_vae:
            input_video = input_video_mask.repeat(1, 3, 1, 1, 1)

    sample = pipeline(
        prompt, 
        num_frames = config.video_model.temporal_window_size,
        negative_prompt = config.video_model.negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = config.video_model.guidance_scale,
        num_inference_steps = config.video_model.num_inference_steps,
        video       = input_video,
        mask_video  = input_video_mask,
        use_trimask = config.video_model.use_trimask,
        zero_out_mask_region = config.video_model.zero_out_mask_region,
        clip_image  = clip_image,
        skip_unet   = config.experiment.skip_unet,

        # skip_unet = config.experiment.skip_unet,
    ).videos

    if not os.path.exists(config.experiment.save_path):
        os.makedirs(config.experiment.save_path, exist_ok=True)

    index = len([path for path in os.listdir(config.experiment.save_path) if path.endswith('_tuple.mp4') and path.startswith(save_video_name)]) + 1
    prefix = save_video_name + f'-{index:04d}'
        
    if video_length == 1:
        save_sample_path = os.path.join(config.experiment.save_path, prefix + f".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(save_sample_path)
    else:
        video_path = os.path.join(config.experiment.save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=config.data.fps)
        save_inout_row(input_video, input_video_mask, sample, video_path[:-4] + "_tuple.mp4", fps=config.data.fps)


def main(_):
    config = FLAGS.config
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(config.to_dict())

    all_seqs_in_dir = sorted(os.listdir(config.data.data_rootdir))
    run_seqs = []
    if '/' in config.experiment.run_seqs:
        run_part, total_parts = config.experiment.run_seqs.split('/')
        run_part = int(run_part)
        total_parts = int(total_parts)
        n_per_part = len(all_seqs_in_dir) // total_parts
        part_start = (run_part - 1) * n_per_part
        part_end = min(run_part * n_per_part, len(all_seqs_in_dir))
        run_seqs = all_seqs_in_dir[part_start:part_end]
    else:
        run_seqs = config.experiment.run_seqs.split(',')
        run_seqs = [seq for seq in run_seqs if seq in all_seqs_in_dir]

    seq_fg_to_run = []
    if not config.experiment.validation:
        for seq in run_seqs:
            fg_ids = [-1]
            num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "mask_*.mp4")))))
            if num_fgs == 0:
                num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "trimask_*.mp4")))))
            assert num_fgs > 0
            if config.experiment.matting_mode == "solo" and num_fgs > 1:
                fg_ids.extend(list(range(num_fgs)))
            for fg_id in fg_ids:
                seq_fg_to_run.append((seq, [fg_id]))
    else:
        # read training videos and random mask generation
        seq_fg_to_run = [(seq, [-1]) for seq in run_seqs]

    pipeline, vae, generator = load_pipeline(config)

    for seq_name, fg_id in seq_fg_to_run:
        logger.info(f'Sequence to run: {seq_name}, fgs to keep: {fg_id}')
        def _run_inference():
            run_inference(
                config=config,
                pipeline=pipeline,
                vae=vae,
                generator=generator,
                input_video_name=seq_name,
                keep_fg_ids=fg_id,
            )
        if config.system.allow_skipping_error:
            try:
                _run_inference()
            except Exception as e:
                logger.info(f'Error in {seq_name}, {fg_id}: {e}')
                continue
        else:
            _run_inference()
        

if __name__ == "__main__":
    app.run(main)
