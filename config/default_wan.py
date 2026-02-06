import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    config.system = get_system_config()
    config.data = get_data_config()
    config.video_model = get_video_model_config()
    config.experiment = get_experiment_config()
    return config


def get_experiment_config():
    config = ml_collections.ConfigDict()
    config.run_seqs = "boys-beach,animator-draw"
    config.matting_mode = "solo"  # "clean_bg" or "solo"
    config.save_path = "casper_outputs"
    config.skip_if_exists = True
    config.validation = False
    config.skip_unet = False
    config.mask_to_vae = False
    return config

def get_data_config():
    config = ml_collections.ConfigDict()
    config.data_rootdir = 'examples'

    config.sample_size = '480x832'
    config.dilate_width = 11
    config.max_video_length = 197
    config.fps = 16
    return config


def get_video_model_config():
    config = ml_collections.ConfigDict()
    config.config_path = "config/wan2.1/wan_civitai.yaml"
    config.model_name = "models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"
    config.transformer_path = ""
    config.vae_path = ""
    config.lora_path = ""
    config.use_trimask = True
    config.zero_out_mask_region = False
    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
    config.sampler_name = "Flow"
    config.denoise_strength = 1.0
    config.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    config.guidance_scale = 1.0
    config.num_inference_steps = 50
    config.lora_weight = 1.0
    config.temporal_window_size = 81
    config.temproal_multidiffusion_stride = 16
    config.merge_mask = False
    config.binarize_mask = False
    config.use_vae_mask = False
    return config


def get_system_config():
    config = ml_collections.ConfigDict()
    config.low_gpu_memory_mode = False
    config.weight_dtype = torch.bfloat16
    config.seed = 43
    config.allow_skipping_error = False
    config.device = 'cuda'
    # GPU memory mode, which can be choosen in [model_full_load, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.

    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.

    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
    # and the transformer model has been quantized to float8, which can save more GPU memory.

    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
    # resulting in slower speeds but saving a large amount of GPU memory.
    config.gpu_memory_mode = "model_full_load"
    # Multi GPUs config
    # Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
    # For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
    # If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
    config.ulysses_degree = 1
    config.ring_degree = 1

    #### WAN2.1 ####
    # Temporal multidiffusion is not working with teacache.
    config.enable_teacache = False
    # Recommended to be set between 0.05 and 0.20. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    config.teacache_threshold = 0.10
    # The number of steps to skip TeaCache at the beginning of the inference process, which can
    # reduce the impact of TeaCache on generated video quality.
    config.num_skip_start_steps = 5
    # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
    config.teacache_offload = False
    # Riflex config
    config.enable_riflex = False
    # Index of intrinsic frequency
    config.riflex_k = 6
    return config
