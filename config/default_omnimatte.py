import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    config.system = get_system_config()
    config.data = get_data_config()
    config.omnimatte = get_omnimatte_config()
    config.experiment = get_experiment_config()
    return config


def get_experiment_config():
    config = ml_collections.ConfigDict()
    config.run_seqs = "davis_dog-agility"
    config.matting_mode = "solo"  # "clean_bg" or "solo"
    config.save_path = "omnimatte_outputs"
    config.skip_if_exists = True
    config.validation = False
    config.skip_unet = False
    config.mask_to_vae = False
    return config


def get_data_config():
    config = ml_collections.ConfigDict()
    config.data_rootdir = 'datasets/test/'

    config.sample_size = '384x672'
    config.dilate_width = 11
    config.max_video_length = 197
    config.fps = 16
    return config


def get_omnimatte_config():
    config = ml_collections.ConfigDict()
    config.source_video_dir = "samples/casper-v1.5-bs=8-gscale=1.0-niters=12000-mask=trilinear_temporalmultidiffusion"
    config.background_video_dir = ""
    config.resegment = True
    config.erode_mask_width = 5

    config.log_dir = 'omnimatte_logs'
    config.freq_log = 50
    config.freq_eval = 100_000

    config.rgb_module_type = "unet"
    config.rgb_lr = 5e-4

    config.alpha_module_type = "unet"
    config.alpha_lr = 1e-3

    config.batch_size = 16
    config.num_steps = 6_000
    config.lr_schedule_milestones = [500, 1_000, 2_000]
    config.lr_schedule_gamma = 0.1

    config.loss_recon_metric = "l2"
    config.loss_mask_super_metric = "l2"
    config.loss_weight_recon = 1.
    config.loss_weight_alpha_reg_l0 = 0.075
    config.loss_weight_alpha_reg_l1 = 0.75
    config.loss_weight_alpha_reg_l0_k = 5.
    config.loss_weight_mask_super = 10.
    config.loss_weight_mask_super_ones = 0.1
    config.loss_weight_smoothness = 0.0

    config.loss_weight_alpha_reg_l0_steps = [1_000, 1_500]
    config.loss_weight_alpha_reg_l1_steps = [1_000, 1_500]
    config.loss_weight_mask_super_steps = [1_000]
    config.loss_weight_mask_super_ones_steps = []
    config.loss_weight_smoothness_steps = []

    config.loss_weight_alpha_reg_l0_gamma = 1.
    config.loss_weight_alpha_reg_l1_gamma = 1.
    config.loss_weight_mask_super_gamma = 0.
    config.loss_weight_smoothness_gamma = 1.
    config.loss_weight_mask_super_ones_gamma = 1.

    config.detail_transfer = True
    config.composite_order = "0,1"
    config.detail_transfer_transmission_thresh = 0.98
    config.detail_transfer_use_input_mask = True
    return config


def get_system_config():
    config = ml_collections.ConfigDict()
    config.low_gpu_memory_mode = False
    config.weight_dtype = torch.bfloat16
    config.seed = 43
    config.allow_skipping_error = False
    config.device = 'cuda'

    return config
