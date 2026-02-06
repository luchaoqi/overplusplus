from absl import app
from absl import flags
from ml_collections import config_flags

from loguru import logger
import json
import os
import sys
import glob
import gc
import cv2
import pprint
import numpy as np
import mediapy as media
import torch
import torch.nn.functional as F
sys.path.append('./')
from omnimatte.optimization import OmnimatteOptimizer
from omnimatte.utils import (
    save_omnimatte,
    refine_mask,
    transfer_detail,
)
from videox_fun.utils.utils import get_video_mask_input, erode_video_mask

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config", "config/default_omnimatte.py", "Path to the python config file"
    )


def read_casper_output(config, seq_name, fg_id):
    source_video_dir = config.omnimatte.source_video_dir
    video_paths = [
        f for f in sorted(glob.glob(os.path.join(source_video_dir, f'{seq_name}-fg={fg_id:02d}-*.mp4')))
        if not f.endswith('tuple.mp4')
    ]
    assert len(video_paths), f'No video found for {seq_name}, fg_id={fg_id}'
    video_path = video_paths[-1]

    video = media.read_video(video_path).astype(float) / 255.
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def run_optimization(config, seq_name, fg_id, num_fgs):
    save_dir = os.path.join(config.experiment.save_path, seq_name, f'fg{fg_id:02d}')
    if config.experiment.skip_if_exists and len(glob.glob(os.path.join(save_dir, '*.png'))) > 0:
        logger.info(f'Skipping {seq_name}, fg_id={fg_id} as {save_dir} exists.')
        return

    logger.info(f'Running optimization: {seq_name}, fg_id: {fg_id}')
    video_length = config.data.max_video_length
    sample_size = config.data.sample_size
    sample_size = tuple(map(int, config.data.sample_size.split('x')))

    input_video, input_mask, prompt, _ = get_video_mask_input(
        seq_name,
        sample_size=sample_size,
        keep_fg_ids=[fg_id],
        max_video_length=video_length,
        apply_temporal_padding=False,
        data_rootdir=config.data.data_rootdir,
        use_trimask=True,
        dilate_width=0,
    )
    # input_video: (1, 3, t, h, w) <float>[0, 1]
    # input_mask: (1, 1, t, h, w) <float>[0, 1]
    input_video = input_video[0].permute(1, 0, 2, 3)  # (t, 3, h, w)
    input_mask = input_mask[0].permute(1, 0, 2, 3)  # (t, 1, h, w)

    def align_video(_video):
        _video = _video[:input_video.shape[0]].to(input_video.device, input_video.dtype)
        assert len(_video) == len(input_video)
        return F.interpolate(_video, size=input_video.shape[-2:], mode='bilinear')

    # determine the video pair to run omnimatte reconstruction
    generated_video = read_casper_output(config, seq_name, fg_id)
    generated_video = align_video(generated_video)

    if num_fgs > 1:
        mask_binary = torch.where(input_mask < 0.25, 1.0, 0.0).to(input_mask.device, input_mask.dtype)  # to preserve
        solo_video = generated_video
        bg_video = read_casper_output(config, seq_name, -1)
        bg_video = align_video(bg_video)
    else:
        mask_binary = torch.where(input_mask > 0.75, 1.0, 0.0).to(input_mask.device, input_mask.dtype)  # to remove
        solo_video = input_video
        bg_video = generated_video

    # resegment when the object is originally occluded.
    # run segmentation on the solo video to obtain a complete object mask
    if config.omnimatte.resegment:
        mask_np = mask_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (t, h, w, 1)
        rgb_np = solo_video.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (t, h, w, 3)
        mask_refined_np = refine_mask(rgb_np, mask_np)[:, None, :, :]  # (t, 1, h, w)
        mask_binary = torch.from_numpy(mask_refined_np).to(input_mask.device, input_mask.dtype)

    # erode object mask a bit in case it is not perfectly accurate
    if config.omnimatte.erode_mask_width:
        mask_np = mask_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (t, h, w, 1)
        mask_np = erode_video_mask(mask_np, config.omnimatte.erode_mask_width).astype(float) / 255.
        mask_np = mask_np.transpose(0, 3, 1, 2)  # (t, 1, h, w)
        mask_binary = torch.from_numpy(mask_np).to(input_mask.device, input_mask.dtype)

    # run optimization
    optimizer = OmnimatteOptimizer(
        config,
        XY=solo_video,
        Y=bg_video,
        init_mask=mask_binary,
        device=config.system.device,
        expname=f'{seq_name}-fg={fg_id}',
    )
    optimization_outputs = optimizer.run()

    optimization_outputs['input_video'] = input_video
    save_omnimatte(
        optimization_outputs, config.experiment.save_path, seq_name, max(0, fg_id), fps=config.data.fps
    )
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()


def main(_):
    config = FLAGS.config
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(config.to_dict())

    torch.manual_seed(config.system.seed)
    np.random.seed(config.system.seed)

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
    for seq in run_seqs:
        fg_ids = [-1]
        num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "mask_*.mp4")))))
        if num_fgs == 0:
            num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "trimask_*.mp4")))))
        assert num_fgs > 0
        if config.experiment.matting_mode == "solo" and num_fgs > 1:
            fg_ids = list(range(num_fgs))

        for fg_id in fg_ids:
            seq_fg_to_run.append((seq, fg_id, num_fgs))

            def _run():
                run_optimization(config=config, seq_name=seq, fg_id=fg_id, num_fgs=num_fgs)

            if config.system.allow_skipping_error:
                try:
                    _run()
                except Exception as e:
                    logger.debug(f'Error in {seq}, {fg_id}: {e}')
                    continue
            else:
                _run()

        if config.omnimatte.detail_transfer:
            if config.system.allow_skipping_error:
                try:
                    transfer_detail(config=config, seq_name=seq, num_fgs=num_fgs)
                except Exception as e:
                    logger.debug(f'Error in detail transfer for {seq}: {e}')
            else:
                transfer_detail(config=config, seq_name=seq, num_fgs=num_fgs)


if __name__ == "__main__":
    app.run(main)
