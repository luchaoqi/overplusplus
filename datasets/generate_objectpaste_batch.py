import os
import sys
import glob
import argparse
import json
import tqdm
import cv2 as cv
import numpy as np
import mediapy as media
import torch
import torch.nn.functional as F
import torchvision.transforms
from decord import VideoReader, cpu

parser = argparse.ArgumentParser()
parser.add_argument("--source_rootdir", type=str, default="PATH/TO/YOUR/DIR/OF/VIDEOS.mp4")
parser.add_argument("--output_dir", type=str, default="outputs/")
parser.add_argument("--num_tuples", type=int, default=1024)
parser.add_argument("--prob_add_preservation_mask", type=float, default=0)
parser.add_argument("--mask_stationary_ratio_thresh", type=float, default=0.05)
parser.add_argument("--mask_mean_ratio_thresh", type=float, default=0.05)
parser.add_argument("--mask_max_ratio_thresh", type=float, default=0.8)
parser.add_argument("--random_scale_range", type=float, nargs=2, default=[0.4, 1.0])
parser.add_argument("--random_shift_range", type=float, nargs=2, default=[-0.2, 0.2])
parser.add_argument("--save_index_offset", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_video_length", type=int, default=256)
args = parser.parse_args()


@torch.no_grad()
def resize_video(video, size, batch_size=8):
    video_resized = []
    i = 0
    while i < len(video): 
        i_end = min(i + batch_size, len(video))
        batch_resized = F.interpolate(
            torch.from_numpy(video[i:i_end]).permute(0, 3, 1, 2),
            size,
            mode='bilinear',
        ).permute(0, 2, 3, 1).detach().cpu().numpy()
        i = i_end
        video_resized.append(batch_resized)
    video_resized = np.concatenate(video_resized, axis=0)
    return video_resized


def align_dimensions(video1, video2, video3, video4):
    min_len = min(len(video1), len(video2), len(video3), len(video4))
    video1 = video1[:min_len]
    video2 = video2[:min_len]
    video3 = video3[:min_len]
    video4 = video4[:min_len]
    if video1.shape[1] != video2.shape[1] or video1.shape[2] != video2.shape[2]:
        video2 = resize_video(video2, (video1.shape[1], video1.shape[2]))
    if video1.shape[1] != video3.shape[1] or video1.shape[2] != video3.shape[2]:
        video3 = resize_video(video3, (video1.shape[1], video1.shape[2]))
    if video1.shape[1] != video4.shape[1] or video1.shape[2] != video4.shape[2]:
        video4 = resize_video(video4, (video1.shape[1], video1.shape[2]))
    return video1, video2, video3, video4

@torch.no_grad()
def random_scale_shift(video, mask):
    _, h, w, _ = video.shape
    scale = np.random.uniform(args.random_scale_range[0], args.random_scale_range[1])
    h_shift = np.random.uniform(args.random_shift_range[0], args.random_shift_range[1]) * h
    w_shift = np.random.uniform(args.random_shift_range[0], args.random_shift_range[1]) * w

    video_transformed = []
    mask_transformed = []
    i = 0
    while i < len(video):
        i_end = min(i + 8, len(video))
        video_transformed.append(
            torchvision.transforms.functional.affine(
                torch.from_numpy(video[i:i_end]).permute(0, 3, 1, 2),
                angle=0,
                translate=(w_shift, h_shift),
                scale=scale,
                shear=0,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                fill=0,
            ).detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
        mask_transformed.append(
            torchvision.transforms.functional.affine(
                torch.from_numpy(mask[i:i_end]).permute(0, 3, 1, 2),
                angle=0,
                translate=(w_shift, h_shift),
                scale=scale,
                shear=0,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                fill=0,
            ).detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
        i = i_end
    video_transformed = np.concatenate(video_transformed)
    mask_transformed = np.concatenate(mask_transformed)
    return video_transformed, mask_transformed


def random_read_video_clip(video_path, mask_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    max_stride = max(1, num_frames // args.max_video_length)
    frame_indices = list(range(0, num_frames, max_stride))
    if len(frame_indices) > args.max_video_length:
        rand_offset = np.random.randint(0, len(frame_indices) - args.max_video_length)
        frame_indices = frame_indices[rand_offset:rand_offset + args.max_video_length]
    video = vr.get_batch(frame_indices).asnumpy()
    video = video.astype(float) / 255.
    del vr

    vr_mask = VideoReader(mask_path, ctx=cpu(0))
    mask = vr_mask.get_batch(frame_indices).asnumpy()
    mask = mask.astype(float)
    del vr_mask

    if len(mask.shape) == 3:
        mask = mask[..., None]
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, 3, axis=-1)
    return video, mask


# def read_caption(video_path):
#     caption_json = os.path.join(args.source_rootdir, "captions.json")
#     with open(caption_json, 'r') as f:
#         captions = json.load(f)
#     video_name = os.path.basename(video_path).split('_')[0]
#     if video_name in captions:
#         caption = captions[video_name]['full']
#     else:
#         caption = "a beautiful scene"
#     return caption

def read_caption(video_path):
    caption_json = os.path.join(args.source_rootdir, "captions.json")
    with open(caption_json, 'r') as f:
        captions = json.load(f)
    # video_name = os.path.basename(video_path).split('_')[0]
    video_name = os.path.basename(video_path).split('.')[0]
    # if video_name in captions:
    #     caption = captions[video_name]['full']
    # else:
    #     caption = "a beautiful scene"
    # return caption
    for caption in captions:
        if caption['path'] == video_name:
            return caption['prompt']
    return "a beautiful scene"


# def make_one_tuple(all_video_list):
#     idx = np.random.randint(0, len(all_video_list))
#     print(f'[DEBUG] selected video {idx}, reading video and mask')
#     video_keep, mask_keep = random_read_video_clip(
#         all_video_list[idx]['video'], all_video_list[idx]['mask']
#     )
#     caption = read_caption(all_video_list[idx]['video'])

#     for attemp_i in tqdm.tqdm(range(100)):
#         idx1 = np.random.randint(0, len(all_video_list))
#         if idx1 == idx: continue
#         video_rm, mask_rm = random_read_video_clip(
#             all_video_list[idx1]['video'], all_video_list[idx1]['mask']
#         )

#         video_keep, video_rm, mask_keep, mask_rm = align_dimensions(video_keep, video_rm, mask_keep, mask_rm)
#         video_rm, mask_rm = random_scale_shift(video_rm, mask_rm)
#         mask_rm = np.where(mask_rm > 0.5, 1, 0)

#         input_video = video_rm * mask_rm + video_keep * (1 - mask_rm)
#         mean_mask_ratio = np.mean(video_rm)
#         max_mask_ratio = np.max(video_rm.mean(axis=(1, 2, 3)))
#         stationary_mask_ratio = (np.mean(video_rm.mean(axis=(0, 3))) > 0.1).mean()

#         if stationary_mask_ratio < args.mask_stationary_ratio_thresh: continue
#         if mean_mask_ratio < args.mask_mean_ratio_thresh: continue
#         if max_mask_ratio > args.mask_max_ratio_thresh: continue

#         if np.random.rand() < args.prob_add_preservation_mask:
#             background_value = 127
#             trimask = np.where(mask_keep > 0.5, 0, background_value).astype(np.uint8)
#             trimask = np.where(mask_rm > 0.5, 255, trimask).astype(np.uint8)
#         else:
#             background_value = 127 if np.random.rand() < 0.5 else 0
#             trimask = np.where(mask_rm > 0.5, 255, background_value).astype(np.uint8)
#         return trimask, input_video, video_keep, caption

#     return None, None, None, None  # failed to generate a valid tuple


def make_one_tuple(all_video_list):
    idx = np.random.randint(0, len(all_video_list))
    print(f'[DEBUG] selected video {idx}, reading video and mask')
    video_keep, mask_keep = random_read_video_clip(
        all_video_list[idx]['video'], all_video_list[idx]['mask']
    )
    # caption = read_caption(all_video_list[idx]['video'])

    for attemp_i in tqdm.tqdm(range(100)):
        idx1 = np.random.randint(0, len(all_video_list))
        if idx1 == idx: continue
        video_rm, mask_rm = random_read_video_clip(
            all_video_list[idx1]['video'], all_video_list[idx1]['mask']
        )

        video_keep, video_rm, mask_keep, mask_rm = align_dimensions(video_keep, video_rm, mask_keep, mask_rm)
        video_rm, mask_rm = random_scale_shift(video_rm, mask_rm)
        mask_rm = np.where(mask_rm > 0.5, 1, 0)

        input_video = video_rm * mask_rm + video_keep * (1 - mask_rm)
        mean_mask_ratio = np.mean(video_rm)
        max_mask_ratio = np.max(video_rm.mean(axis=(1, 2, 3)))
        stationary_mask_ratio = (np.mean(video_rm.mean(axis=(0, 3))) > 0.1).mean()

        if stationary_mask_ratio < args.mask_stationary_ratio_thresh: continue
        if mean_mask_ratio < args.mask_mean_ratio_thresh: continue
        if max_mask_ratio > args.mask_max_ratio_thresh: continue

        # if np.random.rand() < args.prob_add_preservation_mask:
        #     background_value = 127
        #     trimask = np.where(mask_keep > 0.5, 0, background_value).astype(np.uint8)
        #     trimask = np.where(mask_rm > 0.5, 255, trimask).astype(np.uint8)
        # else:
        #     background_value = 127 if np.random.rand() < 0.5 else 0
        #     trimask = np.where(mask_rm > 0.5, 255, background_value).astype(np.uint8)
        # return trimask, input_video, video_keep, caption
        background_value = 255
        trimask = np.where(mask_rm > 0.5, 0, background_value).astype(np.uint8)
        # TODO: handle video_rm and video_keep for the caption that we want to use at the end for over++ training
        return trimask, input_video, video_keep, idx, idx1

    return None, None, None, None  # failed to generate a valid tuple


def main():
    np.random.seed(args.seed)
    all_video_list = []
    # video_paths = sorted(glob.glob(os.path.join(args.source_rootdir, "videos/*.mp4")))
    # for video_path in video_paths:
    #     mask_path = os.path.join(args.source_rootdir, "masks", os.path.basename(video_path))
    #     if os.path.exists(mask_path):
    #         all_video_list.append({
    #             'video': video_path,
    #             'mask': mask_path,
    #         })

    # for i in tqdm.tqdm(range(args.save_index_offset, args.num_tuples + args.save_index_offset)):
    #     trimask, input_video, target_video, caption = make_one_tuple(all_video_list)
    #     if trimask is None: continue

    #     # flip the label: 255 = preserve, 0 = remove
    #     trimask = 255 - trimask

    #     output_dir = os.path.join(args.output_dir, f"{i:06d}")
    #     os.makedirs(output_dir, exist_ok=True)
    #     media.write_video(os.path.join(output_dir, "mask.mp4"), trimask)
    #     media.write_video(os.path.join(output_dir, "rgb_full.mp4"), input_video)
    #     media.write_video(os.path.join(output_dir, "rgb_removed.mp4"), target_video)
    #     data_tuple = np.concatenate([input_video, trimask.astype(float) / 255., target_video], axis=-2)
    #     media.write_video(os.path.join(output_dir, "tuple.mp4"), data_tuple)

    #     with open(os.path.join(output_dir, "caption.txt"), 'w') as f:
    #         f.write(caption)
    #     print(f"[DEBUG] saved {i}th tuple to {output_dir}")



    # ------------------------------- over++ format ------------------------------ #
    subdirs = sorted(os.listdir(args.source_rootdir))
    video_paths = sorted(os.path.join(args.source_rootdir, subdir, "rgb_removed.mp4") for subdir in subdirs)
    for video_path in video_paths:
        subdir = os.path.dirname(video_path)
        mask_path = os.path.join(args.source_rootdir, subdir, "mask.mp4")
        if os.path.exists(mask_path):
            all_video_list.append({
                'video': video_path,
                'mask': mask_path,
            })

    for i in tqdm.tqdm(range(args.save_index_offset, args.num_tuples + args.save_index_offset)):
        # trimask, input_video, target_video, caption = make_one_tuple(all_video_list)
        trimask, input_video, target_video, idx, idx1 = make_one_tuple(all_video_list)
        video_keep_path = all_video_list[idx]['video']
        video_rm_path = all_video_list[idx1]['video']
        if trimask is None: continue

        # flip the label: 255 = preserve, 0 = remove
        trimask = 255 - trimask

        # output_dir = os.path.join(args.output_dir, f"{i:06d}")
        # os.makedirs(output_dir, exist_ok=True)
        # media.write_video(os.path.join(output_dir, "mask.mp4"), trimask)
        # media.write_video(os.path.join(output_dir, "rgb_full.mp4"), input_video)
        # media.write_video(os.path.join(output_dir, "rgb_removed.mp4"), target_video)
        # data_tuple = np.concatenate([input_video, trimask.astype(float) / 255., target_video], axis=-2)
        # media.write_video(os.path.join(output_dir, "tuple.mp4"), data_tuple)

        # with open(os.path.join(output_dir, "caption.txt"), 'w') as f:
        #     f.write(caption)
        # print(f"[DEBUG] saved {i}th tuple to {output_dir}")

        output_name = os.path.dirname(video_rm_path).split('/')[-1]
        output_dir = os.path.join(args.output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        # youtube vos fps
        fps = 7
        media.write_video(os.path.join(output_dir, "mask.mp4"), trimask, fps=fps) 
        media.write_video(os.path.join(output_dir, "rgb_removed.mp4"), input_video, fps=fps)
        media.write_video(os.path.join(output_dir, "rgb_full.mp4"), target_video, fps=fps)
        data_tuple = np.concatenate([input_video, trimask.astype(float) / 255., target_video], axis=-2)
        media.write_video(os.path.join(output_dir, "tuple.mp4"), data_tuple, fps=fps)

if __name__ == "__main__":
    main()
