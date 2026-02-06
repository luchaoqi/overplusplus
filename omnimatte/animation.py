import glob
import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import mediapy as media
import cv2 as cv
import kornia

sys.path.append('./')
from omnimatte.utils import get_alpha_transmission, composite_layers


@torch.no_grad()
def warp(video, points, out_hw, batch_size=8):
    '''
    Input:
        video: torch.tensor [vid_len, ch, in_h, in_w]
        points: torch.tensor [vid_len, 4, 2]  (top-left, top-right, bottom-right, bottom-left)
        out_hw: (out_h, out_w)
        batch_size: int. number ofr frames to process in parallel at once
    '''
    vid_len, ch, in_h, in_w = video.shape

    batch_begin = 0
    video_warped = []
    while batch_begin < vid_len:
        batch_end = min(batch_begin + batch_size, vid_len)

        points_dst = points[batch_begin:batch_end]  # [B, 4, 2]
        b = points_dst.shape[0]
        points_src = torch.FloatTensor(
            [[0, 0], [in_w - 1, 0], [in_w - 1, in_h - 1], [0, in_h - 1]]
        ).repeat(b, 1, 1).to(points_dst.device)  # [B, 4, 2]

        homo = kornia.geometry.homography.find_homography_dlt(points_src, points_dst, solver='svd')

        video_warped_batch = kornia.geometry.transform.warp_perspective(
            video[batch_begin:batch_end].to(points_dst.device),
            homo,
            out_hw,
            mode='bilinear',
            padding_mode='zeros',
        )

        video_warped.append(video_warped_batch.cpu())

        batch_begin = batch_end

    return torch.cat(video_warped)


@torch.no_grad()
def warp_layers(layers, layer_corners, out_hw):
    '''
    Input:
        layers: torch.tensor (n_layers, t, 4, h, w) <float>[0, 1]
        layer_corners: torch.tensor (n_layers, t, 4, 2)
    Return:
        video: torch.tensor (t, 3, h, w) <float>[0, 1]
    '''
    warped_layers = []
    for layer, layer_corner in zip(layers, layer_corners):
        warped_layer = warp(layer.cpu(), layer_corner, out_hw)
        warped_layers.append(warped_layer)

    warped_layers = torch.stack(warped_layers)  # (n_layers, t, c, out_h, out_w)
    composed_video = composite_layers(
        torch.cat([warped_layers, torch.ones_like(warped_layers[:1])])  # add a white canvas background
    )
    return composed_video


@torch.no_grad()
def animate_splitting(
    frame_layers,
    layers_corners,
    out_hw,
    split_delay=0.1,
    split_duration=0.5,
    fps=16,
):
    '''
    Input:
        frame_layers: torch.tensor <float>[0, 1] (n_layers, 4, h, w)
        layers_corners: torch.tensor <float> (n_layers, 4, 2)
    Return:
        animation: torch.tensor <float>[0, 1] (n_frames, 3, h, w)
    '''
    n_layers = len(frame_layers)
    animated_layer_corners = []

    n_animation_frames = int((split_delay + split_duration) * fps * (n_layers - 1))
    for i in range(n_layers - 1, -1, -1):  # from back to front
        if not animated_layer_corners:
            weight = torch.ones(n_animation_frames).to(layers_corners.device)
            animated_layer_corners.append(
                layers_corners[i][None] * weight[:, None, None]  # [n_frames, 4, 2]
            )
        else:
            prev_layer_corners = animated_layer_corners[-1]
            weight = torch.sin(torch.linspace(0, np.pi / 2, int(split_duration * fps)))
            shift_offset = layers_corners[i] - layers_corners[i + 1]  # [4, 2]
            split_begin = int((n_layers - 2 - i) * (split_delay + split_duration) * fps + split_delay * fps)
            split_end = split_begin + len(weight)
            weight_all = torch.zeros(n_animation_frames)
            weight_all[split_begin:split_end] = weight
            weight_all[split_end:] = 1.0
            weight_all = weight_all.reshape(-1).to(layers_corners.device)
            curr_layer_corners = prev_layer_corners + shift_offset[None] * weight_all[:, None, None]
            animated_layer_corners.append(curr_layer_corners)

    animated_layer_corners = torch.stack(animated_layer_corners[::-1])  # [n_layers, n_frames, 4, 2]  front to back
    composed_video = warp_layers(
        frame_layers.unsqueeze(1).expand(-1, n_animation_frames, -1, -1, -1),
        animated_layer_corners,
        out_hw
    )  # [n_frames, 3, out_h, out_w]

    return composed_video

@torch.no_grad()
def animate_moving(
    frame_layers, start_layers_corners, end_layers_corners, out_hw, moving_duration=1.0, fps=16, fade_in=True
):
    '''
    Input:
        frame_layers: torch.tensor <float>[0, 1] (n_layers, 4, h, w)
        start_layers_corners: torch.tensor <float> (n_layers, 4, 2)
        end_layers_corners: torch.tensor <float> (n_layers, 4, 2)
    Return:
        animation: torch.tensor <float>[0, 1] (n_frames, 3, h, w)
    '''
    n_layers = len(frame_layers)

    n_animation_frames = int(moving_duration * fps)
    weight = torch.sin(torch.linspace(0, np.pi / 2, n_animation_frames)).to(start_layers_corners.device)
    animated_layers_corners = (
        start_layers_corners[:, None, :, :] * (1 - weight[None, :, None, None]) +
        end_layers_corners[:, None, :, :] * weight[None, :, None, None]
    )
    composed_video = warp_layers(
        frame_layers.unsqueeze(1).expand(-1, n_animation_frames, -1, -1, -1),
        animated_layers_corners,
        out_hw
    )
    if fade_in:
        white_bg = torch.ones_like(composed_video[0:1])
        fade_in_weight = torch.sin(torch.linspace(0, np.pi / 2, n_animation_frames)).to(frame_layers.device)[:, None, None, None]
        composed_video = composed_video * fade_in_weight + white_bg * (1 - fade_in_weight)
    return composed_video

@torch.no_grad()
def resize_video(video, out_hw, batch_size=8):
    resized_video = []
    batch_begin = 0
    while batch_begin < video.shape[0]:
        batch_end = min(batch_begin + batch_size, video.shape[0])
        resized_video_batch = F.interpolate(
            video[batch_begin:batch_end], out_hw, mode='bilinear', align_corners=False
        )
        resized_video.append(resized_video_batch.cpu())
        batch_begin = batch_end
    resized_video = torch.cat(resized_video)
    return resized_video


@torch.no_grad()
def visualize_layers(
    vid_in,
    layers,
    save_path='layer_vis.mp4',
    fps=16,
    device='cuda',
    padding=20,
    layer_tilt_deg=8,
    layer_shift_x=-120,
    layer_shift_y=120,
    border_width=2,
    border_color=0.5,
    space = 100,
    add_moving_animation=True,
    add_splitting_animation=True,
    pause_time=0,
    split_delay=0.08,
    split_duration=0.6,
    moving_duration=1.0,
):
    """
    Input:
        vid_in: np.array <float>[0, 1] (t, h, w, 3)
        layers: np.array <float>[0, 1] (n_layers, t, h, w, 4)
        save_path: str
        fps: int
        device: 'cuda' or 'cpu'
    """
    vid_in = torch.FloatTensor(vid_in).permute(0, 3, 1, 2)

    layers = torch.FloatTensor(layers).permute(0, 1, 4, 2, 3)  # (n, t, c, h, w)
    if border_width > 0:
        layers = F.pad(
            layers, (border_width, border_width, border_width, border_width),
            mode='constant', value=border_color
        )
        layers[:, :, -1, :border_width, :] = 1.0
        layers[:, :, -1, -border_width:, :] = 1.0
        layers[:, :, -1, :, :border_width] = 1.0
        layers[:, :, -1, :, -border_width:] = 1.0

    n_layers, n_frames, _, height, width = layers.shape
    assert n_frames == vid_in.shape[0], "The number of frames in layers and input video should be the same."
    if vid_in.shape[2] != height or vid_in.shape[3] != width:
        vid_in = resize_video(vid_in, (height, width))

    w_tilt = width * np.cos(layer_tilt_deg / 180. * np.pi)
    h_tilt = width * np.sin(layer_tilt_deg / 180. * np.pi)
    layers_corners = []
    for i in range(n_layers):
        offset_x = i * layer_shift_x
        offset_y = i * layer_shift_y
        layers_corners.append([
            [offset_x, offset_y],
            [offset_x + w_tilt, offset_y + h_tilt],
            [offset_x + w_tilt, offset_y + height + h_tilt],
            [offset_x, offset_y + height],
        ])

    layers_corners = layers_corners[::-1]  # reverse the order of layers to match the order of layers. Now is from front to back
    layers_corners = torch.FloatTensor(layers_corners).to(device)  # (n_layers, 4, 2)
    layers_corners[:, :, 0] -= torch.min(layers_corners[:, :, 0])
    layers_corners[:, :, 1] -= torch.min(layers_corners[:, :, 1])

    layers_w = int(torch.max(layers_corners[:, :, 0])) + 1
    layers_h = int(torch.max(layers_corners[:, :, 1])) + 1

    vid_in_offset_x = padding
    vid_in_offset_y = padding + layers_h // 2 - height // 2
    vid_in_corners = torch.tensor([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).to(device)
    vid_in_corners[:, 0] += vid_in_offset_x
    vid_in_corners[:, 1] += vid_in_offset_y

    out_h = layers_h + padding * 2
    out_w = layers_w + padding * 2 + width + space
    layers_corners[:, :, 0] += padding + width + space
    layers_corners[:, :, 1] += padding

    with media.VideoWriter(save_path, (out_h, out_w), fps=fps) as video_writer:
        def _write_frames(frames):
            for frame in frames:
                frame = np.uint8(frame.permute(1, 2, 0).cpu().numpy() * 255.)
                video_writer.add_image(frame)
            del frames
            gc.collect()
            torch.cuda.empty_cache()

        def overlay_input_video(canvas, video, frame_begin, frame_end):
            canvas[:, :, vid_in_offset_y:vid_in_offset_y + height, vid_in_offset_x:vid_in_offset_x + width] = video[frame_begin:frame_end]
            return canvas

        if pause_time > 0:
            video_beginning = torch.ones((pause_time, 3, out_h, out_w))
            video_beginning = overlay_input_video(video_beginning, vid_in, 0, pause_time)
            _write_frames(video_beginning)

        if add_moving_animation:
            video_moving = animate_moving(
                layers[:, pause_time, :, :, :],
                vid_in_corners[None].expand(n_layers, -1, -1),
                layers_corners[-1:].expand(n_layers, -1, -1).to(device),
                (out_h, out_w),
                moving_duration=moving_duration,
                fps=fps,
            )
            video_moving = overlay_input_video(video_moving, vid_in, pause_time, pause_time + 1)
            _write_frames(video_moving)

        if add_splitting_animation:
            video_splitting = animate_splitting(
                layers[:, pause_time, :, :, :],
                layers_corners,
                (out_h, out_w),
                split_delay=split_delay,
                split_duration=split_duration,
                fps=fps,
            )
            video_splitting = overlay_input_video(video_splitting, vid_in, pause_time, pause_time + 1)
            _write_frames(video_splitting)

        # visualizing layers
        video_layers = warp_layers(
            layers[:, pause_time:], layers_corners.unsqueeze(1).expand(-1, n_frames - pause_time, -1, -1), (out_h, out_w)
        )
        video_layers = overlay_input_video(video_layers, vid_in, pause_time, n_frames)
        _write_frames(video_layers)


def read_layer_from_frames(_dir):
    frame_paths = sorted(glob.glob(os.path.join(_dir, '*.png')))

    def _read_frame_ch4(path):
        frame = media.read_image(path).astype(float) / 255.
        if frame.shape[-1] == 3:
            frame = np.concatenate((frame, np.ones_like(frame[:, :, :1])), axis=-1)
        return frame
    return np.stack([_read_frame_ch4(p) for p in frame_paths])
