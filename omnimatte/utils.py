import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapy as media
import gc
import glob
import uuid
from tqdm import tqdm
from loguru import logger
# from sam2.build_sam import build_sam2_video_predictor
import sys
sam2_dir = '/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/sam2'
sys.path.append(sam2_dir)
from sam2.build_sam import build_sam2_video_predictor

from videox_fun.utils.utils import apply_colormap, get_video_mask_input

def inv_sigmoid(x):
    return torch.nan_to_num(-torch.log(1 / (x + 1e-8) - 1))


def make_checkerboard(height, width, color1=215, color2=174, size=24):
    canvas = np.zeros((height, width, 3)).astype(np.uint8)
    for row in range(int(np.ceil(height / size))):
        for col in range(int(np.ceil(width / size))):
            r1, r2 = row * size, min((row + 1) * size, height)
            c1, c2 = col * size, min((col + 1) * size, width)
            if r1 >= height and c1 >= width: continue
            color = color1 if ((row % 2) + (col % 2)) % 2 == 0 else color2
            canvas[r1:r2, c1:c2] = color
    return canvas


def place_over_checker_background(X_rgb, X_alpha, color1=215, color2=174, size=24):
    bg = make_checkerboard(X_rgb.shape[1], X_rgb.shape[2], color1, color2, size).astype(X_rgb.dtype) / 255.
    bg = bg[None].repeat(X_rgb.shape[0], axis=0)
    return X_rgb * X_alpha + bg * (1 - X_alpha)


def place_over_constant_background(X_rgb, X_alpha, color=1.):
    bg = np.ones_like(X_rgb) * color
    return X_rgb * X_alpha + bg * (1 - X_alpha)


def visualize_omnimatte(pkg, save_path, fps=16):
    row = []
    if 'XY' in pkg:
        row.append(pkg['XY'].cpu().numpy().transpose(0, 2, 3, 1))
    if 'Y' in pkg:
        row.append(pkg['Y'].cpu().numpy().transpose(0, 2, 3, 1))
    if 'mask' in pkg:
        row.append(pkg['mask'].cpu().numpy().transpose(0, 2, 3, 1).repeat(3, axis=-1))

    if 'composited' in pkg:
        row.append(pkg['composited'].cpu().numpy().transpose(0, 2, 3, 1))

    if 'X_rgb' in pkg and 'X_alpha' in pkg:
        X_rgb = pkg['X_rgb'].cpu().numpy().transpose(0, 2, 3, 1)
        X_alpha = pkg['X_alpha'].cpu().numpy().transpose(0, 2, 3, 1).repeat(3, axis=-1)
        row.append(X_rgb)
        row.append(place_over_constant_background(X_rgb, X_alpha))
        row.append(place_over_checker_background(X_rgb, X_alpha))
        row.append(X_alpha)

    elif 'X_alpha' in pkg:
        X_alpha = pkg['X_alpha'].cpu().numpy().transpose(0, 2, 3, 1).repeat(3, axis=-1)
        row.append(X_alpha)

    if 'err' in pkg:
        row.append(pkg['err'].cpu().numpy().transpose(0, 2, 3, 1))

    row = np.concatenate(row, axis=-2)
    if len(row) > 1:
        media.write_video(save_path, row, fps=fps)
    else:
        media.write_image(save_path[:-4] + '.png', row[0])


@torch.no_grad()
def get_alpha_transmission(alphas):
    '''
    get alpha transmission from back to front
    Input:
        alphas: torch.tensor (n_layers, n_frames, 1, h, w) <float>[0, 1]. The first layer is the nearest to the camera
    Return:
        transmission: torch.tensor (n_layers, n_frames, 1, h, w)
    '''
    one_minus_alphas = 1. - alphas
    alpha_cumprod = torch.cumprod(one_minus_alphas, dim=0)
    transmission = alphas.clone()
    transmission[1:] *= alpha_cumprod[:-1]
    return transmission.clamp(0, 1)


@torch.no_grad()
def composite_layers(layers):
    '''
    composite layers from back to front
    Input:
        layers: torch.tensor (n_layers, n_frames, 4, h, w) <float>[0, 1]. The first layer is the nearest to the camera
    Return:
        composed_layers: torch.tensor (n_frames, 3, h, w)
    '''
    alpha_transmit = get_alpha_transmission(layers[:, :, -1:, :, :])  # (n_frames, n_layers, 1, h, w)
    composed = (layers[:, :, :-1, :, :] * alpha_transmit).sum(0)  # (n_frames, 3, h, w)
    return composed


@torch.no_grad()
def transfer_detail(config, seq_name, num_fgs):
    logger.info(f'Transferring detail for {seq_name}')

    # TODO: video depth estimation to automatically determin the composite order
    if config.omnimatte.composite_order and len(config.omnimatte.composite_order.split(',')) == num_fgs:
        composite_order = [int(i) for i in config.omnimatte.composite_order.split(',')]  # front to back
    else:
        composite_order = list(range(num_fgs))
    logger.info(f'Detail transfer - Composite order: {composite_order}')

    save_dir = os.path.join(config.experiment.save_path, seq_name)
    assert os.path.exists(save_dir), f'{save_dir} does not exist'

    device = config.system.device
    video_length = config.data.max_video_length
    sample_size = config.data.sample_size
    sample_size = tuple(map(int, config.data.sample_size.split('x')))

    input_masks = []
    for fg_id in composite_order:
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
        if num_fgs > 1:
            mask_binary = torch.where(input_mask < 0.25, 1.0, 0.0).to(input_mask.device, input_mask.dtype)  # to preserve
        else:
            mask_binary = torch.where(input_mask > 0.75, 1.0, 0.0).to(input_mask.device, input_mask.dtype)  # to remove
        input_masks.append(mask_binary)

    input_video = input_video.to(device)  # (1, 3, t, h, w)
    input_masks = torch.cat(input_masks).to(device)  # (num_fgs, 1, t, h, w)

    def _read_layers(frame_idx):
        layers = []
        for fg_id in composite_order:
            layer_path = os.path.join(save_dir, f'fg{fg_id:02d}', f'{frame_idx:05d}.png')
            layers.append(media.read_image(layer_path))

        bg_path = os.path.join(save_dir, 'bg', f'{frame_idx:05d}.png')
        bg = media.read_image(bg_path)
        bg = np.concatenate([bg, np.ones_like(bg[..., :1])], axis=-1)  # add alpha channel
        layers.append(bg)

        layers = np.stack(layers, axis=0)  # (num_fgs + 1, h, w, 4)
        layers = torch.from_numpy(layers).permute(0, 3, 1, 2)  # (num_fgs + 1, 4, h, w)
        layers = layers.float() / 255.
        return layers

    for i in range(num_fgs):
        dt_fg_dir = os.path.join(save_dir, f'dt_fg{i:02d}')
        os.makedirs(dt_fg_dir, exist_ok=True)

    thresh = config.omnimatte.detail_transfer_transmission_thresh
    for frame_idx in tqdm(range(input_video.shape[2])):
        layers = _read_layers(frame_idx).to(device)  # (num_fgs + 1, 4, h, w)

        composed_alphas = get_alpha_transmission(layers[:, None, -1:, :, :]).squeeze(1)  # (num_fgs + 1, 1, h, w)
        for fg_i in range(num_fgs):
            can_transfer = (composed_alphas[fg_i] > thresh).float()
            if config.omnimatte.detail_transfer_use_input_mask:
                can_transfer = (can_transfer + input_masks[fg_i, :, frame_idx]).clamp(0, 1)
            dt_layer = layers[fg_i].clone()
            dt_layer[:3] = input_video[0, :, frame_idx] * can_transfer + dt_layer[:3] * (1. - can_transfer)

            media.write_image(
                os.path.join(save_dir, f'dt_fg{composite_order[fg_i]:02d}', f'{frame_idx:05d}.png'),
                dt_layer.permute(1, 2, 0).cpu().numpy(),
            )

    # write videos
    for fg_id in range(num_fgs):
        rgba_paths = sorted(glob.glob(os.path.join(save_dir, f'dt_fg{fg_id:02d}', '*.png')))
        rgba = np.stack([media.read_image(p) for p in rgba_paths], axis=0)  # (t, h, w, 4)
        rgba = rgba.astype(float) / 255.
        rgba_over_checker = place_over_checker_background(rgba[..., :3], rgba[..., 3:])
        rgba_over_constant = place_over_constant_background(rgba[..., :3], rgba[..., 3:])
        media.write_video(
            os.path.join(save_dir, f'dt_fg{fg_id:02d}_rgba_checker.mp4'),
            rgba_over_checker,
            fps=config.data.fps,
        )
        media.write_video(
            os.path.join(save_dir, f'dt_fg{fg_id:02d}_rgba_constant.mp4'),
            rgba_over_constant,
            fps=config.data.fps,
        )


def save_omnimatte(pkg, save_path, seq_name, fg_id, fps=16):
    save_dir = os.path.join(save_path, seq_name)
    os.makedirs(save_dir, exist_ok=True)
    fg_row = []
    solo = pkg['XY'].cpu().numpy().transpose(0, 2, 3, 1)
    bg = pkg['Y'].cpu().numpy().transpose(0, 2, 3, 1)
    X_rgb = pkg['X_rgb'].cpu().numpy().transpose(0, 2, 3, 1)
    X_alpha = pkg['X_alpha'].cpu().numpy().transpose(0, 2, 3, 1)
    fg_row.append(solo)
    fg_row.append(bg)

    composition = pkg['X_rgb'] * pkg['X_alpha'] + pkg['Y'] * (1 - pkg['X_alpha'])
    fg_row.append(composition.cpu().numpy().transpose(0, 2, 3, 1))

    fg_row.append(X_rgb)
    X_alpha_repeat = X_alpha.repeat(3, axis=-1)
    X_rgba_over_constant = place_over_constant_background(X_rgb, X_alpha_repeat)
    X_rgba_over_checker = place_over_checker_background(X_rgb, X_alpha_repeat)
    fg_row.append(X_rgba_over_constant)
    fg_row.append(X_rgba_over_checker)
    fg_row.append(X_alpha_repeat)

    with torch.no_grad():
        error = torch.abs(composition - pkg['XY']).mean(1, keepdim=True)
        error = error.detach().cpu().numpy().transpose(0, 2, 3, 1)
        error = apply_colormap(error)
        fg_row.append(error)

    video_row = np.concatenate(fg_row, axis=-2)

    if len(video_row) > 1:
        media.write_video(
            os.path.join(save_dir, f'fg{fg_id:02d}_visualization.mp4'),
            video_row,
            fps=fps,
        )
        media.write_video(
            os.path.join(save_dir, f'fg{fg_id:02d}_rgba_checker.mp4'),
            X_rgba_over_checker,
            fps=fps,
        )
        media.write_video(
            os.path.join(save_dir, f'fg{fg_id:02d}_rgba_constant.mp4'),
            X_rgba_over_constant,
            fps=fps,
        )
        media.write_video(
            os.path.join(save_dir, f'fg{fg_id:02d}_alpha.mp4'),
            X_alpha_repeat,
            fps=fps,
        )
        bg_path = os.path.join(save_dir, 'bg.mp4')
        if not os.path.exists(bg_path):
            media.write_video(
                bg_path,
                bg,
                fps=fps,
            )

    # TODO save per frame png
    bg_dir = os.path.join(save_dir, 'bg')
    os.makedirs(bg_dir, exist_ok=True)
    if len(glob.glob(os.path.join(bg_dir, '*.png'))) != len(bg):
        for i, frame in enumerate(bg):
            media.write_image(os.path.join(bg_dir, f'{i:05d}.png'), frame)

    fg_dir = os.path.join(save_dir, f'fg{fg_id:02d}')
    os.makedirs(fg_dir, exist_ok=True)
    for i, (frame_rgb, frame_alpha) in enumerate(zip(X_rgb, X_alpha)):
        frame_rgba = np.concatenate([frame_rgb, frame_alpha], axis=-1)
        media.write_image(os.path.join(fg_dir, f'{i:05d}.png'), frame_rgba)


# codes from https://github.com/erikalu/omnimatte/blob/main/third_party/models/networks_lnr.py#L104
class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'."""

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
            # padding = .5 * (stride * (output-1) + (k-1)(d-1) + k - input)
            desired_height = height // self.s
            desired_width = width // self.s
            pady = .5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = .5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)
        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)), int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class VideoMaskTracker:
    # def __init__(self, sam_model='large', SAM_DIR="/nfshomes/yclee/disk/projects/sam2"):
    def __init__(self, sam_model='large', SAM_DIR=sam2_dir):
        ckpt, cfg = {
            'tiny': ('sam2.1_hiera_tiny.pt', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
            'small': ('sam2.1_hiera_small.pt', 'configs/sam2.1/sam2.1_hiera_s.yaml'),
            'base+': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
            'large': ('sam2.1_hiera_large.pt', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
        }[sam_model]

        ckpt = os.path.join(SAM_DIR, 'checkpoints', ckpt)
        self.device = torch.device("cuda")
        self.predictor = build_sam2_video_predictor(cfg, ckpt, device=self.device)

    def run(self, video_dir, points, ann_frame_idx=0, labels=None):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            infer_state = self.predictor.init_state(video_path=video_dir)
            self.predictor.reset_state(infer_state)
            num_objs = len(points)
            if labels is not None:
                assert len(labels) == len(points)
            for i in range(num_objs):
                points_i = points[i].astype(np.float32)
                if labels is None:
                    labels_i = np.ones((len(points[i])), dtype=np.int32)
                else:
                    labels_i = np.array(labels[i]).astype(np.int32)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=infer_state,
                    frame_idx=ann_frame_idx,
                    obj_id=i,
                    points=points_i,
                    labels=labels_i,
                )
            video_segments = {}

        def dump_tracking_outputs(tracking):
            for out_frame_idx, out_obj_ids, out_mask_logits in tracking:
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_i = (out_mask_logits[i] > 0.0).cpu().float().numpy()
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = np.zeros_like(mask_i)
                    video_segments[out_frame_idx] = np.where(
                        mask_i > 0.5,
                        out_obj_id + 1,
                        video_segments[out_frame_idx]
                    )

        tracking_fwd = self.predictor.propagate_in_video(
            infer_state, start_frame_idx=ann_frame_idx,
        )
        dump_tracking_outputs(tracking_fwd)

        if ann_frame_idx > 0:
            tracking_bwd = self.predictor.propagate_in_video(
                infer_state, start_frame_idx=ann_frame_idx, reverse=True
            )
            dump_tracking_outputs(tracking_bwd)

        frame_ids = sorted(list(video_segments.keys()))
        video_masks = []
        for i in frame_ids:
            video_masks.append(video_segments[i])
        return np.concatenate(video_masks, axis=0).astype(np.uint8)


def _refine_mask(sam_tracker, frame_dir, vid_mask_init, combine_coarse=True):
    if len(vid_mask_init.shape) == 4:
        vid_mask_init = vid_mask_init[..., 0]
        mask_sum = vid_mask_init.sum(-1).sum(-1)
    # vid_mask_eroded = mask_utils.erode_video_mask(vid_mask_init, 3)
    ref_frame_index = np.argmax(mask_sum)
    ys, xs = np.nonzero(vid_mask_init[ref_frame_index])
    pts = np.stack([xs, ys], -1)
    pts = pts[np.random.choice(np.arange(len(pts)), 20, replace=True)]
    tmp_dir = os.path.join('./tmp', str(uuid.uuid4()))
    video_masks = sam_tracker.run(frame_dir, [pts], ref_frame_index)
    video_mask = (video_masks > 0).astype(float)
    if combine_coarse:
        video_mask = (video_mask + vid_mask_init).clip(0, 1)
    return video_mask


@torch.no_grad()
def refine_mask(video_rgb_np, video_coarse_mask_np):
    tmp_dir = os.path.join('.tmp', str(uuid.uuid4()))
    os.makedirs(tmp_dir)
    for i, rgb in enumerate(video_rgb_np):
        media.write_image(os.path.join(tmp_dir, f'{i:07d}.jpg'), rgb)

    sam_tracker = VideoMaskTracker()

    video_mask_refined = _refine_mask(
        sam_tracker, tmp_dir, video_coarse_mask_np, combine_coarse=True
    )
    del sam_tracker
    gc.collect()
    torch.cuda.empty_cache()
    os.system(f'rm -r {tmp_dir}')
    return video_mask_refined
