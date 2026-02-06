import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from datetime import datetime

from .modules import RGBModule, AlphaModule
from .loss import Loss
from .utils import visualize_omnimatte


class OmnimatteOptimizer:
    def __init__(self, config, XY, Y, init_mask=None, device='cpu', expname=''):
        """Given: XY (full RGB) and Y (background RGB), solve: X (foreground RGBA)."""
        self.config = config
        self.T, self.C, self.H, self.W = XY.shape
        self.device = device
        self.log_dir = config.omnimatte.log_dir if config.omnimatte.log_dir else None
        if self.log_dir is not None:
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            expname = expname + '_' + time_str if expname else time_str
            self.log_dir = os.path.join(self.log_dir, expname)
            os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            json_str = json.dumps(self.config.omnimatte.to_dict(), indent=4)
            with open(os.path.join(self.log_dir, 'config_omnimatte.json'), 'w') as f:
                f.write(json_str)

        self.freq_log = config.omnimatte.freq_log
        self.freq_eval = config.omnimatte.freq_eval

        self.is_rgba_unet = False
        self.optimizers = []
        self.lr_schedulers = []
        def _create_optimizer(module, lr):
            params = module.get_params()
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99))
            return optimizer

        def _create_scheduler(_optimizer):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                _optimizer,
                milestones=config.omnimatte.lr_schedule_milestones,
                gamma=config.omnimatte.lr_schedule_gamma,
            )
            return lr_scheduler
        if config.omnimatte.rgb_module_type == 'unet' and config.omnimatte.alpha_module_type == 'unet':
            self.is_rgba_unet = True
            self.module_rgba = RGBModule(
                config=config,
                num_frames=self.T,
                height=self.H,
                width=self.W,
                device=self.device,
                num_channels=self.C + 1,
            ).to(self.device)
            self.optimizers.append(
                _create_optimizer(self.module_rgba, lr=config.omnimatte.rgb_lr)
            )
            self.lr_schedulers.append(
                _create_scheduler(self.optimizers[-1])
            )
        else:
            self.module_rgb = RGBModule(
                config=config,
                num_frames=self.T,
                height=self.H,
                width=self.W,
                device=self.device,
                num_channels=self.C,
            ).to(self.device)
            init_rgb = XY
            if init_mask is not None:
                init_rgb = XY * init_mask + (1 - init_mask) * 0.5
            self.module_rgb.initialize(value=init_rgb)

            self.module_alpha = AlphaModule(
                config=config,
                num_frames=self.T,
                height=self.H,
                width=self.W,
                device=self.device,
            ).to(self.device)
            self.module_alpha.initialize(value=init_mask)

            self.optimizers.append(
                _create_optimizer(self.module_rgb, lr=config.omnimatte.rgb_lr)
            )
            if config.omnimatte.rgb_module_type == 'unet':
                self.lr_schedulers.append(
                    _create_scheduler(self.optimizers[-1])
                )
            self.optimizers.append(
                _create_optimizer(self.module_alpha, lr=config.omnimatte.alpha_lr)
            )
            if config.omnimatte.alpha_module_type == 'unet':
                self.lr_schedulers.append(
                    _create_scheduler(self.optimizers[-1])
                )

        self.loss = Loss(config=config, mask_super=init_mask)

        self.constants = {
            'XY': XY.cpu(),
            'Y': Y.cpu(),
        }
        if init_mask is not None:
            self.constants['mask'] = init_mask.cpu()

        self.num_steps = config.omnimatte.num_steps
        self.batch_size = config.omnimatte.batch_size

    def run_one_step(self, step_id):
        if self.is_rgba_unet:
            self.module_rgba.train()
        else:
            self.module_rgb.train()
            self.module_alpha.train()

        sample_begin = torch.randint(-self.batch_size, self.T, (1,))
        sample_begin = sample_begin.clamp(0, self.T - self.batch_size).item()
        sample_end = sample_begin + self.batch_size

        constants = {
            'XY': self.constants['XY'][sample_begin:sample_end].to(self.device),
            'Y': self.constants['Y'][sample_begin:sample_end].to(self.device),
        }
        if 'mask' in self.constants:
            constants['mask'] = self.constants['mask'][sample_begin:sample_end].to(self.device)

        sample_indices = torch.arange(sample_begin, sample_end).to(self.device).long()

        alpha_input_tensor = torch.cat([
            constants['XY'],
            torch.abs(constants['XY'] - constants['Y']).detach()
        ], dim=1)

        if self.is_rgba_unet:
            X_rgba = self.module_rgba(input_tensor=alpha_input_tensor, index=sample_indices)
            X_rgb = X_rgba[:, :-1]
            X_alpha = X_rgba[:, -1:]
        else:
            X_rgb = self.module_rgb(input_tensor=constants['XY'], index=sample_indices)
            X_alpha = self.module_alpha(input_tensor=alpha_input_tensor, index=sample_indices)

        variables = {
            'X_rgb': X_rgb,
            'X_alpha': X_alpha,
        }

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        loss, losses, intermediates = self.loss(variables, constants, step_id)
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        if self.log_dir is not None and step_id % self.freq_log == 0:
            self.writer.add_scalar('loss/_total', loss.item(), step_id)
            for k, v in losses.items():
                self.writer.add_scalar(f'loss/{k}', v.item(), step_id)

        return loss.item()

    @torch.no_grad()
    def eval(self):
        if self.is_rgba_unet:
            self.module_rgba.eval()
        else:
            self.module_rgb.eval()
            self.module_alpha.eval()
        index = 0

        X_rgb, X_alpha = [], []
        while index < self.T:
            index_end = min(index + self.batch_size, self.T)
            sample_indices = torch.arange(index, index_end).long()
            input_tensor = self.constants['XY'][sample_indices].to(self.device)
            XY_minus_Y = torch.abs(input_tensor - self.constants['Y'][sample_indices].to(self.device))
            alpha_input_tensor = torch.cat([input_tensor, XY_minus_Y], dim=1)
            sample_indices = sample_indices.to(self.device)

            if self.is_rgba_unet:
                X_rgba_window = self.module_rgba(input_tensor=alpha_input_tensor, index=sample_indices)
                X_rgb_window = X_rgba_window[:, :-1]
                X_alpha_window = X_rgba_window[:, -1:]
            else:
                X_rgb_window = self.module_rgb(input_tensor=input_tensor, index=sample_indices)
                X_alpha_window = self.module_alpha(input_tensor=alpha_input_tensor, index=sample_indices)

            X_rgb.append(X_rgb_window.detach().cpu())
            X_alpha.append(X_alpha_window.detach().cpu())
            index += self.batch_size

        X_rgb = torch.cat(X_rgb, dim=0)
        X_alpha = torch.cat(X_alpha, dim=0)

        pkg = {
            'X_rgb': X_rgb,
            'X_alpha': X_alpha,
            'XY': self.constants['XY'],
            'Y': self.constants['Y'],
            'mask': self.constants['mask'] if 'mask' in self.constants else None,
        }
        return pkg

    def run(self):
        pbar = tqdm.tqdm(np.arange(self.num_steps))
        for step_id in pbar:
            loss = self.run_one_step(step_id)
            if self.log_dir is not None and step_id % self.freq_eval == 0:
                pkg = self.eval()
                visualize_omnimatte(
                    pkg,
                    os.path.join(self.log_dir, f'eval-{step_id:06d}.mp4'),
                )
            pbar.set_description(f"loss: {loss:.4f}")
            pbar.update()

        pkg_final = self.eval()
        return pkg_final
