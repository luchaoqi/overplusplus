import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import inv_sigmoid, ConvBlock

class BaseVariable(nn.Module):
    def __init__(
            self,
            config,
            num_frames,
            height,
            width,
            channels=3,
            device='cpu',
            lr=1e-4,
            name="base_variable",
            **kwargs,
    ):
        super().__init__()
        self.config = config

        self.T = num_frames
        self.H = height
        self.W = width
        self.C = channels
        self.device = device
        self.lr = lr
        self.name = name
        self.inv_activation = lambda x: inv_sigmoid(x).clamp(-4, 4)

    @torch.no_grad()
    def initialize(self):
        pass

    def get_params(self):
        return []

    def forward(self, input_tensor=None, index=None):
        raise NotImplementedError


class TensorVariable(BaseVariable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor = nn.Embedding(self.T, self.C * self.H * self.W).to(self.device)
        self.activation = nn.Sigmoid()

    @torch.no_grad()
    def initialize(self, value=None):
        if value is not None:
            self.tensor.weight.data = self.inv_activation(value.to(self.device)).reshape(self.T, -1)

    def get_params(self):
        return [{
            "params": self.tensor.parameters(),
            "lr": self.lr,
            "name": self.name,
        }]

    def forward(self, input_tensor=None, index=None):
        assert index is not None
        out = self.activation(self.tensor(index)).view(-1, self.C, self.H, self.W)
        return out

# codes from https://github.com/erikalu/omnimatte/blob/main/models/networks.py#L43
class UNetVariable(BaseVariable):
    def __init__(self, in_c=3, nf=32, **kwargs):
        super().__init__(**kwargs)
        self.in_c = in_c
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_c, nf, ksize=4, stride=2),
            ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
        ])
        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d)
        ])
        self.out_conv = ConvBlock(nn.Conv2d, nf, self.C, ksize=4, stride=1, activation='sigmoid')

    @torch.no_grad()
    def initialize(self, value=None):
        for module in [self.encoder, self.decoder, self.out_conv]:
            for name, param in module.named_parameters():
                nn.init.normal_(param, 0, 0.01)
                param.requires_grad = True

    def get_params(self):
        return [{
            "params": m.parameters(),
            "lr": self.lr,
            "name": self.name,
        } for m in [self.encoder, self.decoder, self.out_conv]]

    def forward(self, input_tensor=None, index=None):
        assert input_tensor is not None
        assert input_tensor.size(1) == self.in_c
        x = input_tensor
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), dim=1)
            x = layer(x)
        out = self.out_conv(x)
        if len(out.shape) == 3:
            out = out.unsqueeze(1)
        return out


class RGBModule(nn.Module):
    def __init__(self, config, num_frames, height, width, device='cpu', num_channels=3):
        super().__init__()
        self.variables = {
            "tensor": TensorVariable,
            "unet": UNetVariable,
        }[config.omnimatte.rgb_module_type](
            in_c=6,
            config=config,
            num_frames=num_frames,
            height=height,
            width=width,
            channels=num_channels,
            device=device,
            lr=config.omnimatte.rgb_lr,
            name="rgb",
        )
    def get_params(self):
        return self.variables.get_params()

    def initialize(self, value=None):
        self.variables.initialize(value)

    def forward(self, input_tensor=None, index=None):
        return self.variables(input_tensor=input_tensor, index=index)


class AlphaModule(nn.Module):
    def __init__(self, config, num_frames, height, width, device='cpu'):
        super().__init__()
        self.variables = {
            "tensor": TensorVariable,
            "unet": UNetVariable,
        }[config.omnimatte.alpha_module_type](
            in_c=6,
            config=config,
            num_frames=num_frames,
            height=height,
            width=width,
            channels=1,
            device=device,
            lr=config.omnimatte.alpha_lr,
            name="alpha",
        )

    def get_params(self):
        return self.variables.get_params()

    def initialize(self, value=None):
        self.variables.initialize(value)

    def forward(self, input_tensor=None, index=None):
        return self.variables(input_tensor=input_tensor, index=index)
