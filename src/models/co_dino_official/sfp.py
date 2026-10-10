"""Co-DETR SFP port (upstream license: src/third_party/codetr/LICENSE).

Only input width and the checkpoint API differ from the upstream ViT neck.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + 1e-6)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class OfficialSFP(nn.Module):
    strides = (4, 8, 16, 32, 64)

    def __init__(self, in_channels=384, out_channels=256, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c, d = in_channels, out_channels

        def conv(a, b, k, stride=1):
            return [nn.Conv2d(a, b, k, stride, k // 2, bias=False), ChannelLayerNorm(b)]

        def finish(a):
            return conv(a, d, 1) + [nn.GELU()] + conv(d, d, 3)

        self.p2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
            *conv(c, c // 2, 3), nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"),
            *conv(c // 2, c // 4, 3), nn.GELU(), *finish(c // 4))
        self.p3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
            *conv(c, c // 2, 3), nn.GELU(), *finish(c // 2))
        self.p4 = nn.Sequential(*finish(c))
        self.p5 = nn.Sequential(nn.MaxPool2d(3, 2, 1), *finish(c))
        self.p6 = nn.Sequential(nn.MaxPool2d(3, 2, 1), *conv(c, c, 3, 2),
            nn.GELU(), *finish(c))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            if len(x) != 1:
                raise ValueError("Official SFP requires the single stride-16 ViT map")
            x = x[0]
        branches = (self.p2, self.p3, self.p4, self.p5, self.p6)
        return tuple(checkpoint(branch, x, use_reentrant=False)
                     if self.training and self.use_checkpoint else branch(x)
                     for branch in branches)
