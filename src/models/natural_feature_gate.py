"""Patch adoption after a face crop. Ground-truth geometry is supervision only."""
import torch
from torch import nn
from torch.nn import functional as F


class NaturalFeatureGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2 * channels + 3, 96), nn.GELU(), nn.Linear(96, 1))
        nn.init.constant_(self.network[-1].bias, -2.)

    def forward(self, original, candidate, channel):
        x, r = original.permute(0, 2, 3, 1), candidate.permute(0, 2, 3, 1)
        h, w = x.shape[1:3]
        axis = lambda n: torch.linspace(-1, 1, n, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(axis(h), axis(w), indexing="ij")
        position = torch.stack([xx, yy], -1).expand(len(x), -1, -1, -1)
        difference = 1 - F.cosine_similarity(x, r, dim=-1)
        features = torch.cat([F.layer_norm(x, x.shape[-1:]), F.layer_norm(r, r.shape[-1:]),
                              position, difference[..., None]], -1)
        logits = self.network(features).permute(0, 3, 1, 2)
        fused = original + logits.sigmoid() * channel * (candidate - original)
        return logits, fused


def supervised_losses(logits, coordinates, target, confidence, xy, state):
    entropy = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    classification = (entropy * confidence).sum() / confidence.sum().clamp_min(1.e-8)
    error = F.smooth_l1_loss(coordinates, xy, reduction="none", beta=.02).mean(-1)
    # Each present state receives equal mass, irrespective of predicted presence.
    groups = [error[state == value].mean() for value in (1, 2) if (state == value).any()]
    position = torch.stack(groups).mean() if groups else coordinates.sum() * 0
    return classification, position
