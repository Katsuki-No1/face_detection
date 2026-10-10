"""Regular DINO reference memory; normalized crop x-right/y-down cell rectangles.

Order: ascending block, ascending pooling factor, row-major. Pool BEFORE token
LayerNorm and shared K/V projection. One global FP32 softmax lives in RecoveryLayer.
The adapter is part of the treatment; group token counts are deliberately unequal.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class RecoveryReferenceMemory(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.levels = config.reference_layers if config.multi_level else (11,)
        self.factors = config.reference_factors if config.multi_scale else (1,)
        self.key_position = nn.Linear(8, channels)
        nn.init.normal_(self.key_position.weight, std=0.02)
        nn.init.zeros_(self.key_position.bias)

    def forward(self, final_map, reference_maps=None):
        maps = (final_map,) if reference_maps is None else tuple(reference_maps)
        if len(maps) != len(self.levels) or any(m.shape != final_map.shape for m in maps):
            raise ValueError("Reference maps must match selected DINO levels and final shape")
        b, c, h, w = final_map.shape
        if any(h % f or w % f for f in self.factors):
            raise ValueError("Reference pooling requires divisible spatial dimensions")
        yy, xx = torch.meshgrid(torch.arange(h, device=final_map.device), torch.arange(w, device=final_map.device), indexing="ij")
        qx, qy = xx.flatten(), yy.flatten()
        tokens, metadata, masks, groups = [], [], [], []
        offset = 0
        for level, feature in zip(self.levels, maps):
            for factor in self.factors:
                pooled = F.avg_pool2d(feature, factor, factor) if factor != 1 else feature
                ph, pw = pooled.shape[-2:]
                y, x = torch.meshgrid(torch.arange(ph, device=feature.device), torch.arange(pw, device=feature.device), indexing="ij")
                x0, y0 = x.flatten() * factor, y.flatten() * factor
                x1, y1 = x0 + factor, y0 + factor
                rect = torch.stack(((x0 + x1) / (2 * w), (y0 + y1) / (2 * h),
                    x0 / w, y0 / h, x1 / w, y1 / h,
                    torch.full_like(x0, level / 11, dtype=torch.float32),
                    torch.full_like(x0, math.log2(factor) / 2, dtype=torch.float32)), -1)
                mask = (qx[:, None] >= x0) & (qx[:, None] < x1) & (qy[:, None] >= y0) & (qy[:, None] < y1)
                tokens.append(pooled.flatten(2).transpose(1, 2))
                metadata.append(rect)
                masks.append(mask)
                count = ph * pw
                groups.append(dict(level=level, factor=factor, height=ph, width=pw, start=offset, stop=offset + count, tokens=count))
                offset += count
        coordinates = torch.cat(metadata)
        return dict(tokens=torch.cat(tokens, 1), metadata=coordinates, forbidden=torch.cat(masks, 1), groups=groups,
                    key_position=self.key_position(coordinates.to(final_map.dtype))[None].expand(b, -1, -1))

    def reliability(self, previous_alpha, height, width):
        if previous_alpha is None:
            return None
        rho = (1 - previous_alpha.float()).reshape(-1, 1, height, width)
        spatial = [F.avg_pool2d(rho, f, f).flatten(1) if f != 1 else rho.flatten(1) for f in self.factors]
        return torch.cat(spatial * len(self.levels), -1)
