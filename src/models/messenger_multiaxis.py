"""Independent level/scale recovery branches; no self/occlusion attention mask."""
from itertools import combinations
import torch
from torch import nn
from torch.nn import functional as F
from keypoint_comparison1.landmark_detection.heads import sine_position_encoding_2d


def branch_specs(family):
    if family not in ('scale', 'level', 'level_scale'):
        raise ValueError(family)
    levels = (2,) if family == 'scale' else (0, 1, 2)
    scales = (1,) if family == 'level' else (1, 2, 4)
    return [(l, s) for l in levels for s in scales]


class Branch(nn.Module):
    def __init__(self, channels, width, factor):
        super().__init__()
        self.factor = factor
        # Learned projection of all cells in the patch, rather than average pooling.
        self.patch = nn.Conv2d(channels, width, factor, stride=factor)
        self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(qnorm=nn.LayerNorm(width), knorm=nn.LayerNorm(width),
                               attn=nn.MultiheadAttention(width, 6, dropout=0, batch_first=True),
                               norm=nn.LayerNorm(width),
                               ffn=nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))))
            for _ in range(2)])
        self.decode = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, channels))
        self.score = nn.Linear(width, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        memory = self.patch(x)
        mh, mw = memory.shape[-2:]
        memory = memory.flatten(2).transpose(1, 2)
        width = memory.shape[-1]
        kp = sine_position_encoding_2d(mh, mw, width, device=x.device, dtype=x.dtype)
        qp = sine_position_encoding_2d(h, w, width, device=x.device, dtype=x.dtype)
        q = self.query.expand(b, h*w, -1) + qp
        for layer in self.layers:
            k = layer['knorm'](memory)
            q = q + layer['attn'](layer['qnorm'](q), k+kp, k, need_weights=False)[0]
            q = q + layer['ffn'](layer['norm'](q))
        return self.decode(q).transpose(1, 2).reshape(b, c, h, w), self.score(q).transpose(1, 2).reshape(b, 1, h, w)


class MultiAxisRecovery(nn.Module):
    def __init__(self, family, channels=384, width=96, seed=42):
        super().__init__()
        self.specs = branch_specs(family)
        branches = []
        for level, scale in self.specs:
            # Same branch starts identically across families and A/B/C.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 100*level + scale)
                branches.append(Branch(channels, width, scale))
        self.branches = nn.ModuleList(branches)

    def forward(self, maps):
        if maps.ndim != 5 or maps.shape[1] != 3:
            raise ValueError('Expected B,3,C,H,W in DINO block order 3,7,11')
        values, scores = zip(*(branch(maps[:, level]) for branch, (level, _) in zip(self.branches, self.specs)))
        candidates = torch.stack(values, 1)
        weights = torch.stack(scores, 1).softmax(1)
        return (candidates*weights).sum(1), candidates, weights


def feature_error(pred, target):
    return 1-F.cosine_similarity(pred, target, dim=-3) + (pred-target).square().mean(-3)


def region_mean(error, covered, outside):
    return (error*covered).sum()/covered.sum().clamp_min(1) + .1*(error*outside).sum()/outside.sum().clamp_min(1)


def reconstruction_loss(outputs, target, covered, outside, condition):
    if condition not in ('A', 'B', 'C'):
        raise ValueError(condition)
    fused, candidates, weights = outputs
    main = region_mean(feature_error(fused, target), covered, outside)
    branch = torch.stack([region_mean(feature_error(r, target), covered, outside) for r in candidates.unbind(1)]).mean()
    # Mean pairwise MSE equals 2n/(n-1) times population variance.
    n = candidates.shape[1]
    consistency_map = candidates.var(1, unbiased=False).mean(1) * (2*n/(n-1))
    consistency = region_mean(consistency_map, covered, outside)
    total = main + (branch if condition in ('B', 'C') else 0) + (.1*consistency if condition == 'C' else 0)
    return total, dict(fused=main, branch=branch, consistency=consistency)
