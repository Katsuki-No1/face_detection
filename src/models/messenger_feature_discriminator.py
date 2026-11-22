"""Conditional feature PatchGAN; candidate targets never enter the keypoint head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class FeaturePatchDiscriminator(nn.Module):
    def __init__(self, mean, std, seed=43):
        super().__init__()
        self.register_buffer('mean', mean.detach().float().reshape(1, -1, 1, 1).clone())
        self.register_buffer('std', std.detach().float().clamp_min(.01).reshape(1, -1, 1, 1).clone())
        channels = self.mean.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            specs = [(2*channels+3, 128, 1, 1), (128, 128, 3, 1),
                     (128, 192, 3, 2), (192, 192, 3, 1)]
            self.blocks = nn.ModuleList([
                nn.Sequential(spectral_norm(nn.Conv2d(a, b, k, stride=s, padding=k//2)),
                              nn.LeakyReLU(.2, inplace=False)) for a, b, k, s in specs])
            self.output = spectral_norm(nn.Conv2d(192, 1, 1))

    def forward(self, candidate, context, mask):
        b, _, h, w = candidate.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=candidate.device),
                                torch.linspace(-1, 1, w, device=candidate.device), indexing='ij')
        pos = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        value = torch.cat(((candidate-self.mean)/self.std,
                           (context-self.mean)/self.std, mask[:, None].float(), pos), 1)
        features = []
        for i, block in enumerate(self.blocks):
            value = block(value)
            if i:
                features.append(value)
        return self.output(value)[:, 0], features


def paired_candidates(x, recovered, target, mask):
    """Both labels have exactly the same outside context, including DINO contamination."""
    return torch.where(mask[:, None], target, x), torch.where(mask[:, None], recovered, x)


def receptive_masks(mask):
    value = mask[:, None].float()
    masks = []
    for stride in (1, 2, 1):
        value = F.max_pool2d(value, 3, stride=stride, padding=1)
        masks.append(value[:, 0])
    return masks


def spatial_mean(value, mask):
    """Each example has equal weight; empty supports are forbidden."""
    if value.ndim == 4:
        value = value.mean(1)
    count = mask.sum((1, 2))
    if bool((count == 0).any()):
        raise ValueError('Discriminator requires nonempty restoration support')
    return ((value*mask).sum((1, 2))/count).mean()


def discriminator_objective(real_logits, fake_logits, mask):
    support = receptive_masks(mask)[-1]
    return spatial_mean(F.softplus(-real_logits)+F.softplus(fake_logits), support)


def generator_objectives(discriminator, x, recovered, target, mask):
    """D must be frozen/eval; real features are constant targets, fake path stays live."""
    real, fake = paired_candidates(x, recovered, target, mask)
    fake_logits, fake_features = discriminator(fake, x, mask)
    with torch.no_grad():
        _, real_features = discriminator(real, x, mask)
    supports = receptive_masks(mask)
    adversarial = spatial_mean(F.softplus(-fake_logits), supports[-1])
    matching = torch.stack([spatial_mean((a-b).square(), m)
                            for a, b, m in zip(fake_features, real_features, supports)]).mean()
    return adversarial, matching
