"""Continuous and vector-quantized 16-dimensional Messenger recovery."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.messenger_multiaxis import MultiAxisRecovery, Branch, sine_position_encoding_2d

ARMS = (
    "weighted16",
    "continuous16",
    "vq4096_fixed",
    "vq4096_trainable",
    "vq8192_fixed",
    "vq8192_trainable",
)


def directional_mse(error: torch.Tensor, basis: torch.Tensor, weight: float) -> torch.Tensor:
    """Mean eigen-direction weight is one; basis has orthonormal columns."""
    if weight < 1:
        raise ValueError("Direction weight must be at least one")
    projected = torch.einsum("cr,bchw->brhw", basis, error)
    channels, rank = basis.shape
    return (error.square().sum(1) + (weight - 1) * projected.square().sum(1)) / (
        channels + (weight - 1) * rank
    )


class VectorCodebook(nn.Module):
    def __init__(self, centers: torch.Tensor, trainable: bool, chunk_size: int = 256):
        super().__init__()
        if centers.ndim != 2 or not torch.isfinite(centers).all():
            raise ValueError("Expected finite code vectors with shape K,D")
        self.embedding = nn.Parameter(centers.detach().clone(), requires_grad=trainable)
        self.chunk_size = chunk_size

    def forward(self, latent: torch.Tensor):
        vectors = latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
        # Hard nearest-neighbor assignment. Autograd is needed for selected values,
        # but not for the distance matrix or the discontinuous index selection.
        with torch.no_grad():
            code_norm = self.embedding.square().sum(1)
            indices = []
            for part in vectors.detach().split(self.chunk_size):
                distance = part.square().sum(1, keepdim=True) + code_norm - 2 * part @ self.embedding.t()
                indices.append(distance.argmin(1))
            indices = torch.cat(indices)
        selected = F.embedding(indices, self.embedding)
        selected = selected.reshape(latent.shape[0], *latent.shape[2:], latent.shape[1]).permute(0, 3, 1, 2)
        # Forward is exactly the selected code. Backward reaches both the selected
        # code values and the encoder via a straight-through identity estimator.
        quantized = selected + (latent - latent.detach())
        commitment = (latent - selected.detach()).square().mean(1)
        code_loss = (selected - latent.detach()).square().mean(1)
        return quantized, dict(
            indices=indices.reshape(latent.shape[0], *latent.shape[2:]),
            commitment=commitment,
            code_loss=code_loss,
            quantization_error=(latent.detach() - selected.detach()).square().mean(1),
        )


class LatentBranch(Branch):
    """Existing attention path with 384 input channels and 16 output channels."""

    def forward(self, x):
        b, _, h, w = x.shape
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
        return self.decode(q).transpose(1, 2).reshape(b, 16, h, w), self.score(q).transpose(1, 2).reshape(b, 1, h, w)

class LatentRecovery(nn.Module):
    """Every output path goes through 16 channels, including the decoder skip."""

    def __init__(self, basis: torch.Tensor, mean: torch.Tensor, centers=None, trainable=False, seed=42):
        super().__init__()
        if basis.shape != (384, 16) or mean.shape != (384,):
            raise ValueError("Expected a 384x16 basis and a 384-dimensional mean")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context = MultiAxisRecovery("scale")
            for index, (_, factor) in enumerate(self.context.specs):
                branch = LatentBranch(384, 96, factor)
                branch.load_state_dict(self.context.branches[index].state_dict())
                self.context.branches[index] = branch
                final = branch.decode[-1]
                branch.decode[-1] = nn.Linear(final.in_features, 16)
                nn.init.zeros_(branch.decode[-1].weight)
                nn.init.zeros_(branch.decode[-1].bias)
            self.projection = nn.Conv2d(384, 16, 1)
            self.decoder_linear = nn.Conv2d(16, 384, 1)
            self.decoder_residual = nn.Sequential(nn.Conv2d(16, 96, 1), nn.GELU(), nn.Conv2d(96, 384, 1))
            nn.init.zeros_(self.decoder_residual[-1].weight)
            nn.init.zeros_(self.decoder_residual[-1].bias)
        with torch.no_grad():
            self.projection.weight.copy_(basis.t()[:, :, None, None])
            self.projection.bias.copy_(-basis.t() @ mean)
            self.decoder_linear.weight.copy_(basis[:, :, None, None])
            self.decoder_linear.bias.copy_(mean)
        self.codebook = None if centers is None else VectorCodebook(centers, trainable)

    def encode(self, maps):
        context, _, _ = self.context(maps)
        return self.projection(maps[:, 2]) + context

    def forward(self, maps):
        latent = self.encode(maps)
        details = {}
        if self.codebook is not None:
            latent, details = self.codebook(latent)
        restored = self.decoder_linear(latent) + self.decoder_residual(latent)
        return restored, details


def make_recovery(arm, basis, mean, centers=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm == "weighted16":
        return MultiAxisRecovery("scale")
    if arm == "continuous16":
        return LatentRecovery(basis, mean)
    count = 4096 if arm.startswith("vq4096") else 8192
    if centers is None or centers.shape != (count, 16):
        raise ValueError(f"{arm} requires {count}x16 initial codes")
    return LatentRecovery(basis, mean, centers, trainable=arm.endswith("_trainable"))
