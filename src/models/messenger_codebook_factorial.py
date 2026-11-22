"""Isolated factorial VQ comparison; the historical model remains unchanged."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from src.models.messenger_multiaxis import MultiAxisRecovery, Branch, sine_position_encoding_2d

DIMENSIONS = (16, 32)
COUNTS = (4096, 8192)
ARMS = ("continuous32",) + tuple(f"{mode}_d{dim}_k{count}" for dim in DIMENSIONS for count in COUNTS for mode in "ABCDE")

def arm_spec(arm):
    if arm == "continuous32":
        return dict(dimension=32, count=0, normalized=False, update="none", spatial=False, mode="continuous")
    mode, dim, count = arm.split("_")
    if mode not in "ABCDE" or int(dim[1:]) not in DIMENSIONS or int(count[1:]) not in COUNTS:
        raise ValueError(arm)
    return dict(dimension=int(dim[1:]), count=int(count[1:]), normalized=mode in "BDE",
                update="ema" if mode in "CDE" else "gradient", spatial=mode == "E", mode=mode)

class VectorCodebook(nn.Module):
    """Lookup-only normalization; raw-valued codes, deferred batch-mean EMA."""
    def __init__(self, centers, normalized=False, update="gradient", decay=.99, chunk_size=256):
        super().__init__()
        if centers.ndim != 2 or not torch.isfinite(centers).all() or update not in ("gradient", "ema"):
            raise ValueError("Invalid codebook")
        self.embedding = nn.Parameter(centers.detach().clone(), requires_grad=update == "gradient")
        self.normalized, self.update, self.decay, self.chunk_size = normalized, update, decay, chunk_size
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))
        self._counts = self._sums = None

    def begin_step(self):
        self._counts = self._sums = None

    def forward(self, latent):
        vectors = latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])
        with torch.no_grad():
            keys = F.normalize(self.embedding, dim=1) if self.normalized else self.embedding
            query = F.normalize(vectors.detach(), dim=1) if self.normalized else vectors.detach()
            indices = torch.cat([(part.square().sum(1, keepdim=True) + keys.square().sum(1)
                                  - 2 * part @ keys.T).argmin(1) for part in query.split(self.chunk_size)])
            if self.training and self.update == "ema":
                counts = torch.bincount(indices, minlength=len(keys))
                order = indices.argsort(stable=True)
                sums = torch.segment_reduce(vectors.detach()[order], "sum", lengths=counts)
                if self._counts is None:
                    self._counts, self._sums = counts, sums
                else:
                    self._counts += counts
                    self._sums += sums
        selected = F.embedding(indices, self.embedding)
        selected = selected.reshape(latent.shape[0], *latent.shape[2:], latent.shape[1]).permute(0, 3, 1, 2)
        quantized = selected + (latent - latent.detach())
        commitment = (latent - selected.detach()).square().mean(1)
        code_loss = (selected - latent.detach()).square().mean(1) if self.update == "gradient" else torch.zeros_like(commitment)
        return quantized, dict(indices=indices.reshape(latent.shape[0], *latent.shape[2:]),
            commitment=commitment, code_loss=code_loss,
            quantization_error=(latent.detach() - selected.detach()).square().mean(1))

    @torch.no_grad()
    def end_step(self):
        if self.update == "ema" and self._counts is not None:
            populated = self._counts > 0
            target = self._sums[populated] / self._counts[populated, None]
            self.embedding[populated] = self.decay * self.embedding[populated] + (1-self.decay) * target
            if not torch.isfinite(self.embedding).all():
                raise ValueError("Nonfinite EMA codebook")
            self.ema_steps.add_(1)
        self.begin_step()

class SpatialDecoder(nn.Module):
    """One quantized-map self-attention block, zero-initialized 384D residual."""
    def __init__(self, dimension, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+900)
            self.project = nn.Linear(dimension, 96)
            self.norm = nn.LayerNorm(96)
            self.attention = nn.MultiheadAttention(96, 6, dropout=0, batch_first=True)
            self.ffn_norm = nn.LayerNorm(96)
            self.ffn = nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Linear(192, 96))
            self.output = nn.Linear(96, 384)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, latent):
        b, _, h, w = latent.shape
        x = self.project(latent.flatten(2).transpose(1, 2))
        pos = sine_position_encoding_2d(h, w, 96, device=x.device, dtype=x.dtype)
        value = self.norm(x)
        x = x + self.attention(value+pos, value+pos, value, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return self.output(x).transpose(1, 2).reshape(b, 384, h, w)

class LatentBranch(Branch):
    """Existing attention path with 384 input channels and a configurable latent width."""

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
        return self.decode(q).transpose(1, 2).reshape(b, self.decode[-1].out_features, h, w), self.score(q).transpose(1, 2).reshape(b, 1, h, w)

class LatentRecovery(nn.Module):
    """Every output path goes through the latent channels, including the decoder skip."""

    def __init__(self, basis: torch.Tensor, mean: torch.Tensor, centers=None, normalized=False, update="gradient", spatial=False, seed=42):
        super().__init__()
        dimension = basis.shape[1]
        if basis.shape not in ((384, 16), (384, 32)) or mean.shape != (384,):
            raise ValueError("Expected a 384 x latent-width basis and a 384-dimensional mean")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context = MultiAxisRecovery("scale")
            for index, (_, factor) in enumerate(self.context.specs):
                branch = LatentBranch(384, 96, factor)
                branch.load_state_dict(self.context.branches[index].state_dict())
                self.context.branches[index] = branch
                final = branch.decode[-1]
                branch.decode[-1] = nn.Linear(final.in_features, dimension)
                nn.init.zeros_(branch.decode[-1].weight)
                nn.init.zeros_(branch.decode[-1].bias)
            self.projection = nn.Conv2d(384, dimension, 1)
            self.decoder_linear = nn.Conv2d(dimension, 384, 1)
            self.decoder_residual = nn.Sequential(nn.Conv2d(dimension, 96, 1), nn.GELU(), nn.Conv2d(96, 384, 1))
            nn.init.zeros_(self.decoder_residual[-1].weight)
            nn.init.zeros_(self.decoder_residual[-1].bias)
        with torch.no_grad():
            self.projection.weight.copy_(basis.t()[:, :, None, None])
            self.projection.bias.copy_(-basis.t() @ mean)
            self.decoder_linear.weight.copy_(basis[:, :, None, None])
            self.decoder_linear.bias.copy_(mean)
        self.codebook = None if centers is None else VectorCodebook(centers, normalized, update)
        self.spatial = SpatialDecoder(dimension, seed) if spatial else None

    def encode(self, maps):
        context, _, _ = self.context(maps)
        return self.projection(maps[:, 2]) + context

    def forward(self, maps):
        latent = self.encode(maps)
        details = {}
        if self.codebook is not None:
            latent, details = self.codebook(latent)
        restored = self.decoder_linear(latent) + self.decoder_residual(latent)
        if self.spatial is not None:
            restored = restored + self.spatial(latent)
        return restored, details



def make_recovery(arm, basis, mean, centers=None):
    spec = arm_spec(arm)
    if basis.shape != (384, spec["dimension"]):
        raise ValueError("Basis dimension does not match arm")
    if spec["count"] and (centers is None or centers.shape != (spec["count"], spec["dimension"])):
        raise ValueError("Codebook dimension does not match arm")
    return LatentRecovery(basis, mean, centers if spec["count"] else None,
        normalized=spec["normalized"], update=spec["update"], spatial=spec["spatial"])
