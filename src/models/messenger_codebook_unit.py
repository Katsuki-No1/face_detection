"""VQ with one unit-vector space for assignment, fitting, and decoding."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.messenger_codebook_factorial import ARMS as HISTORICAL_ARMS, arm_spec
from src.models.messenger_codebook_factorial import make_recovery as historical_recovery

ARMS = tuple(arm for arm in HISTORICAL_ARMS if arm.split('_')[0] in ('B', 'D', 'E'))
EPSILON = 1e-6


def unit_centers(centers):
    norms = centers.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= EPSILON).any():
        raise ValueError('Code vectors must be finite and have a nonzero norm')
    return centers / norms


class UnitVectorCodebook(nn.Module):
    def __init__(self, centers, update, decay=.99, chunk_size=256):
        super().__init__()
        if centers.ndim != 2 or update not in ('gradient', 'ema'):
            raise ValueError('Expected K x D centers and a defined update rule')
        self.embedding = nn.Parameter(unit_centers(centers.detach()), requires_grad=update == 'gradient')
        self.update = update
        self.decay = decay
        self.chunk_size = chunk_size
        self.register_buffer('ema_steps', torch.zeros((), dtype=torch.long))
        self.begin_step()

    def begin_step(self):
        self._counts = None
        self._sums = None

    def forward(self, latent):
        batch, width, height, columns = latent.shape
        normalized = F.normalize(latent, dim=1, eps=EPSILON)
        flat = normalized.permute(0, 2, 3, 1).reshape(-1, width)
        # The same code vectors are used for selection, decoding and fitting.
        codes = F.normalize(self.embedding, dim=1, eps=EPSILON)
        with torch.no_grad():
            ids = torch.cat([(part @ codes.T).argmax(1) for part in flat.detach().split(self.chunk_size)])
            if self.training and self.update == 'ema':
                self.accumulate(flat.detach(), ids)
        selected = F.embedding(ids, codes).reshape(batch, height, columns, width).permute(0, 3, 1, 2)
        quantized = selected + (normalized - normalized.detach())
        commitment = (normalized - selected.detach()).square().mean(1)
        code_loss = (selected - normalized.detach()).square().mean(1) if self.update == 'gradient' else torch.zeros_like(commitment)
        return quantized, dict(indices=ids.reshape(batch, height, columns), commitment=commitment,
                              code_loss=code_loss, quantization_error=commitment.detach())

    @torch.no_grad()
    def accumulate(self, vectors, ids):
        counts = torch.bincount(ids, minlength=len(self.embedding))
        order = ids.argsort(stable=True)
        sums = torch.segment_reduce(vectors[order], 'sum', lengths=counts)
        if self._counts is None:
            self._counts, self._sums = counts, sums
        else:
            self._counts += counts
            self._sums += sums

    @torch.no_grad()
    def end_step(self):
        if self.update == 'gradient':
            self.embedding.copy_(unit_centers(self.embedding))
        elif self._counts is not None:
            populated = (self._counts > 0) & (self._sums.norm(dim=1) > EPSILON)
            directions = unit_centers(self._sums[populated])
            blended = self.decay * self.embedding[populated] + (1-self.decay) * directions
            self.embedding[populated] = unit_centers(blended)
            self.ema_steps.add_(1)
        self.begin_step()


def make_recovery(arm, basis, mean, raw_centers):
    if arm not in ARMS:
        raise ValueError(f'Not an affected condition: {arm}')
    model = historical_recovery(arm, basis, mean, raw_centers)
    model.codebook = UnitVectorCodebook(raw_centers, arm_spec(arm)['update'])
    return model
