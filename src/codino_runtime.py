"""Official-SFP resource safety and bounded, scalar-only progress reporting.

The allocator budget is shared by preflight and production, independent of AMP.
It bounds PyTorch reservations, not all driver/other-process GPU allocations.
"""
import math
import time
from datetime import datetime, timezone

import torch

from src.wandb_utils import write_json_atomic


OFFICIAL_TYPES = {"dinov3_codino_official", "dinov3_codino_official_ellipse"}


def memory_budget_policy(config):
    if config.get("model", {}).get("type") not in OFFICIAL_TYPES:
        return None
    settings = config.get("batch_preflight", {})
    fraction = float(settings.get("cuda_memory_fraction", .88))
    safety = float(settings.get("max_vram_fraction", .90))
    if not (math.isfinite(fraction) and math.isfinite(safety)
            and 0 < fraction <= safety <= .90):
        raise ValueError("Official-SFP CUDA budget must be positive and <= max_vram_fraction <= 0.90")
    return dict(allocator_fraction=fraction, max_vram_fraction=safety)


def require_matching_memory_budgets(production, probe):
    if memory_budget_policy(production) != memory_budget_policy(probe):
        raise ValueError("Probe must preserve production CUDA memory budget")


def configure_memory_budget(config, device):
    policy = memory_budget_policy(config)
    if policy is None or torch.device(device).type != "cuda":
        return None
    # Call before building the detector/EMA/optimizer, on EVERY CUDA path.
    # Without this in production WDDM may spill to shared memory instead of
    # raising the CUDA OOM needed by the guarded launcher's existing fallback.
    torch.cuda.set_per_process_memory_fraction(policy["allocator_fraction"], device=device)
    return dict(schema_version=1, **policy, device=str(device),
                scope="pytorch_allocator_not_total_driver_memory")


class RuntimeProgress:
    """One replace-in-place JSON, no tensors/history or extra CUDA synchronize."""
    def __init__(self, path, device, *, interval_seconds=60.):
        self.path, self.device = path, device
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.last_write = float("-inf")

    def update(self, *, phase, epoch, completed_batches, total_batches,
               optimizer_step, force=False, loss=None):
        now = time.monotonic()
        if not force and now - self.last_write < self.interval_seconds:
            return
        record = dict(schema_version=1, phase=phase, epoch=int(epoch),
                      completed_batches=int(completed_batches), total_batches=int(total_batches),
                      optimizer_step=int(optimizer_step), loss=loss,
                      process_elapsed_seconds=now - self.started,
                      updated_at=datetime.now(timezone.utc).isoformat())
        if torch.device(self.device).type == "cuda":
            record.update(cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                          cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        write_json_atomic(self.path, record)
        self.last_write = now
