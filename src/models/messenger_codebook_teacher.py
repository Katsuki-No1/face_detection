"""Fresh controls and factorial recovery with externally supplied teacher g."""
import torch
from src.models.messenger_codebook_factorial import (
    ARMS as FACTORIAL_ARMS, DIMENSIONS, COUNTS, LatentRecovery,
    arm_spec as factorial_spec, make_recovery as factorial_model,
)
from src.models.messenger_multiaxis import MultiAxisRecovery

ARMS = ("continuous32", "regression384", "continuous16") + FACTORIAL_ARMS[1:]

def arm_spec(arm):
    if arm in ("regression384", "continuous16"):
        return dict(dimension=384 if arm == "regression384" else 16, count=0,
                    normalized=False, update="none", spatial=False, mode=arm)
    return factorial_spec(arm)

def make_recovery(arm, basis, mean, centers=None):
    if arm == "regression384":
        return MultiAxisRecovery("scale")
    if arm == "continuous16":
        return LatentRecovery(basis[:,:16], mean)
    return factorial_model(arm, basis, mean, centers)

def teacher_blend(original, restored, teacher_g):
    """Teacher g is an external constant, never predicted or optimized here."""
    if original.shape != restored.shape or teacher_g.shape != (original.shape[0], *original.shape[2:]):
        raise ValueError("Teacher-g shape mismatch")
    if not torch.isfinite(teacher_g).all() or teacher_g.min()<0 or teacher_g.max()>1.00001:
        raise ValueError("Teacher g must be finite and in [0,1]")
    fixed = teacher_g.detach()
    if fixed.dtype == torch.bool:
        return torch.where(fixed[:,None], restored, original)
    fixed = fixed.to(original).clamp(0,1)[:,None]
    return (1-fixed)*original + fixed*restored
