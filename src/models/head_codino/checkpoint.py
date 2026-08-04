from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .backbone import EXPECTED_DINOV3_VITS16PLUS_PARAMETERS
from .codino import EXPECTED_COMPACT_CODINO_PARAMETERS
from .model import EXPECTED_TOTAL_PARAMETERS
from .sfp import EXPECTED_SFP_PARAMETERS


class CheckpointCompatibilityError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_audit(model: nn.Module) -> dict[str, dict[str, int | bool]]:
    components = {
        "dinov3_vits16plus": (
            model.backbone.model,
            EXPECTED_DINOV3_VITS16PLUS_PARAMETERS,
        ),
        "sfp": (model.sfp, EXPECTED_SFP_PARAMETERS),
        "compact_codino": (model.query_head, EXPECTED_COMPACT_CODINO_PARAMETERS),
        "total": (model, EXPECTED_TOTAL_PARAMETERS),
    }
    report: dict[str, dict[str, int | bool]] = {}
    for name, (component, expected) in components.items():
        actual = sum(parameter.numel() for parameter in component.parameters())
        report[name] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return report


def _extract_state_dict(
    checkpoint: Any,
    *,
    prefer_ema: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, Mapping):
        raise CheckpointCompatibilityError("Checkpoint root must be a mapping")
    candidates = ("ema", "model", "state_dict") if prefer_ema else ("model", "state_dict", "ema")
    selected: Any = checkpoint
    source = "<root>"
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            selected = value
            source = key
            break
    if isinstance(selected, Mapping) and isinstance(selected.get("module"), Mapping):
        selected = selected["module"]
        source += ".module"
    state = {
        str(key): value for key, value in selected.items() if isinstance(value, torch.Tensor)
    }
    if not state:
        raise CheckpointCompatibilityError("No tensor state dict was found in the checkpoint")
    return state, source


def _normalize_prefixes(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    normalized: dict[str, torch.Tensor] = {}
    renames: list[dict[str, str]] = []
    strip_model_prefix = bool(state) and all(key.startswith("model.") for key in state)
    for original, value in state.items():
        key = original
        if strip_model_prefix:
            key = key[len("model.") :]
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key in normalized:
            raise CheckpointCompatibilityError(
                f"Prefix normalization produced a duplicate key: {key}"
            )
        normalized[key] = value
        if key != original:
            renames.append({"from": original, "to": key})
    return normalized, renames


def audit_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    prefer_ema: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state, source = _extract_state_dict(checkpoint, prefer_ema=prefer_ema)
    state, renames = _normalize_prefixes(raw_state)
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    shape_mismatches = [
        {
            "key": key,
            "expected": list(expected_state[key].shape),
            "actual": list(state[key].shape),
        }
        for key in sorted(set(expected_state) & set(state))
        if expected_state[key].shape != state[key].shape
    ]
    report: dict[str, Any] = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "state_source": source,
        "prefer_ema": bool(prefer_ema),
        "renamed_keys": renames,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "component_parameters": parameter_audit(model),
    }
    report["strict_compatible"] = not (
        missing
        or unexpected
        or shape_mismatches
        or any(not item["matches"] for item in report["component_parameters"].values())
    )
    return report, state


def load_checkpoint_exact(
    model: nn.Module,
    path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    report, state = audit_checkpoint(model, path, prefer_ema=prefer_ema)
    if not report["strict_compatible"]:
        raise CheckpointCompatibilityError(
            "Exact checkpoint load was rejected: "
            f"missing={len(report['missing_keys'])}, "
            f"unexpected={len(report['unexpected_keys'])}, "
            f"shape_mismatches={len(report['shape_mismatches'])}"
        )
    model.load_state_dict(state, strict=True)
    report["loaded"] = True
    return report


def initialize_detection_head(
    model: nn.Module,
    path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Warm-start SFP and Co-DINO while preserving the configured DINO weights.

    This is intentionally different from resume: optimizer/scheduler state is
    not restored, and the DINO backbone remains the exact weight configured by
    the new run.  It is used when the previously frozen HEAD detector becomes
    the initialization for a new, fully unfrozen detector stage.
    """

    report, state = audit_checkpoint(model, path, prefer_ema=prefer_ema)
    prefixes = ("sfp.", "query_head.")
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }
    compatible = {
        key: state[key]
        for key in expected
        if key in state and tuple(state[key].shape) == tuple(expected[key].shape)
    }
    missing = sorted(set(expected) - set(compatible))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if missing or mismatched:
        raise CheckpointCompatibilityError(
            "HEAD-only initialization was rejected: "
            f"missing={missing[:20]}, shape_mismatches={mismatched[:20]}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "path": report["path"],
        "sha256": report["sha256"],
        "state_source": report["state_source"],
        "prefer_ema": bool(prefer_ema),
        "loaded_components": ["sfp", "query_head"],
        "preserved_component": "dinov3_vits16plus",
        "loaded_tensors": len(compatible),
        "loaded_elements": sum(int(value.numel()) for value in compatible.values()),
        "missing_keys": [],
        "shape_mismatches": [],
    }


__all__ = [
    "CheckpointCompatibilityError",
    "audit_checkpoint",
    "initialize_detection_head",
    "load_checkpoint_exact",
    "parameter_audit",
    "sha256_file",
]
