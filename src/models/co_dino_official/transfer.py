"""Fail-closed new-WIDER EMA -> new-ellipse initialization, never resume."""
import hashlib
from pathlib import Path

import torch
from .provenance import ARCHITECTURE_VERSION


def validate_transfer_state(model, state):
    target = model.state_dict()
    expected_missing = {k for k in target if k.startswith("query_head.ellipse_head.")}
    missing = set(target) - set(state)
    unexpected = set(state) - set(target)
    wrong_shape = {k for k in set(target) & set(state) if target[k].shape != state[k].shape}
    if not expected_missing or missing != expected_missing or unexpected or wrong_shape:
        raise RuntimeError(f"WIDER transfer contract violation: missing={sorted(missing)}, "
            f"expected={sorted(expected_missing)}, unexpected={sorted(unexpected)}, "
            f"shape_mismatch={sorted(wrong_shape)}")
    return sorted(expected_missing)


def load_wider_checkpoint(model, config):
    path = Path(config["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("config", {}).get("model", {})
    if source.get("type") != "dinov3_codino_official" or source.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("Initialization requires the new official-SFP WIDER checkpoint identity")
    if source.get("ellipse_head", {}).get("enabled", False):
        raise RuntimeError("Transfer source must be a bbox-only WIDER model")
    if not config.get("prefer_ema", True) or not isinstance(payload.get("ema"), dict):
        raise RuntimeError("Required WIDER EMA is absent; raw-weight fallback is forbidden")
    if path.name != "best_val_map.pt":
        raise RuntimeError("Use the validation-selected WIDER best_val_map.pt, not last/resume")
    state = payload["ema"]
    if isinstance(state.get("module"), dict):
        state = state["module"]
    missing = validate_transfer_state(model, state)
    model.load_state_dict(state, strict=False)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    expected_hash = config.get("sha256")
    if expected_hash and digest != expected_hash:
        raise RuntimeError("WIDER checkpoint hash mismatch")
    return dict(path=str(path), sha256=digest, source_epoch=payload.get("epoch"),
                source="ema", missing_keys=missing, unexpected_keys=[], shape_mismatch=[],
                architecture_version=ARCHITECTURE_VERSION)
