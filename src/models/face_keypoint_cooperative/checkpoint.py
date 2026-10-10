from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path}")
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover - dependency is optional.
            raise RuntimeError("safetensors is required to load this checkpoint") from error
        value: Any = load_file(str(path), device="cpu")
    else:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    if prefer_ema and isinstance(value.get("ema"), Mapping):
        state: Any = value["ema"]
        if isinstance(state.get("shadow"), Mapping):
            state = state["shadow"]
        model_state: Any = value.get("model")
        if isinstance(model_state, Mapping) and isinstance(
            model_state.get("module"),
            Mapping,
        ):
            model_state = model_state["module"]
        if isinstance(state, Mapping) and isinstance(model_state, Mapping):
            # ModelEMA intentionally tracks floating-point parameters and
            # buffers only. Preserve non-floating buffers such as BatchNorm's
            # num_batches_tracked from the matching model state while keeping
            # every EMA-tracked value authoritative. Do not fill missing
            # floating-point tensors, so a genuinely incomplete EMA state is
            # still rejected by the strict loaders below.
            state = {
                **{
                    str(key): tensor
                    for key, tensor in model_state.items()
                    if isinstance(tensor, torch.Tensor)
                    and not torch.is_floating_point(tensor)
                },
                **state,
            }
    else:
        state = value.get("model", value)
    if isinstance(state, Mapping) and isinstance(state.get("module"), Mapping):
        state = state["module"]
    if isinstance(state, Mapping) and isinstance(state.get("shadow"), Mapping):
        state = state["shadow"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint has no model state: {path}")
    return {
        str(key): tensor
        for key, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
    }


def checkpoint_component_state(
    checkpoint_path: str | Path,
    component: str,
    *,
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract one strict component from a complete cooperative checkpoint.

    Cooperative stage EMA snapshots intentionally contain trainable tensors
    only.  Build the authoritative full state from ``model`` first and then
    overlay the available EMA shadow before selecting the component.  This is
    different from a detector-only strict checkpoint, where an incomplete EMA
    must remain an error instead of being silently filled from ``model``.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path}")
    if path.suffix.lower() == ".safetensors":
        state = checkpoint_state(path, prefer_ema=prefer_ema)
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Checkpoint is not a mapping: {path}")
        model_state: Any = raw.get("model")
        if isinstance(model_state, Mapping) and isinstance(
            model_state.get("module"), Mapping
        ):
            model_state = model_state["module"]
        if not isinstance(model_state, Mapping):
            raise TypeError(
                "A component checkpoint requires a complete 'model' state: "
                f"{path}"
            )
        state = {
            str(key): tensor
            for key, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor)
        }
        if prefer_ema and isinstance(raw.get("ema"), Mapping):
            ema: Any = raw["ema"]
            if isinstance(ema.get("shadow"), Mapping):
                ema = ema["shadow"]
            if isinstance(ema, Mapping):
                state.update(
                    {
                        str(key): tensor
                        for key, tensor in ema.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                )

    prefix = component.strip().rstrip(".") + "."
    selected: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state.items():
        key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
        if key.startswith(prefix):
            selected[key[len(prefix) :]] = tensor
    if not selected:
        raise KeyError(
            f"Checkpoint has no tensors for component {component!r}: {path}"
        )
    return selected


def load_face_checkpoint_strict(
    face_detector: nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
    component: str | None = None,
) -> dict[str, Any]:
    state = (
        checkpoint_component_state(
            checkpoint_path,
            component,
            prefer_ema=prefer_ema,
        )
        if component
        else checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    )
    expected = face_detector.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = {
        key: {
            "checkpoint": tuple(state[key].shape),
            "model": tuple(expected[key].shape),
        }
        for key in set(expected) & set(state)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Face checkpoint is not architecture-compatible: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={dict(list(shape_mismatch.items())[:20])}"
        )
    face_detector.load_state_dict(state, strict=True)
    modules: dict[str, dict[str, int | float]] = {}
    for prefix in ("backbone", "encoder", "decoder", "criterion", "postprocessor"):
        keys = [key for key in expected if key == prefix or key.startswith(prefix + ".")]
        elements = sum(int(expected[key].numel()) for key in keys)
        modules[prefix] = {
            "loaded_tensors": len(keys),
            "total_tensors": len(keys),
            "loaded_elements": elements,
            "total_elements": elements,
            "fraction": 1.0 if keys else 0.0,
        }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "source_component": component,
        "loaded_tensors": len(expected),
        "total_tensors": len(expected),
        "loaded_elements": sum(int(value.numel()) for value in expected.values()),
        "total_elements": sum(int(value.numel()) for value in expected.values()),
        "fraction": 1.0,
        "shape_mismatch": {},
        "missing": [],
        "unexpected": [],
        "modules": modules,
    }


def _pose_key_mapping(
    source_key: str,
    *,
    pose_decoder_layers: int,
) -> tuple[str, ...]:
    prefix = "head.query_head."
    if source_key == prefix + "landmark_queries":
        return ("landmark_queries",)
    if source_key.startswith(prefix + "layers."):
        tail = source_key[len(prefix + "layers.") :]
        layer, separator, remainder = tail.partition(".")
        if not separator:
            return ()
        replacements = {
            "self_attention.": "group_attention.",
            "norm1.": "norm_group.",
            "norm2.": "norm_cross.",
            "norm3.": "norm_ffn.",
            "ffn.": "ffn.",
        }
        for old, new in replacements.items():
            if remainder.startswith(old):
                return (f"layers.{layer}.{new}{remainder[len(old):]}",)
        return ()
    if source_key.startswith(prefix + "coordinate_heads."):
        return (source_key[len(prefix) :],)
    if source_key.startswith(prefix + "visibility_head."):
        return (source_key[len(prefix) :],)
    if source_key.startswith(prefix + "presence_head."):
        parameter = source_key[len(prefix + "presence_head.") :]
        return tuple(
            f"keypoint_confidence_heads.{layer}.{parameter}"
            for layer in range(pose_decoder_layers)
        )
    if source_key.startswith("head.refinement_stages."):
        return (source_key[len("head.") :],)
    return ()


def initialize_pose_from_query_refinement(
    pose_decoder: nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
    minimum_eligible_fraction: float = 0.65,
) -> dict[str, Any]:
    source = checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    target = pose_decoder.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str | list[str]] = {}
    intentional_incompatible: list[str] = []
    shape_mismatch: dict[str, dict[str, Any]] = {}
    eligible_elements = 0
    loaded_elements = 0
    eligible_source_tensors = 0
    loaded_source_tensors = 0
    source_head_elements = 0
    intentional_incompatible_elements = 0
    pose_decoder_layers = len(getattr(pose_decoder, "layers", ()))
    for source_key, tensor in source.items():
        if not (
            source_key.startswith("head.query_head.")
            or source_key.startswith("head.refinement_stages.")
        ):
            continue
        source_head_elements += int(tensor.numel())
        mapped_keys = _pose_key_mapping(
            source_key,
            pose_decoder_layers=pose_decoder_layers,
        )
        if not mapped_keys:
            intentional_incompatible.append(source_key)
            intentional_incompatible_elements += int(tensor.numel())
            continue
        value = tensor.squeeze(0) if source_key.endswith("landmark_queries") else tensor
        destinations = {
            mapped_key: target.get(mapped_key)
            for mapped_key in mapped_keys
        }
        missing_destinations = [
            mapped_key
            for mapped_key, destination in destinations.items()
            if destination is None
        ]
        if missing_destinations:
            intentional_incompatible.append(source_key)
            intentional_incompatible_elements += int(tensor.numel())
            continue
        eligible_source_tensors += 1
        eligible_elements += int(tensor.numel())
        mismatched = {
            mapped_key: tuple(destination.shape)
            for mapped_key, destination in destinations.items()
            if tuple(value.shape) != tuple(destination.shape)
        }
        if mismatched:
            shape_mismatch[source_key] = {
                "mapped_keys": list(mapped_keys),
                "checkpoint": tuple(value.shape),
                "model": mismatched,
            }
            continue
        for mapped_key in mapped_keys:
            mapped[mapped_key] = value
        source_to_target[source_key] = (
            mapped_keys[0]
            if len(mapped_keys) == 1
            else list(mapped_keys)
        )
        loaded_source_tensors += 1
        loaded_elements += int(value.numel())
    if shape_mismatch:
        raise RuntimeError(
            "Compatible Query+refinement parameters had shape mismatches: "
            f"{shape_mismatch}"
        )
    fraction = loaded_elements / max(1, eligible_elements)
    if fraction < float(minimum_eligible_fraction):
        raise RuntimeError(
            "Too little of the compatible Query+refinement checkpoint was loaded: "
            f"fraction={fraction:.4f}, required={minimum_eligible_fraction:.4f}"
        )
    with torch.no_grad():
        for key, value in mapped.items():
            target[key].copy_(value)
    pose_decoder.load_state_dict(target, strict=True)
    modules: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "loaded_tensors": 0,
            "loaded_elements": 0,
        }
    )
    for destination in mapped:
        component = destination.split(".", 1)[0]
        modules[component]["loaded_tensors"] = int(
            modules[component]["loaded_tensors"]
        ) + 1
        modules[component]["loaded_elements"] = int(
            modules[component]["loaded_elements"]
        ) + int(target[destination].numel())
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "target_state_head_mode": str(
            getattr(pose_decoder, "state_head_mode", "legacy_binary")
        ),
        "source_factorized_state_head": any(
            key.startswith("head.query_head.presence_head.")
            for key in source
        ),
        "loaded_tensors": len(mapped),
        "loaded_source_tensors": loaded_source_tensors,
        "eligible_source_tensors": eligible_source_tensors,
        "loaded_elements": loaded_elements,
        "eligible_source_elements": eligible_elements,
        "eligible_fraction": fraction,
        "source_head_elements": source_head_elements,
        "source_head_fraction": loaded_elements / max(1, source_head_elements),
        "intentional_incompatible_elements": intentional_incompatible_elements,
        "source_to_target": source_to_target,
        "shape_mismatch": shape_mismatch,
        "intentional_incompatible": intentional_incompatible,
        "newly_initialized": sorted(set(target) - set(mapped)),
        "modules": dict(modules),
    }


def load_cooperative_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    state = checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(state)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    if missing or unexpected or mismatch:
        raise RuntimeError(
            "Cooperative checkpoint mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={dict(list(mismatch.items())[:20])}"
        )
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_tensors": len(expected),
        "loaded_elements": sum(int(value.numel()) for value in expected.values()),
        "missing": [],
        "unexpected": [],
        "shape_mismatch": {},
    }


def initialize_shared_head_from_head_codino(
    shared_head_branch: nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Load the common SFP P3/P4/P5 and Co-DINO query tensors exactly.

    The standalone detector additionally owns the DINO backbone and an unused
    SFP P6 projection.  The integrated three-stage branch deliberately owns
    neither, so those source tensors are recorded and ignored.  Every tensor
    required by the integrated branch remains a strict requirement.
    """

    source = checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    expected = shared_head_branch.state_dict()
    compatible = {
        key: source[key]
        for key in expected
        if key in source
        and tuple(source[key].shape) == tuple(expected[key].shape)
    }
    missing = sorted(set(expected) - set(compatible))
    mismatch = {
        key: {
            "checkpoint": tuple(source[key].shape),
            "model": tuple(expected[key].shape),
        }
        for key in set(expected) & set(source)
        if tuple(source[key].shape) != tuple(expected[key].shape)
    }
    if missing or mismatch:
        raise RuntimeError(
            "Head Co-DINO checkpoint is not compatible with the shared HEAD "
            f"branch: missing={missing[:20]}, "
            f"shape_mismatch={dict(list(mismatch.items())[:20])}"
        )
    shared_head_branch.load_state_dict(compatible, strict=True)
    ignored = sorted(set(source) - set(expected))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "prefer_ema": bool(prefer_ema),
        "loaded_tensors": len(compatible),
        "loaded_elements": sum(int(value.numel()) for value in compatible.values()),
        "missing": [],
        "shape_mismatch": {},
        "ignored_source_tensors": ignored,
        "ignored_contract": "standalone DINO backbone and SFP P6 only",
    }


def initialize_shared_dino_from_head_codino(
    dino_backbone: nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Load the standalone HEAD detector's DINO into the shared backbone."""

    source = checkpoint_state(checkpoint_path, prefer_ema=prefer_ema)
    prefix = "backbone.model."
    dino_state = {
        key[len(prefix) :]: value
        for key, value in source.items()
        if key.startswith(prefix)
    }
    expected = dino_backbone.state_dict()
    missing = sorted(set(expected) - set(dino_state))
    unexpected = sorted(set(dino_state) - set(expected))
    mismatch = {
        key: {
            "checkpoint": tuple(dino_state[key].shape),
            "model": tuple(expected[key].shape),
        }
        for key in set(expected) & set(dino_state)
        if tuple(dino_state[key].shape) != tuple(expected[key].shape)
    }
    if missing or unexpected or mismatch:
        raise RuntimeError(
            "Head Co-DINO DINO checkpoint is not compatible with the shared "
            f"backbone: missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={dict(list(mismatch.items())[:20])}"
        )
    dino_backbone.load_state_dict(dino_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "prefer_ema": bool(prefer_ema),
        "source_prefix": prefix,
        "loaded_tensors": len(expected),
        "loaded_elements": sum(int(value.numel()) for value in expected.values()),
        "missing": [],
        "unexpected": [],
        "shape_mismatch": {},
    }


__all__ = [
    "checkpoint_state",
    "initialize_shared_head_from_head_codino",
    "initialize_shared_dino_from_head_codino",
    "initialize_pose_from_query_refinement",
    "load_cooperative_checkpoint_strict",
    "load_face_checkpoint_strict",
]
