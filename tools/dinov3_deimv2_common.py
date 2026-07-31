from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402
from src.data.coco_split_audit import audit_coco_splits  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)

    def resolve_in(section: dict[str, Any], key: str) -> None:
        value = section.get(key)
        if value in {None, ""}:
            return
        candidate = Path(str(value)).expanduser()
        section[key] = str(candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve())

    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        for key in ("data_root", "annotation_path", "image_root", "metadata_path"):
            resolve_in(data_cfg, key)
        annotation_paths = data_cfg.get("annotation_paths")
        if isinstance(annotation_paths, dict):
            for key in tuple(annotation_paths):
                value = annotation_paths.get(key)
                if value in {None, ""}:
                    continue
                candidate = Path(str(value)).expanduser()
                annotation_paths[key] = str(candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve())
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        backbone_cfg = model_cfg.get("backbone", {})
        if isinstance(backbone_cfg, dict):
            for key in ("hub_repo", "weights"):
                resolve_in(backbone_cfg, key)
        init_cfg = model_cfg.get("init_checkpoint", {})
        if isinstance(init_cfg, dict):
            for key in ("path", "checkpoint", "weights"):
                resolve_in(init_cfg, key)
    assets_cfg = config.get("assets", {})
    if isinstance(assets_cfg, dict):
        for asset in assets_cfg.values():
            if isinstance(asset, dict):
                resolve_in(asset, "path")
    return config


def class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("model", {}).get("class_names") or config.get("class_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return ["head", "face"]


def category_ids(config: dict[str, Any]) -> list[int]:
    ids = config.get("data", {}).get("category_ids")
    if isinstance(ids, list) and ids:
        return [int(value) for value in ids]
    return list(range(1, len(class_names(config)) + 1))


def run_configured_leakage_audit(config: dict[str, Any]) -> dict[str, Any] | None:
    data_cfg = config.get("data", {})
    audit_cfg = data_cfg.get("leakage_audit", {}) or {}
    if not bool(audit_cfg.get("enabled", False)):
        return None
    data_root = data_cfg.get("data_root")
    if not data_root:
        raise ValueError("data.leakage_audit requires data.data_root")
    configured_category_ids = category_ids(config)
    if len(configured_category_ids) != 1:
        raise ValueError(
            "data.leakage_audit currently requires exactly one configured category id"
        )
    splits = tuple(str(split) for split in audit_cfg.get("splits", ("train", "val", "test")))
    configured_paths = data_cfg.get("annotation_paths")
    return audit_coco_splits(
        Path(data_root),
        annotation_paths=configured_paths if isinstance(configured_paths, dict) else None,
        splits=splits,
        expected_category_id=configured_category_ids[0],
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _annotation_candidates(data_root: Path, split: str) -> list[Path]:
    return [
        data_root / "annotations" / f"instances_{split}.json",
        data_root / "annotations" / f"{split}.json",
        data_root / f"instances_{split}.json",
        data_root / f"{split}.json",
    ]


def resolve_data_paths(config: dict[str, Any], data_root: Path | None, split: str | None = None) -> tuple[dict[str, Any], bool]:
    """Apply --data-root/--split and return (config, should_force_synthetic)."""

    cfg = dict(config)
    cfg["data"] = dict(config.get("data", {}))
    data_cfg = cfg["data"]
    if split is not None:
        data_cfg["split"] = split
    split_name = str(data_cfg.get("split", "train"))
    if data_root is not None:
        data_cfg["data_root"] = str(data_root)

    synthetic_if_missing = bool(data_cfg.get("synthetic_if_missing", False))
    root = Path(data_cfg["data_root"]) if data_cfg.get("data_root") else None
    configured_paths = data_cfg.get("annotation_paths", {})
    split_annotation = configured_paths.get(split_name) if isinstance(configured_paths, dict) else None
    annotation = Path(split_annotation) if split_annotation else (Path(data_cfg["annotation_path"]) if data_cfg.get("annotation_path") else None)
    if root is not None:
        if annotation is None or not annotation.exists():
            for candidate in _annotation_candidates(root, split_name):
                if candidate.exists():
                    annotation = candidate
                    break
        if annotation is not None and not annotation.exists():
            annotation = None
    if annotation is not None:
        data_cfg["annotation_path"] = str(annotation)

    if root is not None and annotation is not None:
        image_root = _choose_image_root(root, annotation, split_name, data_cfg.get("image_root"))
        if image_root is not None:
            data_cfg["image_root"] = str(image_root)

    force_synthetic = synthetic_if_missing and (annotation is None or not annotation.exists())
    if force_synthetic:
        data_cfg.pop("annotation_path", None)
        data_cfg.pop("image_root", None)
        data_cfg.pop("data_root", None)
    return cfg, force_synthetic


def _choose_image_root(root: Path, annotation: Path, split: str, configured: str | Path | None) -> Path | None:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    images = data.get("images", [])
    file_name = str(images[0].get("file_name", "")) if images else ""
    candidates = []
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else root / configured_path)
    candidates.extend([root, root / "images" / split, root / "images", root / split])
    if not file_name:
        return candidates[0] if candidates else None
    file_path = Path(file_name)
    if file_path.is_absolute():
        return None
    for candidate in candidates:
        if (candidate / file_path).exists():
            return candidate
    return candidates[0] if candidates else None


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input image path does not exist: {input_path}")
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())


def image_to_tensor(image: Image.Image, image_size: int | None) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    image = image.convert("RGB")
    original_w, original_h = image.size
    if image_size is not None:
        image = image.resize((int(image_size), int(image_size)), Image.BILINEAR)
    input_w, input_h = image.size
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, (input_h, input_w), (original_h, original_w)


def scale_detections_to_original(
    detections: dict[str, torch.Tensor],
    *,
    input_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    original_h, original_w = int(original_hw[0]), int(original_hw[1])
    scale_x = original_w / float(max(1, input_w))
    scale_y = original_h / float(max(1, input_h))
    scaled = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in detections.items()}
    boxes = scaled.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y
        boxes[:, 0::2].clamp_(0, float(original_w))
        boxes[:, 1::2].clamp_(0, float(original_h))
    ellipses = scaled.get("ellipses")
    if isinstance(ellipses, torch.Tensor) and ellipses.numel() > 0:
        from src.models.dinov3_deimv2_ellipse import ellipse_to_aabb, scale_ellipses_anisotropic

        ellipses = scale_ellipses_anisotropic(ellipses, scale_x=scale_x, scale_y=scale_y)
        scaled["ellipses"] = ellipses
        ellipse_boxes = ellipse_to_aabb(ellipses)
        ellipse_boxes[:, 0::2].clamp_(0, float(original_w))
        ellipse_boxes[:, 1::2].clamp_(0, float(original_h))
        scaled["ellipse_boxes"] = ellipse_boxes
    return scaled


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, strict: bool = False) -> tuple[list[str], list[str]]:
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("Loading .safetensors requires safetensors; install requirements.txt.") from error
        checkpoint: Any = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("ema"), dict):
        ema_state = checkpoint["ema"]
        state_dict = ema_state.get("module", ema_state)
    else:
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def save_run_context(output_dir: Path, config: dict[str, Any], *, config_name: str = "config.yaml") -> None:
    ensure_output_dir(output_dir)
    (output_dir / config_name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for name, command in {
        "git_status.txt": ["git", "status", "--short"],
        "git_diff.patch": ["git", "diff", "--"],
    }.items():
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
            (output_dir / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        except Exception as error:
            (output_dir / name).write_text(f"failed to capture {command}: {error}\n", encoding="utf-8")
