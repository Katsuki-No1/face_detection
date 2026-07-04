from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINO-DEIM-style bbox detector inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None, help="Image file or directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["jsonl", "json", "csv"], default="jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--save-overlay-dir", type=Path, default=None)
    parser.add_argument("--smoke-dummy", action="store_true")
    parser.add_argument("--tiny-model", action="store_true", help="Use a dummy DINO backbone and narrow hidden dims for smoke tests.")
    return parser.parse_args()


def image_paths(source: Path | None, *, smoke_dummy: bool) -> list[Path]:
    if smoke_dummy:
        return []
    if source is None:
        raise ValueError("--source is required unless --smoke-dummy is set")
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def image_to_tensor(image: Image.Image, image_size: int | None) -> tuple[torch.Tensor, tuple[int, int]]:
    image = image.convert("RGB")
    original_w, original_h = image.size
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, (original_h, original_w)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def class_names(config: dict[str, Any]) -> list[str]:
    model_cfg = config.get("model", {})
    names = model_cfg.get("class_names") or config.get("class_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return ["object"]


def apply_tiny_model_overrides(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.setdefault("model", {})
    backbone = model_cfg.setdefault("backbone", {})
    backbone["embed_dim"] = 64
    backbone["dummy"] = True
    backbone["pretrained"] = False
    backbone["hub_repo"] = None
    backbone["hub_model"] = None
    backbone["hub_module"] = None
    backbone["weights"] = None
    backbone["strict_weights"] = False
    backbone["token_key"] = None
    backbone["call_method"] = "auto"
    model_cfg.setdefault("adapter", {})["hidden_dim"] = 64
    model_cfg.setdefault("adapter", {})["out_channels"] = 64
    model_cfg.setdefault("encoder", {})["hidden_dim"] = 64
    model_cfg.setdefault("encoder", {})["num_heads"] = 4
    decoder = model_cfg.setdefault("decoder", {})
    decoder["hidden_dim"] = 64
    decoder["num_heads"] = 4
    decoder["num_layers"] = min(1, int(decoder.get("num_layers", 1)))
    return config


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
    return scaled


def result_records(image_id: str, detections: dict[str, torch.Tensor], names: list[str]) -> list[dict[str, Any]]:
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    labels = detections.get("labels")
    labels = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.zeros((boxes.shape[0],), dtype=torch.long)
    rows = []
    for index in range(boxes.shape[0]):
        label_index = int(labels[index])
        rows.append(
            {
                "image_id": image_id,
                "label": names[label_index] if 0 <= label_index < len(names) else str(label_index),
                "score": float(scores[index]),
                "bbox_xyxy": [float(value) for value in boxes[index].tolist()],
            }
        )
    return rows


def save_records(path: Path, records: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if fmt == "jsonl":
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "label", "score", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for row in records:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            writer.writerow({"image_id": row["image_id"], "label": row["label"], "score": row["score"], "x1": x1, "y1": y1, "x2": x2, "y2": y2})


def save_overlay(path: Path, image: Image.Image, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for row in records:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), f"{row['label']} {row['score']:.2f}", fill="red")
    image.save(path)


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.tiny_model:
        config = apply_tiny_model_overrides(config)
    image_size = args.image_size or config.get("data", {}).get("image_size") or config.get("train", {}).get("imgsz")
    model = build_detector_from_config(config).to(args.device)
    if args.score_thr is not None and hasattr(model, "postprocessor"):
        model.postprocessor.score_threshold = float(args.score_thr)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    names = class_names(config)

    all_records: list[dict[str, Any]] = []
    if args.smoke_dummy:
        size = int(image_size or 64)
        tensor = torch.zeros((1, 3, size, size), dtype=torch.float32, device=args.device)
        with torch.no_grad():
            detections = model(tensor)[0]
        all_records.extend(result_records("smoke_dummy", detections, names))
    else:
        for path in image_paths(args.source, smoke_dummy=False):
            image = Image.open(path).convert("RGB")
            tensor, original_hw = image_to_tensor(image, int(image_size) if image_size else None)
            input_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
            with torch.no_grad():
                detections = model(tensor.to(args.device))[0]
            detections = scale_detections_to_original(detections, input_hw=input_hw, original_hw=original_hw)
            rows = result_records(path.name, detections, names)
            all_records.extend(rows)
            if args.save_overlay_dir is not None:
                save_overlay(args.save_overlay_dir / path.name, image.copy(), rows)
    save_records(args.output, all_records, args.format)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
