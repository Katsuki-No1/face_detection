from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_detector_from_config, draw_ellipse_detections


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINO-SCRFD face/keypoint inference on local images.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None, help="Image file or directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--save-overlay-dir", type=Path, default=None)
    parser.add_argument("--smoke-dummy", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
    orig_w, orig_h = image.size
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, (orig_h, orig_w)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


def result_records(image_id: str, detections: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    boxes = detections.get("bboxes", detections["boxes"]).detach().cpu()
    scores = detections["scores"].detach().cpu()
    keypoints = detections["keypoints"].detach().cpu()
    ellipses = detections.get("ellipses")
    ellipses = ellipses.detach().cpu() if isinstance(ellipses, torch.Tensor) else None
    records = []
    for index in range(boxes.shape[0]):
        record = {
            "image_id": image_id,
            "score": float(scores[index]),
            "bbox": [float(value) for value in boxes[index].tolist()],
            "keypoints": [[float(value) for value in point] for point in keypoints[index].tolist()],
        }
        if ellipses is not None:
            record["ellipse"] = [float(value) for value in ellipses[index].tolist()]
        records.append(record)
    return records


def save_records(path: Path, records: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "score", "x1", "y1", "x2", "y2", "ellipse", "keypoints"])
        writer.writeheader()
        for record in records:
            x1, y1, x2, y2 = record["bbox"]
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "score": record["score"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "ellipse": json.dumps(record.get("ellipse", [])),
                    "keypoints": json.dumps(record["keypoints"]),
                }
            )


def save_overlay(path: Path, image: Image.Image, detections: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "ellipses" in detections:
        draw_ellipse_detections(image, detections).save(path)
        return
    draw = ImageDraw.Draw(image)
    for box, keypoints in zip(detections["boxes"].cpu(), detections["keypoints"].cpu()):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        for x, y in keypoints.tolist():
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="yellow")
    image.save(path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    image_size = args.image_size or config.get("data", {}).get("image_size") or config.get("train", {}).get("imgsz")
    model = build_detector_from_config(config).to(args.device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    all_records: list[dict[str, Any]] = []
    if args.smoke_dummy:
        tensor = torch.zeros((1, 3, int(image_size or 56), int(image_size or 56)), dtype=torch.float32, device=args.device)
        with torch.no_grad():
            detections = model(tensor)[0]
        all_records.extend(result_records("smoke_dummy", detections))
    else:
        for path in image_paths(args.source, smoke_dummy=False):
            image = Image.open(path).convert("RGB")
            tensor, _orig_size = image_to_tensor(image, int(image_size) if image_size else None)
            tensor = tensor.to(args.device)
            with torch.no_grad():
                detections = model(tensor)[0]
            all_records.extend(result_records(path.name, detections))
            if args.save_overlay_dir is not None:
                save_overlay(args.save_overlay_dir / path.name, image.copy(), detections)

    save_records(args.output, all_records, args.format)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
