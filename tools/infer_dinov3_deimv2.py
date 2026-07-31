from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3_deimv2_common import (  # noqa: E402
    category_ids,
    class_names,
    ensure_output_dir,
    image_paths,
    image_to_tensor,
    load_checkpoint,
    load_config,
    save_run_context,
    scale_detections_to_original,
)
from src.models import build_detector_from_config  # noqa: E402


COLORS = {
    "head": "lime",
    "face": "red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv3 + official-style DEIMv2 detector inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output experiment directory.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--use-nms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def detection_records(
    image_id: str,
    detections: dict[str, torch.Tensor],
    names: list[str],
    coco_ids: list[int],
) -> list[dict[str, Any]]:
    boxes = detections["boxes"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    labels = detections.get("labels")
    labels = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.zeros((boxes.shape[0],), dtype=torch.long)
    records = []
    for index in range(boxes.shape[0]):
        score = float(scores[index])
        label_index = int(labels[index])
        name = names[label_index] if 0 <= label_index < len(names) else str(label_index)
        category_id = coco_ids[label_index] if 0 <= label_index < len(coco_ids) else label_index + 1
        x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        records.append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "category_name": name,
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return records


def save_overlay(path: Path, image: Image.Image, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for row in records:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        color = COLORS.get(str(row["category_name"]), "yellow")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), f"{row['category_name']} {row['score']:.2f}", fill=color)
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output)
    config = load_config(args.config)
    save_run_context(output_dir, config)
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    names = class_names(config)
    coco_ids = category_ids(config)
    image_size = int(args.image_size or config.get("data", {}).get("image_size") or config.get("train", {}).get("imgsz", 960))

    model = build_detector_from_config(config).to(device)
    missing: list[str] = []
    unexpected: list[str] = []
    if args.checkpoint.exists():
        missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    elif not args.allow_random_init:
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if hasattr(model, "score_threshold"):
        model.score_threshold = float(args.score_thr)
    if hasattr(model, "postprocessor") and hasattr(model.postprocessor, "score_threshold"):
        model.postprocessor.score_threshold = float(args.score_thr)
    if hasattr(model, "postprocessor"):
        if args.topk is not None:
            if hasattr(model.postprocessor, "topk"):
                model.postprocessor.topk = int(args.topk)
            if hasattr(model.postprocessor, "detections_per_image"):
                model.postprocessor.detections_per_image = int(args.topk)
            if hasattr(model.postprocessor, "num_top_queries"):
                model.postprocessor.num_top_queries = int(args.topk)
        if args.use_nms is not None:
            model.postprocessor.use_nms = bool(args.use_nms)
    if args.topk is not None and hasattr(model, "detections_per_image"):
        model.detections_per_image = int(args.topk)
    model.eval()

    predictions: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    visual_dir = output_dir / "visualizations"
    for path in image_paths(args.input):
        image = Image.open(path).convert("RGB")
        tensor, input_hw, original_hw = image_to_tensor(image, image_size)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            detections = model(tensor.to(device))[0]
        detections = scale_detections_to_original(detections, input_hw=input_hw, original_hw=original_hw)
        rows = detection_records(path.name, detections, names, coco_ids)
        predictions.extend(rows)
        counts = Counter(row["category_name"] for row in rows)
        per_image_rows.append({"image_id": path.name, "detections": len(rows), **{name: counts.get(name, 0) for name in names}})
        save_overlay(visual_dir / path.name, image.copy(), rows)

    class_counts = Counter(row["category_name"] for row in predictions)
    class_rows = [{"category_name": name, "detections": int(class_counts.get(name, 0))} for name in names]
    (output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "per_image_counts.csv", per_image_rows, ["image_id", "detections", *names])
    write_csv(output_dir / "class_counts.csv", class_rows, ["category_name", "detections"])
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "device": device,
        "image_size": image_size,
        "score_threshold": float(args.score_thr),
        "images": len(per_image_rows),
        "detections": len(predictions),
        "class_counts": {name: int(class_counts.get(name, 0)) for name in names},
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "visualization_dir": str(visual_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "infer.log").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
