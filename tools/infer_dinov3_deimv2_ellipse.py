from __future__ import annotations

import argparse
import copy
import json
import math
import sys
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
    parse_image_size,
    prepare_image_tensor,
    load_checkpoint,
    load_config,
    save_run_context,
    scale_detections_to_original,
)
from src.evaluation import ellipse_polygon  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct-ellipse DEIMv2-L image inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", default=None, help="INT or HEIGHTxWIDTH")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--draw-boxes", action="store_true")
    return parser.parse_args()


def detection_records(
    image_id: str,
    detections: dict[str, torch.Tensor],
    names: list[str],
    coco_ids: list[int],
) -> list[dict[str, Any]]:
    boxes = detections["boxes"].detach().cpu()
    ellipses = detections["ellipses"].detach().cpu()
    scores = detections["scores"].detach().cpu()
    labels = detections["labels"].detach().cpu()
    records = []
    for index in range(boxes.shape[0]):
        label_index = int(labels[index])
        category_id = coco_ids[label_index] if 0 <= label_index < len(coco_ids) else label_index + 1
        category_name = names[label_index] if 0 <= label_index < len(names) else str(category_id)
        x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
        cx, cy, major, minor, theta = [float(value) for value in ellipses[index].tolist()]
        records.append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "category_name": category_name,
                "score": float(scores[index]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "ellipse": {
                    "cx": cx,
                    "cy": cy,
                    "semi_major": major,
                    "semi_minor": minor,
                    "angle_deg": math.degrees(theta),
                },
            }
        )
    return records


def save_overlay(path: Path, image: Image.Image, records: list[dict[str, Any]], *, draw_boxes: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for row in records:
        ellipse = row["ellipse"]
        values = [
            ellipse["cx"],
            ellipse["cy"],
            ellipse["semi_major"],
            ellipse["semi_minor"],
            math.radians(ellipse["angle_deg"]),
        ]
        points = ellipse_polygon(values, steps=96)
        draw.line([*points, points[0]], fill="red", width=3)
        if draw_boxes:
            draw.rectangle(row["bbox_xyxy"], outline="yellow", width=2)
        draw.text(
            (ellipse["cx"], max(0.0, ellipse["cy"] - ellipse["semi_major"] - 14.0)),
            f"{row['category_name']} {row['score']:.2f}",
            fill="red",
        )
    image.save(path)


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output)
    config = load_config(args.config)
    save_run_context(output_dir, config)
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    data_cfg = dict(config.get("data", {}) or {})
    image_size = parse_image_size(args.image_size or data_cfg.get("image_size", 960))
    if image_size is None:
        raise ValueError("image_size must be configured")
    resize_mode = str(data_cfg.get("resize_mode", "stretch"))
    pad_color = data_cfg.get("pad_color")
    size_divisor = int(data_cfg.get("size_divisor", 32))
    names = class_names(config)
    coco_ids = category_ids(config)

    build_config = copy.deepcopy(config)
    build_config.get("model", {}).pop("init_checkpoint", None)
    model = build_detector_from_config(build_config).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    model.score_threshold = float(args.score_thr)
    if args.topk is not None:
        model.detections_per_image = int(args.topk)
        model.postprocessor.num_top_queries = int(args.topk)
    model.eval()

    predictions = []
    visual_dir = output_dir / "visualizations"
    for image_path in image_paths(args.input):
        image = Image.open(image_path).convert("RGB")
        tensor, input_hw, original_hw, geometry, padding_mask = prepare_image_tensor(
            image,
            image_size,
            resize_mode=resize_mode,
            pad_color=pad_color,
            size_divisor=size_divisor,
        )
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            detections = model(
                tensor.to(device),
                padding_mask=(
                    padding_mask.unsqueeze(0).to(device)
                    if padding_mask is not None
                    else None
                ),
            )[0]
        detections = scale_detections_to_original(
            detections,
            input_hw=input_hw,
            original_hw=original_hw,
            geometry=geometry,
        )
        rows = detection_records(image_path.name, detections, names, coco_ids)
        predictions.extend(rows)
        save_overlay(visual_dir / image_path.name, image.copy(), rows, draw_boxes=bool(args.draw_boxes))

    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "images": len(image_paths(args.input)),
        "detections": len(predictions),
        "score_threshold": float(args.score_thr),
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "visualization_dir": str(visual_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
