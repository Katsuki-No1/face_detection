from __future__ import annotations

import argparse
import csv
import json
import math
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
from src.data import AugmentedFaceKeypointDataset, FaceKeypointDataset  # noqa: E402
from src.models.ellipse_target_assigner import CenterFaceEllipseTargetAssigner  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview DINO CenterFace ellipse augmentation targets.")
    parser.add_argument("--config", type=Path, default=None, help="Single config to preview.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Directory containing E*.yaml configs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/augmentation_debug"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--respect-probabilities",
        action="store_true",
        help="Use exact train probabilities. By default preview forces enabled stochastic ops to appear.",
    )
    return parser.parse_args()


def config_paths(args: argparse.Namespace) -> list[Path]:
    if args.config:
        return [args.config]
    if args.config_dir:
        return [
            path
            for path in sorted(args.config_dir.glob("E*.yaml"))
            if path.is_file() and load_yaml_config(path).get("experiment", {}).get("active", True) is not False
        ]
    raise ValueError("--config or --config-dir is required")


def experiment_name(config: dict[str, Any], path: Path) -> str:
    return str(config.get("experiment", {}).get("name") or config.get("train", {}).get("name") or path.stem)


def make_dataset(config: dict[str, Any], split: str, seed: int, *, respect_probabilities: bool):
    data_cfg = dict(config.get("data", {}))
    data_cfg["split"] = split
    base = FaceKeypointDataset(
        data_root=data_cfg.get("data_root"),
        split=split,
        annotation_path=data_cfg.get("annotation_path"),
        annotation_format=str(data_cfg.get("format", "auto")),
        image_root=data_cfg.get("image_root"),
        metadata_path=data_cfg.get("metadata_path"),
        image_size=data_cfg.get("image_size") or config.get("train", {}).get("imgsz", 896),
        include_negative=bool(data_cfg.get("include_negative", True)),
    )
    aug_cfg = dict(config.get("augmentation", {}))
    aug_cfg["enabled"] = True
    if not respect_probabilities:
        if float(aug_cfg.get("mosaic", 0.0) or 0.0) > 0.0:
            aug_cfg["mosaic"] = 1.0
        if float(aug_cfg.get("fliplr", 0.0) or 0.0) > 0.0:
            aug_cfg["fliplr"] = 1.0
    return AugmentedFaceKeypointDataset(base, aug_cfg, seed=seed)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def ellipse_polygon(cx: float, cy: float, rx: float, ry: float, theta: float, *, steps: int = 72) -> list[tuple[float, float]]:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    points = []
    for index in range(steps):
        angle = 2.0 * math.pi * index / steps
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append((cx + x * cos_theta - y * sin_theta, cy + x * sin_theta + y * cos_theta))
    return points


def draw_target(image: Image.Image, target: dict[str, Any]) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = target.get("boxes", torch.zeros((0, 4))).detach().cpu()
    ellipses = target.get("ellipses", torch.zeros((0, 5))).detach().cpu()
    keypoints = target.get("keypoints", torch.zeros((0, 5, 2))).detach().cpu()
    masks = target.get("keypoint_mask", torch.zeros((0, 5), dtype=torch.bool)).detach().cpu()
    for index in range(ellipses.shape[0]):
        cx, cy, rx, ry, theta = [float(value) for value in ellipses[index].tolist()]
        polygon = ellipse_polygon(cx, cy, rx, ry, theta)
        draw.line(polygon + [polygon[0]], fill=(255, 50, 50), width=3)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 255, 255))
        if index < boxes.shape[0]:
            x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 190, 0), width=2)
        if index < keypoints.shape[0]:
            for point_index, (x, y) in enumerate(keypoints[index].tolist()):
                if not bool(masks[index, point_index].item()):
                    continue
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 0))
                draw.text((x + 4, y + 4), str(point_index), fill=(255, 255, 0))
        draw.text((cx + 5, cy + 5), f"id:{index} rx:{rx:.1f} ry:{ry:.1f}", fill=(255, 255, 255))
    debug = target.get("metadata", {}).get("augmentation_debug", {}) if isinstance(target.get("metadata"), dict) else {}
    if debug:
        draw.text((8, 8), json.dumps(debug, ensure_ascii=False), fill=(255, 255, 255))
    return image


def dummy_predictions_for_assigner(target: dict[str, Any], image_size: tuple[int, int], config: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
    image_h, image_w = image_size
    patch = int(config.get("model", {}).get("backbone", {}).get("patch_size", 14))
    num_levels = int(config.get("model", {}).get("neck", {}).get("num_levels", 3))
    num_keypoints = int(config.get("model", {}).get("head", {}).get("num_keypoints", 5))
    p4_h = max(1, image_h // patch)
    p4_w = max(1, image_w // patch)
    shapes = [(p4_h, p4_w)]
    if num_levels >= 2:
        shapes = [(p4_h * 2, p4_w * 2), (p4_h, p4_w)]
    if num_levels >= 3:
        shapes.append(((p4_h + 1) // 2, (p4_w + 1) // 2))
    return {
        "center_logits": [torch.zeros((1, 1, h, w), dtype=torch.float32) for h, w in shapes],
        "ellipse_reg": [torch.zeros((1, 5, h, w), dtype=torch.float32) for h, w in shapes],
        "kpt_reg": [torch.zeros((1, num_keypoints * 2, h, w), dtype=torch.float32) for h, w in shapes],
        "bbox_reg": [torch.zeros((1, 4, h, w), dtype=torch.float32) for h, w in shapes],
    }


def save_heatmap(path: Path, assignments: dict[str, Any], image_size: tuple[int, int]) -> None:
    levels = assignments["center_targets"]
    heatmap = torch.zeros((1, 1, image_size[0], image_size[1]), dtype=torch.float32)
    for level in levels:
        resized = torch.nn.functional.interpolate(level.detach().cpu(), size=image_size, mode="nearest")
        heatmap = torch.maximum(heatmap, resized)
    array = (heatmap[0, 0].clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    cards = []
    for row in rows:
        image_name = Path(row["overlay"]).name
        heatmap_name = Path(row["heatmap"]).name
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'<img src="overlays/{image_name}" alt="{image_name}">',
                    f'<img src="heatmaps/{heatmap_name}" alt="{heatmap_name}">',
                    f'<div>{row["image_id"]}</div>',
                    f'<div>instances {row["instances"]} / positives {row["positive_count"]} / ignored {row["ignored_instances"]}</div>',
                    "</article>",
                ]
            )
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #171717; }}
    h1 {{ font-size: 20px; margin: 0 0 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d7d7d2; border-radius: 6px; padding: 8px; }}
    img {{ width: 100%; height: auto; display: block; margin-bottom: 6px; }}
    div {{ font-size: 12px; color: #444; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def safe_name(value: Any, limit: int = 96) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return text[:limit].strip("._") or "sample"


def preview_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(path)
    name = experiment_name(config, path)
    seed = int(args.seed if args.seed is not None else config.get("train", {}).get("seed", 42))
    dataset = make_dataset(config, args.split, seed, respect_probabilities=bool(args.respect_probabilities))
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty for {path}")
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    assigner = CenterFaceEllipseTargetAssigner()
    rows: list[dict[str, Any]] = []
    max_samples = min(int(args.samples), len(dataset))
    for index in range(max_samples):
        image_tensor, target = dataset[index]
        image_h, image_w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        predictions = dummy_predictions_for_assigner(target, (image_h, image_w), config)
        assignments = assigner.assign(predictions, [target], (image_h, image_w))
        filename = safe_name(target.get("image_id", index))
        overlay = output_dir / "overlays" / f"{index:03d}_{filename}.jpg"
        heatmap = output_dir / "heatmaps" / f"{index:03d}_{filename}_center.jpg"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        draw_target(tensor_to_image(image_tensor), target).save(overlay)
        save_heatmap(heatmap, assignments, (image_h, image_w))
        debug = target.get("metadata", {}).get("augmentation_debug", {}) if isinstance(target.get("metadata"), dict) else {}
        rows.append(
            {
                "experiment": name,
                "index": index,
                "image_id": str(target.get("image_id", "")),
                "instances": int(target["boxes"].shape[0]),
                "positive_count": int(assignments["positive_count"].item()),
                "visible_kpt_count": int(assignments["visible_kpt_count"].item()),
                "ignored_instances": int(debug.get("ignored_instances", 0)),
                "clipped_instances": int(debug.get("clipped_instances", 0)),
                "keypoints_outside": int(debug.get("keypoints_outside", 0)),
                "overlay": str(overlay),
                "heatmap": str(heatmap),
            }
        )
    summary = {
        "experiment": name,
        "config": str(path),
        "output_dir": str(output_dir),
        "samples": len(rows),
        "instances": sum(row["instances"] for row in rows),
        "positive_count": sum(row["positive_count"] for row in rows),
        "ignored_instances": sum(row["ignored_instances"] for row in rows),
        "clipped_instances": sum(row["clipped_instances"] for row in rows),
        "keypoints_outside": sum(row["keypoints_outside"] for row in rows),
    }
    (output_dir / "preview_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "preview_index.csv", rows)
    write_html(output_dir / "index.html", rows, f"{name} augmentation preview")
    return summary


def main() -> int:
    args = parse_args()
    summaries = [preview_config(path, args) for path in config_paths(args)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preview_manifest.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_root / "preview_manifest.csv", summaries)
    print(json.dumps({"output_root": str(args.output_root), "experiments": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
