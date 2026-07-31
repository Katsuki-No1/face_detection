from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3_deimv2_common import ensure_output_dir, load_checkpoint, load_config, resolve_data_paths, save_run_context  # noqa: E402
from scripts.train_dino import validate_detection_split  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct-ellipse DEIMv2-L on an ellipse COCO split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, default=0.30)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--bbox-reference-map",
        type=float,
        default=None,
        help="Optional bbox mAP50:95 baseline used for the <=0.01 non-regression gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    config = load_config(args.config)
    config, _ = resolve_data_paths(config, args.data_root, split=args.split)
    save_run_context(output_dir, config)
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    image_size = int(args.image_size or config.get("data", {}).get("image_size", 960))

    build_config = copy.deepcopy(config)
    build_config.get("model", {}).pop("init_checkpoint", None)
    model = build_detector_from_config(build_config).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    metrics = validate_detection_split(
        model,
        config,
        device=device,
        split=args.split,
        image_size=image_size,
        fixed_threshold=float(args.fixed_threshold),
        score_min=float(args.score_min),
        max_images=args.max_images,
        batch_size=int(args.batch_size),
    )
    acceptance_cfg = dict(config.get("acceptance", {}))
    configured_references = acceptance_cfg.get("bbox_reference_map", {})
    configured_reference = configured_references.get(args.split) if isinstance(configured_references, dict) else None
    bbox_reference = args.bbox_reference_map if args.bbox_reference_map is not None else configured_reference
    max_bbox_drop = float(acceptance_cfg.get("max_bbox_map_drop", 0.01))
    bbox_non_regression = (
        None
        if bbox_reference is None
        else float(metrics["mAP"]) >= float(bbox_reference) - max_bbox_drop
    )
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
        "acceptance": {
            "ellipse_beats_bbox_ellipse_baseline": metrics["ellipse_mAP"] > metrics["bbox_ellipse_mAP"],
            "bbox_reference_map": float(bbox_reference) if bbox_reference is not None else None,
            "max_bbox_map_drop": max_bbox_drop,
            "bbox_non_regression": bbox_non_regression,
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
