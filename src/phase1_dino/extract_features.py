from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import apply_cli_overrides, load_config, output_dir
from .dataset import prepare_crops
from .dino import build_extractor, save_feature_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen DINO crop and patch features for Phase 1 analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/dino_phase1_feature_analysis.yaml"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--mock-features", action="store_true")
    return parser.parse_args()


def extract_features(config: dict[str, Any], max_images: int | None = None) -> dict[str, Any]:
    out_dir = output_dir(config)
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    crops, image_records = prepare_crops(config, out_dir, max_images=max_images)
    extractor = build_extractor(config)

    crop_features = []
    for record in crops:
        with Image.open(record["crop_path"]) as image:
            feature, _ = extractor.crop_feature(image.convert("RGB"), record)
        crop_features.append(feature.astype(np.float32))
    feature_array = np.stack(crop_features, axis=0) if crop_features else np.empty((0, 0), dtype=np.float32)
    np.save(features_dir / "crop_features.npy", feature_array)

    patch_features = []
    patch_image_paths = []
    patch_image_ids = []
    patch_widths = []
    patch_heights = []
    patch_bboxes = []
    patch_grids = []
    sample_count = min(int(config.get("analysis", {}).get("similarity_map_samples", 30)), len(image_records))
    for image_record in image_records[:sample_count]:
        with Image.open(image_record["image_path"]) as image:
            tokens, grid = extractor.image_patch_tokens(image.convert("RGB"), image_record)
        patch_features.append(tokens.astype(np.float32))
        patch_image_paths.append(image_record["image_path"])
        patch_image_ids.append(image_record["image_id"])
        patch_widths.append(int(image_record["width"]))
        patch_heights.append(int(image_record["height"]))
        patch_bboxes.append(json.dumps(image_record.get("face_bboxes", [])))
        patch_grids.append(grid)
    if patch_features:
        patch_array = np.stack(patch_features, axis=0)
    else:
        patch_array = np.empty((0, 0, 0, feature_array.shape[-1] if feature_array.ndim == 2 else 0), dtype=np.float32)
    np.savez_compressed(
        features_dir / "patch_features_sample.npz",
        patch_features=patch_array,
        image_paths=np.array(patch_image_paths, dtype=object),
        image_ids=np.array(patch_image_ids, dtype=object),
        widths=np.array(patch_widths, dtype=np.int32),
        heights=np.array(patch_heights, dtype=np.int32),
        bboxes_json=np.array(patch_bboxes, dtype=object),
        grid_hw=np.array(patch_grids, dtype=np.int32),
    )
    save_feature_info(features_dir / "feature_run_info.json", extractor, config, len(crops))
    shutil.copy2(Path(config["_config_path"]), out_dir / "config_used.yaml") if config.get("_config_path") else None

    summary = {
        "output_dir": str(out_dir),
        "crop_count": len(crops),
        "image_count": len(image_records),
        "crop_features": str(features_dir / "crop_features.npy"),
        "crop_features_shape": list(feature_array.shape),
        "crop_features_meta": str(features_dir / "crop_features_meta.csv"),
        "patch_features_sample": str(features_dir / "patch_features_sample.npz"),
        "model_name": extractor.model_name,
        "device": extractor.device,
        "frozen_backbone": True,
        "fine_tuned": False,
        "detection_head_trained": False,
        "keypoint_head_trained": False,
    }
    (out_dir / "extract_features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        max_images=args.max_images,
        output_dir_override=args.output_dir,
        mock_features=True if args.mock_features else None,
        split=args.split,
    )
    summary = extract_features(config, max_images=args.max_images)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
