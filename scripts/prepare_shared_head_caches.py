#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.head_codino.runtime import load_runtime_config, resolve_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_is_current(
    path: Path,
    *,
    checkpoint_sha256: str,
    annotation_sha256: str,
    minimum_score: float,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        return (
            int(payload.get("schema_version", 0)) == 1
            and metadata.get("detector_checkpoint_sha256") == checkpoint_sha256
            and metadata.get("annotation_sha256") == annotation_sha256
            and abs(float(metadata.get("minimum_score")) - minimum_score) < 1.0e-12
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build reusable locked-threshold HEAD predictions/caches once."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.30)
    parser.add_argument("--maximum-per-image", type=int, default=100)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    minimum_score = float(args.minimum_score)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Shared HEAD checkpoint not found: {checkpoint}")
    config = load_runtime_config(config_path)
    checkpoint_digest = sha256_file(checkpoint)
    split_records: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        annotation = resolve_path(config["data"][split]["annotation"], config=config)
        annotation_digest = sha256_file(annotation)
        prediction = output_root / "predictions" / f"{split}.json"
        cache = output_root / "head_cache" / f"{split}.json"
        if not cache_is_current(
            cache,
            checkpoint_sha256=checkpoint_digest,
            annotation_sha256=annotation_digest,
            minimum_score=minimum_score,
        ):
            prediction.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval_head_codino.py"),
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                    "--output",
                    str(prediction),
                ]
            )
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/build_predicted_head_cache.py"),
                    "--annotations",
                    str(annotation),
                    "--predictions",
                    str(prediction),
                    "--output",
                    str(cache),
                    "--split",
                    split,
                    "--detector-name",
                    "shared_head_codino_vits16plus_960_balanced_811_seed42",
                    "--checkpoint",
                    str(checkpoint),
                    "--minimum-score",
                    str(minimum_score),
                    "--maximum-per-image",
                    str(int(args.maximum_per_image)),
                ]
            )
        split_records[split] = {
            "annotation": str(annotation),
            "annotation_sha256": annotation_digest,
            "prediction": str(prediction),
            "cache": str(cache),
            "cache_sha256": sha256_file(cache),
        }

    payload = {
        "status": "complete",
        "shared_component": "DINOv3-ViT-S+/16 + compact Co-DINO HEAD",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "minimum_score": minimum_score,
        "maximum_per_image": int(args.maximum_per_image),
        "splits": split_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "cache_complete.json"
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
