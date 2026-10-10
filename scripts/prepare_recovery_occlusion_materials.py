#!/usr/bin/env python3
"""Reproduce the locked independent material collection; never reads dataset images."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keypoint_comparison1.landmark_detection.occlusion_materials import build_materials, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dataset_adapters/recovery_abc_materials_v1")
    parser.add_argument("--expected-sha256", default="2972f0f3036761f9863a02f05485789c9a64b0568475d1a5d7217c8392daeafd")
    args = parser.parse_args()
    success = args.output_dir / "success.json"
    success.unlink(missing_ok=True)
    manifest = build_materials(args.output_dir)
    if sha256(manifest) != args.expected_sha256:
        raise ValueError("Generated material collection differs from the locked manifest")
    success.write_text(json.dumps(dict(status="passed",manifest=str(manifest),sha256=sha256(manifest)),indent=2)+"\n")
    print(f"{manifest}: {sha256(manifest)}")


if __name__ == "__main__":
    main()
