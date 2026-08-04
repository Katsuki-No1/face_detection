#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.head_pose_geometry.cache import (
    PseudoPoseFilterConfig,
    PseudoPoseRecord,
    filter_pseudo_pose,
    pose_cache_payload,
    rotation_consistency,
    write_pose_cache,
)
from src.head_pose_geometry.crops import HeadCropBuilder
from src.head_pose_geometry.heads import HeadDetection
from src.head_pose_geometry.pose_encoder import build_pose_encoder, sha256_file
from src.head_pose_geometry.rotations import flip_rotation_horizontal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train-only multi-consistency 6DRepNet360 labels."
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-instances", type=int)
    return parser.parse_args()


def resolve_image(root: Path, file_name: str) -> Path:
    value = Path(file_name)
    for candidate in (root / value, root / "images" / value):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(file_name)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in data["images"]}
    teacher = (
        build_pose_encoder(
            checkpoint=args.checkpoint,
            training_mode="frozen",
        )
        .to(device)
        .eval()
    )
    builder = HeadCropBuilder(output_size=224)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[PseudoPoseRecord] = []
    rows = [row for row in data["annotations"] if int(row.get("category_id", -1)) == 1]
    if args.max_instances is not None:
        rows = rows[: args.max_instances]
    config = PseudoPoseFilterConfig()
    for row in rows:
        image_info = images[int(row["image_id"])]
        image = np.asarray(
            Image.open(
                resolve_image(args.image_root, str(image_info["file_name"]))
            ).convert("RGB"),
            dtype=np.float32,
        )
        tensor = (
            torch.from_numpy(image).permute(2, 0, 1).contiguous().to(device) / 255.0
        )
        x, y, width, height = (float(value) for value in row["bbox"])
        detection = HeadDetection(
            box_xyxy=(x, y, x + width, y + height),
            confidence=1.0,
            instance_id=int(row["id"]),
            source="ground_truth",
        )
        base = builder.build(tensor, [detection], training=False)
        jitter = builder.build(
            tensor,
            [detection],
            training=True,
            generator=generator,
        )
        inputs = torch.cat(
            (base.rgb, jitter.rgb, torch.flip(base.rgb, dims=(-1,))),
            dim=0,
        )
        with torch.inference_mode():
            prediction = teacher(inputs)
        rotation = prediction["rotation_matrix"]
        jitter_error = float(rotation_consistency(rotation[0], rotation[1]))
        restored_flip = flip_rotation_horizontal(rotation[2])
        flip_error = float(rotation_consistency(rotation[0], restored_flip))
        # The released 6DRepNet360 regressor has no calibrated confidence
        # head. Do not present the newly initialized adapter head as teacher
        # evidence: derive uncertainty from independent crop/flip stability.
        uncertainty = math.sqrt(0.5 * (jitter_error**2 + flip_error**2))
        confidence = math.exp(-uncertainty)
        accepted, reasons = filter_pseudo_pose(
            confidence=confidence,
            uncertainty=uncertainty,
            crop_jitter_consistency_radians=jitter_error,
            horizontal_flip_consistency_radians=flip_error,
            config=config,
        )
        records.append(
            PseudoPoseRecord(
                image_id=str(row["image_id"]),
                instance_id=str(row["id"]),
                split="train",
                source_video_id=str(image_info.get("source_video_id", "")),
                rotation_matrix=rotation[0].cpu().tolist(),
                teacher_model="6DRepNet360",
                teacher_checkpoint_id=args.checkpoint.name,
                teacher_checkpoint_sha256=sha256_file(args.checkpoint),
                confidence=confidence,
                uncertainty=uncertainty,
                crop_jitter_consistency_radians=jitter_error,
                horizontal_flip_consistency_radians=flip_error,
                temporal_consistency_radians=None,
                second_teacher_consistency_radians=None,
                accepted=accepted,
                rejection_reasons=reasons,
            )
        )
    payload = pose_cache_payload(
        records,
        teacher={
            "name": "6DRepNet360",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "load_diagnostics": teacher.pretrained_load_info,
        },
        filter_config=config,
        seed=args.seed,
    )
    digest = write_pose_cache(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "records": payload["record_count"],
                "accepted": payload["accepted_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
