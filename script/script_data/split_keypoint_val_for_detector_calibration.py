#!/usr/bin/env python3
"""Split validation videos into detector-error calibration and keypoint selection.

The split unit is metadata.source_video_id, so frames from one video can never
appear on both sides.  Video ordering is a stable SHA-256 ranking rather than
the incidental order in either COCO file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-annotation", type=Path, required=True)
    parser.add_argument("--rtmpose-annotation", type=Path, required=True)
    parser.add_argument("--dino-calibration-output", type=Path, required=True)
    parser.add_argument("--dino-keypoint-val-output", type=Path, required=True)
    parser.add_argument("--rtmpose-keypoint-val-output", type=Path, required=True)
    parser.add_argument("--calibration-video-count", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_id(image: dict[str, Any]) -> str:
    value = str(image.get("metadata", {}).get("source_video_id", "")).strip()
    if not value:
        raise ValueError(
            f"image id={image.get('id')} is missing metadata.source_video_id"
        )
    return value


def _ranked_videos(coco: dict[str, Any]) -> list[str]:
    videos = {_video_id(image) for image in coco.get("images", [])}
    return sorted(
        videos,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value),
    )


def _image_identity(coco: dict[str, Any]) -> set[tuple[int, str, str]]:
    return {
        (
            int(image["id"]),
            str(image["file_name"]),
            _video_id(image),
        )
        for image in coco.get("images", [])
    }


def _subset(coco: dict[str, Any], videos: Iterable[str]) -> dict[str, Any]:
    selected = set(videos)
    images = [
        image for image in coco.get("images", []) if _video_id(image) in selected
    ]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    result = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = images
    result["annotations"] = annotations
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(path: Path, coco: dict[str, Any]) -> dict[str, Any]:
    usable = 0
    visibility = {"unlabeled": 0, "occluded": 0, "visible": 0}
    for annotation in coco.get("annotations", []):
        values = list(annotation.get("keypoints", []))[2::3]
        if any(float(value) > 0 for value in values):
            usable += 1
        for value in values:
            state = int(float(value))
            visibility[
                "visible" if state >= 2 else "occluded" if state == 1 else "unlabeled"
            ] += 1
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "usable_keypoint_instances": usable,
        "visibility_counts": visibility,
        "source_video_ids": sorted(
            {_video_id(image) for image in coco.get("images", [])}
        ),
    }


def main() -> int:
    args = parse_args()
    dino = json.loads(args.dino_annotation.read_text(encoding="utf-8"))
    rtmpose = json.loads(args.rtmpose_annotation.read_text(encoding="utf-8"))
    if _image_identity(dino) != _image_identity(rtmpose):
        raise RuntimeError("DINO and RTMPose validation exports do not share image identities")
    ranked = _ranked_videos(dino)
    count = int(args.calibration_video_count)
    if count <= 0 or count >= len(ranked):
        raise ValueError(
            "calibration-video-count must leave at least one video for keypoint validation"
        )
    calibration_videos = ranked[:count]
    keypoint_videos = ranked[count:]
    outputs = (
        (
            args.dino_calibration_output,
            _subset(dino, calibration_videos),
        ),
        (
            args.dino_keypoint_val_output,
            _subset(dino, keypoint_videos),
        ),
        (
            args.rtmpose_keypoint_val_output,
            _subset(rtmpose, keypoint_videos),
        ),
    )
    report = {
        "selection": "sha256(source_video_id), ascending",
        "calibration_videos": calibration_videos,
        "keypoint_validation_videos": keypoint_videos,
        "outputs": {},
    }
    for path, value in outputs:
        _atomic_json(path, value)
        report["outputs"][path.name] = _summary(path, value)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
