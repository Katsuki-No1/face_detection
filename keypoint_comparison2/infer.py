#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_comparison2.config import MODEL_NAMES, load_config  # noqa: E402
from keypoint_comparison2.inference import (  # noqa: E402
    DEIMv2LFaceDetector,
    JointFaceKeypointPipeline,
    TopDownFaceKeypointPipeline,
)
from keypoint_comparison2.training import load_trained_model  # noqa: E402


COLORS = ((255, 70, 70), (70, 150, 255), (255, 220, 0), (255, 140, 0), (160, 80, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production image/video inference for keypoint comparison 2")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "keypoint_comparison2/configs/production.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _pipeline(config, model_name: str, checkpoint: Path | None, device: torch.device):
    model = load_trained_model(config, model_name, checkpoint=checkpoint, device=device)
    if model_name == "deimv2_joint":
        return JointFaceKeypointPipeline(
            model,
            input_size=int(config.raw["models"][model_name]["input_size"]),
            device=device,
            score_threshold=float(config.raw["detector"]["score_threshold"]),
            visibility_threshold=float(config.raw["evaluation"]["visibility_threshold"]),
        )
    detector = DEIMv2LFaceDetector(config.raw["detector"], device=device)
    return TopDownFaceKeypointPipeline(
        detector,
        model,
        input_size=int(config.raw["models"][model_name]["input_size"]),
        device=device,
        crop_scale=float(config.raw["data"]["augmentation"]["scale_factor"]),
        batch_size=int(config.raw["evaluation"]["batch_size"]),
        flip_test=bool(config.raw["models"][model_name].get("test_time_flip", True)),
    )


def _draw(image: Image.Image, predictions: list[dict]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for prediction in predictions:
        ellipse = prediction.get("ellipse")
        if (
            prediction.get("face_geometry_type") == "ellipse"
            and isinstance(ellipse, list)
            and len(ellipse) == 5
        ):
            center_x, center_y, major, minor, theta = (
                float(value) for value in ellipse
            )
            angles = np.linspace(0.0, 2.0 * np.pi, 73)
            cosine, sine = np.cos(theta), np.sin(theta)
            local_x = major * np.cos(angles)
            local_y = minor * np.sin(angles)
            contour = [
                (
                    center_x + cosine * x - sine * y,
                    center_y + sine * x + cosine * y,
                )
                for x, y in zip(local_x, local_y, strict=True)
            ]
            draw.line(contour, fill=(0, 255, 80), width=3)
        else:
            draw.rectangle(
                prediction["bbox_xyxy"],
                outline=(0, 255, 80),
                width=3,
            )
        draw.text(
            (prediction["bbox_xyxy"][0], prediction["bbox_xyxy"][1]),
            f"{prediction['score']:.3f}",
            fill=(0, 255, 80),
        )
        for color, point, visibility in zip(COLORS, prediction["keypoints"], prediction["visibility"]):
            if float(visibility) < 0.5:
                continue
            x, y = point
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    return result


def _run_image(path: Path, output: Path, pipeline) -> None:
    image = Image.open(path).convert("RGB")
    predictions = pipeline(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    _draw(image, predictions).save(output)
    output.with_suffix(".json").write_text(
        json.dumps({"input": str(path), "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_video(path: Path, output: Path, pipeline) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    manifest = output.with_suffix(".jsonl")
    with manifest.open("w", encoding="utf-8") as handle:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = pipeline(rgb)
            rendered = np.asarray(_draw(Image.fromarray(rgb), predictions))
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            handle.write(json.dumps({"frame": frame_index, "predictions": predictions}, ensure_ascii=False) + "\n")
            frame_index += 1
    writer.release()
    capture.release()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, project_root=REPO_ROOT)
    device = torch.device(args.device)
    pipeline = _pipeline(config, args.model, args.checkpoint, device)
    if args.input.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        _run_video(args.input, args.output, pipeline)
    else:
        _run_image(args.input, args.output, pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
