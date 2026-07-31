from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinov3_deimv2_common import image_to_tensor, load_checkpoint, load_config, scale_detections_to_original  # noqa: E402
from src.models import build_detector_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render direct-ellipse DEIMv2-L detections to a video.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--draw-boxes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    image_size = int(args.image_size or config.get("data", {}).get("image_size", 960))
    build_config = copy.deepcopy(config)
    build_config.get("model", {}).pop("init_checkpoint", None)
    model = build_detector_from_config(build_config).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, strict=False)
    model.score_threshold = float(args.score_thr)
    model.detections_per_image = int(args.topk)
    model.postprocessor.num_top_queries = int(args.topk)
    model.eval()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {args.output_video}")
    jsonl_path = args.output_jsonl or args.output_video.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    detection_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor, input_hw, original_hw = image_to_tensor(Image.fromarray(rgb), image_size)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                detections = model(tensor.to(device))[0]
            detections = scale_detections_to_original(detections, input_hw=input_hw, original_hw=original_hw)
            boxes = detections["boxes"].detach().cpu()
            ellipses = detections["ellipses"].detach().cpu()
            scores = detections["scores"].detach().cpu()
            labels = detections["labels"].detach().cpu()
            records = []
            for index in range(scores.shape[0]):
                cx, cy, major, minor, theta = [float(value) for value in ellipses[index].tolist()]
                box = [float(value) for value in boxes[index].tolist()]
                score = float(scores[index])
                cv2.ellipse(
                    frame,
                    (round(cx), round(cy)),
                    (max(1, round(major)), max(1, round(minor))),
                    math.degrees(theta),
                    0.0,
                    360.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                if args.draw_boxes:
                    cv2.rectangle(
                        frame,
                        (round(box[0]), round(box[1])),
                        (round(box[2]), round(box[3])),
                        (0, 255, 255),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (round(cx), max(12, round(cy - major))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                records.append(
                    {
                        "label": int(labels[index]),
                        "score": score,
                        "bbox_xyxy": box,
                        "ellipse": {
                            "cx": cx,
                            "cy": cy,
                            "semi_major": major,
                            "semi_minor": minor,
                            "angle_deg": math.degrees(theta),
                        },
                    }
                )
            jsonl_file.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / max(fps, 1.0e-6),
                        "detections": records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.write(frame)
            frame_index += 1
            detection_count += len(records)

    capture.release()
    writer.release()
    summary = {
        "input": str(args.input),
        "output_video": str(args.output_video),
        "output_jsonl": str(jsonl_path),
        "frames": frame_index,
        "detections": detection_count,
        "fps": fps,
        "load_state_missing_keys": missing,
        "load_state_unexpected_keys": unexpected,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
