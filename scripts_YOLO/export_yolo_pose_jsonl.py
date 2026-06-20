import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


KEYPOINT_NAMES = ["e1", "e2", "n", "m1", "m2"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
FRAME_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO Pose predictions as JSONL.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_pose_predictions.jsonl"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def source_fps(path: Path) -> float | None:
    if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.exists():
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else None


def image_time_seconds(path: str) -> float | None:
    match = FRAME_TIME_RE.search(Path(path).name)
    return None if match is None else float(match.group(1))


def tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def result_objects(result: Any) -> list[dict[str, Any]]:
    boxes_xyxy = tensor_to_list(result.boxes.xyxy) if result.boxes is not None else []
    scores = tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    keypoints_xy = tensor_to_list(result.keypoints.xy) if result.keypoints is not None else []
    keypoints_conf = tensor_to_list(result.keypoints.conf) if result.keypoints is not None else []
    objects = []
    for index, bbox in enumerate(boxes_xyxy):
        keypoints = {}
        xy_points = keypoints_xy[index] if index < len(keypoints_xy) else []
        conf_points = keypoints_conf[index] if index < len(keypoints_conf) else []
        for point_index, name in enumerate(KEYPOINT_NAMES):
            if point_index >= len(xy_points):
                keypoints[name] = [0.0, 0.0, 0]
                continue
            x, y = xy_points[point_index]
            point_conf = conf_points[point_index] if point_index < len(conf_points) else 1.0
            visibility = 2 if point_conf > 0 else 0
            keypoints[name] = [float(x), float(y), visibility]
        objects.append(
            {
                "class_name": "face",
                "bbox": [float(value) for value in bbox],
                "score": float(scores[index]) if index < len(scores) else 0.0,
                "keypoints": keypoints,
            }
        )
    return objects


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt before export.") from error

    model = YOLO(str(args.weights))
    predict_kwargs = {
        "source": str(args.source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "stream": True,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    fps = source_fps(args.source)
    is_video = args.source.suffix.lower() in VIDEO_EXTENSIONS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as file:
        for frame_index, result in enumerate(model.predict(**predict_kwargs)):
            result_path = getattr(result, "path", str(args.source))
            time_sec = frame_index / fps if is_video and fps else image_time_seconds(result_path)
            record = {
                "source": str(args.source),
                "video": str(args.source) if is_video else None,
                "image": result_path if not is_video else None,
                "frame_index": frame_index if is_video else 0,
                "time_sec": time_sec,
                "objects": result_objects(result),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
