import argparse
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/fine-tune a YOLO Pose model for face pose.")
    parser.add_argument("--data", type=Path, default=Path("datasets/face_pose/face_pose.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", default="auto", help="Batch size or 'auto'.")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--project", default="runs/face_pose")
    parser.add_argument("--name", default="yolo11n_face_pose")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def parse_batch(value: str) -> int | float | str:
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt before training.") from error

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": parse_batch(args.batch),
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "pretrained": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
