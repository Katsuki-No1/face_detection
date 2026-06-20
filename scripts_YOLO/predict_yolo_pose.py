import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO Pose inference on images, folders, or videos.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/yolo_pose_predictions"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt before prediction.") from error

    model = YOLO(str(args.weights))
    predict_kwargs = {
        "source": str(args.source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "project": str(args.out_dir.parent),
        "name": args.out_dir.name,
        "save": True,
        "exist_ok": True,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    model.predict(**predict_kwargs)
    print(f"output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
