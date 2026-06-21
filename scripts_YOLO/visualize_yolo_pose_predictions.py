import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_yolo_pose_metrics import (  # noqa: E402
    bbox_iou,
    image_paths,
    label_path_for_image,
    load_ground_truth,
    result_predictions,
    visible_keypoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO Pose GT/prediction overlays.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for image overlays.") from error
    return cv2


def draw_label(cv2: Any, image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(image, text, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_xyxy_bbox(
    cv2: Any,
    image: Any,
    bbox: list[float],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        draw_label(cv2, image, label, (x1, y1 - 5), color)


def draw_gt_keypoints(cv2: Any, image: Any, gt: dict[str, Any], color: tuple[int, int, int]) -> None:
    for point in visible_keypoints(gt):
        center = (int(round(point["x"])), int(round(point["y"])))
        cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)
        draw_label(cv2, image, point["name"], (center[0] + 4, center[1] - 4), color)


def draw_prediction_keypoints(cv2: Any, image: Any, prediction: dict[str, Any], color: tuple[int, int, int]) -> None:
    for point in prediction.get("keypoints", []):
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        if x <= 0 and y <= 0:
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)
        draw_label(cv2, image, point.get("name", ""), (center[0] + 4, center[1] + 12), color)


def best_iou(prediction: dict[str, Any], ground_truth: list[dict[str, Any]]) -> float:
    if not ground_truth:
        return 0.0
    return max(bbox_iou(prediction["bbox"], gt["bbox"]) for gt in ground_truth)


def render_overlay(
    cv2: Any,
    image_path: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gt_color = (0, 220, 0)
    pred_color = (0, 0, 255)
    fp_color = (180, 0, 180)
    for gt_index, gt in enumerate(ground_truth):
        draw_xyxy_bbox(cv2, image, gt["bbox"], gt_color, f"GT {gt_index}")
        draw_gt_keypoints(cv2, image, gt, gt_color)
    for pred_index, prediction in enumerate(predictions):
        iou = best_iou(prediction, ground_truth)
        color = pred_color if iou >= iou_threshold else fp_color
        draw_xyxy_bbox(cv2, image, prediction["bbox"], color, f"P {pred_index} {prediction['score']:.2f} IoU {iou:.2f}")
        draw_prediction_keypoints(cv2, image, prediction, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {
        "image": str(image_path),
        "visualization": str(output_path),
        "gt_count": len(ground_truth),
        "prediction_count": len(predictions),
        "false_positive_like": sum(1 for prediction in predictions if best_iou(prediction, ground_truth) < iou_threshold),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    cards = []
    for row in rows:
        image_path = Path(row["visualization"]).name
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f'<img src="overlays/{image_path}" alt="{Path(row["image"]).name}">',
                    f'<div class="caption">{Path(row["image"]).name}</div>',
                    f'<div class="meta">GT {row["gt_count"]} / Pred {row["prediction_count"]} / FP-like {row["false_positive_like"]}</div>',
                    "</article>",
                ]
            )
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #171717; }}
    h1 {{ font-size: 20px; margin: 0 0 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d7d7d2; border-radius: 6px; padding: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    .caption {{ font-size: 12px; margin-top: 6px; word-break: break-all; }}
    .meta {{ font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">
    {"".join(cards)}
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2 = import_cv2()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt first.") from error

    model = YOLO(str(args.weights))
    images = image_paths(args.data_root, args.split)[: args.max_images]
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.prediction_conf_min,
        device=args.device,
        stream=True,
        verbose=False,
    )

    rows = []
    for image_path, result in zip(images, results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        ground_truth = load_ground_truth(label_path_for_image(args.data_root, args.split, image_path), width, height)
        predictions = [
            prediction
            for prediction in result_predictions(result)
            if prediction["score"] >= args.conf_threshold
        ]
        rows.append(
            render_overlay(
                cv2,
                image_path,
                args.output_dir / "overlays" / image_path.name,
                ground_truth,
                predictions,
                args.iou_threshold,
            )
        )

    summary = {
        "weights": str(args.weights),
        "data_root": str(args.data_root),
        "split": args.split,
        "images": len(rows),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "visualization_dir": str(args.output_dir / "overlays"),
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "visualization_index.csv", rows)
    write_html(args.output_dir / "index.html", rows, f"YOLO Pose {args.split} inference overlays")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
