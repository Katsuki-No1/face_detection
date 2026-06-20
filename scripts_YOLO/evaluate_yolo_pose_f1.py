import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pose bbox F1 across confidence thresholds.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        default="0.01:0.95:0.01",
        help="Either start:end:step or comma-separated thresholds.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def parse_thresholds(value: str, fixed_threshold: float | None) -> list[float]:
    if fixed_threshold is not None:
        return [fixed_threshold]
    if ":" in value:
        start, end, step = [float(item) for item in value.split(":", 2)]
        thresholds = []
        current = start
        while current <= end + 1e-12:
            thresholds.append(round(current, 6))
            current += step
        return thresholds
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as error:
        raise RuntimeError("OpenCV is required for image size loading.") from error
    return cv2


def image_paths(data_root: Path, split: str) -> list[Path]:
    root = data_root / "images" / split
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(data_root: Path, split: str, image_path: Path) -> Path:
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def load_ground_truth_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        xc, yc, bw, bh = values[1:5]
        box_width = bw * width
        box_height = bh * height
        cx = xc * width
        cy = yc * height
        boxes.append(
            {
                "bbox": [
                    cx - box_width / 2.0,
                    cy - box_height / 2.0,
                    cx + box_width / 2.0,
                    cy + box_height / 2.0,
                ]
            }
        )
    return boxes


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = bbox_area([x1, y1, x2, y2])
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / union if union > 0 else 0.0


def greedy_counts(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    filtered = [prediction for prediction in predictions if prediction["score"] >= threshold]
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(filtered):
            iou = bbox_iou(gt["bbox"], prediction["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
    tp = len(matched_gt)
    fp = len(filtered) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


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


def result_predictions(result: Any) -> list[dict[str, Any]]:
    boxes = tensor_to_list(result.boxes.xyxy) if result.boxes is not None else []
    scores = tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    return [
        {"bbox": [float(value) for value in box], "score": float(scores[index])}
        for index, box in enumerate(boxes)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cv2 = import_cv2()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Install requirements.txt first.") from error

    thresholds = parse_thresholds(args.thresholds, args.fixed_threshold)
    model = YOLO(str(args.weights))
    images = image_paths(args.data_root, args.split)
    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    ground_truth_by_image: dict[str, list[dict[str, Any]]] = {}
    prediction_records = []

    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.prediction_conf_min,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for image_path, result in zip(images, results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        predictions = result_predictions(result)
        ground_truth = load_ground_truth_boxes(
            label_path_for_image(args.data_root, args.split, image_path),
            width,
            height,
        )
        predictions_by_image[image_path.name] = predictions
        ground_truth_by_image[image_path.name] = ground_truth
        prediction_records.append(
            {
                "image": str(image_path),
                "objects": predictions,
                "ground_truth_count": len(ground_truth),
            }
        )

    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for image_name, ground_truth in ground_truth_by_image.items():
            counts = greedy_counts(
                ground_truth,
                predictions_by_image.get(image_name, []),
                threshold,
                args.iou_threshold,
            )
            tp += counts[0]
            fp += counts[1]
            fn += counts[2]
        row = {
            "split": args.split,
            "threshold": threshold,
            "iou_threshold": args.iou_threshold,
            **metrics_from_counts(tp, fp, fn),
        }
        rows.append(row)

    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"], -row["threshold"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{args.split}_threshold_sweep.csv", rows)
    (args.output_dir / f"{args.split}_best_threshold.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.write_predictions:
        with (args.output_dir / f"{args.split}_raw_predictions.jsonl").open("w", encoding="utf-8") as file:
            for record in prediction_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"best": best, "rows": len(rows), "images": len(images)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
