import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw YOLO pose predictions and GT for mined failure images.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--failure-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-threshold", type=float, default=0.51)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--pck-threshold", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def lookup_failure_row(by_image: dict[str, dict[str, str]], image_path: Path) -> dict[str, str]:
    candidates = [str(image_path)]
    try:
        candidates.append(str(image_path.relative_to(Path.cwd())))
    except ValueError:
        pass
    candidates.append(image_path.name)
    for candidate in candidates:
        if candidate in by_image:
            return by_image[candidate]
    for key, row in by_image.items():
        if Path(key).name == image_path.name:
            return row
    raise KeyError(f"Missing failure row for {image_path}")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_yolo_pose_metrics import (  # type: ignore
        greedy_matches,
        import_cv2,
        keypoint_errors,
        label_path_for_image,
        load_ground_truth,
        result_predictions,
    )

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as error:
        raise RuntimeError("Missing ultralytics. Use the repo .venv for this helper.") from error

    cv2 = import_cv2()
    failure_rows = read_csv(args.failure_csv)
    by_image: dict[str, dict[str, str]] = {}
    for row in failure_rows:
        by_image.setdefault(row["image_path"], row)
    image_paths = [resolve_path(path_text) for path_text in sorted(by_image)]

    model = YOLO(str(args.weights))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.prediction_conf_min,
        device=args.device,
        stream=True,
        verbose=False,
    )

    with args.output_jsonl.open("w", encoding="utf-8") as file:
        for image_path, result in zip(image_paths, results):
            failure_row = lookup_failure_row(by_image, image_path)
            split = failure_row["split"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            height, width = image.shape[:2]
            label_path = label_path_for_image(args.data_root, split, image_path)
            ground_truth = load_ground_truth(label_path, width, height)
            predictions = [
                prediction
                for prediction in result_predictions(result)
                if prediction["score"] >= args.conf_threshold
            ]
            matches = greedy_matches(ground_truth, predictions, args.iou_threshold)
            matched_gt = {gt_index for gt_index, _, _ in matches}
            matched_pred = {pred_index for _, pred_index, _ in matches}
            match_details = []
            for gt_index, pred_index, iou in matches:
                errors = keypoint_errors(ground_truth[gt_index], predictions[pred_index])
                nmes = [float(error["normalized_error"]) for error in errors]
                match_details.append(
                    {
                        "gt_index": gt_index,
                        "prediction_index": pred_index,
                        "confidence": predictions[pred_index]["score"],
                        "bbox_iou": iou,
                        "keypoint_nme": mean(nmes),
                        "pck005_pass": bool(nmes) and max(nmes) <= args.pck_threshold,
                        "keypoint_errors": errors,
                    }
                )
            record: dict[str, Any] = {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "split": split,
                "auto_failure_type": failure_row.get("auto_failure_type", ""),
                "manual_failure_type": failure_row.get("manual_failure_type", ""),
                "ground_truth": ground_truth,
                "predictions": predictions,
                "matches": match_details,
                "unmatched_gt_indices": [index for index in range(len(ground_truth)) if index not in matched_gt],
                "unmatched_prediction_indices": [index for index in range(len(predictions)) if index not in matched_pred],
                "aggregate": failure_row,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"failure_images": len(image_paths), "output_jsonl": str(args.output_jsonl)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
