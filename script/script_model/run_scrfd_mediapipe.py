import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRFD_STRIDES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SCRFD face detection and MediaPipe eye landmark detection."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/reports/ground_truth.json"),
        help="Ground truth JSON. Used to keep simple/skeleton evaluation axes identical.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/scrfd/det_10g.onnx"),
        help="SCRFD ONNX model path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/scrfd_mediapipe_baseline/predictions/predictions.json"),
        help="Prediction JSON output path.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square SCRFD input size. Default: 640.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="SCRFD confidence threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="NMS IoU threshold. Default: 0.4.",
    )
    parser.add_argument(
        "--mediapipe-crop-margin",
        type=float,
        default=0.15,
        help="Margin ratio around each SCRFD bbox before running MediaPipe. Default: 0.15.",
    )
    parser.add_argument(
        "--skip-mediapipe",
        action="store_true",
        help="Skip MediaPipe FaceMesh and only save SCRFD detections/5-point landmarks.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Progress print interval. Use 0 to disable per-image progress. Default: 1.",
    )
    return parser.parse_args()


def import_dependencies(skip_mediapipe: bool = False) -> tuple[Any, Any, Any]:
    missing = []
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")
    if skip_mediapipe:
        mp = None
    else:
        try:
            import mediapipe as mp  # type: ignore
        except Exception:
            mp = None
            missing.append("mediapipe")
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")

    if missing:
        raise RuntimeError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + ". Use Python 3.11/3.12 and install the project requirements."
        )
    return cv2, mp, np


def load_ground_truth(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def unique_image_records(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    seen = set()
    records = []
    for record in ground_truth.get("records", []):
        key = (record["image_stem"], record["image_name"])
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def resize_with_padding(image: Any, input_size: int, cv2: Any, np: Any) -> tuple[Any, float, int, int]:
    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    padded = np.zeros((input_size, input_size, 3), dtype=image.dtype)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return padded, scale, pad_x, pad_y


def generate_anchor_centers(input_size: int, stride: int, np: Any) -> Any:
    height = input_size // stride
    width = input_size // stride
    y, x = np.mgrid[:height, :width]
    centers = np.stack((x, y), axis=-1).astype(np.float32) * stride
    centers = centers.reshape((-1, 2))
    return np.repeat(centers, 2, axis=0)


def distance_to_bbox(anchor_centers: Any, distances: Any, np: Any) -> Any:
    return np.stack(
        (
            anchor_centers[:, 0] - distances[:, 0],
            anchor_centers[:, 1] - distances[:, 1],
            anchor_centers[:, 0] + distances[:, 2],
            anchor_centers[:, 1] + distances[:, 3],
        ),
        axis=-1,
    )


def distance_to_keypoints(anchor_centers: Any, distances: Any, np: Any) -> Any:
    keypoints = []
    for index in range(0, distances.shape[1], 2):
        keypoints.append(anchor_centers + distances[:, index : index + 2])
    return np.stack(keypoints, axis=1)


def flatten_output(output: Any) -> Any:
    if len(output.shape) == 3 and output.shape[0] == 1:
        return output[0]
    return output


def decode_scrfd_outputs(
    outputs: list[Any],
    input_size: int,
    score_threshold: float,
    np: Any,
) -> tuple[list[list[float]], list[float], list[Any]]:
    outputs = [flatten_output(output) for output in outputs]
    if len(outputs) < 6:
        raise RuntimeError(f"Expected at least 6 SCRFD outputs, got {len(outputs)}")

    if len(outputs) >= 9 and outputs[1].shape[-1] == 4:
        score_outputs = [outputs[0], outputs[3], outputs[6]]
        bbox_outputs = [outputs[1], outputs[4], outputs[7]]
        keypoint_outputs = [outputs[2], outputs[5], outputs[8]]
    else:
        score_outputs = outputs[0:3]
        bbox_outputs = outputs[3:6]
        keypoint_outputs = outputs[6:9] if len(outputs) >= 9 else [None, None, None]

    bboxes: list[list[float]] = []
    scores: list[float] = []
    keypoints: list[Any] = []

    for stride, score_output, bbox_output, keypoint_output in zip(
        SCRFD_STRIDES, score_outputs, bbox_outputs, keypoint_outputs
    ):
        anchor_centers = generate_anchor_centers(input_size, stride, np)
        score_output = score_output.reshape((-1,))
        bbox_output = bbox_output.reshape((-1, 4)) * stride
        keep = np.where(score_output >= score_threshold)[0]
        if len(keep) == 0:
            continue

        decoded_bboxes = distance_to_bbox(anchor_centers, bbox_output, np)
        decoded_keypoints = None
        if keypoint_output is not None:
            keypoint_output = keypoint_output.reshape((-1, 10)) * stride
            decoded_keypoints = distance_to_keypoints(anchor_centers, keypoint_output, np)

        for index in keep:
            bboxes.append(decoded_bboxes[index].tolist())
            scores.append(float(score_output[index]))
            if decoded_keypoints is None:
                keypoints.append([])
            else:
                keypoints.append(decoded_keypoints[index].tolist())

    return bboxes, scores, keypoints


def unpad_bbox(bbox: list[float], scale: float, pad_x: int, pad_y: int, width: int, height: int) -> dict[str, float]:
    xtl = max(0.0, min(width, (bbox[0] - pad_x) / scale))
    ytl = max(0.0, min(height, (bbox[1] - pad_y) / scale))
    xbr = max(0.0, min(width, (bbox[2] - pad_x) / scale))
    ybr = max(0.0, min(height, (bbox[3] - pad_y) / scale))
    return {
        "xtl": xtl,
        "ytl": ytl,
        "xbr": xbr,
        "ybr": ybr,
        "width": max(0.0, xbr - xtl),
        "height": max(0.0, ybr - ytl),
    }


def unpad_keypoints(keypoints: list[list[float]], scale: float, pad_x: int, pad_y: int) -> list[dict[str, float]]:
    labels = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"]
    points = []
    for label, point in zip(labels, keypoints):
        points.append({"label": label, "x": (point[0] - pad_x) / scale, "y": (point[1] - pad_y) / scale})
    return points


def nms_indices(bboxes: list[list[float]], scores: list[float], score_threshold: float, nms_threshold: float, cv2: Any) -> list[int]:
    if not bboxes:
        return []
    xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in bboxes]
    raw_indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold, nms_threshold)
    if len(raw_indices) == 0:
        return []
    indices = []
    for index in raw_indices:
        if isinstance(index, (list, tuple)):
            indices.append(int(index[0]))
        elif hasattr(index, "item"):
            indices.append(int(index.item()))
        else:
            indices.append(int(index))
    return indices


def mediapipe_eye_landmarks(
    image: Any,
    bbox: dict[str, float],
    face_mesh: Any,
    cv2: Any,
    crop_margin: float,
) -> list[dict[str, float]]:
    if face_mesh is None:
        return []

    height, width = image.shape[:2]
    box_width = bbox["width"]
    box_height = bbox["height"]
    x1 = max(0, int(math.floor(bbox["xtl"] - box_width * crop_margin)))
    y1 = max(0, int(math.floor(bbox["ytl"] - box_height * crop_margin)))
    x2 = min(width, int(math.ceil(bbox["xbr"] + box_width * crop_margin)))
    y2 = min(height, int(math.ceil(bbox["ybr"] + box_height * crop_margin)))
    if x2 <= x1 or y2 <= y1:
        return []

    crop = image[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return []

    landmarks = result.multi_face_landmarks[0].landmark
    eye_indices = {
        "right_eye": [33, 133],
        "left_eye": [362, 263],
    }
    points = []
    crop_h, crop_w = crop.shape[:2]
    for label, indices in eye_indices.items():
        xs = [landmarks[index].x * crop_w + x1 for index in indices]
        ys = [landmarks[index].y * crop_h + y1 for index in indices]
        points.append({"label": label, "x": sum(xs) / len(xs), "y": sum(ys) / len(ys)})
    return points


def detect_image(
    image_path: Path,
    net: Any,
    face_mesh: Any,
    args: argparse.Namespace,
    cv2: Any,
    np: Any,
) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    model_input, scale, pad_x, pad_y = resize_with_padding(image, args.input_size, cv2, np)
    blob = cv2.dnn.blobFromImage(
        model_input,
        scalefactor=1.0 / 128.0,
        size=(args.input_size, args.input_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    bboxes, scores, keypoints = decode_scrfd_outputs(outputs, args.input_size, args.score_threshold, np)
    keep = nms_indices(bboxes, scores, args.score_threshold, args.nms_threshold, cv2)

    detections = []
    for index in keep:
        bbox = unpad_bbox(bboxes[index], scale, pad_x, pad_y, width, height)
        landmarks_5pt = unpad_keypoints(keypoints[index], scale, pad_x, pad_y) if keypoints[index] else []
        detections.append(
            {
                "bbox": bbox,
                "score": scores[index],
                "scrfd_landmarks_5pt": landmarks_5pt,
                "mediapipe_eye_landmarks": mediapipe_eye_landmarks(
                    image,
                    bbox,
                    face_mesh,
                    cv2,
                    args.mediapipe_crop_margin,
                ),
            }
        )
    return detections


def build_prediction_records(
    ground_truth: dict[str, Any],
    detections_by_image: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for record in ground_truth.get("records", []):
        key = (record["image_stem"], record["image_name"])
        records.append(
            {
                "annotation_kind": record["annotation_kind"],
                "annotation_mode": record["annotation_mode"],
                "image_stem": record["image_stem"],
                "image_name": record["image_name"],
                "image_path": record["image_path"],
                "detections": detections_by_image.get(key, []),
            }
        )
    return records


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"SCRFD model not found: {args.model}")

    cv2, mp, np = import_dependencies(skip_mediapipe=args.skip_mediapipe)
    ground_truth = load_ground_truth(args.ground_truth)
    image_records = unique_image_records(ground_truth)
    if args.max_images is not None:
        image_records = image_records[: args.max_images]

    net = cv2.dnn.readNetFromONNX(str(args.model))
    face_mesh = None
    if args.skip_mediapipe:
        print("warning: --skip-mediapipe enabled; MediaPipe eye landmarks will be empty.")
    elif hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    else:
        print("warning: mediapipe.solutions.face_mesh is unavailable; MediaPipe eye landmarks will be empty.")

    detections_by_image = {}
    try:
        for index, record in enumerate(image_records, start=1):
            image_path = Path(record["image_path"])
            detections_by_image[(record["image_stem"], record["image_name"])] = detect_image(
                image_path=image_path,
                net=net,
                face_mesh=face_mesh,
                args=args,
                cv2=cv2,
                np=np,
            )
            if args.print_every and (index == 1 or index % args.print_every == 0 or index == len(image_records)):
                count = len(detections_by_image[(record["image_stem"], record["image_name"])])
                print(f"{index}/{len(image_records)} {record['image_name']}: {count}")
    finally:
        if face_mesh is not None:
            face_mesh.close()

    output = {
        "schema_version": "1.0",
        "model": {
            "detector": "SCRFD",
            "detector_path": str(args.model),
            "landmark_refiner": "skipped" if args.skip_mediapipe else ("MediaPipe FaceMesh" if face_mesh is not None else "unavailable"),
            "input_size": args.input_size,
            "score_threshold": args.score_threshold,
            "nms_threshold": args.nms_threshold,
            "mediapipe_crop_margin": args.mediapipe_crop_margin,
        },
        "records": build_prediction_records(ground_truth, detections_by_image),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
