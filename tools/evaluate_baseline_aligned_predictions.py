#!/usr/bin/env python3
"""Adapt saved model predictions to the shared Face-DINO evaluation contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baseline_aligned import (  # noqa: E402
    evaluate_normalized_predictions,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=(
            "cooperative",
            "head_coco",
            "keypoint_comparison2",
            "yolo_pose",
        ),
        required=True,
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--architecture-id", required=True)
    parser.add_argument("--comparison-stage", default="refined")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-partial-inventory", action="store_true")
    parser.add_argument("--validation-face-threshold", type=float)
    parser.add_argument("--validation-point-threshold", type=float)
    parser.add_argument(
        "--face-score-candidate-floor",
        type=float,
        help=(
            "Lowest FACE score retained in the prediction artifact. Val "
            "threshold selection must not search below this floor."
        ),
    )
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _image_lookup(annotation_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    exact, basename = {}, {}
    for image in coco["images"]:
        name = str(image["file_name"])
        exact[name] = int(image["id"])
        basename[Path(name).name] = int(image["id"])
    return exact, basename


def _resolve_image_id(
    value: str,
    exact: dict[str, int],
    basename: dict[str, int],
) -> int:
    normalized = value.replace("\\", "/")
    candidates = (normalized, normalized.removeprefix("./"))
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    name = Path(normalized).name
    if name in basename:
        return basename[name]
    raise KeyError(f"Prediction image is absent from V3 annotations: {value}")


def _cooperative(
    path: Path,
    annotation_path: Path,
    stage: str,
) -> dict[int, dict[str, Any]]:
    exact, basename = _image_lookup(annotation_path)
    output: dict[int, dict[str, Any]] = {}
    if stage == "causal":
        face_key, pose_key = "independent_face", "refined_pose"
    else:
        face_key, pose_key = f"{stage}_face", f"{stage}_pose"
    for row in _jsonl(path):
        image_id = _resolve_image_id(str(row["file_name"]), exact, basename)
        decoded = row["predictions"]
        if face_key not in decoded or pose_key not in decoded:
            raise KeyError(
                f"Cooperative prediction lacks {face_key}/{pose_key} for "
                f"comparison stage {stage!r}"
            )
        face = decoded.get(face_key, {})
        pose = decoded.get(pose_key, {})
        faces = [
            {
                "bbox_xyxy": box,
                "ellipse": ellipse,
                "score": score,
            }
            for box, ellipse, score in zip(
                face.get("boxes", []),
                face.get("ellipses", []),
                face.get("scores", []),
                strict=True,
            )
        ]
        poses = []
        states = pose.get("state_probabilities")
        for index, (box, score, points, point_scores) in enumerate(
            zip(
                pose.get("boxes", []),
                pose.get("scores", []),
                pose.get("keypoints", []),
                pose.get("keypoint_scores", []),
                strict=True,
            )
        ):
            poses.append(
                {
                    "bbox_xyxy": box,
                    "score": score,
                    "keypoints": points,
                    "point_scores": point_scores,
                    "state_probabilities": (
                        states[index] if states is not None else None
                    ),
                }
            )
        shared = decoded.get("shared_head", {})
        heads = [
            {"bbox_xyxy": box, "score": score}
            for box, score in zip(
                shared.get("boxes", []),
                shared.get("scores", []),
                strict=True,
            )
        ]
        output[image_id] = {"heads": heads, "faces": faces, "poses": poses}
    return output


def _yolo_pose(
    path: Path,
    annotation_path: Path,
) -> dict[int, dict[str, Any]]:
    exact, basename = _image_lookup(annotation_path)
    output: dict[int, dict[str, Any]] = {}
    for row in _jsonl(path):
        image_id = _resolve_image_id(str(row["image"]), exact, basename)
        faces, poses = [], []
        for item in row.get("objects", []):
            box = item["bbox"]
            score = float(item.get("score", 0.0))
            faces.append({"bbox_xyxy": box, "score": score, "ellipse": None})
            keypoints = item.get("keypoints")
            if keypoints:
                poses.append(
                    {
                        "bbox_xyxy": box,
                        "score": score,
                        "keypoints": [
                            [float(point["x"]), float(point["y"])]
                            for point in keypoints
                        ],
                        "point_scores": [
                            (
                                float(point["score"])
                                if point.get("score") is not None
                                else score
                            )
                            for point in keypoints
                        ],
                        "state_probabilities": None,
                    }
                )
        output[image_id] = {"heads": [], "faces": faces, "poses": poses}
    return output


def _head_coco(
    path: Path,
    annotation_path: Path,
) -> dict[int, dict[str, Any]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    output = {
        int(image["id"]): {"heads": [], "faces": [], "poses": []}
        for image in coco["images"]
    }
    predictions = json.loads(path.read_text(encoding="utf-8"))
    for row in predictions:
        image_id = int(row["image_id"])
        if image_id not in output:
            raise KeyError(f"HEAD prediction has an unknown image id: {image_id}")
        x, y, width, height = (float(value) for value in row["bbox"])
        output[image_id]["heads"].append(
            {
                "bbox_xyxy": [x, y, x + width, y + height],
                "score": float(row["score"]),
            }
        )
    return output


def _keypoint_comparison2(
    path: Path,
    annotation_path: Path,
) -> dict[int, dict[str, Any]]:
    exact, basename = _image_lookup(annotation_path)
    output: dict[int, dict[str, Any]] = {}
    for row in _jsonl(path):
        image_id = _resolve_image_id(str(row["file_name"]), exact, basename)
        faces, poses = [], []
        for prediction in row.get("predictions", []):
            face = {
                "bbox_xyxy": prediction["bbox_xyxy"],
                "score": float(prediction["score"]),
                "ellipse": prediction.get("ellipse"),
            }
            faces.append(face)
            poses.append(
                {
                    "bbox_xyxy": prediction["bbox_xyxy"],
                    "score": float(prediction["score"]),
                    "keypoints": prediction["keypoints"],
                    # Legacy DEIM joint had only a visibility output, so its
                    # historical artifacts retain the old fallback.  The
                    # factorized variant must use presence for point gating;
                    # using visibility would turn correctly occluded points
                    # into false missing predictions.
                    "point_scores": prediction.get(
                        "presence", prediction["visibility"]
                    ),
                    "state_probabilities": prediction.get(
                        "state_probabilities"
                    ),
                }
            )
        output[image_id] = {"heads": [], "faces": faces, "poses": poses}
    return output


def main() -> int:
    args = parse_args()
    if args.format == "cooperative":
        predictions = _cooperative(
            args.predictions,
            args.annotations,
            args.comparison_stage,
        )
    elif args.format == "head_coco":
        predictions = _head_coco(args.predictions, args.annotations)
    elif args.format == "keypoint_comparison2":
        predictions = _keypoint_comparison2(
            args.predictions,
            args.annotations,
        )
    else:
        predictions = _yolo_pose(args.predictions, args.annotations)
    runtime: dict[str, Any] | None = None
    if args.source_metrics and args.source_metrics.is_file():
        source = json.loads(args.source_metrics.read_text(encoding="utf-8"))
        runtime = source.get("runtime")
    evaluator_path = REPO_ROOT / "src/evaluation/baseline_aligned.py"
    adapter_path = Path(__file__).resolve()
    provenance: dict[str, Any] = {
        "prediction_format": args.format,
        "prediction_path": str(args.predictions),
        "prediction_sha256": sha256_file(args.predictions),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "seed": args.seed,
    }
    for name in ("config", "checkpoint"):
        path = getattr(args, name)
        if path is not None:
            provenance[name] = str(path)
            provenance[f"{name}_sha256"] = (
                sha256_file(path) if path.is_file() else None
            )
    metrics = evaluate_normalized_predictions(
        annotation_path=args.annotations,
        predictions_by_image=predictions,
        split=args.split,
        architecture_id=args.architecture_id,
        comparison_stage=args.comparison_stage,
        output_dir=args.output_dir,
        provenance=provenance,
        runtime=runtime,
        allow_partial_inventory=args.allow_partial_inventory,
        validation_selected_face_threshold=args.validation_face_threshold,
        validation_selected_point_threshold=args.validation_point_threshold,
        face_score_candidate_floor=args.face_score_candidate_floor,
    )
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "architecture_id": metrics["architecture_id"],
                "split": metrics["split"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
