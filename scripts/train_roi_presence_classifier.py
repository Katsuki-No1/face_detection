#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.active_learning.adapters import DinoDeimAdapter, create_adapter
from src.active_learning.config import ModelConfig
from src.active_learning.presence import crop_roi, padded_roi, sha256_file
from src.active_learning.tracking import bbox_iou


@dataclass(slots=True)
class Sample:
    image_path: Path
    bbox: tuple[float, float, float, float]
    label: tuple[float, float]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-DINO ROI face/head presence classifier.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/roi_presence_dinov3_v1.yaml")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def jitter_bbox(box: Sequence[float], width: int, height: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width, box_height = x2 - x1, y2 - y1
    scale = float(rng.uniform(0.85, 1.20))
    center_x = (x1 + x2) / 2 + float(rng.uniform(-0.12, 0.12)) * box_width
    center_y = (y1 + y2) / 2 + float(rng.uniform(-0.12, 0.12)) * box_height
    half_width, half_height = box_width * scale / 2, box_height * scale / 2
    return (max(0.0, center_x - half_width), max(0.0, center_y - half_height), min(width, center_x + half_width), min(height, center_y + half_height))


def build_samples(
    annotation_path: Path,
    image_root: Path,
    *,
    padding: float,
    backgrounds_per_image: int,
    seed: int,
) -> tuple[list[Sample], set[str]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]).lower() for row in data["categories"]}
    images = {int(row["id"]): row for row in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    sources: set[str] = set()
    for image_id, image in images.items():
        image_path = image_root / str(image["file_name"])
        width, height = int(image["width"]), int(image["height"])
        source = str((image.get("metadata") or {}).get("source_video_id") or Path(str(image["file_name"])).stem.split("_frame_")[0])
        sources.add(source)
        annotations = annotations_by_image[image_id]
        boxes: list[tuple[float, float, float, float]] = []
        face_boxes: list[tuple[float, float, float, float]] = []
        for annotation in annotations:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            box = (x, y, x + box_width, y + box_height)
            boxes.append(box)
            if categories[int(annotation["category_id"])] == "face":
                face_boxes.append(box)
        for annotation, box in zip(annotations, boxes, strict=True):
            name = categories[int(annotation["category_id"])]
            if name not in {"face", "head"}:
                continue
            face_present = name == "face" or any(bbox_iou(box, face_box) > 0.05 for face_box in face_boxes)
            label = (float(face_present), 1.0)
            roi = padded_roi(box, width=width, height=height, padding=padding, min_side=16)
            if roi is not None:
                samples.append(Sample(image_path, roi, label, name))
                samples.append(Sample(image_path, jitter_bbox(roi, width, height, rng), label, f"{name}_jitter"))
        for _ in range(backgrounds_per_image):
            for _attempt in range(100):
                side = float(rng.uniform(32, max(33, min(width, height) * 0.45)))
                x1 = float(rng.uniform(0, max(1.0, width - side)))
                y1 = float(rng.uniform(0, max(1.0, height - side)))
                box = (x1, y1, min(float(width), x1 + side), min(float(height), y1 + side))
                if all(bbox_iou(box, target) < 0.05 for target in boxes):
                    samples.append(Sample(image_path, box, (0.0, 0.0), "background"))
                    break
    return samples, sources


def extract_embeddings(
    samples: Sequence[Sample],
    dino: DinoDeimAdapter,
    *,
    input_size: int,
    cache_path: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file() and not force:
        cached = np.load(cache_path)
        return cached["features"], cached["labels"]
    features: list[np.ndarray] = []
    labels: list[tuple[float, float]] = []
    batch: list[np.ndarray] = []
    image_path: Path | None = None
    image: np.ndarray | None = None
    for sample in samples:
        if sample.image_path != image_path:
            image_path = sample.image_path
            image = cv2.imread(image_path.as_posix())
            if image is None:
                raise FileNotFoundError(image_path)
        assert image is not None
        batch.append(crop_roi(image, sample.bbox, input_size))
        labels.append(sample.label)
        if len(batch) >= dino.config.batch_size:
            features.append(dino.embed_batch(batch, input_size=input_size))
            batch.clear()
    if batch:
        features.append(dino.embed_batch(batch, input_size=input_size))
    matrix = np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=matrix, labels=targets)
    return matrix, targets


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = float(labels.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    precision = true_positive / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def class_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    ap = average_precision(labels, scores)
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels > 0.5)))
        fp = int(np.sum(predicted & (labels <= 0.5)))
        fn = int(np.sum(~predicted & (labels > 0.5)))
        rows.append({
            "threshold": float(threshold), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
        })
    feasible = [row for row in rows if row["precision"] >= 0.50]
    selected = max(feasible, key=lambda row: (row["recall"], row["precision"], row["threshold"]), default=max(rows, key=lambda row: row["precision"]))
    return {"pr_auc": ap, "selected": selected}


def evaluate(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray, temperature: torch.Tensor | None = None) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).float())
        if temperature is not None:
            logits = logits / temperature.clamp_min(1e-3)
        probabilities = torch.sigmoid(logits).numpy()
    face = class_metrics(labels[:, 0], probabilities[:, 0])
    head = class_metrics(labels[:, 1], probabilities[:, 1])
    return {"face": face, "head": head, "macro_pr_auc": (face["pr_auc"] + head["pr_auc"]) / 2.0}


def train_model(
    architecture: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    input_dim = train_features.shape[1]
    model: torch.nn.Module = (
        torch.nn.Sequential(torch.nn.Linear(input_dim, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 2))
        if architecture == "mlp" else torch.nn.Linear(input_dim, 2)
    )
    positives = torch.from_numpy(train_labels.sum(axis=0)).float()
    negatives = len(train_labels) - positives
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives.clamp_min(1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    best_state = None
    best_score = -1.0
    for _epoch in range(epochs):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_features, val_labels)
        if metrics["macro_pr_auc"] > best_score:
            best_score = float(metrics["macro_pr_auc"])
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, evaluate(model, val_features, val_labels)


def fit_temperature(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).float()).detach()
    log_temperature = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    targets = torch.from_numpy(labels).float()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.05, 20.0)


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project_root = resolve(args.config.parent, raw.get("project_root", ".."))
    output_dir = resolve(project_root, raw["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(raw["seed"])
    random.seed(seed)
    np.random.seed(seed)
    model_raw = dict(raw["dino_model"])
    model_raw["weights"] = resolve(project_root, model_raw["weights"])
    model_raw["config"] = resolve(project_root, model_raw["config"])
    adapter = create_adapter(ModelConfig(**model_raw))
    if not isinstance(adapter, DinoDeimAdapter):
        raise TypeError("dino_model must create a DinoDeimAdapter")
    split_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_sources: dict[str, set[str]] = {}
    sample_counts: dict[str, int] = {}
    for split, split_raw in raw["splits"].items():
        samples, sources = build_samples(
            resolve(project_root, split_raw["annotation"]), resolve(project_root, split_raw["image_root"]),
            padding=float(raw["roi_padding"]), backgrounds_per_image=int(raw["background_crops_per_image"]), seed=seed,
        )
        split_data[split] = extract_embeddings(
            samples, adapter, input_size=int(raw["roi_input_size"]),
            cache_path=output_dir / f"{split}_features.npz", force=args.force_features,
        )
        split_sources[split] = sources
        sample_counts[split] = len(samples)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sources[left] & split_sources[right]
        if overlap:
            raise RuntimeError(f"Source leakage between {left} and {right}: {sorted(overlap)}")
    train_features, train_labels = split_data["train"]
    val_features, val_labels = split_data["val"]
    linear, linear_val = train_model(
        "linear", train_features, train_labels, val_features, val_labels,
        epochs=int(raw["epochs"]), batch_size=int(raw["batch_size"]), learning_rate=float(raw["learning_rate"]), seed=seed,
    )
    face_selected = linear_val["face"]["selected"]
    head_selected = linear_val["head"]["selected"]
    needs_mlp = (
        linear_val["macro_pr_auc"] < 0.90
        or face_selected["recall"] < 0.95
        or head_selected["recall"] < 0.95
    )
    model, architecture, validation = linear, "linear", linear_val
    mlp_validation = None
    if needs_mlp:
        mlp, mlp_validation = train_model(
            "mlp", train_features, train_labels, val_features, val_labels,
            epochs=int(raw["epochs"]), batch_size=int(raw["batch_size"]), learning_rate=float(raw["learning_rate"]), seed=seed,
        )
        if (
            mlp_validation["macro_pr_auc"] - linear_val["macro_pr_auc"] >= 0.02
            and mlp_validation["face"]["selected"]["recall"] >= face_selected["recall"]
            and mlp_validation["head"]["selected"]["recall"] >= head_selected["recall"]
        ):
            model, architecture, validation = mlp, "mlp", mlp_validation
    temperature = fit_temperature(model, val_features, val_labels)
    test_features, test_labels = split_data["test"]
    test_metrics = evaluate(model, test_features, test_labels, temperature)
    thresholds = {
        "face": float(validation["face"]["selected"]["threshold"]),
        "head": float(validation["head"]["selected"]["threshold"]),
    }
    metadata = {
        "dataset_version": raw["dataset_version"], "seed": seed, "sample_counts": sample_counts,
        "split_sources": {key: sorted(value) for key, value in split_sources.items()},
        "dino_weights": model_raw["weights"].resolve().as_posix(),
        "dino_weights_sha256": sha256_file(model_raw["weights"]),
        "validation": validation, "linear_validation": linear_val, "mlp_validation": mlp_validation,
        "test": test_metrics, "architecture": architecture, "thresholds": thresholds,
    }
    torch.save({
        "model": model.state_dict(), "architecture": architecture, "input_dim": int(train_features.shape[1]),
        "temperature": temperature.tolist(), "thresholds": thresholds, "metadata": metadata,
    }, output_dir / "best.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"architecture": architecture, "thresholds": thresholds, "validation_macro_pr_auc": validation["macro_pr_auc"], "test_macro_pr_auc": test_metrics["macro_pr_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
