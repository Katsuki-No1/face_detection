from __future__ import annotations

import csv
import html
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .dataset import read_crop_metadata
from .dino import l2_normalize


def load_features(output_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = np.load(output_dir / "features" / "crop_features.npy")
    records = read_crop_metadata(output_dir / "features" / "crop_features_meta.csv")
    if len(records) != len(features):
        raise ValueError(f"Feature/meta count mismatch: {len(features)} features vs {len(records)} metadata rows")
    return features.astype(np.float32), records


def binary_metrics(y_true: list[str], y_pred: list[str], positive_label: str = "face_like") -> dict[str, Any]:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive_label and pred == positive_label)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true != positive_label and pred != positive_label)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != positive_label and pred == positive_label)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == positive_label and pred != positive_label)
    total = max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "labels": [positive_label, "non_face"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def knn_predict(features: np.ndarray, labels: list[str], k: int) -> list[str]:
    if len(features) == 0:
        return []
    normalized = l2_normalize(features.astype(np.float32), axis=1)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    k = max(1, min(k, max(1, len(labels) - 1)))
    predictions = []
    for row in similarities:
        neighbors = np.argsort(-row)[:k]
        votes: dict[str, int] = {}
        for index in neighbors:
            votes[labels[int(index)]] = votes.get(labels[int(index)], 0) + 1
        predictions.append(max(sorted(votes), key=lambda label: votes[label]))
    return predictions


def write_knn_metrics(features: np.ndarray, records: list[dict[str, Any]], output_path: Path, k: int) -> dict[str, Any]:
    labels = [str(record.get("label", "")) for record in records]
    predictions = knn_predict(features, labels, k=k)
    metrics = binary_metrics(labels, predictions)
    metrics.update(
        {
            "k": k,
            "sample_count": len(records),
            "note": "This is a frozen DINO feature-space diagnostic, not detection Precision/Recall/mAP.",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def pair_indices(
    records: list[dict[str, Any]],
    left_filter: Callable[[dict[str, Any]], bool],
    right_filter: Callable[[dict[str, Any]], bool],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    left = [index for index, record in enumerate(records) if left_filter(record)]
    right = [index for index, record in enumerate(records) if right_filter(record)]
    if not left or not right:
        return []
    pair_count = len(left) * len(right)
    same_pool = left == right
    if same_pool:
        pair_count = len(left) * max(0, len(left) - 1) // 2
    if pair_count <= max_pairs:
        pairs = []
        if same_pool:
            for i, first in enumerate(left):
                for second in left[i + 1 :]:
                    pairs.append((first, second))
        else:
            for first in left:
                for second in right:
                    if first != second:
                        pairs.append((first, second))
        return pairs
    pairs = set()
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 20:
        attempts += 1
        first = rng.choice(left)
        second = rng.choice(right)
        if first == second:
            continue
        pair = tuple(sorted((first, second))) if same_pool else (first, second)
        pairs.add(pair)
    return sorted(pairs)


def similarity_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def write_similarity_metrics(
    features: np.ndarray,
    records: list[dict[str, Any]],
    output_path: Path,
    histogram_path: Path,
    max_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    normalized = l2_normalize(features.astype(np.float32), axis=1)
    rng = random.Random(seed)
    pair_specs = {
        "face-face": (
            lambda record: record.get("crop_type") == "face",
            lambda record: record.get("crop_type") == "face",
        ),
        "face-background": (
            lambda record: record.get("crop_type") == "face",
            lambda record: record.get("crop_type") == "background",
        ),
        "background-background": (
            lambda record: record.get("crop_type") == "background",
            lambda record: record.get("crop_type") == "background",
        ),
        "face-hard_negative": (
            lambda record: record.get("crop_type") == "face",
            lambda record: record.get("crop_type") == "hard_negative",
        ),
    }
    raw_values: dict[str, np.ndarray] = {}
    rows = []
    for pair_type, (left_filter, right_filter) in pair_specs.items():
        pairs = pair_indices(records, left_filter, right_filter, max_pairs, rng)
        values = np.array([float(normalized[first] @ normalized[second]) for first, second in pairs], dtype=np.float32)
        raw_values[pair_type] = values
        row = {"pair_type": pair_type, **similarity_stats(values)}
        rows.append(row)
    face_face_mean = rows[0]["mean"] if rows and rows[0]["mean"] is not None else None
    face_background_mean = rows[1]["mean"] if len(rows) > 1 and rows[1]["mean"] is not None else None
    separation_margin = None
    if face_face_mean is not None and face_background_mean is not None:
        separation_margin = float(face_face_mean) - float(face_background_mean)
    for row in rows:
        row["separation_margin"] = separation_margin

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["pair_type", "count", "mean", "std", "median", "p25", "p75", "separation_margin"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_similarity_histogram(raw_values, histogram_path)
    return rows


def write_similarity_histogram(values_by_type: dict[str, np.ndarray], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair_type, values in values_by_type.items():
        if values.size:
            ax.hist(values, bins=40, alpha=0.45, label=pair_type, density=True)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("DINO crop feature cosine similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_retrieval(
    features: np.ndarray,
    records: list[dict[str, Any]],
    output_html: Path,
    metrics_path: Path,
    topk: int,
) -> dict[str, Any]:
    normalized = l2_normalize(features.astype(np.float32), axis=1)
    face_queries = [index for index, record in enumerate(records) if record.get("crop_type") == "face"]
    query_rows = []
    top_hits: dict[int, list[float]] = {1: [], 5: [], 10: []}
    for query_index in face_queries:
        similarities = normalized @ normalized[query_index]
        similarities[query_index] = -np.inf
        order = np.argsort(-similarities)[: max(topk, 10)]
        retrieved = []
        for rank, index in enumerate(order[:topk], start=1):
            record = records[int(index)]
            retrieved.append({"rank": rank, "score": float(similarities[index]), "record": record})
        for k_value in top_hits:
            subset = order[: min(k_value, len(order))]
            if len(subset):
                top_hits[k_value].append(
                    sum(1 for index in subset if records[int(index)].get("label") == "face_like") / len(subset)
                )
        query_rows.append({"query": records[query_index], "retrieved": retrieved})
    metrics = {
        "query_count": len(face_queries),
        "top1_precision": float(np.mean(top_hits[1])) if top_hits[1] else None,
        "top5_precision": float(np.mean(top_hits[5])) if top_hits[5] else None,
        "top10_precision": float(np.mean(top_hits[10])) if top_hits[10] else None,
        "note": "Retrieval uses crop-level DINO features and is not a detector metric.",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_retrieval_html(query_rows, output_html)
    return metrics


def rel_image(path: str, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def write_retrieval_html(query_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>DINO Retrieval</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px} .query{margin:0 0 28px;border-bottom:1px solid #ddd;padding-bottom:20px} .row{display:flex;gap:12px;align-items:flex-start;overflow-x:auto}.card{width:150px;flex:0 0 auto}.card img{max-width:140px;max-height:140px;border:1px solid #ccc}.meta{font-size:12px;color:#444;word-break:break-word}</style>",
        "</head><body>",
        "<h1>DINO crop retrieval</h1>",
        "<p>This is a frozen DINO feature-space diagnostic, not detection Precision/Recall/mAP.</p>",
    ]
    for row in query_rows:
        query = row["query"]
        lines.append("<section class='query'>")
        lines.append(f"<h2>Query: {html.escape(query['crop_id'])}</h2>")
        lines.append("<div class='row'>")
        lines.append(card_html(query, base, prefix="query"))
        for item in row["retrieved"]:
            record = item["record"]
            lines.append(card_html(record, base, prefix=f"#{item['rank']} score={item['score']:.3f}"))
        lines.append("</div></section>")
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def card_html(record: dict[str, Any], base: Path, prefix: str) -> str:
    crop_path = html.escape(rel_image(str(record.get("crop_path", "")), base))
    title = html.escape(str(record.get("crop_type", "")))
    label = html.escape(str(record.get("label", "")))
    tags = html.escape("|".join(record.get("tags", [])) if isinstance(record.get("tags"), list) else str(record.get("tags", "")))
    return (
        "<div class='card'>"
        f"<img src='{crop_path}' alt='{title}'>"
        f"<div class='meta'><b>{html.escape(prefix)}</b><br>{title}<br>{label}<br>{tags}</div>"
        "</div>"
    )


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(features) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    x = features.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:2].T
    coords = x @ components
    variance = singular_values**2
    explained = variance[:2] / max(1e-12, variance.sum())
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
        explained = np.array([explained[0], 0.0])
    return coords.astype(np.float32), explained.astype(np.float32)


def write_pca_figures(features: np.ndarray, records: list[dict[str, Any]], figures_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    coords, explained = pca_2d(features)
    outputs = {}
    for field, filename, title in [
        ("crop_type", "pca_by_crop_type.png", "PCA by crop type"),
        ("label", "pca_by_label.png", "PCA by label"),
        ("face_size_category", "pca_by_face_size.png", "PCA by face size"),
    ]:
        values = [str(record.get(field, "")) for record in records]
        output_path = figures_dir / filename
        scatter_by_category(coords, values, title, output_path, explained)
        outputs[field] = str(output_path)
    return outputs


def scatter_by_category(coords: np.ndarray, values: list[str], title: str, output_path: Path, explained: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    categories = sorted(set(values))
    for category in categories:
        mask = np.array([value == category for value in values])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.75, label=category)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}%)" if len(explained) else "PC1")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}%)" if len(explained) > 1 else "PC2")
    ax.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_patch_similarity_outputs(output_dir: Path, features: np.ndarray, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    patch_path = output_dir / "features" / "patch_features_sample.npz"
    if not patch_path.exists() or len(features) == 0:
        return []
    face_indices = [index for index, record in enumerate(records) if record.get("crop_type") == "face"]
    if not face_indices:
        return []
    prototype = l2_normalize(features[face_indices].mean(axis=0, keepdims=True), axis=1)[0]
    data = np.load(patch_path, allow_pickle=True)
    patch_features = data["patch_features"]
    image_paths = data["image_paths"]
    widths = data["widths"]
    heights = data["heights"]
    bboxes_json = data["bboxes_json"]
    overlay_dir = output_dir / "overlays" / "similarity_map_samples"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in range(len(patch_features)):
        tokens = l2_normalize(patch_features[sample_index].astype(np.float32), axis=-1)
        similarity = tokens @ prototype
        image_path = Path(str(image_paths[sample_index]))
        width = int(widths[sample_index])
        height = int(heights[sample_index])
        bboxes = json.loads(str(bboxes_json[sample_index]))
        inside_mask = patch_inside_mask(similarity.shape, width, height, bboxes)
        outside_mask = ~inside_mask
        inside_mean = float(similarity[inside_mask].mean()) if inside_mask.any() else None
        outside_mean = float(similarity[outside_mask].mean()) if outside_mask.any() else None
        margin = None
        if inside_mean is not None and outside_mean is not None:
            margin = inside_mean - outside_mean
        overlay_path = overlay_dir / f"{sample_index:03d}_{image_path.stem}.png"
        with Image.open(image_path) as image:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(image.convert("RGB"))
            ax.imshow(similarity, cmap="jet", alpha=0.45, extent=[0, width, height, 0], vmin=-1.0, vmax=1.0)
            for box in bboxes:
                x1, y1, x2, y2 = [float(value) for value in box]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="lime", facecolor="none"))
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            fig.savefig(overlay_path, dpi=160, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        rows.append(
            {
                "image_path": str(image_path),
                "overlay_path": str(overlay_path),
                "bbox_inside_similarity_mean": inside_mean,
                "bbox_outside_similarity_mean": outside_mean,
                "inside_outside_margin": margin,
            }
        )
    metrics_path = output_dir / "metrics" / "patch_similarity_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "image_path",
            "overlay_path",
            "bbox_inside_similarity_mean",
            "bbox_outside_similarity_mean",
            "inside_outside_margin",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def patch_inside_mask(shape: tuple[int, int], width: int, height: int, bboxes: list[list[float]]) -> np.ndarray:
    grid_h, grid_w = shape
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    if not bboxes:
        return mask
    for gy in range(grid_h):
        cy = (gy + 0.5) / grid_h * height
        for gx in range(grid_w):
            cx = (gx + 0.5) / grid_w * width
            for box in bboxes:
                x1, y1, x2, y2 = [float(value) for value in box]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    mask[gy, gx] = True
                    break
    return mask
