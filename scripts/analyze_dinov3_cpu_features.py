import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize DINOv3 CPU feature JSONL outputs.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            image_path = Path(row["image"]).resolve()
            row["classes_list"] = [item for item in row.get("classes", "").split("|") if item]
            row["is_positive"] = row.get("positive") == "True"
            row["is_negative"] = row.get("negative") == "True"
            row["objects"] = int(row.get("objects") or 0)
            row["metadata_face_occluded"] = int(row.get("metadata_face_occluded") or 0)
            row["metadata_ellipse_out_of_image"] = int(row.get("metadata_ellipse_out_of_image") or 0)
            rows[str(image_path)] = row
    return rows


def read_features(path: Path, manifest: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            resolved = str(Path(item["image"]).resolve())
            meta = manifest.get(resolved)
            if meta is None:
                raise KeyError(f"Feature image was not found in manifest: {item['image']} -> {resolved}")
            feature = np.array(item["embedding"], dtype=np.float32)
            record = {**meta, **{key: value for key, value in item.items() if key != "embedding"}}
            record["resolved_image"] = resolved
            records.append(record)
            features.append(feature)
    if not features:
        raise RuntimeError(f"No features were read from {path}")
    return np.stack(features, axis=0), records


def l2_normalize(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norm, eps)


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular_values**2
    explained = variance[:2] / max(1e-12, variance.sum())
    return coords.astype(np.float32), explained.astype(np.float32)


def category_values(records: list[dict[str, Any]], field: str) -> list[str]:
    if field == "posneg":
        return ["negative" if record["is_negative"] else "positive" for record in records]
    if field == "source":
        return [str(record["source"]) for record in records]
    if field == "stratum":
        return [str(record["selection_stratum"]) for record in records]
    if field == "has_small":
        return ["small_or_tiny" if {"small_face", "tiny_face"} & set(record["classes_list"]) else "not_small" for record in records]
    if field == "has_occlusion":
        return [
            "occluded" if {"occluded_face", "occluded_keypoint"} & set(record["classes_list"]) else "not_occluded"
            for record in records
        ]
    if field == "has_edge_oob":
        return ["edge_or_oob" if {"edge_face", "ellipse_oob"} & set(record["classes_list"]) else "not_edge_oob" for record in records]
    if field == "has_multi":
        return [
            "multi_or_crowded" if {"multi_face", "crowded", "incomplete_keypoints"} & set(record["classes_list"]) else "single_or_complete"
            for record in records
        ]
    return [str(record.get(field, "")) for record in records]


def plot_pca(coords: np.ndarray, explained: np.ndarray, records: list[dict[str, Any]], field: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = category_values(records, field)
    categories = sorted(set(values))
    fig, ax = plt.subplots(figsize=(8, 6))
    for category in categories:
        mask = np.array([value == category for value in values])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=54, alpha=0.82, label=category)
    ax.axhline(0, color="#ddd", linewidth=0.8)
    ax.axvline(0, color="#ddd", linewidth=0.8)
    ax.set_title(f"DINOv3 image embedding PCA by {field}")
    ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}%)")
    ax.legend(fontsize=8, markerscale=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_similarity_heatmap(similarity: np.ndarray, records: list[dict[str, Any]], output_path: Path) -> list[int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(records)), key=lambda idx: (records[idx]["source"], records[idx]["selection_stratum"], Path(records[idx]["image"]).name))
    ordered = similarity[np.ix_(order, order)]
    labels = [f"{records[idx]['source']}\\n{records[idx]['selection_stratum']}" for idx in order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(ordered, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_title("DINOv3 cosine similarity heatmap")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return order


def pair_type(record_a: dict[str, Any], record_b: dict[str, Any]) -> str:
    if record_a["is_negative"] and record_b["is_negative"]:
        return "negative-negative"
    if record_a["is_positive"] and record_b["is_positive"]:
        return "positive-positive"
    return "positive-negative"


def write_pairwise_metrics(similarity: np.ndarray, records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(similarity[i, j])
            values_by_key[pair_type(records[i], records[j])].append(value)
            values_by_key["same_source" if records[i]["source"] == records[j]["source"] else "different_source"].append(value)
            values_by_key[
                "same_stratum" if records[i]["selection_stratum"] == records[j]["selection_stratum"] else "different_stratum"
            ].append(value)

    rows = []
    summary = {}
    for key in sorted(values_by_key):
        values = np.array(values_by_key[key], dtype=np.float32)
        row = {
            "pair_group": key,
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }
        rows.append(row)
        summary[key] = row
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["pair_group", "count", "mean", "std", "median", "p25", "p75"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def plot_pairwise_hist(similarity: np.ndarray, records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_by_key: dict[str, list[float]] = defaultdict(list)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            values_by_key[pair_type(records[i], records[j])].append(float(similarity[i, j]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["positive-positive", "positive-negative", "negative-negative"]:
        values = values_by_key.get(key, [])
        if values:
            ax.hist(values, bins=24, alpha=0.55, density=True, label=key)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("DINOv3 image embedding similarity by positive/negative pair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def nearest_neighbors(similarity: np.ndarray, records: list[dict[str, Any]], topk: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    consistency: dict[str, list[float]] = defaultdict(list)
    fields = ["source", "selection_stratum", "posneg", "has_small", "has_occlusion", "has_edge_oob", "has_multi"]
    for index, record in enumerate(records):
        order = np.argsort(-similarity[index])
        order = [int(item) for item in order if int(item) != index][:topk]
        neighbors = []
        for rank, neighbor_index in enumerate(order, start=1):
            neighbor = records[neighbor_index]
            neighbors.append(
                {
                    "rank": rank,
                    "index": neighbor_index,
                    "score": float(similarity[index, neighbor_index]),
                    "record": neighbor,
                }
            )
        for field in fields:
            query_value = category_values([record], field)[0]
            neighbor_values = category_values([records[item] for item in order], field)
            consistency[field].append(sum(value == query_value for value in neighbor_values) / max(1, len(neighbor_values)))
        rows.append({"index": index, "record": record, "neighbors": neighbors})
    metrics = {field: float(np.mean(values)) for field, values in consistency.items()}
    metrics["topk"] = topk
    return rows, metrics


def rel_to(path: str | Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def write_neighbors_html(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>DINOv3 CPU nearest neighbors</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px}.query{border-bottom:1px solid #ddd;margin-bottom:24px;padding-bottom:18px}.row{display:flex;gap:12px;overflow-x:auto}.card{width:170px;flex:0 0 auto}.card img{max-width:160px;max-height:120px;border:1px solid #ccc}.meta{font-size:12px;color:#444;word-break:break-word}</style>",
        "</head><body><h1>DINOv3 CPU nearest neighbors</h1>",
        "<p>Image-level frozen DINOv3 cosine retrieval. This is not face detection Precision/Recall/mAP.</p>",
    ]
    for row in rows:
        record = row["record"]
        lines.append("<section class='query'>")
        lines.append(
            f"<h2>{html.escape(record['source'])} / {html.escape(record['selection_stratum'])} / "
            f"{'negative' if record['is_negative'] else 'positive'}</h2>"
        )
        lines.append("<div class='row'>")
        lines.append(card_html(record, base, "query"))
        for neighbor in row["neighbors"]:
            lines.append(card_html(neighbor["record"], base, f"#{neighbor['rank']} {neighbor['score']:.3f}"))
        lines.append("</div></section>")
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def card_html(record: dict[str, Any], base: Path, prefix: str) -> str:
    image = html.escape(rel_to(record["image"], base))
    classes = html.escape(str(record.get("classes", "")))
    return (
        "<div class='card'>"
        f"<img src='{image}' alt='sample'>"
        f"<div class='meta'><b>{html.escape(prefix)}</b><br>"
        f"{html.escape(record['source'])}<br>{html.escape(record['selection_stratum'])}<br>{classes}</div>"
        "</div>"
    )


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    pair = summary["pairwise_summary"]
    nn = summary["nearest_neighbor_consistency"]
    lines = [
        "# DINOv3 CPU Feature Visualization Analysis",
        "",
        "This analyzes image-level frozen DINOv3 embeddings for the stratified 30-image sample.",
        "It is not a face detector evaluation and does not report Precision / Recall / mAP.",
        "",
        "## Sample",
        "",
        f"- records: `{summary['record_count']}`",
        f"- embedding_dim: `{summary['embedding_dim']}`",
        f"- sources: `{summary['source_counts']}`",
        f"- strata: `{summary['stratum_counts']}`",
        f"- positive/negative: `{summary['positive_negative_counts']}`",
        "",
        "## Pairwise Cosine Similarity",
        "",
    ]
    for key in [
        "positive-positive",
        "positive-negative",
        "negative-negative",
        "same_source",
        "different_source",
        "same_stratum",
        "different_stratum",
    ]:
        if key in pair:
            row = pair[key]
            lines.append(
                f"- {key}: mean=`{row['mean']:.4f}`, median=`{row['median']:.4f}`, count=`{row['count']}`"
            )
    lines.extend(
        [
            "",
            "## Nearest-Neighbor Consistency",
            "",
        ]
    )
    for key, value in nn.items():
        if key != "topk":
            lines.append(f"- {key}: `{value:.3f}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `figures/pca_by_source.png`",
            "- `figures/pca_by_stratum.png`",
            "- `figures/pca_by_posneg.png`",
            "- `figures/cosine_similarity_heatmap.png`",
            "- `figures/similarity_pair_histogram.png`",
            "- `retrieval/nearest_neighbors.html`",
            "- `metrics/pairwise_similarity_summary.csv`",
            "- `metrics/nearest_neighbor_consistency.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    retrieval_dir = output_dir / "retrieval"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)
    features, records = read_features(args.features, manifest)
    normalized = l2_normalize(features, axis=1)
    similarity = normalized @ normalized.T
    coords, explained = pca_2d(normalized)

    for field in ["source", "stratum", "posneg", "has_small", "has_occlusion", "has_edge_oob", "has_multi"]:
        plot_pca(coords, explained, records, field, figures_dir / f"pca_by_{field}.png")
    order = plot_similarity_heatmap(similarity, records, figures_dir / "cosine_similarity_heatmap.png")
    pairwise_summary = write_pairwise_metrics(similarity, records, metrics_dir / "pairwise_similarity_summary.csv")
    plot_pairwise_hist(similarity, records, figures_dir / "similarity_pair_histogram.png")
    neighbor_rows, nn_metrics = nearest_neighbors(similarity, records, args.topk)
    write_neighbors_html(neighbor_rows, retrieval_dir / "nearest_neighbors.html")
    (metrics_dir / "nearest_neighbor_consistency.json").write_text(
        json.dumps(nn_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "record_count": len(records),
        "embedding_dim": int(features.shape[1]),
        "features": str(args.features),
        "manifest": str(args.manifest),
        "explained_variance_pc1_pc2": [float(explained[0]), float(explained[1])],
        "source_counts": dict(Counter(record["source"] for record in records)),
        "stratum_counts": dict(Counter(record["selection_stratum"] for record in records)),
        "positive_negative_counts": dict(Counter("negative" if record["is_negative"] else "positive" for record in records)),
        "class_counts": dict(Counter(class_name for record in records for class_name in record["classes_list"])),
        "pairwise_summary": pairwise_summary,
        "nearest_neighbor_consistency": nn_metrics,
        "heatmap_order": order,
        "outputs": {
            "figures_dir": str(figures_dir),
            "metrics_dir": str(metrics_dir),
            "retrieval_html": str(retrieval_dir / "nearest_neighbors.html"),
        },
    }
    (output_dir / "dinov3_feature_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "dinov3_feature_analysis_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
