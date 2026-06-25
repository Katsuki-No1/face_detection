from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import apply_cli_overrides, load_config, output_dir
from .metrics import (
    load_features,
    write_knn_metrics,
    write_patch_similarity_outputs,
    write_pca_figures,
    write_retrieval,
    write_similarity_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen DINO Phase 1 crop features.")
    parser.add_argument("--config", type=Path, default=Path("configs/dino_phase1_feature_analysis.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def analyze_features(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = output_dir(config)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    retrieval_dir = out_dir / "retrieval"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    features, records = load_features(out_dir)
    analysis = config.get("analysis", {})
    knn_metrics = write_knn_metrics(
        features,
        records,
        metrics_dir / "knn_metrics.json",
        k=int(analysis.get("knn_k", 5)),
    )
    similarity_rows = write_similarity_metrics(
        features,
        records,
        metrics_dir / "similarity_metrics.csv",
        figures_dir / "similarity_histogram.png",
        max_pairs=int(analysis.get("max_pair_samples", 200000)),
        seed=int(analysis.get("seed", 42)),
    )
    retrieval_metrics = write_retrieval(
        features,
        records,
        retrieval_dir / "knn_retrieval.html",
        metrics_dir / "retrieval_metrics.json",
        topk=int(analysis.get("retrieval_topk", 10)),
    )
    pca_outputs = write_pca_figures(features, records, figures_dir)
    umap_outputs = write_optional_umap(features, records, figures_dir)
    patch_rows = write_patch_similarity_outputs(out_dir, features, records)

    summary = {
        "output_dir": str(out_dir),
        "feature_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] else 0,
        "knn_metrics": knn_metrics,
        "similarity_metrics": similarity_rows,
        "retrieval_metrics": retrieval_metrics,
        "pca_outputs": pca_outputs,
        "umap_outputs": umap_outputs,
        "patch_similarity_rows": len(patch_rows),
        "note": "Phase 1 analyzes frozen DINO features only. It does not train or evaluate a detector/keypoint head.",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_optional_umap(features: np.ndarray, records: list[dict[str, Any]], figures_dir: Path) -> dict[str, str]:
    try:
        import umap  # type: ignore
    except Exception:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(features) < 3:
        return {}
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(features)
    outputs = {}
    for field, filename, title in [
        ("crop_type", "umap_by_crop_type.png", "UMAP by crop type"),
        ("label", "umap_by_label.png", "UMAP by label"),
    ]:
        values = [str(record.get(field, "")) for record in records]
        output_path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(7, 6))
        for category in sorted(set(values)):
            mask = np.array([value == category for value in values])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.75, label=category)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        outputs[field] = str(output_path)
    return outputs


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, output_dir_override=args.output_dir)
    summary = analyze_features(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
