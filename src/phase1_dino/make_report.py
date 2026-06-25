from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from .config import apply_cli_overrides, load_config, output_dir
from .dataset import read_crop_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 1 frozen-DINO HTML report.")
    parser.add_argument("--config", type=Path, default=Path("configs/dino_phase1_feature_analysis.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_similarity_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def crop_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {"crop_type": {}, "label": {}, "face_size_category": {}}
    for record in records:
        for field in counts:
            value = str(record.get(field, ""))
            counts[field][value] = counts[field].get(value, 0) + 1
    return counts


def phase2_recommendation(knn: dict[str, Any], similarity_rows: list[dict[str, str]]) -> str:
    f1 = float(knn.get("f1") or 0.0)
    margin = 0.0
    for row in similarity_rows:
        if row.get("separation_margin"):
            try:
                margin = float(row["separation_margin"])
                break
            except ValueError:
                pass
    if f1 >= 0.70 and margin >= 0.10:
        return (
            "DINO特徴量は face / non-face の分離に有効そうである。"
            "ただし、これは検出性能ではない。実際のPrecision / Recall / mAPを確認するには、"
            "Phase 2で frozen DINO + detection head を実装する必要がある。"
        )
    return (
        "face / non-face の分離が弱い、またはhard negativeがface clusterに混ざる可能性がある。"
        "Phase 2に進む前に、入力解像度、使用するDINO層、crop設計、patch tokenの扱いを見直すべきである。"
    )


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_report(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = output_dir(config)
    report_path = out_dir / "report.html"
    features_info = read_json(out_dir / "features" / "feature_run_info.json")
    extract_summary = read_json(out_dir / "extract_features_summary.json")
    analysis_summary = read_json(out_dir / "analysis_summary.json")
    knn = read_json(out_dir / "metrics" / "knn_metrics.json")
    retrieval = read_json(out_dir / "metrics" / "retrieval_metrics.json")
    similarity_rows = read_similarity_csv(out_dir / "metrics" / "similarity_metrics.csv")
    crop_meta_path = out_dir / "features" / "crop_features_meta.csv"
    records = read_crop_metadata(crop_meta_path) if crop_meta_path.exists() else []
    counts = crop_counts(records)
    recommendation = phase2_recommendation(knn, similarity_rows)
    overlay_dir = out_dir / "overlays" / "similarity_map_samples"
    overlays = sorted(overlay_dir.glob("*.png"))[:10] if overlay_dir.exists() else []

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Phase 1 DINO Feature Analysis</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:28px;line-height:1.55;max-width:1100px} table{border-collapse:collapse;margin:12px 0 24px} th,td{border:1px solid #ddd;padding:6px 9px;text-align:right} th:first-child,td:first-child{text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}.grid img{max-width:100%;border:1px solid #ccc}.note{background:#fff8d6;border-left:4px solid #d6a600;padding:12px}</style>",
        "</head><body>",
        "<h1>Phase 1: Frozen DINO Feature Analysis</h1>",
        "<p>DINO backbone が fine-tune なしでも顔・頭・keypoint周辺・背景を分離しやすい特徴を出しているかを診断するレポートです。</p>",
        "<div class='note'><b>重要:</b> 今回の実装ではDINO backboneはfine-tuneしていません。detection head / keypoint headの学習も行っていません。そのため、Precision / Recall / mAPはまだ評価対象ではありません。</div>",
        "<h2>Run Info</h2>",
        "<table>",
        "<tr><th>item</th><th>value</th></tr>",
        row("DINO model", features_info.get("model_name", config.get("dino", {}).get("model_name", ""))),
        row("device", features_info.get("device", "")),
        row("frozen_backbone", features_info.get("frozen_backbone", True)),
        row("fine_tuned", features_info.get("fine_tuned", False)),
        row("detection_head_trained", features_info.get("detection_head_trained", False)),
        row("keypoint_head_trained", features_info.get("keypoint_head_trained", False)),
        row("crop_features.npy shape", extract_summary.get("crop_features_shape", "")),
        "</table>",
        "<h2>Crop Counts</h2>",
        dict_table("crop_type", counts["crop_type"]),
        dict_table("label", counts["label"]),
        dict_table("face_size_category", counts["face_size_category"]),
        "<h2>k-NN Classification</h2>",
        "<p>これは特徴空間上の face_like / non_face 分離診断であり、検出性能ではありません。</p>",
        dict_table("metric", {key: value for key, value in knn.items() if key not in {"confusion_matrix", "note"}}),
        "<h2>Cosine Similarity</h2>",
        similarity_table(similarity_rows),
        img_link("figures/similarity_histogram.png", "similarity histogram"),
        "<h2>Retrieval</h2>",
        dict_table("metric", retrieval),
        f"<p><a href='{html.escape(rel(out_dir / 'retrieval' / 'knn_retrieval.html', report_path.parent))}'>retrieval HTML</a></p>",
        "<h2>PCA / UMAP</h2>",
        figure_grid(out_dir, ["figures/pca_by_crop_type.png", "figures/pca_by_label.png", "figures/pca_by_face_size.png", "figures/umap_by_crop_type.png", "figures/umap_by_label.png"]),
        "<h2>Patch Similarity Map</h2>",
        "<p>face cropの平均特徴をprototypeとして、画像patch tokenとのcosine similarityをoverlayしています。</p>",
        overlay_grid(out_dir, overlays),
        "<h2>このPhase 1で判断できること</h2>",
        "<ul><li>face cropとbackground cropがDINO特徴空間で分離しやすいか</li><li>head / face / eye regionが近い特徴を持つか</li><li>hard negativeがface featureに混ざりやすいか</li><li>face prototypeがGT bbox周辺patchで強く反応するか</li></ul>",
        "<h2>このPhase 1では判断できないこと</h2>",
        "<ul><li>顔検出のPrecision / Recall / mAP</li><li>keypoint headの精度</li><li>実運用thresholdでの検出性能</li><li>DINOをfine-tuneした場合の性能</li></ul>",
        "<h2>Phase 2 判定</h2>",
        f"<p>{html.escape(recommendation)}</p>",
        "<h2>Generated Artifacts</h2>",
        "<ul>",
        artifact(out_dir, "features/crop_features.npy"),
        artifact(out_dir, "features/crop_features_meta.csv"),
        artifact(out_dir, "features/patch_features_sample.npz"),
        artifact(out_dir, "metrics/knn_metrics.json"),
        artifact(out_dir, "metrics/similarity_metrics.csv"),
        artifact(out_dir, "metrics/patch_similarity_metrics.csv"),
        artifact(out_dir, "retrieval/knn_retrieval.html"),
        "</ul>",
        "</body></html>",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "report": str(report_path),
        "phase2_recommendation": recommendation,
        "output_dir": str(out_dir),
        "analysis_summary": analysis_summary,
    }
    (out_dir / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def row(key: str, value: Any) -> str:
    return f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"


def dict_table(first_header: str, values: dict[str, Any]) -> str:
    lines = [f"<table><tr><th>{html.escape(first_header)}</th><th>value</th></tr>"]
    for key, value in values.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(row(str(key), value))
    lines.append("</table>")
    return "\n".join(lines)


def similarity_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p>No similarity metrics found.</p>"
    fields = ["pair_type", "count", "mean", "std", "median", "p25", "p75", "separation_margin"]
    lines = ["<table><tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fields) + "</tr>"]
    for item in rows:
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(item.get(field, '')))}</td>" for field in fields) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def img_link(relative_path: str, alt: str) -> str:
    return f"<p><img src='{html.escape(relative_path)}' alt='{html.escape(alt)}' style='max-width:760px;width:100%'></p>"


def figure_grid(out_dir: Path, figure_paths: list[str]) -> str:
    items = []
    for figure in figure_paths:
        path = out_dir / figure
        if path.exists():
            items.append(f"<figure><img src='{html.escape(figure)}'><figcaption>{html.escape(Path(figure).name)}</figcaption></figure>")
    if not items:
        return "<p>No PCA/UMAP figures found.</p>"
    return "<div class='grid'>" + "\n".join(items) + "</div>"


def overlay_grid(out_dir: Path, overlays: list[Path]) -> str:
    if not overlays:
        return "<p>No similarity map overlays found.</p>"
    items = []
    for path in overlays:
        relative = rel(path, out_dir)
        items.append(f"<figure><img src='{html.escape(relative)}'><figcaption>{html.escape(path.name)}</figcaption></figure>")
    return "<div class='grid'>" + "\n".join(items) + "</div>"


def artifact(out_dir: Path, relative_path: str) -> str:
    path = out_dir / relative_path
    if not path.exists():
        return f"<li>{html.escape(relative_path)}: missing</li>"
    return f"<li><a href='{html.escape(relative_path)}'>{html.escape(relative_path)}</a></li>"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, output_dir_override=args.output_dir)
    summary = make_report(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
