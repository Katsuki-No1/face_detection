from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_scalars(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def latest_scalars(work_dir: Path) -> Path:
    candidates = sorted(work_dir.glob("*/vis_data/scalars.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"scalars.json not found under {work_dir}")
    return candidates[-1]


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if "epoch" in row]
    val_rows = [row for row in rows if "epoch" not in row]
    return train_rows, val_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, row in enumerate(rows, start=1):
        value = row.get(key)
        if value is None:
            continue
        xs.append(float(row.get("epoch", row.get("step", index))))
        ys.append(float(value))
    return xs, ys


def plot_detector(name: str, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows, val_rows = split_rows(rows)
    summary_rows: list[dict[str, Any]] = []
    for row in train_rows:
        summary_rows.append(
            {
                "model": name,
                "phase": "train",
                "epoch": row.get("epoch"),
                "loss": row.get("loss"),
                "loss_cls": row.get("loss_cls"),
                "loss_bbox": row.get("loss_bbox"),
                "lr": row.get("lr"),
            }
        )
    for row in val_rows:
        summary_rows.append(
            {
                "model": name,
                "phase": "val",
                "epoch": row.get("step"),
                "bbox_mAP": row.get("coco/bbox_mAP"),
                "bbox_mAP_50": row.get("coco/bbox_mAP_50"),
                "bbox_mAP_75": row.get("coco/bbox_mAP_75"),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    for key in ("loss", "loss_cls", "loss_bbox"):
        xs, ys = series(train_rows, key)
        if xs:
            axes[0].plot(xs, ys, label=key)
    axes[0].set_title(f"{name} train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key, label in (("coco/bbox_mAP", "mAP"), ("coco/bbox_mAP_50", "mAP50"), ("coco/bbox_mAP_75", "mAP75")):
        xs, ys = series(val_rows, key)
        if xs:
            axes[1].plot(xs, ys, label=label)
    axes[1].set_title(f"{name} validation bbox")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_curves.png")
    plt.close(fig)
    return summary_rows


def plot_pose(name: str, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows, val_rows = split_rows(rows)
    summary_rows: list[dict[str, Any]] = []
    for row in train_rows:
        summary_rows.append(
            {
                "model": name,
                "phase": "train",
                "epoch": row.get("epoch"),
                "loss": row.get("loss"),
                "loss_kpt": row.get("loss_kpt"),
                "acc_pose": row.get("acc_pose"),
                "lr": row.get("lr"),
            }
        )
    for row in val_rows:
        summary_rows.append({"model": name, "phase": "val", "epoch": row.get("step"), "NME": row.get("NME")})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    xs, ys = series(train_rows, "loss")
    if xs:
        axes[0].plot(xs, ys, label="loss")
    xs, ys = series(train_rows, "loss_kpt")
    if xs:
        axes[0].plot(xs, ys, label="loss_kpt")
    axes[0].set_title(f"{name} train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    xs, ys = series(val_rows, "NME")
    if xs:
        axes[1].plot(xs, ys, label="NME")
    axes[1].set_title(f"{name} validation NME")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_curves.png")
    plt.close(fig)
    return summary_rows


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    out_name = args.out_name or (f"curves_{args.run_id}" if args.run_id else "curves")
    out_dir = output_dir / "reports" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.run_id}" if args.run_id else ""
    specs = [
        (
            "rtmdet_hf_full",
            args.detector_work_dir or output_dir / "work_dirs" / "mmdet" / f"rtmdet_hf_full{suffix}",
            "detector",
        ),
        (
            "rtmdet_headcrop_face",
            args.crop_face_detector_work_dir
            or output_dir / "work_dirs" / "mmdet" / f"rtmdet_headcrop_face_full100_640{suffix}",
            "detector",
        ),
        (
            "rtmpose_face",
            args.pose_face_work_dir or output_dir / "work_dirs" / "mmpose" / f"rtmpose_face{suffix}",
            "pose",
        ),
        (
            "rtmpose_head",
            args.pose_head_work_dir or output_dir / "work_dirs" / "mmpose" / f"rtmpose_head{suffix}",
            "pose",
        ),
    ]
    all_rows: list[dict[str, Any]] = []
    for name, work_dir, kind in specs:
        scalar_path = latest_scalars(work_dir)
        rows = read_scalars(scalar_path)
        if kind == "detector":
            all_rows.extend(plot_detector(name, rows, out_dir))
        else:
            all_rows.extend(plot_pose(name, rows, out_dir))

    write_csv(out_dir / "training_curves.csv", all_rows)
    print(f"saved: {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot FDP-series fine-tuning curves.")
    parser.add_argument("--output-dir", type=Path, default=Path("outsput/experience"))
    parser.add_argument("--run-id", default=None, help="Optional suffix appended to the standard work_dir names.")
    parser.add_argument("--out-name", default=None, help="Output directory name under reports/.")
    parser.add_argument("--detector-work-dir", type=Path, default=None)
    parser.add_argument("--crop-face-detector-work-dir", type=Path, default=None)
    parser.add_argument("--pose-face-work-dir", type=Path, default=None)
    parser.add_argument("--pose-head-work-dir", type=Path, default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
