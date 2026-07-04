from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_utils import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run DINO CenterFace ellipse augmentation ablation.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/dino_centerface_ellipse_aug_ablation"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/dino_centerface_ellipse_aug_ablation"))
    parser.add_argument("--eval-root", type=Path, default=Path("outputs_experiment/dino_centerface_ellipse_aug_ablation_eval"))
    parser.add_argument("--preview-root", type=Path, default=Path("outputs/augmentation_debug"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Execute commands instead of only writing the manifest.")
    parser.add_argument("--allow-training", action="store_true", help="Required with --execute to run real training.")
    parser.add_argument("--dry-run-smoke", action="store_true", help="Execute one-batch dry-run for each config. No training.")
    parser.add_argument("--preview", action="store_true", help="Execute augmentation preview for each config. No training.")
    parser.add_argument("--preview-samples", type=int, default=20)
    parser.add_argument("--manifest", type=Path, default=Path("outputs_experiment/dino_centerface_ellipse_aug_ablation_manifest/commands.json"))
    return parser.parse_args()


def config_paths(config_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(config_dir.glob("E*.yaml"))
        if path.is_file() and load_yaml_config(path).get("experiment", {}).get("active", True) is not False
    ]
    if not paths:
        raise FileNotFoundError(f"No active E*.yaml configs found under {config_dir}")
    return paths


def experiment_name(config: dict[str, Any], path: Path) -> str:
    return str(config.get("experiment", {}).get("name") or config.get("train", {}).get("name") or path.stem)


def command_row(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(path)
    name = experiment_name(config, path)
    run_dir = args.run_root / name
    eval_dir = args.eval_root / name
    train_cmd = [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(path),
        "--output-dir",
        str(run_dir),
        "--save-every",
        "0",
    ]
    dry_run_cmd = [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(path),
        "--dry-run",
    ]
    if args.device:
        train_cmd.extend(["--device", str(args.device)])
        dry_run_cmd.extend(["--device", str(args.device)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
        dry_run_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        train_cmd.extend(["--num-workers", str(args.num_workers)])
        dry_run_cmd.extend(["--num-workers", str(args.num_workers)])
    if args.prefetch_factor is not None:
        train_cmd.extend(["--prefetch-factor", str(args.prefetch_factor)])
        dry_run_cmd.extend(["--prefetch-factor", str(args.prefetch_factor)])
    eval_cfg = config.get("evaluation", {})
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_dino_centerface_ellipse.py",
        "--config",
        str(path),
        "--checkpoint",
        str(run_dir / "checkpoints" / "best_loss.pt"),
        "--data-root",
        str(config.get("data", {}).get("data_root")),
        "--split",
        str(eval_cfg.get("split", "test")),
        "--imgsz",
        str(eval_cfg.get("imgsz", config.get("train", {}).get("imgsz", 896))),
        "--prediction-conf-min",
        str(eval_cfg.get("prediction_conf_min", 0.001)),
        "--iou-threshold",
        str(eval_cfg.get("iou_threshold", 0.5)),
        "--thresholds",
        str(eval_cfg.get("thresholds", "0.01:0.95:0.01")),
        "--output-dir",
        str(eval_dir),
    ]
    if args.device:
        eval_cmd.extend(["--device", str(args.device)])
    preview_cmd = [
        sys.executable,
        "scripts/preview_dino_centerface_aug_ablation.py",
        "--config",
        str(path),
        "--output-root",
        str(args.preview_root),
        "--samples",
        str(args.preview_samples),
    ]
    collect_cmd = [
        sys.executable,
        "scripts/collect_dino_centerface_aug_ablation_results.py",
        "--config-dir",
        str(args.config_dir),
        "--eval-root",
        str(args.eval_root),
        "--output-dir",
        str(args.eval_root / "_summary"),
    ]
    return {
        "experiment": name,
        "config": str(path),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "train_command": train_cmd,
        "dry_run_command": dry_run_cmd,
        "eval_command": eval_cmd,
        "preview_command": preview_cmd,
        "collect_command": collect_cmd,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "config", "run_dir", "eval_dir", "train_command", "dry_run_command", "eval_command", "preview_command"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: " ".join(row[key]) if isinstance(row.get(key), list) else row.get(key, "") for key in fields})


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    rows = [command_row(path, args) for path in config_paths(args.config_dir)]
    manifest = {
        "note": "Default mode only writes commands. Real training requires --execute --allow-training.",
        "config_dir": str(args.config_dir),
        "run_root": str(args.run_root),
        "eval_root": str(args.eval_root),
        "preview_root": str(args.preview_root),
        "experiments": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.manifest.with_suffix(".csv"), rows)

    if args.execute and args.allow_training:
        for row in rows:
            run_command(row["train_command"])
            run_command(row["eval_command"])
        run_command(rows[0]["collect_command"])
    elif args.execute and args.dry_run_smoke:
        for row in rows:
            run_command(row["dry_run_command"])
    elif args.execute and args.preview:
        for row in rows:
            run_command(row["preview_command"])
    elif args.execute:
        raise RuntimeError("Refusing to execute without --allow-training, --dry-run-smoke, or --preview.")

    print(json.dumps({"manifest": str(args.manifest), "experiments": len(rows), "executed": bool(args.execute)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
