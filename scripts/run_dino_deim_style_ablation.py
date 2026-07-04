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


TARGETS = ("face", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run DINO-DEIM-style E1-E5 ablation.")
    parser.add_argument("--config-root", type=Path, default=Path("configs/dino_deim_style_ablation"))
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="face")
    parser.add_argument("--run-root", type=Path, default=Path("runs/dino_deim_style_ablation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Execute commands instead of only writing the manifest.")
    parser.add_argument("--allow-training", action="store_true", help="Required with --execute to run real training.")
    parser.add_argument("--dry-run-smoke", action="store_true", help="Execute CPU synthetic one-batch dry-runs for each config.")
    parser.add_argument("--smoke-image-size", type=int, default=64)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=Path("outputs_experiment/dino_deim_style_ablation_manifest/commands.json"))
    return parser.parse_args()


def active_config_paths(config_root: Path, target: str) -> list[Path]:
    target_dirs = TARGETS if target == "all" else (target,)
    paths: list[Path] = []
    for target_dir in target_dirs:
        config_dir = config_root / target_dir
        found = [
            path
            for path in sorted(config_dir.glob("E*.yaml"))
            if path.is_file() and load_yaml_config(path).get("experiment", {}).get("active", True) is not False
        ]
        if not found:
            raise FileNotFoundError(f"No active E*.yaml configs found under {config_dir}")
        paths.extend(found)
    return paths


def experiment_name(config: dict[str, Any], path: Path) -> str:
    return str(config.get("experiment", {}).get("name") or config.get("train", {}).get("name") or path.stem)


def _append_common_train_options(command: list[str], args: argparse.Namespace) -> None:
    if args.device:
        command.extend(["--device", str(args.device)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.prefetch_factor is not None:
        command.extend(["--prefetch-factor", str(args.prefetch_factor)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])


def command_row(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(path)
    name = experiment_name(config, path)
    target = path.parent.name
    run_dir = args.run_root / target / name
    train_cmd = [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(path),
        "--output-dir",
        str(run_dir),
        "--save-every",
        str(args.save_every),
    ]
    _append_common_train_options(train_cmd, args)

    dry_run_cmd = [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(path),
        "--dry-run",
        "--synthetic",
        "--tiny-model",
        "--device",
        "cpu",
        "--image-size",
        str(args.smoke_image_size),
        "--batch-size",
        str(args.smoke_batch_size),
    ]
    return {
        "experiment": name,
        "target": target,
        "stage": path.stem.split("_", 1)[0],
        "config": str(path),
        "run_dir": str(run_dir),
        "train_command": train_cmd,
        "dry_run_command": dry_run_cmd,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "target", "stage", "config", "run_dir", "train_command", "dry_run_command"]
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
    rows = [command_row(path, args) for path in active_config_paths(args.config_root, args.target)]
    manifest = {
        "note": "Default mode only writes commands. Real training requires --execute --allow-training.",
        "ablation": {
            "E1": "DINO + adapter + RT-DETR-style detector, standard Hungarian matching, cls/L1/GIoU loss.",
            "E2": "E1 + controlled mosaic/mixup/geometry/photometric augmentation.",
            "E3": "E2 + Dense O2O-style extra positive query-target matches.",
            "E4": "E3 + MAL-style quality-aware positive loss weighting.",
            "E5": "E4 + STA/decoder-refinement style toggles.",
        },
        "config_root": str(args.config_root),
        "target": args.target,
        "run_root": str(args.run_root),
        "experiments": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.manifest.with_suffix(".csv"), rows)

    if args.execute and args.allow_training:
        for row in rows:
            run_command(row["train_command"])
    elif args.execute and args.dry_run_smoke:
        for row in rows:
            run_command(row["dry_run_command"])
    elif args.execute:
        raise RuntimeError("Refusing to execute without --allow-training or --dry-run-smoke.")

    print(json.dumps({"manifest": str(args.manifest), "experiments": len(rows), "executed": bool(args.execute)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
