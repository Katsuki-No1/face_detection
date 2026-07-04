from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outsput" / "experience" / "dino_hfdp"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    started = now_text()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n# started_at={started}\n# command={' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        code = process.wait()
        log_file.write(f"# finished_at={now_text()}\n# returncode={code}\n")
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def train_command(
    *,
    config: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int | None,
    num_workers: int | None,
    save_every: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(config),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--output-dir",
        str(output_dir),
        "--save-every",
        str(save_every),
        "--feature-cache",
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    last = output_dir / "checkpoints" / "last.pt"
    metrics = read_metrics(output_dir / "metrics.csv")
    if last.exists() and metrics:
        completed = max(int(float(row["epoch"])) for row in metrics if row.get("epoch"))
        if completed < epochs:
            cmd.extend(["--resume-from", str(last)])
    return cmd


def dry_run_command(config: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "scripts/train_dino.py",
        "--config",
        str(config),
        "--dry-run",
        "--device",
        device,
        "--image-size",
        "224",
        "--batch-size",
        "1",
        "--num-workers",
        "0",
    ]


def checkpoint_for(run_dir: Path) -> Path:
    best = run_dir / "checkpoints" / "best_loss.pt"
    if best.exists():
        return best
    last = run_dir / "checkpoints" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, smoke-test, train, plot and evaluate DINOv3 HFDP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "outsput" / "experience" / "data")
    parser.add_argument("--head-config", type=Path, default=REPO_ROOT / "configs" / "dino_hfdp" / "head_detector.yaml")
    parser.add_argument("--face-config", type=Path, default=REPO_ROOT / "configs" / "dino_hfdp" / "headcrop_face_keypoint.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--only", choices=["both", "head", "face"], default="both")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    runs_dir = output_dir / "runs"
    head_run = runs_dir / "dino_hfdp_head_detector"
    face_run = runs_dir / "dino_hfdp_headcrop_face_keypoint"

    if not args.skip_prepare:
        run_logged(
            [
                sys.executable,
                "script/script_experiment/prepare_dino_hfdp_data.py",
                "--source-root",
                str(args.source_root),
                "--output-dir",
                str(output_dir),
            ],
            logs_dir / "prepare_data.log",
        )

    if not args.skip_dry_run:
        if args.only in {"both", "head"}:
            run_logged(dry_run_command(args.head_config, args.device), logs_dir / "dry_run_head.log")
        if args.only in {"both", "face"}:
            run_logged(dry_run_command(args.face_config, args.device), logs_dir / "dry_run_face.log")

    if not args.skip_train:
        if args.only in {"both", "head"}:
            run_logged(
                train_command(
                    config=args.head_config,
                    output_dir=head_run,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    save_every=args.save_every,
                ),
                logs_dir / "train_head_detector.log",
            )
        if args.only in {"both", "face"}:
            run_logged(
                train_command(
                    config=args.face_config,
                    output_dir=face_run,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    save_every=args.save_every,
                ),
                logs_dir / "train_headcrop_face_keypoint.log",
            )

    run_logged(
        [
            sys.executable,
            "scripts/plot_dino_centerface_aug_learning_curves.py",
            "--run-root",
            str(runs_dir),
            "--output-dir",
            str(output_dir / "reports" / "learning_curves"),
        ],
        logs_dir / "plot_learning_curves.log",
    )

    if not args.skip_eval and args.only == "both":
        run_logged(
            [
                sys.executable,
                "script/script_experiment/evaluate_dino_hfdp.py",
                "--data-root",
                str(args.source_root),
                "--output-dir",
                str(output_dir / "eval"),
                "--head-config",
                str(args.head_config),
                "--head-checkpoint",
                str(checkpoint_for(head_run)),
                "--face-config",
                str(args.face_config),
                "--face-checkpoint",
                str(checkpoint_for(face_run)),
                "--device",
                args.device,
            ],
            logs_dir / "evaluate_dino_hfdp.log",
        )

    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": now_text(),
            "source_root": str(args.source_root),
            "output_dir": str(output_dir),
            "device": args.device,
            "epochs": args.epochs,
            "head_config": str(args.head_config),
            "face_config": str(args.face_config),
            "head_run": str(head_run),
            "face_run": str(face_run),
            "logs_dir": str(logs_dir),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
