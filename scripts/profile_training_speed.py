from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a free GPU, run one training profile, and summarize nvidia-smi samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gpu_profiles"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progressive-resizing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wait-for-free-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--free-gpu-util", type=float, default=10.0)
    parser.add_argument("--free-memory-used-mib", type=float, default=3000.0)
    parser.add_argument("--free-samples", type=int, default=5)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0, help="0 waits indefinitely.")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--wait-log-interval", type=float, default=30.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def query_gpu() -> dict[str, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    first = result.stdout.strip().splitlines()[0]
    util, memory_used, memory_total, power, temp = [float(item.strip()) for item in first.split(",")]
    return {
        "gpu_util": util,
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "power_w": power,
        "temperature_c": temp,
    }


def wait_for_free_gpu(args: argparse.Namespace) -> list[dict[str, float]]:
    if not args.wait_for_free_gpu:
        return []
    started = time.monotonic()
    last_log = 0.0
    consecutive = 0
    samples: list[dict[str, float]] = []
    while True:
        sample = query_gpu()
        samples.append(sample)
        is_free = sample["gpu_util"] <= args.free_gpu_util and sample["memory_used_mib"] <= args.free_memory_used_mib
        consecutive = consecutive + 1 if is_free else 0
        if consecutive >= max(1, int(args.free_samples)):
            return samples
        if args.wait_timeout_seconds > 0 and time.monotonic() - started > args.wait_timeout_seconds:
            raise TimeoutError(f"GPU did not become free within {args.wait_timeout_seconds}s; latest={sample}")
        now = time.monotonic()
        if now - last_log >= max(1.0, float(args.wait_log_interval)):
            last_log = now
            print(
                "waiting_for_gpu: "
                f"util={sample['gpu_util']:.0f}% memory={sample['memory_used_mib']:.0f}/{sample['memory_total_mib']:.0f}MiB "
                f"free_streak={consecutive}/{args.free_samples}",
                flush=True,
            )
        time.sleep(max(0.5, float(args.sample_interval)))


def profile_dir(args: argparse.Namespace) -> Path:
    name = args.name or f"{args.config.stem}_b{args.batch_size or 'cfg'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name).strip("._")
    return args.output_root / safe


def train_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_dino_scrfd_face.py",
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--save-every",
        "0",
        "--device",
        args.device,
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.prefetch_factor is not None:
        command.extend(["--prefetch-factor", str(args.prefetch_factor)])
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    for flag, value in (
        ("--torch-compile", args.torch_compile),
        ("--feature-cache", args.feature_cache),
        ("--fused-optimizer", args.fused_optimizer),
        ("--progressive-resizing", args.progressive_resizing),
    ):
        if value is True:
            command.append(flag)
        elif value is False:
            command.append("--no-" + flag[2:])
    return command


def parse_number(value: str) -> float:
    cleaned = re.sub(r"[^0-9.+-]", "", value.strip())
    return float(cleaned) if cleaned else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * quantile)))]


def aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {"samples": len(rows)}
    for key in ("gpu_util", "mem_util", "mem_used", "power", "temp"):
        values = [row[key] for row in rows]
        output[key] = {
            "min": min(values) if values else 0.0,
            "mean": statistics.fmean(values) if values else 0.0,
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else 0.0,
        }
    return output


def summarize_wait_samples(samples: list[dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {"samples": len(samples)}
    if not samples:
        return output
    output["first"] = samples[0]
    output["last"] = samples[-1]
    for key in ("gpu_util", "memory_used_mib", "power_w", "temperature_c"):
        values = [sample[key] for sample in samples]
        output[key] = {
            "min": min(values),
            "mean": statistics.fmean(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    return output


def summarize(profile: Path, wait_samples: list[dict[str, float]], status: int, wall_seconds: float, command: list[str]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with (profile / "gpu_samples.csv").open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "gpu_util": parse_number(row[" utilization.gpu [%]"]),
                    "mem_util": parse_number(row[" utilization.memory [%]"]),
                    "mem_used": parse_number(row[" memory.used [MiB]"]),
                    "power": parse_number(row[" power.draw [W]"]),
                    "temp": parse_number(row[" temperature.gpu"]),
                }
            )
    active_rows = [row for row in rows if row["mem_used"] >= 5000.0 and row["gpu_util"] >= 50.0]
    train_dir = profile / "train"
    run_meta = json.loads((train_dir / "run_meta.json").read_text(encoding="utf-8")) if (train_dir / "run_meta.json").exists() else {}
    metrics_rows = list(csv.DictReader((train_dir / "metrics.csv").open(encoding="utf-8"))) if (train_dir / "metrics.csv").exists() else []
    metrics = metrics_rows[-1] if metrics_rows else {}
    epoch_seconds = float(metrics.get("elapsed_seconds") or wall_seconds)
    dataset_size = float(run_meta.get("dataset_size") or 0)
    summary = {
        "profile_dir": str(profile),
        "status": status,
        "command": command,
        "wall_seconds": round(wall_seconds, 3),
        "epoch_seconds": epoch_seconds,
        "images_per_second": dataset_size / epoch_seconds if epoch_seconds > 0 else 0.0,
        "wait": summarize_wait_samples(wait_samples),
        "all_samples": aggregate(rows),
        "active_samples": aggregate(active_rows),
        "run_meta": run_meta,
        "metrics": metrics,
    }
    (profile / "gpu_profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    wait_samples = wait_for_free_gpu(args)
    profile = profile_dir(args)
    train_dir = profile / "train"
    profile.mkdir(parents=True, exist_ok=True)
    command = train_command(args, train_dir)
    command_meta = {"command": command, "wait": summarize_wait_samples(wait_samples)}
    (profile / "command.json").write_text(json.dumps(command_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu",
            "--format=csv",
            "-l",
            str(max(1, int(args.sample_interval))),
        ],
        cwd=REPO_ROOT,
        stdout=(profile / "gpu_samples.csv").open("w", encoding="utf-8"),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    started = time.monotonic()
    with (profile / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    wall_seconds = time.monotonic() - started
    monitor.terminate()
    try:
        monitor.wait(timeout=5)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=5)
    summary = summarize(profile, wait_samples, process.returncode, wall_seconds, command)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
