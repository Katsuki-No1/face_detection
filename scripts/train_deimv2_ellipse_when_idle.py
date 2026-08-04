#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_ellipse.yaml"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_960_ellipse"
)
DEFAULT_QUEUE_DIR = PROJECT_ROOT / "outputs/training_queues/dinov3_deimv2_ellipse"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for existing model-training processes and a stably idle GPU, "
            "then train the DEIMv2-L direct-ellipse model."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--log-interval-seconds", type=float, default=60.0)
    parser.add_argument("--idle-samples", type=int, default=4)
    parser.add_argument("--max-gpu-utilization", type=float, default=10.0)
    parser.add_argument("--max-memory-used-mib", type=float, default=2500.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum wait time. Zero waits indefinitely.",
    )
    parser.add_argument(
        "--extra-process-pattern",
        action="append",
        default=[],
        help="Additional regular expression treated as a blocking training command. Repeatable.",
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="sato-waseda-university")
    parser.add_argument("--wandb-project", default="face_detection")
    parser.add_argument(
        "--wandb-group",
        default="face_pose_drive_v2_balanced_811_deimv2_ellipse",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_ellipse_seed0",
    )
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the current blockers, GPU state, and future command without waiting or starting training.",
    )
    return parser.parse_args()


def read_processes(proc_root: Path = Path("/proc")) -> list[ProcessInfo]:
    processes = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
            ppid = int(stat_fields[3])
            raw_command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
        arguments = [part.decode("utf-8", errors="replace") for part in raw_command.split(b"\0") if part]
        if not arguments:
            continue
        processes.append(ProcessInfo(pid=int(entry.name), ppid=ppid, command=shlex.join(arguments)))
    return processes


def is_training_command(command: str, *, extra_patterns: list[re.Pattern[str]] | None = None) -> bool:
    lowered = command.lower()
    tokens = shlex.split(command)
    basenames = [Path(token).name.lower() for token in tokens]
    scripts = [name for name in basenames if name.endswith((".py", ".sh"))]
    has_train_stage = "--stage" in tokens and any(
        tokens[index + 1].lower() == "train"
        for index, token in enumerate(tokens[:-1])
        if token == "--stage"
    )
    known_runner = any(
        name.startswith("train")
        or name in {
            "run_landmark_experiments.py",
            "run_yolo26_pose.py",
            "run_face_pose_drive_v2_deimv2_l_yolo26m.sh",
        }
        for name in scripts
    ) and not any("when_idle" in name or "training_queue" in name for name in scripts)
    distributed_training = any(name in {"torchrun", "deepspeed", "accelerate"} for name in basenames) and "train" in lowered
    built_in_match = (known_runner and ("python" in lowered or has_train_stage)) or has_train_stage or distributed_training
    return built_in_match or any(pattern.search(command) for pattern in extra_patterns or [])


def blocking_training_processes(
    processes: list[ProcessInfo],
    *,
    current_pid: int,
    extra_patterns: list[re.Pattern[str]] | None = None,
) -> list[ProcessInfo]:
    by_pid = {process.pid: process for process in processes}
    matched = {
        process.pid: process
        for process in processes
        if process.pid != current_pid and is_training_command(process.command, extra_patterns=extra_patterns)
    }
    roots = []
    for process in matched.values():
        ancestor_pid = process.ppid
        has_matching_ancestor = False
        visited = set()
        while ancestor_pid > 1 and ancestor_pid not in visited:
            visited.add(ancestor_pid)
            if ancestor_pid in matched:
                has_matching_ancestor = True
                break
            ancestor = by_pid.get(ancestor_pid)
            if ancestor is None:
                break
            ancestor_pid = ancestor.ppid
        if not has_matching_ancestor:
            roots.append(process)
    return sorted(roots, key=lambda process: process.pid)


def query_gpu(gpu_index: int) -> dict[str, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(gpu_index)}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
    return {
        "utilization_percent": values[0],
        "memory_used_mib": values[1],
        "memory_total_mib": values[2],
    }


def write_status(queue_dir: Path, state: str, **fields: Any) -> None:
    payload = {"state": state, "updated_at": now_iso(), **fields}
    temporary = queue_dir / ".status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, queue_dir / "status.json")


def log_line(log_file: TextIO, message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def wait_until_available(args: argparse.Namespace, log_file: TextIO) -> dict[str, float]:
    extra_patterns = [re.compile(value, re.IGNORECASE) for value in args.extra_process_pattern]
    started = time.monotonic()
    last_log = float("-inf")
    idle_streak = 0
    latest_gpu: dict[str, float] = {}
    while True:
        blockers = blocking_training_processes(
            read_processes(),
            current_pid=os.getpid(),
            extra_patterns=extra_patterns,
        )
        try:
            latest_gpu = query_gpu(args.gpu_index)
            gpu_idle = (
                latest_gpu["utilization_percent"] <= float(args.max_gpu_utilization)
                and latest_gpu["memory_used_mib"] <= float(args.max_memory_used_mib)
            )
            gpu_error = None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
            latest_gpu = {}
            gpu_idle = False
            gpu_error = f"{type(error).__name__}: {error}"

        idle_streak = idle_streak + 1 if not blockers and gpu_idle else 0
        blocker_rows = [asdict(process) for process in blockers]
        write_status(
            args.queue_dir,
            "waiting",
            waiting_since_seconds=round(time.monotonic() - started, 1),
            blockers=blocker_rows,
            gpu=latest_gpu,
            gpu_error=gpu_error,
            idle_streak=idle_streak,
            idle_samples_required=int(args.idle_samples),
        )
        if idle_streak >= max(1, int(args.idle_samples)):
            log_line(
                log_file,
                "GPU is stably idle and no training processes remain; starting queued ellipse training.",
            )
            return latest_gpu

        if args.timeout_seconds > 0 and time.monotonic() - started >= float(args.timeout_seconds):
            raise TimeoutError(
                f"Training/GPU wait exceeded {args.timeout_seconds}s; blockers={blocker_rows}, gpu={latest_gpu}"
            )
        if time.monotonic() - last_log >= max(1.0, float(args.log_interval_seconds)):
            last_log = time.monotonic()
            blocker_summary = "; ".join(
                f"pid={process.pid} {process.command}" for process in blockers
            ) or "none"
            gpu_summary = (
                f"util={latest_gpu.get('utilization_percent', float('nan')):.0f}% "
                f"memory={latest_gpu.get('memory_used_mib', float('nan')):.0f}/"
                f"{latest_gpu.get('memory_total_mib', float('nan')):.0f}MiB"
            )
            log_line(
                log_file,
                f"waiting: blockers=[{blocker_summary}] {gpu_summary} "
                f"idle_streak={idle_streak}/{args.idle_samples}",
            )
        time.sleep(max(1.0, float(args.poll_seconds)))


def training_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        str(PROJECT_ROOT / "tools/train_dinov3_deimv2.py"),
        "--config",
        str(args.config.resolve()),
        "--data-root",
        str((PROJECT_ROOT / "outputs/dataset_cache/face_pose_current").resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--device",
        str(args.device),
    ]
    last_checkpoint = args.output_dir / "checkpoints/last.pt"
    completion_marker = args.output_dir / "summary.md"
    if last_checkpoint.is_file() and not completion_marker.is_file():
        command.extend(["--resume-from", str(last_checkpoint.resolve())])
    if args.wandb:
        command.extend(
            [
                "--wandb",
                "--wandb-entity",
                str(args.wandb_entity),
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-run-name",
                str(args.wandb_run_name),
                "--wandb-group",
                str(args.wandb_group),
                "--wandb-mode",
                "online",
                "--dataset-version",
                "face_pose_drive_v2_balanced_811",
            ]
        )
    return command


def stream_training(command: list[str], log_file: TextIO) -> int:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return int(process.wait())


def main() -> int:
    args = parse_args()
    args.config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    args.queue_dir = args.queue_dir if args.queue_dir.is_absolute() else PROJECT_ROOT / args.queue_dir
    if not args.config.is_file():
        raise FileNotFoundError(f"Ellipse training config not found: {args.config}")
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(args)

    blockers = blocking_training_processes(
        read_processes(),
        current_pid=os.getpid(),
        extra_patterns=[re.compile(value, re.IGNORECASE) for value in args.extra_process_pattern],
    )
    gpu = query_gpu(args.gpu_index)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "state": "dry_run",
                    "blockers": [asdict(process) for process in blockers],
                    "gpu": gpu,
                    "command": command,
                    "command_shell": shlex.join(command),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    lock_path = args.queue_dir / "queue.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another ellipse-training queue watcher already holds {lock_path}"
            ) from error

        log_path = args.queue_dir / "queue.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            if (args.output_dir / "summary.md").is_file():
                log_line(log_file, f"Training is already complete; skipping: {args.output_dir}")
                write_status(args.queue_dir, "complete", skipped=True, reason="summary.md already exists")
                return 0
            try:
                log_line(log_file, f"Queue watcher started: {shlex.join(command)}")
                if args.wait:
                    wait_until_available(args, log_file)
                else:
                    log_line(log_file, "Waiting disabled by --no-wait.")
                write_status(
                    args.queue_dir,
                    "running",
                    command=command,
                    output_dir=str(args.output_dir),
                    started_at=now_iso(),
                )
                exit_code = stream_training(command, log_file)
                if exit_code != 0:
                    write_status(args.queue_dir, "failed", exit_code=exit_code, output_dir=str(args.output_dir))
                    log_line(log_file, f"Ellipse training failed with exit code {exit_code}.")
                    return exit_code
                write_status(args.queue_dir, "complete", exit_code=0, output_dir=str(args.output_dir))
                log_line(log_file, "Ellipse training completed successfully.")
                return 0
            except BaseException as error:
                write_status(
                    args.queue_dir,
                    "failed",
                    error=f"{type(error).__name__}: {error}",
                    output_dir=str(args.output_dir),
                )
                log_line(log_file, f"Queue watcher failed: {type(error).__name__}: {error}")
                raise


if __name__ == "__main__":
    raise SystemExit(main())
