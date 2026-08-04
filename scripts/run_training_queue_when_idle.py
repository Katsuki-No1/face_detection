#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_CONFIG = (
    PROJECT_ROOT
    / "configs/training_queues/pending_models_tests_then_training.yaml"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_deimv2_ellipse_when_idle import (  # noqa: E402
    blocking_training_processes,
    log_line,
    now_iso,
    query_gpu,
    read_processes,
    stream_training,
    wait_until_available,
    write_status,
)


class QueueConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingJob:
    name: str
    command: tuple[str, ...]
    completion_marker: Path
    depends_on: tuple[str, ...]
    resume_checkpoint: Path | None
    resume_arguments: tuple[str, ...]
    enabled: bool = True
    phase: str = "other"
    architecture: str | None = None
    gate_stage: str | None = None


GATE_STAGES = ("schema", "gpu_step", "overfit", "small_scale")
JOB_PHASES = ("setup", "gate", "production_train", "evaluation", "postprocess", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an appendable YAML training queue after current training processes and the GPU become idle."
    )
    parser.add_argument("--queue-config", type=Path, default=DEFAULT_QUEUE_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start the queue runner in a background session and return immediately.",
    )
    parser.add_argument(
        "--after-pid",
        type=int,
        help=(
            "Defer loading and running the queue until this process exits. "
            "This safely hands an updated appendable queue to a follow-up runner."
        ),
    )
    return parser.parse_args()


def _expand(value: str, *, queue_config: Path) -> str:
    replacements = {
        "project_root": str(PROJECT_ROOT),
        "queue_config_dir": str(queue_config.parent),
    }
    try:
        return str(value).format_map(replacements)
    except KeyError as error:
        raise QueueConfigError(f"Unknown queue placeholder {error} in {value!r}") from error


def _resolve_path(value: str | Path, *, queue_config: Path) -> Path:
    expanded = Path(_expand(str(value), queue_config=queue_config)).expanduser()
    return expanded if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def _load_raw_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise QueueConfigError(f"Could not read queue config {path}: {error}") from error
    if not isinstance(value, dict):
        raise QueueConfigError(f"Queue config must be a mapping: {path}")
    if int(value.get("version", 0)) != 1:
        raise QueueConfigError(f"Unsupported queue config version: {value.get('version')!r}")
    return value


def load_queue(path: Path) -> tuple[SimpleNamespace, list[TrainingJob], bool]:
    queue_config = path.resolve()
    raw = _load_raw_config(queue_config)
    queue = raw.get("queue")
    if not isinstance(queue, dict):
        raise QueueConfigError("queue section is required")
    gpu = queue.get("gpu", {})
    gpu = gpu if isinstance(gpu, dict) else {}
    wait = queue.get("wait", {})
    wait = wait if isinstance(wait, dict) else {}
    queue_dir = _resolve_path(queue.get("state_dir", "outputs/training_queues/default"), queue_config=queue_config)
    settings = SimpleNamespace(
        queue_dir=queue_dir,
        gpu_index=int(gpu.get("index", 0)),
        poll_seconds=float(wait.get("poll_seconds", 15.0)),
        log_interval_seconds=float(wait.get("log_interval_seconds", 60.0)),
        idle_samples=int(wait.get("idle_samples", 4)),
        max_gpu_utilization=float(wait.get("max_gpu_utilization", 10.0)),
        max_memory_used_mib=float(wait.get("max_memory_used_mib", 2500.0)),
        timeout_seconds=float(wait.get("timeout_seconds", 0.0)),
        extra_process_pattern=[str(value) for value in wait.get("extra_process_patterns", [])],
        gate_contract=dict(queue.get("gate_contract", {})),
        continue_on_failure_phases=tuple(
            str(value).strip()
            for value in queue.get("continue_on_failure_phases", [])
        ),
    )
    continue_on_failure = bool(queue.get("continue_on_failure", False))

    if any(
        phase not in JOB_PHASES for phase in settings.continue_on_failure_phases
    ):
        raise QueueConfigError(
            "queue.continue_on_failure_phases must contain only "
            f"{list(JOB_PHASES)}"
        )

    raw_jobs = raw.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise QueueConfigError("jobs must contain at least one job")
    jobs = []
    seen_names = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise QueueConfigError(f"jobs[{index}] must be a mapping")
        name = str(raw_job.get("name", "")).strip()
        if not name:
            raise QueueConfigError(f"jobs[{index}].name is required")
        if name in seen_names:
            raise QueueConfigError(f"Duplicate job name: {name!r}")
        seen_names.add(name)
        raw_command = raw_job.get("command")
        if not isinstance(raw_command, list) or not raw_command or not all(isinstance(value, str) for value in raw_command):
            raise QueueConfigError(f"jobs[{index}].command must be a non-empty string list")
        command = tuple(_expand(value, queue_config=queue_config) for value in raw_command)
        completion_value = raw_job.get("completion_marker")
        if not completion_value:
            raise QueueConfigError(f"jobs[{index}].completion_marker is required")
        raw_dependencies = raw_job.get("depends_on", [])
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(value, str) and value.strip()
            for value in raw_dependencies
        ):
            raise QueueConfigError(f"jobs[{index}].depends_on must be a string list")
        resume = raw_job.get("resume", {})
        resume = resume if isinstance(resume, dict) else {}
        checkpoint_value = resume.get("checkpoint")
        raw_arguments = resume.get("arguments", ["--resume-from", "{checkpoint}"])
        if not isinstance(raw_arguments, list) or not all(isinstance(value, str) for value in raw_arguments):
            raise QueueConfigError(f"jobs[{index}].resume.arguments must be a string list")
        jobs.append(
            TrainingJob(
                name=name,
                command=command,
                completion_marker=_resolve_path(completion_value, queue_config=queue_config),
                depends_on=tuple(str(value).strip() for value in raw_dependencies),
                resume_checkpoint=(
                    _resolve_path(checkpoint_value, queue_config=queue_config)
                    if checkpoint_value
                    else None
                ),
                resume_arguments=tuple(raw_arguments),
                enabled=bool(raw_job.get("enabled", True)),
                phase=str(raw_job.get("phase", "other")),
                architecture=(
                    str(raw_job["architecture"]).strip()
                    if raw_job.get("architecture")
                    else None
                ),
                gate_stage=(
                    str(raw_job["gate_stage"]).strip()
                    if raw_job.get("gate_stage")
                    else None
                ),
            )
        )
    job_indices = {job.name: index for index, job in enumerate(jobs)}
    for index, job in enumerate(jobs):
        for dependency in job.depends_on:
            if dependency not in job_indices:
                raise QueueConfigError(
                    f"Job {job.name!r} depends on unknown job {dependency!r}"
                )
            if job_indices[dependency] >= index:
                raise QueueConfigError(
                    f"Job {job.name!r} dependency {dependency!r} must appear earlier"
                )
    _validate_gate_contract(settings, jobs)
    return settings, jobs, continue_on_failure


def _validate_gate_contract(
    settings: SimpleNamespace,
    jobs: list[TrainingJob],
) -> None:
    contract = settings.gate_contract
    if not bool(contract.get("enabled", False)):
        return
    required = tuple(contract.get("required_stages", GATE_STAGES))
    if set(required) != set(GATE_STAGES) or len(required) != len(GATE_STAGES):
        raise QueueConfigError(
            "gate_contract.required_stages must contain exactly "
            f"{list(GATE_STAGES)}"
        )
    for job in jobs:
        if job.phase not in JOB_PHASES:
            raise QueueConfigError(
                f"Job {job.name!r} has unknown phase {job.phase!r}"
            )
        if job.phase in {"gate", "production_train"} and not job.architecture:
            raise QueueConfigError(
                f"Job {job.name!r} phase={job.phase} requires architecture"
            )
        if job.phase == "gate" and job.gate_stage not in required:
            raise QueueConfigError(
                f"Gate {job.name!r} must set gate_stage to one of {list(required)}"
            )
        if job.phase != "gate" and job.gate_stage is not None:
            raise QueueConfigError(
                f"Non-gate job {job.name!r} cannot set gate_stage"
            )

    active_trains = [
        (index, job)
        for index, job in enumerate(jobs)
        if job.enabled
        and job.phase == "production_train"
        and not job.completion_marker.is_file()
    ]
    if not active_trains:
        return
    active_architectures = {job.architecture for _index, job in active_trains}
    relevant_gates = [
        (index, job)
        for index, job in enumerate(jobs)
        if job.enabled
        and job.phase == "gate"
        and job.architecture in active_architectures
    ]
    if bool(contract.get("tests_first", True)):
        last_gate_index = max((index for index, _job in relevant_gates), default=-1)
        first_train_index = min(index for index, _job in active_trains)
        if last_gate_index >= first_train_index:
            raise QueueConfigError(
                "tests_first requires every active architecture gate to appear "
                "before the first incomplete production_train job"
            )

    for _index, train in active_trains:
        gates = [
            gate
            for _gate_index, gate in relevant_gates
            if gate.architecture == train.architecture
        ]
        by_stage: dict[str, list[TrainingJob]] = {
            stage: [gate for gate in gates if gate.gate_stage == stage]
            for stage in required
        }
        invalid = {
            stage: len(stage_gates)
            for stage, stage_gates in by_stage.items()
            if len(stage_gates) != 1
        }
        if invalid:
            raise QueueConfigError(
                f"Production job {train.name!r} requires exactly one enabled "
                f"gate per stage for architecture {train.architecture!r}; got {invalid}"
            )
        required_dependencies = {
            stage_gates[0].name for stage_gates in by_stage.values()
        }
        missing_dependencies = required_dependencies.difference(train.depends_on)
        if missing_dependencies:
            raise QueueConfigError(
                f"Production job {train.name!r} must depend on all four gates; "
                f"missing={sorted(missing_dependencies)}"
            )


def command_for_job(job: TrainingJob) -> list[str]:
    command = list(job.command)
    checkpoint = job.resume_checkpoint
    if (
        checkpoint is not None
        and checkpoint.is_file()
        and not job.completion_marker.is_file()
    ):
        command.extend(
            value.replace("{checkpoint}", str(checkpoint.resolve()))
            for value in job.resume_arguments
        )
    return command


def job_state(job: TrainingJob) -> str:
    if not job.enabled:
        return "disabled"
    if job.completion_marker.is_file():
        return "complete"
    if job.resume_checkpoint is not None and job.resume_checkpoint.is_file():
        return "resume_pending"
    return "pending"


def _best_checkpoint(checkpoints: list[Path]) -> Path:
    preferred_names = (
        "best_val_loss.pt",
        "best_val_loss.pth",
        "best_validation_loss.pt",
        "best_validation_loss.pth",
        "best_val_ellipse_map.pt",
        "best_loss.pt",
        "best.pt",
        "best.pth",
        "last.pt",
        "last.pth",
    )
    ranks = {name: index for index, name in enumerate(preferred_names)}
    return min(
        checkpoints,
        key=lambda path: (
            ranks.get(path.name, len(ranks) if path.name.startswith("best") else len(ranks) + 1),
            path.name,
        ),
    )


def retain_best_gate_checkpoints(job: TrainingJob) -> list[Path]:
    """After a successful overfit/small-scale gate, keep one best per model."""
    if job.phase != "gate" or job.gate_stage not in {"overfit", "small_scale"}:
        return []
    root = job.completion_marker.parent
    while root.name != job.gate_stage and root != PROJECT_ROOT:
        root = root.parent
    if root.name != job.gate_stage or not root.is_dir():
        return []
    removed: list[Path] = []
    by_directory: dict[Path, list[Path]] = {}
    for checkpoint in root.rglob("*"):
        if checkpoint.is_file() and checkpoint.suffix in {".pt", ".pth"}:
            by_directory.setdefault(checkpoint.parent, []).append(checkpoint)
    for checkpoints in by_directory.values():
        keep = _best_checkpoint(checkpoints)
        for checkpoint in checkpoints:
            if checkpoint != keep:
                checkpoint.unlink()
                removed.append(checkpoint)
    for archive in root.parent.glob(f"{root.name}.previous_*"):
        if archive.is_dir():
            shutil.rmtree(archive)
            removed.append(archive)
    return removed


def status_rows(jobs: list[TrainingJob], completed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        row = {
            "name": job.name,
            "state": job_state(job),
            "completion_marker": str(job.completion_marker),
            "depends_on": list(job.depends_on),
            "resume_checkpoint": str(job.resume_checkpoint) if job.resume_checkpoint else None,
            "phase": job.phase,
            "architecture": job.architecture,
            "gate_stage": job.gate_stage,
        }
        row.update(completed.get(job.name, {}))
        rows.append(row)
    return rows


def dry_run_payload(
    settings: SimpleNamespace,
    jobs: list[TrainingJob],
) -> dict[str, Any]:
    patterns = [re.compile(value, re.IGNORECASE) for value in settings.extra_process_pattern]
    blockers = blocking_training_processes(
        read_processes(),
        current_pid=os.getpid(),
        extra_patterns=patterns,
    )
    return {
        "state": "dry_run",
        "queue_dir": str(settings.queue_dir),
        "blockers": [asdict(process) for process in blockers],
        "gpu": query_gpu(settings.gpu_index),
        "jobs": [
            {
                "name": job.name,
                "state": job_state(job),
                "depends_on": list(job.depends_on),
                "phase": job.phase,
                "architecture": job.architecture,
                "gate_stage": job.gate_stage,
                "command": command_for_job(job),
                "command_shell": shlex.join(command_for_job(job)),
            }
            for job in jobs
        ],
    }


def run_queue(
    settings: SimpleNamespace,
    jobs: list[TrainingJob],
    *,
    wait: bool,
    continue_on_failure: bool,
) -> int:
    settings.queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = settings.queue_dir / "queue.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another queue runner already holds {lock_path}") from error

        completed: dict[str, dict[str, Any]] = {}
        continued_failures: list[str] = []
        log_path = settings.queue_dir / "queue.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_line(log_file, f"Training queue started with {len(jobs)} configured jobs.")
            for job_index, job in enumerate(jobs):
                state = job_state(job)
                if state in {"disabled", "complete"}:
                    completed[job.name] = {"state": state, "skipped": True}
                    log_line(log_file, f"SKIP {job.name}: state={state}")
                    continue
                missing_dependencies = [
                    dependency
                    for dependency in job.depends_on
                    if not next(
                        candidate.completion_marker
                        for candidate in jobs
                        if candidate.name == dependency
                    ).is_file()
                ]
                if missing_dependencies:
                    completed[job.name] = {
                        "state": "dependency_skipped",
                        "skipped": True,
                        "missing_dependencies": missing_dependencies,
                    }
                    log_line(
                        log_file,
                        f"SKIP {job.name}: missing dependency markers for "
                        f"{missing_dependencies}",
                    )
                    continue
                command = command_for_job(job)
                if wait:
                    log_line(log_file, f"WAIT {job.name}: checking training processes and GPU.")
                    wait_until_available(settings, log_file)
                write_status(
                    settings.queue_dir,
                    "running",
                    current_job=job.name,
                    current_job_index=job_index,
                    total_jobs=len(jobs),
                    command=command,
                    jobs=status_rows(jobs, completed),
                    started_at=now_iso(),
                )
                log_line(log_file, f"START {job.name}: {shlex.join(command)}")
                exit_code = stream_training(command, log_file)
                if exit_code == 0 and not job.completion_marker.is_file():
                    exit_code = 1
                    log_line(
                        log_file,
                        f"FAIL {job.name}: command exited successfully but did "
                        f"not create {job.completion_marker}",
                    )
                if exit_code == 0:
                    removed = retain_best_gate_checkpoints(job)
                    if removed:
                        log_line(
                            log_file,
                            f"RETENTION {job.name}: removed {len(removed)} redundant artifacts",
                        )
                completed[job.name] = {
                    "state": "complete" if exit_code == 0 else "failed",
                    "exit_code": exit_code,
                }
                log_line(log_file, f"END {job.name}: exit={exit_code}")
                failure_is_continuable = (
                    continue_on_failure
                    or job.phase in settings.continue_on_failure_phases
                )
                if exit_code != 0 and failure_is_continuable:
                    continued_failures.append(job.name)
                    log_line(
                        log_file,
                        f"CONTINUE after {job.name}: phase={job.phase} allows failure",
                    )
                    continue
                if exit_code != 0:
                    for remaining in jobs[job_index + 1 :]:
                        if job.name in remaining.depends_on:
                            completed[remaining.name] = {
                                "state": "dependency_skipped",
                                "skipped": True,
                                "missing_dependencies": [job.name],
                            }
                    write_status(
                        settings.queue_dir,
                        "failed",
                        failed_job=job.name,
                        exit_code=exit_code,
                        jobs=status_rows(jobs, completed),
                    )
                    return exit_code
            if continued_failures:
                write_status(
                    settings.queue_dir,
                    "complete_with_failures",
                    failed_jobs=continued_failures,
                    jobs=status_rows(jobs, completed),
                )
                log_line(
                    log_file,
                    "Training queue completed with allowed failures: "
                    f"{continued_failures}",
                )
                return 1
            write_status(
                settings.queue_dir,
                "complete",
                jobs=status_rows(jobs, completed),
            )
            log_line(log_file, "Training queue completed.")
            return 0


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def recover_stale_watcher_state(settings: SimpleNamespace) -> bool:
    """Remove a dead watcher PID and mark stale running state interrupted."""
    pid_path = settings.queue_dir / "watcher.pid"
    if not pid_path.is_file():
        return False
    try:
        existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        existing_pid = -1
    if _pid_is_alive(existing_pid):
        return False
    pid_path.unlink(missing_ok=True)
    status_path = settings.queue_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        status = {}
    if status.get("state") == "running":
        status["state"] = "interrupted"
        status["interrupted_at"] = now_iso()
        status["interruption_reason"] = "stale_watcher_pid"
        temporary = status_path.with_name(f".{status_path.name}.tmp")
        temporary.write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, status_path)
    return True


def wait_for_process_exit(pid: int, *, poll_seconds: float = 15.0) -> None:
    if pid <= 0:
        raise ValueError("--after-pid must be positive")
    if pid == os.getpid():
        raise ValueError("--after-pid cannot refer to the current process")
    interval = max(0.1, float(poll_seconds))
    while _pid_is_alive(pid):
        time.sleep(interval)


def start_detached(
    *,
    queue_config: Path,
    settings: SimpleNamespace,
    wait_override: bool | None,
    after_pid: int | None = None,
) -> dict[str, Any]:
    settings.queue_dir.mkdir(parents=True, exist_ok=True)
    pid_path = settings.queue_dir / "watcher.pid"
    recover_stale_watcher_state(settings)
    if pid_path.is_file():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if _pid_is_alive(existing_pid):
            raise RuntimeError(f"Queue watcher is already running with pid={existing_pid}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--queue-config",
        str(queue_config.resolve()),
    ]
    if wait_override is True:
        command.append("--wait")
    elif wait_override is False:
        command.append("--no-wait")
    if after_pid is not None:
        command.extend(["--after-pid", str(after_pid)])
    launcher_log_path = settings.queue_dir / "watcher.log"
    environment = dict(os.environ)
    environment["TRAINING_QUEUE_DETACHED"] = "1"
    with launcher_log_path.open("a", encoding="utf-8") as launcher_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
    temporary = settings.queue_dir / ".watcher.pid.tmp"
    temporary.write_text(f"{process.pid}\n", encoding="utf-8")
    os.replace(temporary, pid_path)
    return {
        "state": "detached",
        "pid": process.pid,
        "pid_file": str(pid_path),
        "watcher_log": str(launcher_log_path),
        "status_file": str(settings.queue_dir / "status.json"),
        "command": command,
    }


def main() -> int:
    args = parse_args()
    queue_config = args.queue_config if args.queue_config.is_absolute() else PROJECT_ROOT / args.queue_config
    settings, jobs, continue_on_failure = load_queue(queue_config)
    if args.detach:
        if args.dry_run:
            raise QueueConfigError("--detach and --dry-run cannot be used together")
        print(
            json.dumps(
                start_detached(
                    queue_config=queue_config,
                    settings=settings,
                    wait_override=args.wait,
                    after_pid=args.after_pid,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.after_pid is not None:
        wait_for_process_exit(
            args.after_pid,
            poll_seconds=settings.poll_seconds,
        )
        # The queue is intentionally appendable. Reload only after the prior
        # runner exits so jobs added while it was active are included.
        settings, jobs, continue_on_failure = load_queue(queue_config)
    if args.dry_run:
        print(json.dumps(dry_run_payload(settings, jobs), ensure_ascii=False, indent=2))
        return 0
    wait = True if args.wait is None else bool(args.wait)
    try:
        return run_queue(
            settings,
            jobs,
            wait=wait,
            continue_on_failure=continue_on_failure,
        )
    finally:
        if os.environ.get("TRAINING_QUEUE_DETACHED") == "1":
            pid_path = settings.queue_dir / "watcher.pid"
            try:
                recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError):
                recorded_pid = -1
            if recorded_pid == os.getpid():
                pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
