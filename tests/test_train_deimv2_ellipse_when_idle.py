from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.train_deimv2_ellipse_when_idle import (
    ProcessInfo,
    blocking_training_processes,
    is_training_command,
    training_command,
)


def test_training_command_detection_covers_current_landmark_runner() -> None:
    assert is_training_command(
        ".venv/bin/python scripts/landmarks/run_landmark_experiments.py "
        "--stage train --variants all --device cuda:0"
    )
    assert is_training_command(
        ".venv/bin/python scripts_YOLO/yolo26_pose/run_yolo26_pose.py --stage train"
    )
    assert not is_training_command(
        ".venv/bin/python tools/infer_dinov3_deimv2_ellipse.py --input sample.jpg"
    )


def test_worker_processes_are_collapsed_to_top_level_training_process() -> None:
    processes = [
        ProcessInfo(100, 1, ".venv/bin/python train_model.py"),
        ProcessInfo(101, 100, ".venv/bin/python train_model.py"),
        ProcessInfo(102, 100, ".venv/bin/python train_model.py"),
        ProcessInfo(200, 1, "python inference.py"),
    ]

    blockers = blocking_training_processes(processes, current_pid=999)

    assert blockers == [processes[0]]


def test_command_automatically_resumes_an_incomplete_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    checkpoint = output_dir / "checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = tmp_path / "config.yaml"
    config.touch()
    args = Namespace(
        config=config,
        output_dir=output_dir,
        device="cuda:0",
        wandb=False,
    )

    command = training_command(args)

    assert "--resume-from" in command
    assert str(checkpoint.resolve()) in command
