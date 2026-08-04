from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from keypoint_comparison2.training import (
    comparison_wandb_metric_row,
    comparison_wandb_run_id,
)
from scripts.run_training_queue_when_idle import load_queue
from scripts.train_head_codino import (
    head_codino_wandb_metric_row,
    head_codino_wandb_run_id,
)
from src.wandb_utils import flatten_wandb_summary, sync_evaluation_to_wandb


class _FakeArtifact:
    def __init__(
        self,
        *,
        name: str,
        type: str,
        metadata: dict[str, Any],
    ) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class _FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.finished = False

    def log_artifact(
        self,
        artifact: _FakeArtifact,
        *,
        aliases: list[str],
    ) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run


def test_pending_production_and_evaluation_jobs_require_wandb_completion() -> None:
    _settings, jobs, _continue = load_queue(
        Path("configs/training_queues/pending_models_tests_then_training.yaml")
    )
    production = [
        job for job in jobs if job.enabled and job.phase == "production_train"
    ]
    evaluation = [job for job in jobs if job.enabled and job.phase == "evaluation"]

    assert production
    assert all("--wandb" in job.command for job in production)
    wandb_evaluation = [
        job
        for job in evaluation
        if job.completion_marker.name == "wandb_evaluation_sync.json"
    ]
    assert len(wandb_evaluation) == 12
    assert all(
        job.completion_marker.name == "wandb_evaluation_sync.json"
        for job in wandb_evaluation
    )
    for architecture in {job.architecture for job in wandb_evaluation}:
        architecture_evaluations = [
            job for job in wandb_evaluation if job.architecture == architecture
        ]
        assert len(architecture_evaluations) == 2
        val_job = next(
            job for job in architecture_evaluations if "_val_" in job.name
        )
        test_job = next(
            job for job in architecture_evaluations if "_test_" in job.name
        )
        assert test_job.depends_on == (val_job.name,)


def test_head_wandb_identity_and_epoch_schema_are_resume_safe() -> None:
    config = {
        "data": {
            "image_size": 960,
            "train": {"annotation": "annotations/train.json"},
        },
        "train": {"seed": 42, "output_dir": "runs/shared-head"},
        "wandb": {"metric_schema_version": 2},
    }
    first = head_codino_wandb_run_id(
        config,
        entity="team",
        project="face_detection",
    )
    assert first == head_codino_wandb_run_id(
        config,
        entity="team",
        project="face_detection",
    )
    changed = {**config, "train": {**config["train"], "seed": 43}}
    assert first != head_codino_wandb_run_id(
        changed,
        entity="team",
        project="face_detection",
    )

    row: dict[str, int | float] = {
        "epoch": 3,
        "global_step": 120,
        "elapsed_seconds": 180.0,
        "learning_rate_min": 1.0e-6,
        "learning_rate_max": 1.0e-4,
    }
    for split, scale in (("train", 1.0), ("val", 2.0)):
        for index, loss in enumerate(
            ("loss_cls", "loss_bbox", "loss_giou", "loss_total"),
            start=1,
        ):
            row[f"{split}/{loss}"] = scale * index
    logged = head_codino_wandb_metric_row(
        row,
        previous_elapsed_seconds=60.0,
    )
    assert logged["overview/train_total_loss"] == 4.0
    assert logged["overview/val_total_loss"] == 8.0
    assert logged["loss_components/val_giou"] == 6.0
    assert logged["timing/epoch_minutes"] == 2.0


def test_joint_wandb_identity_and_epoch_schema_are_resume_safe() -> None:
    config = {
        "experiment_name": "comparison2",
        "data": {"dataset_version": "balanced-811"},
        "train": {
            "seed": 42,
            "wandb": {"metric_schema_version": 2},
        },
    }
    first = comparison_wandb_run_id(
        config,
        "deimv2_joint",
        entity="team",
        project="face_detection",
    )
    assert first == comparison_wandb_run_id(
        config,
        "deimv2_joint",
        entity="team",
        project="face_detection",
    )
    assert first != comparison_wandb_run_id(
        config,
        "dinov3_fpn_rtmcc",
        entity="team",
        project="face_detection",
    )
    logged = comparison_wandb_metric_row(
        {
            "epoch": 4,
            "updates": 20,
            "batch_size": 2,
            "effective_batch_size": 8,
            "lr": 1.0e-6,
            "maximum_lr": 1.0e-4,
            "backbone_frozen": False,
            "elapsed_seconds": 300.0,
            "loss_total": 5.0,
            "loss_bbox": 1.0,
            "face_loss_state": 0.5,
            "validation_loss": 4.0,
        },
        previous_elapsed_seconds=120.0,
    )
    assert logged["overview/train_total_loss"] == 5.0
    assert logged["overview/val_total_loss"] == 4.0
    assert logged["loss_components/train_bbox"] == 1.0
    assert logged["loss_components/train_face_loss_state"] == 0.5
    assert logged["timing/epoch_minutes"] == 3.0


def test_evaluation_sync_uploads_full_metrics_and_updates_training_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "wandb_run.json"
    state_path.write_text(
        json.dumps(
            {
                "run_id": "stable-run-id",
                "url": "https://wandb.example/run",
                "entity": "team",
                "project": "face_detection",
                "mode": "online",
            }
        ),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "evaluation" / "metrics.json"
    diagnostic.parent.mkdir()
    diagnostic.write_text(
        json.dumps(
            {
                "face": {"f1": 0.75, "counts": [1, 2, 3]},
                "unsupported": None,
            }
        ),
        encoding="utf-8",
    )
    common = tmp_path / "baseline_aligned" / "metrics.json"
    common.parent.mkdir()
    common.write_text(
        json.dumps({"head": {"AP": 0.5}, "status": "passed"}),
        encoding="utf-8",
    )
    success = common.with_name("success.json")
    success.write_text('{"status":"passed"}\n', encoding="utf-8")
    report_path = tmp_path / "wandb_evaluation_sync.json"
    fake = _FakeWandb()

    report = sync_evaluation_to_wandb(
        fake,
        state_path=state_path,
        split="test",
        metric_sources={
            "diagnostic": diagnostic,
            "baseline_aligned": common,
        },
        artifact_files=(success,),
        headline={"face_f1": 0.75},
        report_path=report_path,
    )

    assert report["status"] == "synced"
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["id"] == "stable-run-id"
    assert fake.init_kwargs["resume"] == "allow"
    assert fake.run.summary[
        "evaluation/test/baseline_aligned/head/AP"
    ] == 0.5
    assert fake.run.summary["evaluation/test/headline/face_f1"] == 0.75
    assert "evaluation/test/diagnostic/face/counts" not in fake.run.summary
    assert fake.run.finished is True
    assert len(fake.run.artifacts[0][0].files) == 3
    assert report_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["evaluation_sync"]["test"]["status"] == "synced"


def test_flatten_wandb_summary_keeps_only_finite_compact_scalars() -> None:
    flattened = flatten_wandb_summary(
        {
            "metric.value": 1.0,
            "bad": float("nan"),
            "array": [1, 2],
            "status": "complete",
        }
    )
    assert flattened == {"metric_value": 1.0, "status": "complete"}
