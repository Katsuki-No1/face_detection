from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_training_queue_when_idle as queue_runner
from scripts.run_training_queue_when_idle import (
    QueueConfigError,
    command_for_job,
    job_state,
    load_queue,
    run_queue,
    retain_best_gate_checkpoints,
    recover_stale_watcher_state,
    start_detached,
)


def test_gate_retention_keeps_one_best_checkpoint_per_model(tmp_path: Path) -> None:
    root = tmp_path / "training_gates" / "model" / "small_scale"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name in ("epoch_001.pt", "last.pt", "best_val_loss.pt"):
        (checkpoints / name).write_bytes(name.encode())
    marker = root / "success.json"
    marker.write_text("{}", encoding="utf-8")
    job = queue_runner.TrainingJob(
        name="gate",
        command=("true",),
        completion_marker=marker,
        depends_on=(),
        resume_checkpoint=None,
        resume_arguments=(),
        phase="gate",
        architecture="model",
        gate_stage="small_scale",
    )

    retain_best_gate_checkpoints(job)

    assert [path.name for path in checkpoints.iterdir()] == ["best_val_loss.pt"]


def test_recover_stale_watcher_marks_running_status_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SimpleNamespace(queue_dir=tmp_path)
    (tmp_path / "watcher.pid").write_text("999999\n", encoding="utf-8")
    (tmp_path / "status.json").write_text('{"state":"running"}\n', encoding="utf-8")
    monkeypatch.setattr(queue_runner, "_pid_is_alive", lambda _pid: False)

    assert recover_stale_watcher_state(settings) is True
    assert not (tmp_path / "watcher.pid").exists()
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted"


def write_queue(path: Path, *, duplicate_name: bool = False) -> None:
    second_name = "first" if duplicate_name else "second"
    path.write_text(
        f"""
version: 1
queue:
  state_dir: state
  gpu: {{index: 0}}
  wait: {{idle_samples: 2}}
jobs:
  - name: first
    command: ["{{project_root}}/.venv/bin/python", "train.py"]
    completion_marker: "{{queue_config_dir}}/out/first/summary.md"
    resume:
      checkpoint: "{{queue_config_dir}}/out/first/checkpoints/last.pt"
      arguments: [--resume-from, "{{checkpoint}}"]
  - name: {second_name}
    enabled: false
    command: ["echo", "second"]
    completion_marker: "{{queue_config_dir}}/out/second/done"
""",
        encoding="utf-8",
    )


def test_queue_config_is_appendable_and_preserves_job_order(tmp_path: Path) -> None:
    path = tmp_path / "queue.yaml"
    write_queue(path)

    settings, jobs, continue_on_failure = load_queue(path)

    assert settings.idle_samples == 2
    assert [job.name for job in jobs] == ["first", "second"]
    assert jobs[1].enabled is False
    assert continue_on_failure is False


def test_production_queue_retrains_detector_before_keypoint_jobs() -> None:
    queue_path = Path("configs/training_queues/after_current_models.yaml")

    _settings, jobs, _continue = load_queue(queue_path)
    names = [job.name for job in jobs]

    assert names[:4] == [
        "dinov3_deimv2_l_face_pose_drive_v2_balanced_811_ellipse_from_coco",
        "calibrate_ellipse_deim_crop_residuals",
        "keypoint_comparison2_extract_balanced_811_dinov3_backbone",
        "keypoint_comparison2_preflight",
    ]
    assert "balanced_811_960_ellipse_from_coco.yaml" in " ".join(jobs[0].command)
    assert str(jobs[1].completion_marker).endswith(
        "ellipse_deim_val_residuals.json"
    )
    assert all(
        names.index(keypoint_job) > names.index("keypoint_comparison2_preflight")
        for keypoint_job in (
            "keypoint_comparison2_rtmpose_face5",
            "keypoint_comparison2_dinov3_fpn_rtmcc",
            "keypoint_comparison2_dinov3_orformer_rtmcc",
            "keypoint_comparison2_deimv2_joint",
        )
    )
    assert names[-3:] == [
        "keypoint_comparison1_vitpose_376",
        "keypoint_comparison1_landmark_query",
        "keypoint_comparison1_vitpose_376_evaluate",
    ]
    assert "--variants vitpose_376" in " ".join(jobs[-3].command)
    assert "--variants query" in " ".join(jobs[-2].command)
    vitpose_config = jobs[-3].command[jobs[-3].command.index("--config") + 1]
    query_config = jobs[-2].command[jobs[-2].command.index("--config") + 1]
    assert query_config == vitpose_config
    assert "--variants query vitpose_376" in " ".join(jobs[-1].command)
    assert "G_vitpose_376_deim" in str(jobs[-1].completion_marker)


def test_cooperative_queue_waits_and_runs_safety_gates_before_training() -> None:
    queue_path = Path(
        "configs/training_queues/face_keypoint_cooperative_after_current.yaml"
    )

    settings, jobs, continue_on_failure = load_queue(queue_path)
    names = [job.name for job in jobs]

    assert names == [
        "keypoint_comparison1_landmark_query",
        "keypoint_comparison1_query_vs_vitpose_376_evaluate",
        "keypoint_comparison1_query_refinement_gt_ellipse_long_axis_square",
        "face_keypoint_cooperative_960_audit",
        "face_keypoint_cooperative_real_data_overfit",
        "face_keypoint_cooperative_detr_l_960_train",
        "yolo26m_pose_balanced_811_960_train",
        "face_keypoint_cooperative_detr_l_960_val_evaluate",
        "face_keypoint_cooperative_detr_l_960_evaluate",
        "keypoint_comparison1_factorized_state_preflight",
        "keypoint_comparison1_factorized_state_query_train",
        "keypoint_comparison1_factorized_state_query_refinement_train",
        "keypoint_comparison1_factorized_state_query_refinement_evaluate",
        "yolo26m_pose_balanced_811_960_evaluate",
    ]
    assert settings.extra_process_pattern == [
        r"run_training_queue_when_idle\.py.*after_current_models\.yaml"
    ]
    assert continue_on_failure is False
    assert jobs[0].enabled is False
    assert jobs[1].enabled is False
    assert jobs[2].enabled is False
    assert "--variants" in jobs[2].command
    assert jobs[2].command[jobs[2].command.index("--variants") + 1] == (
        "query_refinement"
    )
    assert "--overfit-smoke" in jobs[4].command
    assert jobs[5].resume_checkpoint is not None
    assert jobs[5].completion_marker.name == "training_complete.json"
    assert "--wandb" in jobs[5].command
    assert "--models" in jobs[6].command
    assert jobs[6].command[jobs[6].command.index("--models") + 1] == "m"
    assert jobs[6].resume_arguments == ("--resume",)
    assert jobs[7].completion_marker.name == "success.json"
    assert jobs[8].completion_marker.name == "success.json"
    assert jobs[8].depends_on == (
        "face_keypoint_cooperative_detr_l_960_val_evaluate",
    )
    cooperative_evaluation = names.index(
        "face_keypoint_cooperative_detr_l_960_evaluate"
    )
    corrected_query = names.index("keypoint_comparison1_factorized_state_query_train")
    corrected_refinement = names.index(
        "keypoint_comparison1_factorized_state_query_refinement_train"
    )
    corrected_evaluation = names.index(
        "keypoint_comparison1_factorized_state_query_refinement_evaluate"
    )
    preflight = names.index(
        "keypoint_comparison1_factorized_state_preflight"
    )
    yolo_evaluation = names.index(
        "yolo26m_pose_balanced_811_960_evaluate"
    )
    assert (
        cooperative_evaluation
        < preflight
        < corrected_query
        < corrected_refinement
        < corrected_evaluation
        < yolo_evaluation
    )
    assert all(
        "coco_visibility" in str(jobs[index].completion_marker)
        for index in (
            preflight,
            corrected_query,
            corrected_refinement,
            corrected_evaluation,
        )
    )
    assert jobs[corrected_query].enabled is False
    assert jobs[corrected_refinement].depends_on == (
        "keypoint_comparison1_factorized_state_preflight",
    )
    assert jobs[corrected_evaluation].depends_on == (
        "keypoint_comparison1_factorized_state_query_refinement_train",
    )
    assert "--resume" in jobs[corrected_query].command
    assert "--resume" in jobs[corrected_refinement].command
    assert jobs[yolo_evaluation].completion_marker.name == "success.json"


def test_three_stage_queue_freezes_val_head_threshold_before_both_arms() -> None:
    queue_path = Path(
        "configs/training_queues/"
        "three_stage_codino_deimv2_landmark_query_after_current.yaml"
    )

    settings, jobs, continue_on_failure = load_queue(queue_path)
    names = [job.name for job in jobs]

    assert settings.extra_process_pattern == [
        (
            r"run_training_queue_when_idle\.py.*"
            r"face_keypoint_cooperative_after_current\.yaml"
        )
    ]
    assert continue_on_failure is False
    threshold_index = names.index(
        "head_codino_balanced_811_select_val_threshold"
    )
    cache_indices = [
        names.index(f"three_stage_build_{split}_head_cache")
        for split in ("train", "val", "test")
    ]
    assert all(index > threshold_index for index in cache_indices)
    for index in cache_indices:
        assert "--threshold-file" in jobs[index].command

    baseline_train = names.index("three_stage_baseline_train")
    hpg_train = names.index("three_stage_hpg_train")
    head_overfit = jobs[names.index("head_codino_balanced_811_real_data_overfit")]
    head_train = jobs[names.index("head_codino_balanced_811_train")]
    assert baseline_train > max(cache_indices)
    assert hpg_train > max(cache_indices)
    assert head_overfit.completion_marker.name == "overfit_success.json"
    assert head_train.depends_on == (
        "head_codino_balanced_811_preflight",
        "head_codino_balanced_811_real_data_overfit",
    )
    assert jobs[names.index("three_stage_baseline_audit")].completion_marker.name == (
        "audit_success.json"
    )
    assert jobs[names.index("three_stage_baseline_real_data_overfit")].completion_marker.name == (
        "overfit_success.json"
    )
    assert jobs[baseline_train].depends_on == (
        "three_stage_baseline_audit",
        "three_stage_baseline_real_data_overfit",
    )
    assert jobs[hpg_train].depends_on == (
        "three_stage_hpg_audit",
        "three_stage_hpg_real_data_overfit",
    )
    assert jobs[baseline_train].resume_checkpoint is not None
    assert jobs[hpg_train].resume_checkpoint is not None
    for arm in ("baseline", "hpg"):
        val_evaluation = jobs[
            names.index(f"three_stage_{arm}_val_evaluate")
        ]
        test_evaluation = jobs[
            names.index(f"three_stage_{arm}_test_evaluate")
        ]
        assert val_evaluation.completion_marker.name == "success.json"
        assert test_evaluation.completion_marker.name == "success.json"
        assert test_evaluation.depends_on == (
            f"three_stage_{arm}_val_evaluate",
        )
    original_comparison = names.index("three_stage_hpg_comparison")
    border_crop_jobs = [
        "three_stage_ellipse_border_crop_audit",
        "three_stage_ellipse_border_crop_real_data_overfit",
        "three_stage_ellipse_border_crop_train",
        "three_stage_ellipse_border_crop_val_evaluate",
        "three_stage_ellipse_border_crop_test_evaluate",
        "three_stage_ellipse_border_crop_comparison",
    ]
    assert names[-len(border_crop_jobs) :] == border_crop_jobs
    assert original_comparison < names.index(border_crop_jobs[0])
    border_crop_train = jobs[names.index("three_stage_ellipse_border_crop_train")]
    assert border_crop_train.depends_on == (
        "three_stage_ellipse_border_crop_audit",
        "three_stage_ellipse_border_crop_real_data_overfit",
    )
    assert jobs[names.index(border_crop_jobs[0])].completion_marker.name == (
        "audit_success.json"
    )
    assert jobs[names.index(border_crop_jobs[1])].completion_marker.name == (
        "overfit_success.json"
    )
    assert jobs[names.index(border_crop_jobs[3])].completion_marker.name == (
        "success.json"
    )
    assert jobs[names.index(border_crop_jobs[4])].completion_marker.name == (
        "success.json"
    )
    assert jobs[-1].completion_marker.name == "comparison.md"


def test_queue_job_adds_resume_arguments_only_for_incomplete_run(tmp_path: Path) -> None:
    path = tmp_path / "queue.yaml"
    write_queue(path)
    _settings, jobs, _continue = load_queue(path)
    first = jobs[0]
    assert first.resume_checkpoint is not None
    first.resume_checkpoint.parent.mkdir(parents=True)
    first.resume_checkpoint.touch()

    command = command_for_job(first)

    assert command[-2:] == ["--resume-from", str(first.resume_checkpoint.resolve())]
    assert job_state(first) == "resume_pending"
    first.completion_marker.parent.mkdir(parents=True, exist_ok=True)
    first.completion_marker.touch()
    assert job_state(first) == "complete"
    assert command_for_job(first)[-2:] != ["--resume-from", str(first.resume_checkpoint.resolve())]


def test_duplicate_job_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queue.yaml"
    write_queue(path, duplicate_name=True)

    with pytest.raises(QueueConfigError, match="Duplicate job name"):
        load_queue(path)


def test_queue_dependencies_must_reference_earlier_jobs(tmp_path: Path) -> None:
    path = tmp_path / "queue.yaml"
    path.write_text(
        """
version: 1
queue:
  state_dir: state
jobs:
  - name: production
    depends_on: [missing_gate]
    command: ["echo", "train"]
    completion_marker: production.done
""",
        encoding="utf-8",
    )

    with pytest.raises(QueueConfigError, match="unknown job"):
        load_queue(path)


def test_pending_models_queue_runs_all_gates_before_all_training() -> None:
    path = Path(
        "configs/training_queues/pending_models_tests_then_training.yaml"
    )
    settings, jobs, continue_on_failure = load_queue(path)
    assert queue_runner.DEFAULT_QUEUE_CONFIG.resolve() == path.resolve()
    assert settings.gate_contract["enabled"] is True
    assert continue_on_failure is False
    assert settings.continue_on_failure_phases == ("evaluation", "postprocess")
    train_indices = [
        index for index, job in enumerate(jobs)
        if job.phase == "production_train"
    ]
    gate_indices = [
        index for index, job in enumerate(jobs) if job.phase == "gate"
    ]
    assert len(train_indices) == 13
    assert len(gate_indices) == 44
    assert max(gate_indices) < min(train_indices)
    for train in (job for job in jobs if job.phase == "production_train"):
        stages = {
            gate.gate_stage
            for gate in jobs
            if gate.name in train.depends_on and gate.phase == "gate"
        }
        assert stages == {"schema", "gpu_step", "overfit", "small_scale"}

    by_name = {job.name: job for job in jobs}
    head_small = by_name["shared_head_codino_unfrozen_small_scale_gate"]
    face_gpu = by_name["three_stage_face_stage_gpu_step_gate"]
    face_overfit = by_name["three_stage_face_stage_overfit_gate"]
    face_small = by_name["three_stage_face_stage_small_scale_gate"]
    keypoint_gpu = by_name["three_stage_keypoint_stage_gpu_step_gate"]
    keypoint_overfit = by_name["three_stage_keypoint_stage_overfit_gate"]
    keypoint_small = by_name["three_stage_keypoint_stage_small_scale_gate"]
    head_train = by_name["shared_head_codino_unfrozen_train"]
    face_train = by_name["three_stage_face_stage_train"]
    keypoint_train = by_name["three_stage_keypoint_stage_train"]

    head_improvement = head_small.command.index("--minimum-relative-improvement")
    assert head_small.command[head_improvement + 1] == "0.003"

    assert head_small.name in face_gpu.depends_on
    for job in (face_gpu, face_overfit, face_small):
        assert "--initialize-shared-head-from" in job.command
        checkpoint = job.command[
            job.command.index("--initialize-shared-head-from") + 1
        ]
        assert checkpoint.endswith(
            "shared_head_codino_vits16plus_960_unfrozen/"
            "small_scale/best_val_loss.pth"
        )
        assert "--initialize-shared-dino-from-head-checkpoint" in job.command

    assert face_small.name in keypoint_gpu.depends_on
    for job in (keypoint_gpu, keypoint_overfit, keypoint_small):
        option = "--checkpoint" if job is keypoint_gpu else "--initialize-from"
        checkpoint = job.command[job.command.index(option) + 1]
        assert checkpoint.endswith(
            "three_stage_face_stage/small_scale/checkpoints/best_val_loss.pt"
        )

    assert head_train.name in face_train.depends_on
    assert "--initialize-shared-dino-from-head-checkpoint" in face_train.command
    face_head_checkpoint = face_train.command[
        face_train.command.index("--initialize-shared-head-from") + 1
    ]
    assert face_head_checkpoint.endswith(
        "shared_head_codino_vits16plus_960_balanced_811_unfrozen/"
        "best_val_loss.pth"
    )
    assert face_train.name in keypoint_train.depends_on
    keypoint_face_checkpoint = keypoint_train.command[
        keypoint_train.command.index("--initialize-from") + 1
    ]
    assert keypoint_face_checkpoint.endswith(
        "codino_deimv2_l_face_stage_seed42/checkpoints/best_val_loss.pt"
    )
    assert jobs.index(head_train) < jobs.index(face_train) < jobs.index(keypoint_train)

    strict_three_stage_configs = {
        "three_stage_baseline": (
            "codino_deimv2_l_landmark_query_baseline.yaml"
        ),
        "three_stage_hpg": "codino_deimv2_l_landmark_query_hpg.yaml",
        "three_stage_ellipse_border_crop": (
            "codino_deimv2_l_landmark_query_ellipse_border_crop.yaml"
        ),
    }
    for architecture, config_name in strict_three_stage_configs.items():
        architecture_jobs = [
            job for job in jobs if job.architecture == architecture
        ]
        schema = next(
            job for job in architecture_jobs if job.gate_stage == "schema"
        )
        gpu_step = next(
            job for job in architecture_jobs if job.gate_stage == "gpu_step"
        )
        assert Path(schema.command[schema.command.index("--config") + 1]).name == (
            config_name
        )
        assert Path(
            gpu_step.command[gpu_step.command.index("--config") + 1]
        ).name == config_name
        production = next(
            job for job in architecture_jobs if job.phase == "production_train"
        )
        assert {
            job.gate_stage
            for job in architecture_jobs
            if job.name in production.depends_on and job.phase == "gate"
        } == {"schema", "gpu_step", "overfit", "small_scale"}

    deim_overfit = next(job for job in jobs if job.name == "deimv2_joint_overfit_gate")
    epoch_index = deim_overfit.command.index("--epochs")
    assert deim_overfit.command[epoch_index + 1] == "12"
    freeze_index = deim_overfit.command.index("--freeze-backbone-epochs")
    assert deim_overfit.command[freeze_index + 1] == "12"
    deim_small_scale = next(job for job in jobs if job.name == "deimv2_joint_small_scale_gate")
    small_scale_epoch_index = deim_small_scale.command.index("--epochs")
    assert deim_small_scale.command[small_scale_epoch_index + 1] == "12"
    improvement_index = deim_small_scale.command.index("--minimum-relative-improvement")
    assert deim_small_scale.command[improvement_index + 1] == "0.003"
    transition_index = deim_small_scale.command.index("--minimum-improving-transition-fraction")
    assert deim_small_scale.command[transition_index + 1] == "0.4"

    active_learning_jobs = jobs[-3:]
    expected_datasets = (
        "active_learning08-02-4-4",
        "active_learning08-02-5-4",
        "active_learning08-02-6-4",
    )
    assert [job.name for job in active_learning_jobs] == [
        "active_learning08_02_4_4_infer_extract",
        "active_learning08_02_5_4_infer_extract",
        "active_learning08_02_6_4_infer_extract",
    ]
    for job, dataset_name in zip(active_learning_jobs, expected_datasets):
        assert job.phase == "postprocess"
        assert job.depends_on == ()
        assert job.command[job.command.index("--dataset-name") + 1] == dataset_name
        assert job.command[job.command.index("--stage") + 1] == "all"
        assert job.completion_marker == (
            Path("data/output/active_learning") / dataset_name / "summary.json"
        ).resolve()
    assert "--allow-known-videos" in active_learning_jobs[0].command
    assert all(
        "--allow-known-videos" not in job.command
        for job in active_learning_jobs[1:]
    )


def test_gate_contract_rejects_production_without_all_four_gates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.yaml"
    path.write_text(
        """
version: 1
queue:
  state_dir: state
  gate_contract: {enabled: true, tests_first: true}
jobs:
  - name: schema
    phase: gate
    architecture: model_a
    gate_stage: schema
    command: ["schema"]
    completion_marker: schema.done
  - name: production
    phase: production_train
    architecture: model_a
    depends_on: [schema]
    command: ["train"]
    completion_marker: training_complete.json
""",
        encoding="utf-8",
    )

    with pytest.raises(QueueConfigError, match="exactly one enabled gate"):
        load_queue(path)


def test_gate_contract_rejects_gate_appended_after_production(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.yaml"
    path.write_text(
        """
version: 1
queue:
  state_dir: state
  gate_contract: {enabled: true, tests_first: true}
jobs:
  - name: production
    phase: production_train
    architecture: model_a
    depends_on: [schema, gpu, overfit, small]
    command: ["train"]
    completion_marker: training_complete.json
  - {name: schema, phase: gate, architecture: model_a, gate_stage: schema, command: ["schema"], completion_marker: schema.done}
  - {name: gpu, phase: gate, architecture: model_a, gate_stage: gpu_step, command: ["gpu"], completion_marker: gpu.done}
  - {name: overfit, phase: gate, architecture: model_a, gate_stage: overfit, command: ["overfit"], completion_marker: overfit.done}
  - {name: small, phase: gate, architecture: model_a, gate_stage: small_scale, command: ["small"], completion_marker: small.done}
""",
        encoding="utf-8",
    )

    with pytest.raises(QueueConfigError, match="must appear earlier"):
        load_queue(path)


def test_failed_gate_records_production_dependency_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queue.yaml"
    path.write_text(
        """
version: 1
queue:
  state_dir: state
jobs:
  - name: gate
    command: ["gate"]
    completion_marker: gate_success.json
  - name: production
    depends_on: [gate]
    command: ["train"]
    completion_marker: training_complete.json
""",
        encoding="utf-8",
    )
    settings, jobs, continue_on_failure = load_queue(path)
    monkeypatch.setattr(
        "scripts.run_training_queue_when_idle.stream_training",
        lambda _command, _log_file: 1,
    )

    exit_code = run_queue(
        settings,
        jobs,
        wait=False,
        continue_on_failure=continue_on_failure,
    )

    assert exit_code == 1
    status = json.loads((settings.queue_dir / "status.json").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in status["jobs"]}
    assert rows["gate"]["state"] == "failed"
    assert rows["production"]["state"] == "dependency_skipped"
    assert rows["production"]["missing_dependencies"] == ["gate"]


def test_evaluation_failure_continues_to_independent_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queue.yaml"
    path.write_text(
        """
version: 1
queue:
  state_dir: state
  continue_on_failure_phases: [evaluation, postprocess]
jobs:
  - name: failed_evaluation
    phase: evaluation
    command: ["fail-evaluation"]
    completion_marker: "{queue_config_dir}/failed.done"
  - name: dependent_test
    phase: evaluation
    depends_on: [failed_evaluation]
    command: ["must-not-run"]
    completion_marker: "{queue_config_dir}/dependent.done"
  - name: independent_inference
    phase: evaluation
    command: ["run-inference"]
    completion_marker: "{queue_config_dir}/independent.done"
""",
        encoding="utf-8",
    )
    settings, jobs, continue_on_failure = load_queue(path)
    commands: list[str] = []

    def fake_stream(command: list[str], _log_file: object) -> int:
        commands.append(command[0])
        if command[0] == "fail-evaluation":
            return 1
        (tmp_path / "independent.done").touch()
        return 0

    monkeypatch.setattr(
        "scripts.run_training_queue_when_idle.stream_training",
        fake_stream,
    )

    exit_code = run_queue(
        settings,
        jobs,
        wait=False,
        continue_on_failure=continue_on_failure,
    )

    assert exit_code == 1
    assert commands == ["fail-evaluation", "run-inference"]
    status = json.loads(
        (settings.queue_dir / "status.json").read_text(encoding="utf-8")
    )
    rows = {row["name"]: row for row in status["jobs"]}
    assert status["state"] == "complete_with_failures"
    assert status["failed_jobs"] == ["failed_evaluation"]
    assert rows["failed_evaluation"]["state"] == "failed"
    assert rows["dependent_test"]["state"] == "dependency_skipped"
    assert rows["independent_inference"]["state"] == "complete"


def test_detached_start_records_pid_without_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_config = tmp_path / "queue.yaml"
    queue_config.touch()
    settings = SimpleNamespace(queue_dir=tmp_path / "state")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 43210

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "scripts.run_training_queue_when_idle.subprocess.Popen",
        fake_popen,
    )

    payload = start_detached(
        queue_config=queue_config,
        settings=settings,
        wait_override=None,
        after_pid=12345,
    )

    assert payload["pid"] == 43210
    assert (settings.queue_dir / "watcher.pid").read_text(encoding="utf-8") == "43210\n"
    assert captured["command"] == [
        sys.executable,
        str(Path(queue_runner.__file__).resolve()),
        "--queue-config",
        str(queue_config.resolve()),
        "--after-pid",
        "12345",
    ]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is not None
