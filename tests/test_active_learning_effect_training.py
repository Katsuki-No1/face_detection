from __future__ import annotations

from pathlib import Path

import yaml

from keypoint_comparison1.landmark_detection.config import load_landmark_config
from scripts.run_active_learning_effect_gate import stage_commands
from scripts.run_training_queue_when_idle import load_queue
from src.config_utils import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "configs/active_learning_effect/experiment.yaml"
QUEUE_PATH = (
    PROJECT_ROOT
    / "configs/training_queues/pending_models_tests_then_training.yaml"
)


def _experiment() -> dict:
    return yaml.safe_load(EXPERIMENT_PATH.read_text(encoding="utf-8"))


def test_training_arms_share_recipes_and_use_corrected_query_state() -> None:
    experiment = _experiment()
    detector_configs = {
        arm: load_yaml_config(PROJECT_ROOT / raw["detector_config"])
        for arm, raw in experiment["arms"].items()
    }
    query_configs = {
        arm: load_landmark_config(
            PROJECT_ROOT / raw["query_config"], project_root=PROJECT_ROOT
        ).raw
        for arm, raw in experiment["arms"].items()
    }

    for arm in ("normal400", "active357"):
        detector = detector_configs[arm]
        query = query_configs[arm]
        assert detector["model"]["type"] == "dinov3_deimv2_ellipse"
        assert detector["train"]["seed"] == 0
        assert query["model"]["state_head_mode"] == (
            "factorized_presence_visibility"
        )
        assert query["train"]["seed"] == 42
        assert query["train"]["refinement_initialize_from_query"] is False
        assert query["detector"]["score_threshold"] == 0.80
        assert query["evaluation"]["default_sets"] == ["v2_val", "v2_test"]
        assert query["data"]["evaluation_sets"]["external_v1_v2"][
            "enabled"
        ] is False
        assert query["evaluation"]["baseline_aligned_output_root"].endswith(
            f"active_learning_effect/{arm}/evaluation"
        )

    assert detector_configs["normal400"]["model"] == detector_configs["active357"]["model"]
    assert detector_configs["normal400"]["loss"] == detector_configs["active357"]["loss"]
    assert query_configs["normal400"]["loss"] == query_configs["active357"]["loss"]
    assert query_configs["normal400"]["train"] == query_configs["active357"]["train"]


def test_gate_commands_cover_both_components_and_bound_small_scale_data() -> None:
    experiment = _experiment()
    for arm in ("normal400", "active357"):
        commands = stage_commands(
            stage="small_scale",
            arm_name=arm,
            arm=experiment["arms"][arm],
            latest=experiment["latest_model_contract"],
            device="cuda:0",
        )
        assert len(commands) == 2
        joined = [" ".join(command) for command in commands]
        assert "train_dinov3_deimv2.py" in joined[0]
        assert "--max-train-batches 32" in joined[0]
        assert "ellipse_mAP" in joined[0]
        assert "run_landmark_experiments.py" in joined[1]
        assert "--variants query_refinement" in joined[1]
        assert "--max-train-samples 128" in joined[1]
        assert "--max-val-samples 128" in joined[1]
        assert "val/nme" in joined[1]


def test_queue_places_all_data_effect_tests_before_production() -> None:
    _settings, jobs, _continue = load_queue(QUEUE_PATH)
    names = [job.name for job in jobs]
    first_production = min(
        index for index, job in enumerate(jobs) if job.phase == "production_train"
    )
    for arm in ("normal400", "active357"):
        prefix = f"active_learning_effect_{arm}"
        gates = [
            job
            for job in jobs
            if job.architecture == prefix and job.phase == "gate"
        ]
        assert {job.gate_stage for job in gates} == {
            "schema",
            "gpu_step",
            "overfit",
            "small_scale",
        }
        assert all(names.index(job.name) < first_production for job in gates)
        detector = jobs[names.index(f"{prefix}_detector_train")]
        query = jobs[names.index(f"{prefix}_query_refinement_train")]
        assert all(job.name in detector.depends_on for job in gates)
        assert all(job.name in query.depends_on for job in gates)
        assert detector.name in query.depends_on
        val = jobs[names.index(f"{prefix}_val_evaluate")]
        test = jobs[names.index(f"{prefix}_test_evaluate")]
        assert val.depends_on == (query.name,)
        assert test.depends_on == (val.name,)
        assert val.completion_marker.name == "success.json"
        assert test.completion_marker.name == "success.json"

    compare = jobs[names.index("active_learning_effect_compare_visualize")]
    assert compare.depends_on == (
        "active_learning_effect_normal400_test_evaluate",
        "active_learning_effect_active357_test_evaluate",
    )
