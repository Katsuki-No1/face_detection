from pathlib import Path
import hashlib

import yaml

from src.config_utils import load_yaml_config
from tools.dinov3_deimv2_common import load_config
from keypoint_comparison1.landmark_detection.config import load_landmark_config

ROOT = Path(__file__).resolve().parents[1]


def test_all_retained_configuration_inheritance_resolves():
    # In particular, ../datasets/default_face_pose.yaml must survive cleanup.
    for folder in (ROOT / "configs", ROOT / "keypoint_comparison1/configs"):
        for path in folder.rglob("*.yaml"):
            assert isinstance(load_yaml_config(path), dict), path


def test_registered_components_load_without_local_weights_or_datasets():
    registry = yaml.safe_load((ROOT / "configs/model_registry/local_baseline.yaml").read_text())
    face_asset = registry["config"]
    point_asset = registry["baseline_pipeline"]["components"]["keypoints"]["config"]
    for asset in (face_asset, point_asset):
        assert hashlib.sha256((ROOT / asset["path"]).read_bytes()).hexdigest() == asset["sha256"]
    face = load_config(ROOT / face_asset["path"])
    points = load_landmark_config(ROOT / point_asset["path"], project_root=ROOT).raw
    assert face["model"]["type"] == "dinov3_deimv2_ellipse"
    assert face["loss"]["mal_alpha"] == 2.0
    assert face["data"]["dataset_version"] == "face_pose_drive_v2_balanced_811"
    assert "query_refinement" in points["model"]["variants"]
