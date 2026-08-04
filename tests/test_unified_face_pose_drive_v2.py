from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from mmengine.config import Config
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "outputs/dataset_cache/face_pose_current"
ADAPTER_ROOT = ROOT / "outputs/dataset_adapters/face_pose_drive_v2_balanced_811"
SOURCE_ARCHIVE = ROOT / "datasets/face_pose_drive_v2_balanced_811_training_export_20260724.zip"
RTMDET_CONFIG = ROOT / "configs/rtmdet/rtmdet_s_hf_drive_v2_960.py"
RTMPOSE_CONFIG = ROOT / "configs/rtmpose/rtmpose_s_face_pose_drive_v2_256.py"


def test_rtmdet_v2_uses_unified_image_root_and_annotation_subdirectory() -> None:
    cfg = Config.fromfile(RTMDET_CONFIG)
    assert cfg.data_root == "outputs/dataset_cache/face_pose_current/"
    assert cfg.dataset_version == "face_pose_drive_v2_balanced_811_head_face"
    assert cfg.annotation_root == "annotations/rtmdet_hf/"
    assert cfg.train_dataloader.dataset.ann_file == "annotations/rtmdet_hf/train.json"
    assert cfg.train_dataloader.dataset.data_prefix.img == ""
    assert cfg.val_evaluator.ann_file.endswith("annotations/rtmdet_hf/val.json")

    for split in ("train", "val", "test"):
        path = DATA_ROOT / cfg.annotation_root / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["images"]) == cfg.dataset_contract.images[split]
        assert len(data["annotations"]) == cfg.dataset_contract.annotations[split]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg.dataset_contract.coco_sha256[split]
        categories = {int(row["id"]): row["name"] for row in data["categories"]}
        counts = Counter(categories[int(row["category_id"])] for row in data["annotations"])
        assert dict(counts) == dict(cfg.dataset_contract.category_annotations[split])
        assert all((DATA_ROOT / image["file_name"]).is_file() for image in data["images"])


def test_rtmpose_v2_uses_the_same_unified_image_root() -> None:
    cfg = Config.fromfile(RTMPOSE_CONFIG)
    assert cfg.data_root == "outputs/dataset_cache/face_pose_current/"
    assert cfg.dataset_version == "face_pose_drive_v2_balanced_811"
    assert Path(cfg.annotation_root).resolve() == (ADAPTER_ROOT / "rtmpose").resolve()
    assert not Path(cfg.model.backbone.init_cfg.checkpoint).is_absolute()
    for split, filename in (
        ("train", "person_keypoints_train.json"),
        ("val", "person_keypoints_val.json"),
        ("test", "person_keypoints_test.json"),
    ):
        dataset = cfg[f"{split}_dataloader"].dataset
        assert dataset.data_root == cfg.data_root
        path = Path(dataset.ann_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["images"]) == cfg.dataset_contract.images[split]
        assert len(data["annotations"]) == cfg.dataset_contract.annotations[split]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            cfg.dataset_contract.coco_sha256[split]
        )
        assert all((DATA_ROOT / image["file_name"]).is_file() for image in data["images"])
        assert data["info"]["source_format"] == "canonical_cvat_metadata"
        visibility = {
            int(value)
            for annotation in data["annotations"]
            for value in annotation["keypoints"][2::3]
        }
        assert visibility <= {0, 1, 2}
        assert 1 in visibility
        assert 2 in visibility
        assert all(len(row["ellipse"]) == 5 for row in data["annotations"])


def test_deim_dino_v2_is_dedicated_and_preserves_ellipses() -> None:
    for split in ("train", "val", "test"):
        path = ADAPTER_ROOT / f"face_keypoints_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["info"]["source_format"] == "canonical_cvat_metadata"
        assert {int(row["category_id"]) for row in data["annotations"]} == {2}
        assert all(len(row["ellipse"]) == 5 for row in data["annotations"])
        assert all(
            float(row["ellipse"][2]) > 0 and float(row["ellipse"][3]) > 0
            for row in data["annotations"]
        )


def test_unified_dataset_contains_only_one_image_tree() -> None:
    image_directories = [
        path
        for path in DATA_ROOT.rglob("images")
        if path.is_dir()
    ]
    assert image_directories == [DATA_ROOT / "images"]
    assert sum(1 for path in (DATA_ROOT / "images").rglob("*") if path.is_file()) == 5295
    assert not (DATA_ROOT / "annotations" / "rtmdet_hf" / "images").exists()


def test_source_archive_is_immutable_and_adapters_are_external() -> None:
    digest = hashlib.sha256()
    with SOURCE_ARCHIVE.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == (
        "f493be94f9c30e5417ddbac505f5fc11042d7aa427ee7ce9da491991d4c06ea7"
    )
    assert DATA_ROOT.resolve() != (ROOT / "datasets/face_pose_drive_v2").resolve()
    assert ADAPTER_ROOT.is_dir()
    assert ROOT / "datasets" not in ADAPTER_ROOT.parents


def test_balanced_manifest_matches_the_fixed_split() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/datasets/default_face_pose.yaml").read_text(encoding="utf-8")
    )["default_dataset"]
    assert contract["version"] == "face_pose_drive_v2_balanced_811"
    assert contract["expected"]["images"] == {"train": 4237, "val": 530, "test": 528}
    assert contract["split_policy"] == "group-stratified-balanced-80-10-10"
