from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rtmdet/rtmdet_s_hf_full_960_wandb.py"


def test_rtmdet_hf_config_keeps_two_class_data_and_wandb() -> None:
    cfg = Config.fromfile(CONFIG)
    assert tuple(cfg.classes) == ("head", "face")
    assert tuple(cfg.img_scale) == (960, 960)
    assert cfg.train_cfg.max_epochs == 60
    assert cfg.train_dataloader.batch_size == 16
    assert cfg.data_root == "datasets/rtmdet_hf_drive_v1/"
    assert cfg.dataset_version == "rtmdet_hf_drive_v1"
    assert cfg.custom_hooks[0].type == "EMAHook"
    assert cfg.custom_hooks[1].type == "PipelineSwitchHook"
    assert cfg.custom_hooks[2].type == "EarlyStoppingHook"
    assert cfg.custom_hooks[2].patience == 15
    wandb = cfg.vis_backends[1].init_kwargs
    assert wandb.entity == "sato-waseda-university"
    assert wandb.project == "face_detection"
    assert wandb.group == "rtmdet_hf_production"
    assert wandb.id == "rtmdet-s-hf-drive-v1-960-low-lr-s42-20260719"
    assert wandb.config.model_family == "RTMDet"
    assert "model" not in wandb.config
    assert all(
        key not in cfg or cfg[key] == value
        for key, value in wandb.config.items()
    )
    assert wandb.config.dataset_version == "rtmdet_hf_drive_v1"
    assert cfg.max_epochs == 60
    assert cfg.optim_wrapper.type == "AmpOptimWrapper"
    assert cfg.optim_wrapper.optimizer.lr == 0.0002
    assert cfg.param_scheduler[1].begin == 0
    assert cfg.param_scheduler[1].end == 60
    assert cfg.param_scheduler[1].T_max == 60
    assert cfg.load_from.endswith("rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth")


def test_rtmdet_hf_dataset_contract_matches_local_coco_files() -> None:
    cfg = Config.fromfile(CONFIG)
    root = ROOT / cfg.data_root
    for split in ("train", "val", "test"):
        path = root / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["images"]) == cfg.dataset_contract.images[split]
        assert len(data["annotations"]) == cfg.dataset_contract.annotations[split]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg.dataset_contract.coco_sha256[split]
        categories = {int(row["id"]): row["name"] for row in data["categories"]}
        assert categories == {1: "head", 2: "face"}
        category_counts = Counter(categories[int(row["category_id"])] for row in data["annotations"])
        assert dict(category_counts) == dict(cfg.dataset_contract.category_annotations[split])
        assert all((root / image["file_name"]).is_file() for image in data["images"])
