from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from scripts.train_dino import (
    ModelEMA,
    _validation_loss_mode,
    build_epoch_scheduler,
    save_checkpoint,
    training_wandb_run_id,
    validate_production_training_contract,
)
from src.config_utils import load_yaml_config
from src.models.dinov3_deimv2.official import _load_pretrained_checkpoint
from tools.dinov3_deimv2_common import load_checkpoint


CONFIG_PATH = Path(
    "configs/dinov3_deimv2/"
    "dinov3_deimv2_l_vits16_face_pose_drive_v1_960_full.yaml"
)
FREEZE_CONFIG_PATH = Path(
    "configs/dinov3_deimv2/"
    "dinov3_deimv2_l_vits16_face_pose_drive_v1_960_freeze.yaml"
)


class _LocalDinoWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.dino = nn.Module()
        self.backbone.dino.backbone = nn.Module()
        self.backbone.dino.backbone.proj = nn.Linear(2, 2, bias=False)


def test_official_safetensors_dinov3_prefix_is_remapped(tmp_path: Path) -> None:
    model = _LocalDinoWrapper()
    checkpoint_path = tmp_path / "official_l.safetensors"
    save_file({"backbone.dinov3.proj.weight": torch.ones(2, 2)}, checkpoint_path)

    info = _load_pretrained_checkpoint(
        model,  # type: ignore[arg-type]
        {"path": checkpoint_path, "skip_class_heads": False, "verbose": False},
    )

    assert info["loaded"] == 1
    assert info["remapped_keys"] == 1
    assert info["skipped_shape"] == 0
    assert info["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert torch.equal(model.backbone.dino.backbone.proj.weight, torch.ones(2, 2))


def test_model_ema_updates_and_round_trips() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.9, warmups=0)
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.update(model)

    assert ema.updates == 1
    assert torch.allclose(ema.module.weight, torch.full_like(ema.module.weight, 1.2))
    restored = ModelEMA(model, decay=0.9, warmups=0)
    restored.load_state_dict(ema.module.state_dict(), updates=ema.updates)
    assert restored.updates == 1
    assert torch.equal(restored.module.weight, ema.module.weight)


def test_training_checkpoint_loader_prefers_ema(tmp_path: Path) -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5, warmups=0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        step=1,
        config={},
        metrics={"loss_total": 1.0},
        ema=ema,
    )

    loaded = nn.Linear(2, 1, bias=False)
    missing, unexpected = load_checkpoint(loaded, checkpoint_path, strict=True)

    assert not missing
    assert not unexpected
    assert torch.equal(loaded.weight, ema.module.weight)


def test_deimv2_l_config_keeps_official_architecture_and_full_data() -> None:
    config = load_yaml_config(CONFIG_PATH)

    assert config["model"]["backbone"]["name"] == "dinov3_vits16"
    assert config["model"]["backbone"]["freeze"] is False
    assert config["model"]["adapter"]["interaction_indexes"] == [5, 8, 11]
    assert config["model"]["adapter"]["hidden_dim"] == 224
    assert config["model"]["decoder"]["num_layers"] == 4
    assert config["model"]["decoder"]["dim_feedforward"] == 1792
    assert config["data"]["image_size"] == 960
    assert config["data"]["dataset_version"] == "face_pose_drive_v1"
    assert config["data"]["synthetic_if_missing"] is False
    assert config["train"]["production_mode"] is True
    assert config["train"]["backbone_policy"] == "full_unfreeze"
    assert config["train"]["ema"]["enabled"] is True
    assert config["train"]["early_stopping"] == {
        "enabled": True,
        "monitor": "val/mAP50_95",
        "minimum_epoch": 68,
        "patience_validations": 6,
        "min_delta": 0.0005,
    }
    assert config["verification"]["expected_images"] == {
        "train": 2460,
        "val": 300,
        "test": 330,
    }


def test_deimv2_l_freeze_ablation_changes_only_policy_and_output() -> None:
    full = load_yaml_config(CONFIG_PATH)
    frozen = load_yaml_config(FREEZE_CONFIG_PATH)

    assert full["model"]["backbone"]["freeze"] is False
    assert frozen["model"]["backbone"]["freeze"] is True
    assert frozen["model"]["backbone"]["trainable_last_blocks"] == 0
    assert frozen["train"]["backbone_policy"] == "freeze"
    assert frozen["verification"]["backbone_policy"] == "freeze"
    for section in ("adapter", "encoder", "decoder"):
        assert frozen["model"][section] == full["model"][section]
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "seed"):
        assert frozen["train"][key] == full["train"][key]
    assert frozen["augmentation"] == full["augmentation"]


def test_production_contract_requires_ema() -> None:
    config = load_yaml_config(CONFIG_PATH)
    config["train"]["ema"]["enabled"] = False

    with pytest.raises(RuntimeError, match="requires EMA"):
        validate_production_training_contract(
            config,
            force_synthetic=False,
            max_train_batches=None,
        )


def test_flat_cosine_scheduler_has_flat_and_decay_phases() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    config = {
        "train": {
            "scheduler": "flat_cosine",
            "warmup_epochs": 0,
            "flat_epochs": 2,
            "min_lr_ratio": 0.1,
        }
    }
    scheduler = build_epoch_scheduler(optimizer, config, epochs=4)
    assert scheduler is not None
    assert scheduler.lr_lambdas[0](0) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](1) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](3) == pytest.approx(0.1)


def test_wandb_run_id_is_reproducible_and_condition_specific() -> None:
    common = {
        "entity": "sato-waseda-university",
        "project": "face_detection",
        "group": "face_pose_drive_v1_four_model_comparison",
        "run_name": "dinov3_deimv2_l_vits16_unfreeze_seed0",
        "seed": 0,
    }
    first = training_wandb_run_id(**common)
    assert first == training_wandb_run_id(**common)
    assert len(first) == 16
    assert first != training_wandb_run_id(**{**common, "seed": 1})
    assert first != training_wandb_run_id(**{**common, "run_name": "yolo26s_pose_full-seed0"})


def test_validation_loss_mode_only_enables_wrapper_and_decoder() -> None:
    model = nn.Module()
    model.decoder = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    model.backbone = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    model.eval()

    with _validation_loss_mode(model):
        assert model.training is True
        assert model.decoder.training is True
        assert model.decoder[0].training is False
        assert model.backbone.training is False
        assert model.backbone[0].training is False

    assert model.training is False
    assert model.decoder.training is False
    assert model.backbone.training is False
