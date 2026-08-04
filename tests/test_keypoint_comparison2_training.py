from __future__ import annotations

import torch
from torch import nn

from keypoint_comparison2.training import _joint_validation_loss_mode


def test_joint_validation_loss_mode_enables_only_loss_branch_wrappers() -> None:
    model = nn.Module()
    model.face_detector = nn.Module()
    model.face_detector.decoder = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2, 2),
    )
    model.face_detector.backbone = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2, 2),
    )
    model.eval()

    with _joint_validation_loss_mode(model):
        assert model.training is True
        assert model.face_detector.training is True
        assert model.face_detector.decoder.training is True
        assert model.face_detector.decoder[0].training is False
        assert model.face_detector.backbone.training is False
        assert model.face_detector.backbone[0].training is False

    assert model.training is False
    assert model.face_detector.training is False
    assert model.face_detector.decoder.training is False
    assert model.face_detector.backbone.training is False


def test_joint_validation_loss_mode_restores_existing_training_flags() -> None:
    model = nn.Module()
    model.face_detector = nn.Module()
    model.face_detector.decoder = nn.Linear(2, 2)
    model.train()
    model.face_detector.eval()
    model.face_detector.decoder.train()

    with _joint_validation_loss_mode(model):
        assert model.training is True
        assert model.face_detector.training is True
        assert model.face_detector.decoder.training is True

    assert model.training is True
    assert model.face_detector.training is False
    assert model.face_detector.decoder.training is True
