from scripts.train_head_codino import assess_overfit_run


def test_overfit_allows_quality_focal_loss_to_rise() -> None:
    assertions = assess_overfit_run(
        {"loss_cls": 0.0001, "loss_bbox": 6.0, "loss_giou": 3.0, "loss_total": 48.0},
        {"loss_cls": 0.003, "loss_bbox": 4.0, "loss_giou": 2.0, "loss_total": 31.0},
        optimizer_steps=18,
        box_movement=0.3,
    )

    assert all(assertions.values())


def test_overfit_still_requires_finite_classification_loss() -> None:
    assertions = assess_overfit_run(
        {"loss_cls": 0.0001, "loss_bbox": 6.0, "loss_giou": 3.0, "loss_total": 48.0},
        {"loss_cls": float("nan"), "loss_bbox": 4.0, "loss_giou": 2.0, "loss_total": 31.0},
        optimizer_steps=18,
        box_movement=0.3,
    )

    assert assertions["losses_are_finite"] is False
    assert assertions["required_losses_decreased"] is False
