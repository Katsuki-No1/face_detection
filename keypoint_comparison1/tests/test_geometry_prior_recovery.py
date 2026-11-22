from __future__ import annotations

import copy

import pytest
import torch

from keypoint_comparison1.landmark_detection.geometry_prior_recovery import (
    ConditionalFeatureDiscriminator, GeometryPriorBank, attach_geometry_prior,
    connected_occlusion, discriminator_loss, feature_similarity, generator_losses,
    geometry_embedding, research_train_step, backbone_state_sha256,
)
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.tests.test_continuous_recovery import config, targets


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metadata():
    return dict(split="train", annotation_sha256="a" * 64, backbone_sha256="b" * 64, backbone_state_sha256="d" * 64,
                detector_sha256="c" * 64, crop_convention="test aligned crop x-right/y-down",
                clean_eligibility="synthetic fixture only", fit_video_ids=["train-a"],
                calibration_video_ids=["train-b"], seed=42)


def fit_bank(k=1, channels=24):
    torch.manual_seed(81)
    fit, calibration, geometry, calibration_geometry = [], [], [], []
    for j in range(k):
        center = torch.eye(channels)[j, :, None, None].expand(channels, 4, 4)
        for count, xs, gs in ((6, fit, geometry), (4, calibration, calibration_geometry)):
            xs.extend(center + .001 * torch.randn(count, channels, 4, 4))
            gs.extend([[.1 + j * .1, 1 + j * .3, j * .1]] * count)
    return GeometryPriorBank.fit(torch.stack(fit), torch.tensor(geometry), torch.stack(calibration),
        torch.tensor(calibration_geometry), k=k, metadata=metadata())


def model_case(anomalous=True):
    torch.manual_seed(48)
    base, _ = build_landmark_model(config(adoption="optimal"), variant="query_refinement", initialize_from_detector=False)
    images, truth = torch.randn(2, 3, 64, 64), targets()
    criterion = LandmarkLoss()
    optimizer = torch.optim.Adam(base.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        criterion(base(images, recovery_mode="raw"), truth)["loss_total"].backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    bank = fit_bank()
    bank.metadata["backbone_state_sha256"] = backbone_state_sha256(base.backbone)
    # Deterministic masks for gradient tests, not a calibrated performance result.
    bank.thresholds.fill_(1 if anomalous else -1)
    model = attach_geometry_prior(base, bank, enabled=True, observation_backbone_sha256="b" * 64)
    geometry = torch.tensor([[.1, 1., 0.], [.1, 1., 0.]])
    support = torch.ones(2, 4, 4, dtype=torch.bool)
    support[:, 0, 0] = False
    return model, images, geometry, support, truth, criterion


@pytest.mark.parametrize("k", range(1, 8))
def test_fit_k1_to_7_is_repeatable_and_serializable(k):
    bank = fit_bank(k)
    again = fit_bank(k)
    torch.testing.assert_close(bank.centers, again.centers, rtol=0, atol=0)
    assert bank.centers.shape == (k, 24, 4, 4)
    assert bank.calibration_counts.tolist() == [4] * k
    restored = GeometryPriorBank(bank.payload())
    restored.load_state_dict(bank.state_dict(), strict=True)
    torch.testing.assert_close(restored.thresholds, bank.thresholds)


def test_geometry_periodicity_and_circle_angle_invariance():
    g = torch.tensor([[.2, 3., .35], [.1, 1., 1.]])
    other = g.clone()
    other[0, 2] += torch.pi
    other[1, 2] += .75
    torch.testing.assert_close(geometry_embedding(g), geometry_embedding(other))
    for bad in (torch.tensor([[0., 2., 0.]]), torch.tensor([[.1, .5, 0.]])):
        with pytest.raises(ValueError):
            geometry_embedding(bad)


@pytest.mark.parametrize("change", [dict(split="test"), dict(fit_video_ids=["train-b"]), dict(backbone_sha256="invalid")])
def test_bank_rejects_leakage_and_missing_authentication(change):
    payload = fit_bank().payload()
    payload["metadata"].update(change)
    with pytest.raises(ValueError):
        GeometryPriorBank(payload)


def test_nonfinite_and_collapsed_clusters_fail_closed():
    g = torch.tensor([[.2, 2., 0.]] * 8)
    for x in (torch.ones(8, 24, 4, 4), torch.full((8, 24, 4, 4), float("nan"))):
        with pytest.raises(ValueError):
            GeometryPriorBank.fit(x, g, x, g, k=2, metadata=metadata())


def test_identical_features_and_constant_channels_have_finite_scores():
    x = torch.randn(2, 24, 4, 4)
    torch.testing.assert_close(feature_similarity(x, x, torch.ones(24)), torch.ones(2, 4, 4))
    assert torch.isfinite(feature_similarity(torch.ones_like(x), torch.ones_like(x), torch.ones(24))).all()


def test_connected_area_boundary_and_small_local_mask():
    support = torch.ones(2, 10, 10, dtype=torch.bool)
    anomaly = torch.zeros_like(support)
    anomaly[0, 0, :5] = True
    anomaly[1, 0, :4] = True
    result = connected_occlusion(anomaly, support, .05)
    assert result["face_occluded"].tolist() == [True, False]
    assert result["large_component_mask"].sum().item() == 5
    assert anomaly[1].sum() == 4  # face decision must not discard the local anomaly mask
    # Diagonal cells do not form a four-connected component.
    diagonal = torch.eye(10, dtype=torch.bool)[None]
    assert not connected_occlusion(diagonal, support[:1], .05)["face_occluded"].item()


def test_disabled_is_same_model_and_checkpoint():
    base, _ = build_landmark_model(config(), variant="query_refinement", initialize_from_detector=False)
    state = set(base.state_dict())
    assert attach_geometry_prior(base, enabled=False) is base
    assert set(base.state_dict()) == state


def test_visible_features_are_exactly_preserved_and_observer_stays_frozen():
    model, images, geometry, support, _, _ = model_case()
    model.train()
    assert not model.observer.training
    out = model(images, geometry=geometry, support=support)
    d = out["prior_diagnostics"]
    torch.testing.assert_close(d["fused"][:, :, 0, 0], d["original"][:, :, 0, 0], rtol=0, atol=0)
    assert not d["anomalous"].requires_grad and not d["observation"].requires_grad
    assert all(not p.requires_grad for p in model.observer.parameters())
    assert d["anomalous"].sum() == 30


def nonzero_grad(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return bool(grads) and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


def test_adversarial_and_keypoint_gradient_ownership():
    model, images, geometry, support, truth, criterion = model_case()
    discriminator = ConditionalFeatureDiscriminator(24, hidden=8)
    out = model(images, geometry=geometry, support=support)
    clean = torch.randn_like(out["prior_diagnostics"]["fused"], requires_grad=True)
    discriminator_loss(discriminator, out, clean, geometry).backward()
    assert nonzero_grad(discriminator)
    assert all(p.grad is None for p in model.parameters())
    assert clean.grad is None
    discriminator.zero_grad(set_to_none=True)
    losses = generator_losses(out, clean, criterion(out, truth)["loss_total"],
        discriminator=discriminator, geometry=geometry, adversarial_weight=.01)
    losses["total"].backward()
    assert nonzero_grad(model.base.backbone)
    assert nonzero_grad(model.base.messenger_recovery.decoder)
    assert nonzero_grad(model.base.head)
    assert model.channel_adoption.grad.abs().sum() > 0
    assert all(p.grad is None for p in discriminator.parameters())
    assert clean.grad is None
    # Task-only loss must reach G and fusion, independent of reconstruction/GAN.
    model.zero_grad(set_to_none=True)
    out = model(images, geometry=geometry, support=support)
    criterion(out, truth)["loss_coordinates"].backward()
    assert nonzero_grad(model.base.messenger_recovery.decoder)
    assert model.channel_adoption.grad.abs().sum() > 0


@pytest.mark.parametrize("adversarial", [False, True])
def test_repeated_research_updates_and_resume_roundtrip(adversarial):
    model, images, geometry, support, truth, criterion = model_case()
    discriminator = ConditionalFeatureDiscriminator(24, hidden=8)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    dopt = torch.optim.Adam(discriminator.parameters(), lr=.001)
    observer_before = copy.deepcopy(model.observer.state_dict())
    for _ in range(3):
        result = research_train_step(model, images, images, geometry, support, truth, criterion, opt,
            discriminator=discriminator, discriminator_optimizer=dopt, adversarial_weight=.01 if adversarial else 0.)
        assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    for key, value in observer_before.items():
        torch.testing.assert_close(model.observer.state_dict()[key], value, rtol=0, atol=0)
    assert nonzero_grad(model.base.messenger_recovery.layers[0])
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval(); restored.eval()
    torch.testing.assert_close(model(images, geometry=geometry, support=support)["decoded_coordinates"],
                               restored(images, geometry=geometry, support=support)["decoded_coordinates"], rtol=0, atol=0)


def test_no_occlusion_does_not_update_discriminator_even_with_weight_decay():
    model, images, geometry, support, truth, criterion = model_case(anomalous=False)
    discriminator = ConditionalFeatureDiscriminator(24, hidden=8)
    before = copy.deepcopy(discriminator.state_dict())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    dopt = torch.optim.AdamW(discriminator.parameters(), weight_decay=.1)
    result = research_train_step(model, images, images, geometry, support, truth, criterion, opt,
        discriminator=discriminator, discriminator_optimizer=dopt, adversarial_weight=.01)
    assert result["discriminator"] == result["adversarial"] == 0
    for key, value in before.items():
        torch.testing.assert_close(discriminator.state_dict()[key], value, rtol=0, atol=0)


def test_checkpoint_cannot_change_prior_provenance():
    bank = fit_bank()
    state = copy.deepcopy(bank.state_dict())
    state["_extra_state"]["metadata"]["seed"] = 99
    with pytest.raises(ValueError, match="provenance"):
        bank.load_state_dict(state)


def test_actual_observer_state_is_authenticated_and_matched_control_is_exact():
    model, images, geometry, support, _, _ = model_case()
    model.eval()
    model.fusion_mode = "messenger"
    torch.testing.assert_close(model(images, geometry=geometry, support=support)["decoded_coordinates"],
                               model.base(images)["decoded_coordinates"], rtol=0, atol=0)
    with torch.no_grad():
        next(p for p in model.base.backbone.parameters() if p.numel()).add_(.01)
    with pytest.raises(ValueError, match="tensor state"):
        attach_geometry_prior(model.base, model.bank, enabled=True, observation_backbone_sha256="b" * 64)


def test_adversarial_loss_alone_reaches_generator_not_discriminator():
    model, images, geometry, support, _, _ = model_case()
    d = ConditionalFeatureDiscriminator(24, hidden=8)
    out = model(images, geometry=geometry, support=support)
    clean = out["prior_diagnostics"]["observation"]
    loss = generator_losses(out, clean, out["decoded_coordinates"].sum() * 0,
                            discriminator=d, geometry=geometry, reconstruction_weight=0, adversarial_weight=.01)
    loss["total"].backward()
    assert nonzero_grad(model.base.messenger_recovery.decoder)
    assert all(p.grad is None for p in d.parameters())


def test_model_resume_rejects_changed_area_or_fusion_contract():
    model, _, _, _, _, _ = model_case()
    checkpoint = copy.deepcopy(model.state_dict())
    model.minimum_fraction = .1
    with pytest.raises(ValueError, match="checkpoint contract"):
        model.load_state_dict(checkpoint)
