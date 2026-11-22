import copy

import pytest
import torch

from keypoint_comparison1.tests.test_geometry_prior_recovery import fit_bank, model_case
from src.models.messenger_prototype_observer import (
    LearnablePrototypeBank, attach_learnable_prototypes, prototype_loss, sparse_point_samples,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_disabled_identity_and_initial_inference_parity():
    original, images, geometry, support, _, _ = model_case()
    original.eval()
    assert attach_learnable_prototypes(original) is original
    adapted = attach_learnable_prototypes(original, enabled=True)
    with torch.no_grad():
        expected = original(images, geometry=geometry, support=support)
        actual = adapted(images, geometry=geometry, support=support)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            # Freezing can select a different attention kernel; allow float32 roundoff.
            torch.testing.assert_close(actual[key], value, rtol=1e-6, atol=2e-7)
    for key in ("score", "anomalous", "selected_center", "fused"):
        torch.testing.assert_close(actual["prior_diagnostics"][key], expected["prior_diagnostics"][key], rtol=0, atol=0)
    assert isinstance(original.bank.centers, torch.Tensor) and not original.bank.centers.requires_grad


def test_only_centers_receive_gradients_and_change():
    original, images, geometry, support, _, _ = model_case()
    model = attach_learnable_prototypes(original, enabled=True).train()
    assert not model.pipeline.training and not model.pipeline.observer.training
    before = {n: t.detach().clone() for n, t in model.state_dict().items() if isinstance(t, torch.Tensor)}
    images.requires_grad_(); geometry.requires_grad_()
    points = torch.tensor([[[.4, .4], [.7, .7]]] * 2, requires_grad=True)
    states = torch.tensor([[1, 2]] * 2)
    optimizer = model.optimizer()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prior = model.observe(images, geometry=geometry, support=support)
        loss = prototype_loss(model.bank, prior, points, states, support, input_hw=(64, 64))
        loss["loss"].backward()
        assert images.grad is None and geometry.grad is None and points.grad is None
        assert [(n, p.grad is not None) for n, p in model.named_parameters() if p.grad is not None] == [
            ("pipeline.bank.centers", True)]
        assert torch.isfinite(model.bank.centers.grad).all() and model.bank.centers.grad.abs().sum() > 0
        optimizer.step()
    changed = [n for n, t in model.state_dict().items() if isinstance(t, torch.Tensor) and not torch.equal(before[n], t)]
    assert changed == ["pipeline.bank.centers"]
    assert len(optimizer.param_groups[0]["params"]) == 1
    model.pipeline.channel_adoption.requires_grad_(True)
    with pytest.raises(ValueError, match="Only prototype"):
        model.optimizer()


def test_sparse_labels_conflicts_boundaries_missing_and_support():
    logits = torch.zeros(1, 4, 4, requires_grad=True)
    points = torch.tensor([[[.4, .4], [.4, .4], [.7, .7], [-.1, .5], [1., 1.], [float('nan'), .5], [0., 0.]]])
    states = torch.tensor([[1, 2, 0, 1, 2, 1, 2]])
    support = torch.ones_like(logits, dtype=torch.bool); support[0, 0, 0] = False
    values, labels, counts = sparse_point_samples(logits, points, states, support, input_hw=(64, 64))
    assert labels.tolist() == [1., 0., 0.]
    assert counts == dict(supervised=3, positives=1, negatives=2, missing=1, outside_crop=2, outside_support=1)
    assert len(values) == 3  # opposite labels in the same patch both survive


@pytest.mark.parametrize("state,direction", [(1, -1), (2, 1)])
def test_sparse_loss_moves_similarity_in_correct_direction(state, direction):
    fixed = fit_bank()
    bank = LearnablePrototypeBank(fixed)
    features = fixed.centers.clone() + .05 * torch.randn_like(fixed.centers)
    features.requires_grad_()
    geometry = torch.tensor([[.1, 1., 0.]], requires_grad=True)
    support = torch.ones(1, 4, 4, dtype=torch.bool)
    points = torch.tensor([[[.4, .4]]]); states = torch.tensor([[state]])
    optimizer = torch.optim.Adam([bank.centers], lr=.001)
    before = bank(features, geometry, support)["score"][0, 1, 1].item()
    for _ in range(5):
        optimizer.zero_grad()
        prior = bank(features, geometry, support)
        prototype_loss(bank, prior, points, states, support, input_hw=(64, 64))["loss"].backward()
        optimizer.step()
    after = bank(features, geometry, support)["score"][0, 1, 1].item()
    assert direction * (after - before) > 0
    assert features.grad is None and geometry.grad is None


def test_empty_supervision_cannot_step_and_checkpoint_resume_is_exact():
    fixed = fit_bank(); bank = LearnablePrototypeBank(fixed)
    x = fixed.centers.clone() + .01 * torch.randn_like(fixed.centers)
    geo = torch.tensor([[.1, 1., 0.]]); support = torch.ones(1, 4, 4, dtype=torch.bool)
    points = torch.tensor([[[.4, .4]]]); states = torch.ones(1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="No supervised points"):
        prototype_loss(bank, bank(x, geo, support), points, states * 0, support, input_hw=(64, 64))
    optimizer = torch.optim.Adam([bank.centers], lr=.001)
    def step(b, opt):
        opt.zero_grad()
        prototype_loss(b, b(x, geo, support), points, states, support, input_hw=(64, 64))["loss"].backward()
        opt.step()
    step(bank, optimizer)
    checkpoint = copy.deepcopy(dict(model=bank.state_dict(), optimizer=optimizer.state_dict()))
    restored = LearnablePrototypeBank(fixed)
    restored.load_state_dict(checkpoint["model"], strict=True)
    opt2 = torch.optim.Adam([restored.centers], lr=.001)
    opt2.load_state_dict(checkpoint["optimizer"])
    step(bank, optimizer); step(restored, opt2)
    torch.testing.assert_close(bank.centers, restored.centers, rtol=0, atol=0)
    changed = copy.deepcopy(fixed); changed.centers.add_(.1)
    for wrong in (LearnablePrototypeBank(changed), LearnablePrototypeBank(fixed, logit_temperature=.2)):
        with pytest.raises(ValueError, match="differs from checkpoint"):
            wrong.load_state_dict(checkpoint["model"], strict=True)
