import numpy as np
import pytest
import torch
from src.models.messenger_feature_discriminator import (
    FeaturePatchDiscriminator, paired_candidates, receptive_masks,
    spatial_mean, discriminator_objective, generator_objectives)


def data():
    torch.set_num_threads(2); torch.manual_seed(8)
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn_like(x)
    r = torch.randn_like(x, requires_grad=True)
    m = torch.zeros(2, 16, 16, dtype=torch.bool); m[:, 5:9, 4:8] = True
    d = FeaturePatchDiscriminator(torch.zeros(12), torch.ones(12))
    return x, z, r, m, d


def test_pair_has_identical_outside_and_correct_inside():
    x, z, r, m, _ = data(); real, fake = paired_candidates(x, r, z, m)
    outside = (~m)[:, None].expand_as(x); inside = ~outside
    assert torch.equal(real[outside], fake[outside])
    assert torch.equal(real[inside], z[inside])
    assert torch.equal(fake[inside], r[inside])


def test_patch_shapes_and_receptive_field_masks():
    x, z, r, m, d = data()
    logit, features = d(r, x, m)
    assert logit.shape == (2, 8, 8)
    assert [f.shape[1:] for f in features] == [(128, 16, 16), (192, 8, 8), (192, 8, 8)]
    assert [v.shape for v in receptive_masks(m)] == [(2, 16, 16), (2, 8, 8), (2, 8, 8)]
    single = torch.zeros_like(m); single[:, 7, 7] = True
    assert int(receptive_masks(single)[0][0].sum()) == 9


def test_generator_losses_have_R_gradient_no_D_or_target_gradient():
    x, z, r, m, d = data(); z.requires_grad_(True)
    d.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in d.state_dict().items()}
    adv, fm = generator_objectives(d, x, r, z, m)
    for loss in (adv, fm):
        grad = torch.autograd.grad(loss, r, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.norm() > 0
        assert (grad*(~m)[:, None]).abs().max() == 0
    (adv+fm).backward()
    assert z.grad is None and all(p.grad is None for p in d.parameters())
    assert all(torch.equal(v, before[k]) for k, v in d.state_dict().items())


def test_identical_candidates_have_zero_feature_matching():
    x, z, _, m, d = data(); d.eval().requires_grad_(False)
    _, fm = generator_objectives(d, x, z, z, m)
    assert fm == 0


def test_D_updates_without_R_gradient():
    x, z, r, m, d = data(); d.train()
    optimizer = torch.optim.Adam(d.parameters(), lr=1e-4)
    real, fake = paired_candidates(x, r.detach(), z, m)
    rl, _ = d(real, x, m); fl, _ = d(fake, x, m)
    loss = discriminator_objective(rl, fl, m); loss.backward()
    assert r.grad is None
    assert any(p.grad is not None and p.grad.norm() > 0 for p in d.parameters())
    optimizer.step()


def test_masked_mean_equal_example_weight_and_empty_rejected():
    value = torch.stack((torch.ones(4, 4), torch.full((4, 4), 3.)))
    m = torch.ones_like(value); m[0, 1:] = 0
    assert spatial_mean(value, m) == 2
    with pytest.raises(ValueError):
        spatial_mean(value, torch.zeros_like(m))


def test_checkpoint_roundtrip_preserves_D_scores_and_optimizer(tmp_path):
    x, z, r, m, d = data(); d.train()
    opt = torch.optim.AdamW(d.parameters(), lr=1e-4)
    rl, _ = d(z, x, m); fl, _ = d(r.detach(), x, m)
    discriminator_objective(rl, fl, m).backward(); opt.step()
    d.eval(); before = d(z, x, m)[0]
    torch.save(dict(d=d.state_dict(), opt=opt.state_dict()), tmp_path/'state.pt')
    restored = FeaturePatchDiscriminator(torch.ones(12), torch.ones(12)*2)
    ck = torch.load(tmp_path/'state.pt', weights_only=True)
    restored.load_state_dict(ck['d']); restored.eval()
    torch.testing.assert_close(before, restored(z, x, m)[0], rtol=0, atol=0)


def test_plan_preserves_position_primary_baseline():
    from scripts.run_messenger_adversarial_comparison import PLAN, ARMS
    assert len(ARMS) == 3 and PLAN['mse_weight'] == PLAN['cosine_weight'] == .01
    assert PLAN['clean_head'] is False and PLAN['gate_bce_weight'] == .01
    assert PLAN['microbatch'] == 4 and PLAN['effective_batch'] == 16


def test_D_seed_does_not_change_global_R_randomness():
    torch.manual_seed(42); before = torch.get_rng_state().clone()
    FeaturePatchDiscriminator(torch.zeros(12), torch.ones(12))
    assert torch.equal(before, torch.get_rng_state())
