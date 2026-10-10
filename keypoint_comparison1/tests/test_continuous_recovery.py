from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from keypoint_comparison1.tests.test_messenger_recovery import _messenger_config
from keypoint_comparison1.landmark_detection.continuous_recovery import (
    ContinuousRecoveryBlock, ContinuousRecoveryConfig, ContinuousRecoveryLoss,
    oracle_adoption, relative_structure_targets,
)
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.training import _optimizer


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def config(**kwargs):
    raw = _messenger_config(enabled=False)
    raw["model"]["continuous_recovery"] = dict(enabled=True, structure_dim=24, structure_heads=4, **kwargs)
    return raw


def targets():
    return [dict(keypoints=torch.rand(5, 2), labeled=torch.ones(5, dtype=torch.bool),
                 visible=torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool),
                 structure_pairs=torch.randn(10, 2) * .1,
                 structure_pairs_valid=torch.ones(10, dtype=torch.bool)) for _ in range(2)]


def assert_grad(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.parametrize("adoption,structure", [("alpha", False), ("local", False), ("optimal", False), ("local", True), ("optimal", True)])
def test_inference_and_task_only_gradients_before_after_updates_and_unfreeze(adoption, structure):
    torch.manual_seed(47)
    model, _ = build_landmark_model(config(adoption=adoption, structure_enabled=structure), variant="query_refinement", initialize_from_detector=False)
    block = model.messenger_recovery
    optimizer = _optimizer(model, dict(lr_backbone=1e-3, lr_head=1e-3, lr_messenger=1e-3, weight_decay=0.0))
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    images, truth = torch.randn(2, 3, 64, 64), targets()
    criterion = LandmarkLoss()
    # Production warm-starts a trained Query Refinement. The tiny test head's
    # coordinate output starts at zero, so prime only its raw path first.
    for _ in range(2):
        criterion(model(images, recovery_mode="raw"), truth)["loss_coordinates"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    def task_forward():
        out = model(images, return_recovery_diagnostics=True)
        return out, criterion(out, truth)["loss_coordinates"]
    out, loss = task_forward()
    out["recovery_diagnostics"]["recovered_tokens"].retain_grad()
    loss.backward()
    assert_grad(model.backbone.parameters())
    assert_grad(block.decoder.parameters())
    assert out["recovery_diagnostics"]["recovered_tokens"].grad.abs().sum() > 0
    # Zero-output residual/prototype correction intentionally delays upstream gradients.
    assert block.layers[0].q.weight.grad.abs().sum() == 0
    if structure:
        torch.testing.assert_close(out["recovery_diagnostics"]["local_gate"], out["recovery_diagnostics"]["adoption_gate"], rtol=0, atol=0)
        assert_grad(block.structure_gate.parameters())
        assert block.structure.prototypes.grad.abs().sum() == 0
    optimizer.zero_grad(set_to_none=True)
    model.backbone.requires_grad_(False)
    clean = torch.randn(2, 24, 4, 4, requires_grad=True)
    for _ in range(4):
        out, loss = task_forward()
        recovery = ContinuousRecoveryLoss(block.config)(out["recovery_diagnostics"], clean, torch.ones(2, 64, 64), channel_scale=block.channel_scale, targets=truth)
        (loss + recovery["loss_messenger_total"]).backward()
        assert clean.grad is None
        assert all(p.grad is None for p in model.backbone.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.backbone.requires_grad_(True)
    out, loss = task_forward()
    loss.backward()  # ONLY deployed final keypoint coordinate loss, no auxiliary losses.
    assert_grad(model.head.parameters())
    assert_grad(model.backbone.parameters())
    assert_grad(block.decoder.parameters())
    assert_grad([block.messenger_token])
    for layer in block.layers:
        assert_grad(layer.q.parameters())
        assert_grad(layer.k.parameters())
        assert_grad(layer.v.parameters())
        assert_grad(layer.alpha.parameters())
    if block.local_gate is not None:
        assert_grad(block.local_gate.parameters())
    if structure:
        assert_grad(block.structure_gate.parameters())
        assert_grad(block.structure.attention.parameters())
        assert_grad(block.structure.project.parameters())
        assert_grad([block.structure.prototypes])
        assert all(p.grad is None for p in block.structure.descriptor.parameters())
    updated_parameter = next(p for p in model.backbone.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    before = updated_parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, updated_parameter)
    model.eval()
    # No GT Head, clean images, or teacher on the inference signature.
    assert torch.isfinite(model(images)["decoded_coordinates"]).all()


def test_disabled_checkpoint_contract_and_query_refinement_structure():
    plain = _messenger_config(enabled=False)
    disabled = copy.deepcopy(plain)
    disabled["model"]["continuous_recovery"] = {"enabled": False}
    a, _ = build_landmark_model(plain, variant="query_refinement", initialize_from_detector=False)
    b, _ = build_landmark_model(disabled, variant="query_refinement", initialize_from_detector=False)
    b.load_state_dict(a.state_dict(), strict=True)
    c, _ = build_landmark_model(config(), variant="query_refinement", initialize_from_detector=False)
    assert a.head.state_dict().keys() == c.head.state_dict().keys()
    assert {k: v.shape for k, v in a.head.state_dict().items()} == {k: v.shape for k, v in c.head.state_dict().items()}
    a.eval(); b.eval()
    images = torch.randn(2, 3, 64, 64)
    torch.testing.assert_close(a(images)["decoded_coordinates"], b(images)["decoded_coordinates"], rtol=0, atol=0)


@pytest.mark.parametrize("size", [1, 4])
def test_attention_finite_empty_context_and_amp(size):
    block = ContinuousRecoveryBlock(24, {"enabled": True})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        recovered, diag = block(torch.randn(2, 24, size, size), return_diagnostics=True)
    assert torch.isfinite(recovered).all()
    assert torch.count_nonzero(diag["attention"].diagonal(dim1=-2, dim2=-1)) == 0
    if size == 1:
        assert diag["attention"].sum() == 0
    # This host's oneDNN lacks bf16 backward; independently test FP32 backward.
    recovered, _ = block(torch.randn(2, 24, size, size))
    recovered.square().mean().backward()
    assert_grad(block.decoder.parameters())


def test_oracle_is_detached_scaled_and_degenerate_is_raw():
    x = torch.zeros(1, 3, 2, requires_grad=True)
    r = torch.tensor([[[2., 4.], [0., 0.], [2., 4.]]], requires_grad=True)
    z = torch.tensor([[[1., 2.], [1., 1.], [8., 8.]]], requires_grad=True)
    oracle = oracle_adoption(x, r, z, torch.tensor([1., 2.]), 1e-6)
    torch.testing.assert_close(oracle, torch.tensor([[.5, 0., 1.]]))
    assert not oracle.requires_grad


def test_clean_and_empty_pair_losses_descriptor_grad_and_consistency():
    cfg = ContinuousRecoveryConfig.from_mapping(dict(enabled=True, structure_enabled=True, structure_dim=24, consistency_weight=.01, gram_weight=.05))
    block = ContinuousRecoveryBlock(24, cfg)
    _, diagnostics = block(torch.randn(2, 24, 4, 4))
    truth = targets()
    for target in truth:
        target["structure_pairs_valid"].zero_()
    clean = torch.randn(2, 24, 4, 4, requires_grad=True)
    clean_pi = torch.full((2, 8), 1/8, requires_grad=True)
    criterion = ContinuousRecoveryLoss(cfg)
    losses = criterion(diagnostics, clean, torch.zeros(2, 64, 64), channel_scale=block.channel_scale, targets=truth, clean_pi=clean_pi)
    assert losses["loss_structure_descriptor"] == 0
    assert losses["loss_recovery_candidate"] > 0  # unoccluded samples remain supervised
    losses["loss_messenger_total"].backward()
    assert clean.grad is None and clean_pi.grad is None
    block.zero_grad(set_to_none=True)
    _, diagnostics = block(torch.randn(2, 24, 4, 4))
    losses = criterion(diagnostics, clean, torch.ones(2, 64, 64), channel_scale=block.channel_scale, targets=targets(), clean_pi=clean_pi)
    losses["loss_structure_descriptor"].backward()
    assert_grad(block.structure.descriptor.parameters())
    assert_grad(block.structure.attention.parameters())


def test_structure_normalization_semantic_flip_and_invalid_points():
    q = torch.tensor([[.1, .2], [.8, .3], [.5, .5], [.2, .8], [.7, .8]])
    inverse = torch.tensor([[2., 0., 4.], [0., 2., 8.]])
    pairs, valid = relative_structure_targets(q, torch.ones(5, dtype=torch.bool), inverse, 101, 100.)
    torch.testing.assert_close(pairs[0], (q[0] - q[1]) * 2)
    assert valid.all()
    flipped = q[[1, 0, 2, 4, 3]].clone(); flipped[:, 0] = 1 - flipped[:, 0]
    flip_pairs, _ = relative_structure_targets(flipped, torch.ones(5, dtype=torch.bool), inverse, 101, 100.)
    torch.testing.assert_close(flip_pairs[0], pairs[0] * torch.tensor([1., -1.]))
    q[0] = float("nan")
    values, valid = relative_structure_targets(q, torch.tensor([1, 1, 0, 0, 0]), inverse, 101, 100.)
    assert torch.isfinite(values).all() and not valid.any()


def test_mixing_warmup_only_affects_training_and_structure_sensitivity():
    block = ContinuousRecoveryBlock(24, dict(enabled=True, structure_enabled=True, structure_dim=24))
    x = torch.randn(2, 24, 4, 4)
    block.mixing_enabled = False
    y, _ = block(x)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    block.eval()
    y, _ = block(x)
    assert not torch.equal(x, y)
    with torch.no_grad():
        block.structure_gate[-1].weight.fill_(0.2)
    _, actual = block(x)
    _, zero = block(x, structure_ablation="zero")
    _, shuffled = block(x, structure_ablation="shuffle")
    torch.testing.assert_close(zero["adoption_gate"], zero["local_gate"])
    assert not torch.equal(actual["adoption_gate"], zero["adoption_gate"])
    assert not torch.equal(actual["adoption_gate"], shuffled["adoption_gate"])


def test_all_ablation_configs_share_downstream_and_combined_uses_option1():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "keypoint_comparison1/configs").glob("continuous_recovery_*.yaml"))
    baseline = load_landmark_config(root / "keypoint_comparison1/configs/dinov3_vits16_face_pose_drive_v2_messenger_control.yaml").raw
    for path in paths:
        raw = load_landmark_config(path).raw
        assert raw["model"]["variants"] == baseline["model"]["variants"]
        assert raw["train"]["microbatch_candidates"] == [4, 8, 16]
        assert raw["train"]["flat_epochs"] == raw["train"]["warmup_epochs"]
        assert raw["train"]["checkpoint_every"] == 0
        assert raw["evaluation"]["pipeline_contract"]["stages"] == ["full_image_face_ellipse", "face_crop_keypoint"]
        if "combined" in path.name:
            cfg = raw["model"]["continuous_recovery"]
            assert cfg["adoption"] == "optimal" and cfg["structure_enabled"] and cfg["gram_weight"] > 0


def test_trainer_teacher_provenance_no_periodic_checkpoints_and_resume(tmp_path, monkeypatch):
    """One tiny fixture epoch exercises the real trainer; validation is stubbed.

    This catches integration/checkpoint bugs without running a production gate.
    """
    import hashlib
    import json
    from keypoint_comparison1.tests.test_messenger_recovery import _write_dataset
    from keypoint_comparison1.landmark_detection import training
    from keypoint_comparison1.landmark_detection.config import LandmarkExperimentConfig

    annotation, image_root = _write_dataset(tmp_path)
    heads = tmp_path / "train_heads.json"
    heads.write_text(json.dumps({"images": [{"id": 1, "file_name": "face.png"}], "annotations": [
        {"id": 1, "image_id": 1, "bbox": [10, 10, 80, 80], "ellipse": [48, 50, 28, 23, 0],
         "attribute_valid": True, "keypoint_states": [2, 2, 2, 2, 1]}]}))
    raw = config(adoption="optimal", structure_enabled=True, consistency_weight=.01, mixing_warmup_epochs=0, scale_samples=2)
    raw.update(experiment_name="test", data=dict(root=str(image_root), annotations={"train": str(annotation), "val": str(annotation)},
              input_size=64, category_ids=[2], include_missing_keypoint_instances=True, structure_head_annotations=str(heads),
              augmentation=dict(paired_occlusion=dict(probability=0.3, kinds=["rectangle", "irregular", "boundary"], face_overlap=[.1, .4]))),
               outputs={"root": str(tmp_path / "training")})
    plain = copy.deepcopy(raw); plain["model"]["continuous_recovery"]["enabled"] = False
    model, _ = build_landmark_model(plain, variant="query_refinement", initialize_from_detector=False)
    checkpoint = tmp_path / "baseline.pt"
    torch.save(dict(model=model.state_dict(), variant="query_refinement", keypoint_target_policy={}, state_head_mode="legacy_binary"), checkpoint)
    raw["train"] = dict(seed=42, batch_size=1, gradient_accumulation_steps=1, epochs=1, num_workers=0, amp=False,
                        refinement_initialize_from_query=False, freeze_backbone_epochs=0, checkpoint_every=0,
                        finetune_checkpoint=dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), strict=False, allowed_missing_prefixes=["messenger_recovery."]))
    monkeypatch.setattr(training, "evaluate_gt_boxes", lambda *a, **kw: (dict(nme=.1, **{"pck_0.05": .5, "pck_0.10": .8}), []))
    experiment = LandmarkExperimentConfig(tmp_path / "config.yaml", tmp_path, raw)
    summary = training.train_variant(experiment, variant="query_refinement", device=torch.device("cpu"))
    assert summary["status"] == "complete"
    root = experiment.output_root / "query_refinement"
    assert {p.name for p in (root / "checkpoints").glob("*.pt")} == {"last.pt", "best_nme.pt"}
    provenance = json.loads((root / "continuous_teacher_provenance.json").read_text())
    assert provenance["split"] == "train" and provenance["structure_supervision_inventory"]["excluded"] == 0
    assert raw["continuous_teacher_provenance"] == provenance
    raw["train"]["epochs"] = 2
    resumed = training.train_variant(experiment, variant="query_refinement", device=torch.device("cpu"), resume_from=root / "checkpoints/last.pt")
    assert resumed["last_epoch"] == 2
    raw["model"]["continuous_recovery"]["gram_weight"] = .05
    with pytest.raises(RuntimeError, match="different continuous recovery"):
        training.train_variant(experiment, variant="query_refinement", device=torch.device("cpu"), resume_from=root / "checkpoints/last.pt")
