import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from keypoint_comparison1.landmark_detection.continuous_recovery import ContinuousRecoveryBlock, ContinuousRecoveryConfig, ContinuousRecoveryLoss, RecoveryLayer
from keypoint_comparison1.landmark_detection.recovery_acceptance import common_state_hash
from keypoint_comparison1.landmark_detection.dataset import PairedOcclusionConfig
from keypoint_comparison1.landmark_detection.occlusion_materials import build_materials, load_materials, sha256, apply_material_occlusion
from keypoint_comparison1.tests.test_continuous_recovery import targets, assert_grad, config
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("scale,level,nkeys,masked", [(False,False,256,1),(True,False,336,3),(False,True,768,3),(True,True,1008,9)])
def test_reference_shapes_masks_and_perturbation(scale, level, nkeys, masked):
    block = ContinuousRecoveryBlock(24, dict(enabled=True, multi_scale=scale, multi_level=level))
    maps = [torch.randn(2,24,16,16) for _ in range(3 if level else 1)]
    _, diag = block(maps[-1], reference_maps=maps if level else None, return_diagnostics=True)
    assert diag["attention"].shape == (2,2,6,256,nkeys)
    assert diag["recovered_tokens"].shape == (2,256,24)
    if block.reference_memory is None:
        assert torch.count_nonzero(diag["attention"].diagonal(dim1=-2, dim2=-1)) == 0
        return
    ref = block.reference_memory(maps[-1], maps)
    forbidden = ref["forbidden"]
    assert (forbidden.sum(-1) == masked).all()
    assert diag["attention"].masked_select(forbidden[None,None,None]).count_nonzero() == 0
    torch.testing.assert_close(diag["reference_attention_mass"].sum(-1), torch.ones(2,2))
    # Test every query and level after DINO extraction. Other cells/alpha fixed.
    for i in range(256):
        perturbed = [m.clone() for m in maps]
        for m in perturbed:
            m[:,:,i//16,i%16] += 100
        changed = block.reference_memory(perturbed[-1], perturbed)
        torch.testing.assert_close(ref["tokens"][:,~forbidden[i]], changed["tokens"][:,~forbidden[i]], atol=0, rtol=0)
    previous = torch.rand(2,256,requires_grad=True)
    rho = block.reference_memory.reliability(previous,16,16)
    for group in ref["groups"]:
        f = group["factor"]
        expected = torch.nn.functional.avg_pool2d((1-previous).reshape(2,1,16,16), f, f).flatten(1)
        torch.testing.assert_close(rho[:,group["start"]:group["stop"]],expected)
    rho.sum().backward()
    assert previous.grad.abs().sum() > 0
    first = ref["metadata"][0]
    torch.testing.assert_close(first[:6],torch.tensor([1/32,1/32,0,0,1/16,1/16]))


def test_all_forbidden_rectangular_attention_has_zero_context_and_finite_backward():
    layer = RecoveryLayer(24, ContinuousRecoveryConfig())
    state, alpha, attention = layer(torch.randn(2,4,24), torch.randn(2,4,24), None,
        memory=torch.randn(2,7,24), forbidden=torch.ones(4,7,dtype=torch.bool))
    assert torch.isfinite(state).all() and torch.isfinite(alpha).all()
    assert attention.count_nonzero() == 0
    (state.square().mean()+alpha.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_c0_strict_compatibility_common_initial_state_and_rng():
    blocks = []
    after_rng = []
    for scale,level in [(False,False),(True,False),(False,True),(True,True)]:
        torch.manual_seed(42)
        block = ContinuousRecoveryBlock(24,dict(enabled=True,structure_enabled=True,structure_dim=24,multi_scale=scale,multi_level=level))
        blocks.append(block)
        after_rng.append(torch.rand(3))
    assert len({common_state_hash(b) for b in blocks}) == 1
    for value in after_rng[1:]:
        torch.testing.assert_close(value,after_rng[0],atol=0,rtol=0)
    legacy = ContinuousRecoveryBlock(24,dict(enabled=True,structure_enabled=True,structure_dim=24))
    legacy.load_state_dict(blocks[0].state_dict(),strict=True)
    x = torch.randn(2,24,4,4)
    torch.testing.assert_close(legacy(x)[0],blocks[0](x)[0],atol=0,rtol=0)
    assert not any("reference_memory" in k for k in legacy.state_dict())


def test_alpha_zero_has_no_direct_teacher_but_preserves_other_coefficients():
    cfg = ContinuousRecoveryConfig.from_mapping(dict(enabled=True,alpha_weight=0,adoption="optimal"))
    block = ContinuousRecoveryBlock(24,cfg)
    _, d = block(torch.randn(2,24,4,4))
    clean = torch.randn(2,24,4,4)
    loss = ContinuousRecoveryLoss(cfg)(d,clean,torch.rand(2,64,64),channel_scale=block.channel_scale)
    assert not loss["loss_recovery_alpha"].requires_grad
    torch.testing.assert_close(loss["loss_messenger_total"], .1*loss["loss_recovery_candidate"]+.1*loss["loss_recovery_output"]+.01*loss["loss_recovery_adoption"])
    # Isolate direct alpha supervision with all other terms disabled.
    cfg2 = ContinuousRecoveryConfig.from_mapping(dict(enabled=True,alpha_weight=0,candidate_weight=0,output_weight=0,adoption_weight=0))
    for mask in (torch.zeros(2,64,64),torch.ones(2,64,64)):
        total=ContinuousRecoveryLoss(cfg2)(d,clean,mask,channel_scale=block.channel_scale)["loss_messenger_total"]
        assert total == 0


@pytest.mark.parametrize("objective",["output_reconstruction","downstream_coordinates"])
def test_alpha_and_reference_receive_gradients_after_warmup(objective):
    torch.manual_seed(42)
    model,_=build_landmark_model(config(alpha_weight=0,multi_scale=True,structure_enabled=True),variant="query_refinement",initialize_from_detector=False)
    block=model.messenger_recovery
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    images=torch.randn(2,3,64,64)
    truth=targets()
    criterion=LandmarkLoss()
    # Warm-start tiny downstream coordinate outputs, then pass zero-initialized residuals.
    for _ in range(8):
        output=model(images,return_recovery_diagnostics=True)
        loss=criterion(output,truth)["loss_coordinates"]
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    output=model(images,return_recovery_diagnostics=True)
    if objective == "downstream_coordinates":
        loss=criterion(output,truth)["loss_coordinates"]
    else:
        d=output["recovery_diagnostics"]
        loss=(d["recovered_tokens"]-torch.randn_like(d["recovered_tokens"])).square().mean()
    loss.backward()
    for layer in block.layers:
        assert_grad(layer.alpha.parameters())
        assert_grad(layer.k.parameters())
        assert_grad(layer.v.parameters())
    assert_grad(block.reference_memory.parameters())
    assert_grad(block.local_gate.parameters())
    assert_grad(model.backbone.parameters())
    block.mixing_enabled=False
    optimizer.zero_grad(set_to_none=True)
    output=model(images,return_recovery_diagnostics=True)
    d=output["recovery_diagnostics"]
    torch.testing.assert_close(d["recovered_tokens"],d["original_tokens"],atol=0,rtol=0)


def test_materials_support_same_texture_mask_and_no_fallback(tmp_path):
    manifest=build_materials(tmp_path)
    digest=sha256(manifest)
    image=np.full((128,128,3),127,dtype=np.uint8)
    points=np.array([[40,50],[85,50],[64,65],[48,80],[80,80]],dtype=np.float32)
    states=np.array([2,1,2,2,0])
    kwargs=dict(face_ellipse=[64,64,48,40,0],keypoints=points,keypoint_visibility=states)
    mapping=dict(appearance="texture",material_manifest=str(manifest),material_manifest_sha256=digest)
    cfg=PairedOcclusionConfig.from_mapping(mapping)
    for kind in cfg.kinds:
        solid,smask,ss,_=apply_landmark_occlusion_numpy(image,**kwargs,config=cfg.landmark_mapping(),rng=random.Random(99),force=True,force_kind=kind)
        texture,tmask,ts,meta=apply_material_occlusion(image,**kwargs,config=cfg,rng=random.Random(99),force_kind=kind)
        np.testing.assert_array_equal(smask,tmask)
        np.testing.assert_array_equal(ss,ts)
        np.testing.assert_array_equal(texture[~tmask],image[~tmask])
        assert not np.array_equal(solid,texture)
        ocfg=PairedOcclusionConfig.from_mapping({**mapping,"appearance":"object"})
        obj,mask,updated,meta=apply_material_occlusion(image,**kwargs,config=ocfg,rng=random.Random(99),force_kind=kind)
        assert .1 <= meta["face_overlap"] <= .4
        assert meta["mask_source"] == "actual_binary_alpha"
        x0,y0,x1,y1=meta["bbox_xyxy"]
        assert not mask[y0:y1,x0:x1].all()
        np.testing.assert_array_equal(obj[~mask],image[~mask])
        assert updated[4] == 0
    assert np.all(image == 127)  # teacher/source unmodified
    with pytest.raises(ValueError,match="manifest"):
        load_materials(str(manifest),"0"*64)


@pytest.mark.parametrize("kwargs",[dict(reference_layers=[11,7,3]),dict(reference_layers=[3,3,11]),dict(reference_layers=[3,7,12]),dict(reference_factors=[1,3]),dict(multi_level="true")])
def test_reference_schema_rejects_unlocked_or_ambiguous_settings(kwargs):
    with pytest.raises(ValueError):
        ContinuousRecoveryConfig.from_mapping(kwargs)
