"""The scheduled face-only policy must not inherit the mixed policy's fallback."""
import json
import random
import numpy as np
import pytest
import torch
from test_real_mixed_occlusion import fixture_bank
from keypoint_comparison1.landmark_detection.real_occlusion import digest, identity
from keypoint_comparison1.landmark_detection.real_face_occlusion import RealFaceOcclusionBank, apply_real_face
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation, TopDownFaceLandmarkDataset


def strict_bank(fixture_bank):
    bank, annotation, manifest, images, anns = fixture_bank
    data = bank.data
    data.update(face_only_policy="real_face_w48_s2_v1",
        recipient_contexts={identity(f"{i}.png", a["ellipse"]): dict(video=f"v{i}", descriptor=[30,0,0,10]) for i,a in enumerate(anns)},
        strict_faces=[dict(f, video=f"v{i}", ratio=1.1) for i,f in enumerate(data["faces"])])
    manifest.write_text(json.dumps(data))
    return RealFaceOcclusionBank(manifest, digest(manifest), annotation_path=annotation, image_root=bank.root)


def test_strict_context_profile_and_video_selection(fixture_bank):
    bank = strict_bank(fixture_bank)
    _, _, _, images, anns = fixture_bank
    sample = dict(image=images[0], annotation=anns[0])
    donor = bank.select(sample, random.Random(42), False)
    assert donor[2]["donor_video"] == "v1"
    assert bank.select(sample, random.Random(42), True) is None
    bank.strict_faces[1]["descriptor"][0] = 43
    assert bank.select(sample, random.Random(42), False) is None
    bank.strict_faces[1]["descriptor"] = [30,13,0,10]
    assert bank.select(sample, random.Random(42), False) is None
    bank.strict_faces[1]["descriptor"] = [30,0,0,26]
    assert bank.select(sample, random.Random(42), False) is None


def test_failed_placement_preserves_clean_image_and_missing(fixture_bank, monkeypatch):
    bank = strict_bank(fixture_bank)
    _, _, _, images, anns = fixture_bank
    # Force donor availability; placement failure is a skip, never a downgraded route.
    bank.select = lambda *args: (np.zeros((32,32,3),np.uint8), [16,16,8,8,0], {})
    monkeypatch.setattr("keypoint_comparison1.landmark_detection.real_face_occlusion.place",
        lambda target, donor, rng, route, **kwargs: (target[0].copy(), np.zeros((256,256),bool), dict(applied=False)))
    rgb = np.full((256,256,3),70,np.uint8)
    out, mask, states, meta = apply_real_face(rgb, sample=dict(image=images[0],annotation=anns[0]),
        face_ellipse=anns[0]["ellipse"], keypoints=[[-1,100],[50,50],[60,60],[70,70],[80,80]],
        keypoint_visibility=[2]*5, bank=bank, rng=random.Random(42))
    assert not meta["applied"] and not mask.any() and np.array_equal(rgb,out)
    assert states[0] == 0


def test_actual_loader_face_only_rgb_and_teacher(fixture_bank):
    bank = strict_bank(fixture_bank)
    _, annotation, manifest, images, anns = fixture_bank
    aug = CropAugmentation(scale_jitter=0,center_jitter=0,rotation_deg=0,
        brightness=0,contrast=0,saturation=0,blur_probability=0,horizontal_flip_probability=1,
        occlusion_probability=0,paired_occlusion=dict(probability=1,appearance="real_face",
            material_manifest=str(manifest),material_manifest_sha256=digest(manifest)))
    ds = TopDownFaceLandmarkDataset(annotation_path=annotation,image_root=bank.root,training=True,augmentation=aug)
    applied = 0
    for epoch in range(1,12):
        ds.set_epoch(epoch)
        random.seed(42)
        x,t = ds[0]
        mask = t["occlusion_mask"]
        assert torch.equal(x[:,~mask],t["clean_image"][:,~mask])
        assert not t["synthetic_truncated"].any()
        meta = t["paired_occlusion"]
        if meta["applied"]:
            applied += 1
            assert meta["requested_family"] == "face" and not meta["requested_profile"]
            assert meta["donor_video"] != meta["recipient_video"]
            assert 0 < meta["overlap"] <= .4 and meta["donor_face_overlap"] > 0
    assert applied > 0


def test_schema_rejects_relaxed_face_only_contract():
    import copy
    from pathlib import Path
    from keypoint_comparison1.landmark_detection.config import load_landmark_config
    from scripts.run_continuous_recovery_gate import validate_real_face_contract
    config = Path(__file__).resolve().parents[1]/"keypoint_comparison1/configs/continuous_recovery_abc_b_real_face_w48_s2.yaml"
    raw = load_landmark_config(config).raw
    assert validate_real_face_contract(raw)
    bad = copy.deepcopy(raw)
    bad["augmentation_contract"]["profile_donor_probability"] = 0
    with pytest.raises(ValueError,match="contract"):
        validate_real_face_contract(bad)


class _EdgePlacementRng:
    def uniform(self, *_): return 1.0
    def randrange(self, *_): return 2
    def randint(self, low, high): return 1 if low == 1 else 0


def test_area_only_light_accepts_no_hidden_points_and_rejects_exact_ten_percent(monkeypatch):
    from scripts import prototype_cross_video_face_occlusion as prototype
    monkeypatch.setattr(prototype, "ellipse_region_mask", lambda h,w,e: np.ones((h,w),bool))
    image=np.zeros((10,10,3),np.uint8);donor=np.full((2,10,3),200,np.uint8)
    points=np.array([[5,5]]*5);states=np.array([2]*5)
    # Nine out of 100 face pixels covered; all five landmarks remain visible.
    target=(image,[5,5,4.5,4.5,0],points,states)
    _,mask,meta=prototype.place(target,(donor,[5,1,5,1,0]),_EdgePlacementRng(),"light",light_area_only=True)
    assert meta["applied"] and meta["overlap"]==.09 and meta["occluded_keypoints"]==0
    assert meta["states"]==[2]*5 and mask.sum()==9
    # Legacy calls still require a covered keypoint.
    assert not prototype.place(target,(donor,[5,1,5,1,0]),_EdgePlacementRng(),"light")[2]["applied"]
    target=(image,[5,5,5,5,0],points,states)
    assert not prototype.place(target,(donor,[5,1,5,1,0]),_EdgePlacementRng(),"light",light_area_only=True)[2]["applied"]


def test_light_allows_sub_one_percent_face_overlap_but_not_background_only(monkeypatch):
    from scripts import prototype_cross_video_face_occlusion as prototype
    def masks(h,w,e):
        out=np.ones((h,w),bool) if h==100 else np.zeros((h,w),bool)
        if h!=100:out[-1,0]=True
        return out
    monkeypatch.setattr(prototype,"ellipse_region_mask",masks)
    image=np.zeros((100,100,3),np.uint8);donor=np.ones((2,100,3),np.uint8)
    target=(image,[50,50,45,45,0],np.array([[50,50]]*5),np.array([2]*5))
    _,_,meta=prototype.place(target,(donor,[50,1,50,1,0]),_EdgePlacementRng(),"light",light_area_only=True)
    assert meta["applied"] and 0<meta["donor_face_overlap"]<.01
    monkeypatch.setattr(prototype,"ellipse_region_mask",lambda h,w,e: np.ones((h,w),bool) if h==100 else np.zeros((h,w),bool))
    assert not prototype.place(target,(donor,[50,1,50,1,0]),_EdgePlacementRng(),"light",light_area_only=True)[2]["applied"]


def test_revised_light_policy_does_not_change_medium_or_strong():
    from scripts.prototype_cross_video_face_occlusion import place
    image=np.zeros((64,64,3),np.uint8);donor=np.full((40,40,3),128,np.uint8)
    target=(image,[32,32,24,20,0],np.array([[25,25],[39,25],[32,32],[27,39],[37,39]]),np.array([2]*5))
    for route in ["medium","strong"]:
        old=place(target,(donor,[20,20,18,16,0]),random.Random(42),route)
        new=place(target,(donor,[20,20,18,16,0]),random.Random(42),route,light_area_only=True)
        assert np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1]) and old[2]==new[2]


def test_user_aggregate_seven_percent_audit_contract():
    from scripts.audit_real_face_occlusion import summarize, distribution_passed
    def rows(skips):
        return [dict(applied=i>=skips, requested_route="strong", reason="placement_budget") for i in range(100)]
    assert summarize(rows(7), .07)["checks"]["skip_within_limit"]
    assert not summarize(rows(8), .07)["checks"]["skip_within_limit"]
    groups={str(i):summarize(rows(8 if i==0 else 6), .07) for i in range(6)}
    total=summarize(rows(8)+rows(6)*5, .07)
    assert distribution_passed(total,groups,"aggregate")
    assert not distribution_passed(total,groups,"each_seed_epoch")
    groups['0']['checks']['strong_at_least_25_percent']=False
    assert not distribution_passed(total,groups,"aggregate")
