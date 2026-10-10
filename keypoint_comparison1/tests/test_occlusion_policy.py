import random
from dataclasses import replace

import numpy as np
import pytest

from keypoint_comparison1.landmark_detection.dataset import PairedOcclusionConfig
from keypoint_comparison1.landmark_detection.occlusion_materials import apply_material_occlusion, build_materials, sha256
from keypoint_comparison1.landmark_detection.occlusion_policy import ObjectOverlapPolicy, assess_overlap, keypoint_regions


def test_exempt_material_and_local_exception_keep_upper_cap():
    p = ObjectOverlapPolicy()
    face = np.ones((100, 100), bool)
    mask = np.zeros_like(face); mask[20:25, 20:40] = True
    region = np.zeros_like(face); region[20:30, 20:40] = True
    assert assess_overlap(mask, face, {}, "glasses", p)["acceptance_reason"] == "exempt_material"
    assert not assess_overlap(mask, face, {}, "bottle", p)["accepted"]
    assert assess_overlap(mask, face, {0: region}, "bottle", p)["acceptance_reason"] == "keypoint_neighborhood"
    assert not assess_overlap(np.zeros_like(face), face, {}, "glasses", p)["accepted"]
    assert not assess_overlap(face, face, {0: region}, "glasses", p)["accepted"]
    mask[:20] = True
    assert assess_overlap(mask, face, {}, "bottle", p)["acceptance_reason"] == "standard"


def test_missing_outside_or_nonfinite_keypoints_cannot_enable_exception():
    face = np.ones((100, 100), bool)
    points = np.array([[20,20],[-1,50],[np.nan,3],[40,40],[500,500]])
    regions = keypoint_regions(face.shape, face, [50,50,40,40,0], points, [0,2,2,2,1], ObjectOverlapPolicy())
    assert set(regions) == {3}


@pytest.mark.parametrize("raw", [{"version":"bad"}, {"route_probabilities":[.1,.1,.4,.5]}, {"minimum_local_coverage":0}, {"maximum_attempts":0}, {"keypoint_slots":[5]}, {"route_probabilities":[float("nan"),0,0,0]}, {"typo":1}])
def test_policy_rejects_invalid_config(raw):
    with pytest.raises(ValueError):
        ObjectOverlapPolicy.from_mapping(raw)


@pytest.fixture
def cfg(tmp_path):
    manifest = build_materials(tmp_path)
    return PairedOcclusionConfig.from_mapping(dict(appearance="object", material_manifest=str(manifest),
        material_manifest_sha256=sha256(manifest), object_overlap_policy={}))


@pytest.mark.parametrize("route", range(4))
def test_each_sampling_route_is_deterministic_and_preserves_clean_and_states(cfg, route):
    weights = [0.,0.,0.,0.]; weights[route] = 1.
    cfg = replace(cfg, object_overlap_policy={"route_probabilities":weights})
    image = np.full((128,128,3), 127, np.uint8)
    points = np.array([[44,48],[84,48],[64,64],[48,84],[80,84]],float)
    states = np.array([2,2,2,1,0])
    args = dict(face_ellipse=[64,64,48,40,0], keypoints=points, keypoint_visibility=states, config=cfg, force_kind="boundary")
    a = apply_material_occlusion(image, **args, rng=random.Random(53))
    b = apply_material_occlusion(image, **args, rng=random.Random(53))
    for x,y in zip(a[:3], b[:3]): np.testing.assert_array_equal(x,y)
    assert a[3] == b[3]
    out, mask, updated, meta = a
    assert meta["applied"]
    assert meta["kind"] == "object"  # No false boundary/truncation labels.
    if route < 2: assert 0 < meta["face_overlap"] < .1
    if route == 1: assert meta["qualifying_keypoint_slots"]
    if route == 2: assert .1 <= meta["face_overlap"] < .2
    if route == 3: assert .2 <= meta["face_overlap"] <= .4
    assert updated[-1] == 0 and updated[3] == 1
    np.testing.assert_array_equal(out[~mask],image[~mask])
    assert np.all(image == 127)
    assert set(np.unique(mask)) <= {False,True}


def test_unplaceable_request_is_explicit_noop_not_solid_fallback(cfg):
    cfg = replace(cfg, object_overlap_policy={"route_probabilities":[0,1,0,0]})
    image = np.zeros((64,64,3),np.uint8)
    states = np.zeros(5,int)
    out, mask, updated, meta = apply_material_occlusion(image, face_ellipse=[32,32,20,20,0],
        keypoints=np.zeros((5,2)), keypoint_visibility=states, config=cfg, rng=random.Random(7),force_kind="irregular")
    assert not meta["applied"] and meta["reason"] == "no_eligible_keypoint"
    assert not mask.any()
    np.testing.assert_array_equal(out,image)
    np.testing.assert_array_equal(updated,states)


def test_opt_in_does_not_change_legacy_default():
    assert PairedOcclusionConfig.from_mapping({}).object_overlap_policy is None
    with pytest.raises(ValueError, match="requires object"):
        PairedOcclusionConfig.from_mapping({"object_overlap_policy":{}})


def test_distribution_check_rejects_light_bias_and_skip_bias():
    from scripts.audit_occlusion_distribution import summarize, distribution_checks
    p = ObjectOverlapPolicy()
    rows = [dict(applied=True, face_overlap=.05, requested_route="material_exception") for _ in range(9)]
    rows += [dict(applied=True, face_overlap=.3, requested_route="strong")]
    check = distribution_checks(summarize(rows, p), p)
    assert not check["light_fraction_within_budget"]
    assert not check["strong_fraction_sufficient"]
    rows = [dict(applied=False, reason="placement_budget_exhausted")]
    assert not distribution_checks(summarize(rows,p),p)["skip_fraction_within_budget"]


def test_placement_budget_exhaustion_returns_recorded_noop(cfg):
    cfg = replace(cfg, object_overlap_policy={"route_probabilities":[0,0,0,1], "side_face_factors":[.01,.01], "maximum_attempts":1})
    image=np.full((128,128,3),127,np.uint8)
    out, mask, states, meta = apply_material_occlusion(image, face_ellipse=[64,64,48,40,0],
        keypoints=np.full((5,2),64), keypoint_visibility=np.full(5,2),config=cfg,rng=random.Random(1),force_kind="rectangle")
    assert not meta["applied"] and meta["reason"] == "placement_budget_exhausted"
    assert meta["attempts"] == 1 and not mask.any()
    np.testing.assert_array_equal(out,image)
    assert (states == 2).all()
