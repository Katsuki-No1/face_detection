import copy

import pytest

from scripts.prepare_messenger_prior_inventory import prepare_inventory


def fixture():
    images = [dict(id=i, file_name=f"train/{i}.jpg", width=100, height=100,
                   source_video_id=f"train_video_{i}") for i in range(1, 5)]
    canonical = {"train": dict(images=images)}
    for split in ["val", "test"]:
        canonical[split] = dict(images=[dict(id=1, file_name=f"{split}/1.jpg",
            width=100, height=100, source_video_id=f"{split}_video")])
    adapter = dict(images=[dict(id=im["id"], file_name=im["file_name"], width=100, height=100,
        metadata=dict(source_video_id=im["source_video_id"], split="train")) for im in images],
        annotations=[dict(id=i, image_id=i, category_id=2, bbox=[30,30,40,40],
            ellipse=[50,50,20,10,0], keypoints=[50,50,2]*5,
            attributes=dict(face_occluded=False)) for i in range(1,5)])
    return adapter, canonical


def test_partition_is_video_disjoint_deterministic_and_independent_of_record_order():
    adapter, canonical = fixture()
    result = prepare_inventory(adapter, canonical)
    partition = result["partition"]
    assert len(partition["fit_video_ids"]) == 3
    assert len(partition["calibration_video_ids"]) == 1
    assert not set(partition["fit_video_ids"]) & set(partition["calibration_video_ids"])
    adapter["images"].reverse()
    adapter["annotations"].reverse()
    assert prepare_inventory(adapter, canonical) == result
    assert result["gate_success"] is False


@pytest.mark.parametrize("change,reason", [
    (lambda a: a.pop("attributes"), "contour_occluded_or_unknown"),
    (lambda a: a["attributes"].update(face_occluded=True), "contour_occluded_or_unknown"),
    (lambda a: a["keypoints"].__setitem__(2, 0), "not_all_five_visible"),
    (lambda a: a["keypoints"].__setitem__(5, 1), "not_all_five_visible"),
    (lambda a: a["keypoints"].__setitem__(0, 100), "keypoint_outside_image"),
    (lambda a: a["keypoints"].__setitem__(0, float("nan")), "nonfinite_keypoints"),
    (lambda a: a["ellipse"].__setitem__(2, 0), "invalid_face_geometry"),
])
def test_clean_eligibility_never_uses_missing_or_occluded_as_clean(change, reason):
    adapter, canonical = fixture()
    change(adapter["annotations"][0])
    result = prepare_inventory(adapter, canonical)
    assert result["counts"]["excluded"] == {reason: 1}
    assert "1" not in {r["record_id"] for r in result["records"]}


def test_fail_closed_on_leakage_adapter_mismatch_duplicates_and_orphans():
    adapter, canonical = fixture()
    leaked = copy.deepcopy(canonical)
    leaked["val"]["images"][0]["source_video_id"] = "train_video_1"
    with pytest.raises(ValueError, match="leakage"):
        prepare_inventory(adapter, leaked)
    changed = copy.deepcopy(adapter)
    changed["images"][0]["metadata"]["source_video_id"] = "unknown_video"
    with pytest.raises(ValueError, match="differs"):
        prepare_inventory(changed, canonical)
    changed = copy.deepcopy(adapter)
    changed["annotations"].append(copy.deepcopy(changed["annotations"][0]))
    with pytest.raises(ValueError, match="Duplicate annotation"):
        prepare_inventory(changed, canonical)
    adapter["annotations"][0]["image_id"] = 999
    with pytest.raises(ValueError, match="Orphan"):
        prepare_inventory(adapter, canonical)
