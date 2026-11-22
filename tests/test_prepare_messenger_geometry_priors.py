from argparse import Namespace
import json

import pytest
import torch

from scripts.prepare_messenger_geometry_priors import prepare, sha256
from keypoint_comparison1.tests.test_geometry_prior_recovery import metadata


def fixture_args(tmp_path):
    paths = {}
    for name in ("annotations", "backbone_checkpoint", "detector_checkpoint"):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    meta = metadata()
    meta.update(annotation_sha256=sha256(paths["annotations"]), backbone_sha256=sha256(paths["backbone_checkpoint"]),
                detector_sha256=sha256(paths["detector_checkpoint"]))
    torch.manual_seed(42)
    # Identical geometry deliberately cannot support k>1 routing; report it.
    center = torch.randn(1, 8, 2, 2)
    cache = dict(metadata=meta, features=center + torch.randn(20, 8, 2, 2) * .01,
                 geometry=torch.tensor([[.1, 2., 0.]] * 20),
                 records=[dict(record_id=str(i), source_video_id="a" if i < 12 else "b",
                               split="train", clean_eligible=True) for i in range(20)])
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(dict(schema="geometry-prior-clean-inventory-v1",
        status="inventory_prepared", split="train",
        provenance=dict(adapter=dict(sha256=sha256(paths["annotations"]))),
        partition=dict(fit_video_ids=["a"], calibration_video_ids=["b"]),
        records=[dict(r, partition="fit" if r["source_video_id"] == "a" else "calibration")
                 for r in cache["records"]])))
    cache["metadata"]["clean_inventory_sha256"] = sha256(inventory)
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    return Namespace(cache=cache_path, inventory=inventory, **paths, calibration_video=["b"], output=tmp_path / "priors", seed=42, quantile=.05)


def test_prepare_records_all_seven_without_fabricated_pass(tmp_path):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = fixture_args(tmp_path)
        report = prepare(args)
        assert report["status"] == "prepared_not_evaluated"
        assert [row["k"] for row in report["candidates"]] == list(range(1, 8))
        assert report["candidates"][0]["status"] == "prepared"
        assert all(row["status"] == "unsupported" for row in report["candidates"][1:])
        assert not (args.output / "success.json").exists()
        assert json.loads((args.output / "preparation_report.json").read_text())["test"] == "not run"
        with pytest.raises(FileExistsError):
            prepare(args)
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.parametrize("fault", ["test", "unclean", "hash", "video", "metadata_split", "empty_calibration",
    "inventory_hash", "unknown_record", "wrong_record_video", "valid_but_changed_partition"])
def test_prepare_rejects_bad_cache_before_writing(tmp_path, fault):
    args = fixture_args(tmp_path)
    cache = torch.load(args.cache, weights_only=True)
    if fault == "test":
        cache["records"][0]["split"] = "test"
    elif fault == "unclean":
        cache["records"][0]["clean_eligible"] = False
    elif fault == "hash":
        cache["metadata"]["backbone_sha256"] = "f" * 64
    elif fault == "metadata_split":
        cache["metadata"]["split"] = "val"
    elif fault == "empty_calibration":
        args.calibration_video = []
    elif fault == "inventory_hash":
        cache["metadata"]["clean_inventory_sha256"] = "f" * 64
    elif fault == "unknown_record":
        cache["records"][0]["record_id"] = "outside-clean-inventory"
    elif fault == "wrong_record_video":
        cache["records"][0]["source_video_id"] = "b"
    elif fault == "valid_but_changed_partition":
        args.calibration_video = ["a"]
    else:
        args.calibration_video = ["missing-video"]
    torch.save(cache, args.cache)
    with pytest.raises(ValueError):
        prepare(args)
    assert not args.output.exists()
