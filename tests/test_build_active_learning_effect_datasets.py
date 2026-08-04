from __future__ import annotations

from scripts.build_active_learning_effect_datasets import (
    filter_training_eligible_records,
    select_uniform_records,
    training_eligibility,
)


def _records(count: int) -> list[dict[str, int]]:
    return [{"index": index} for index in range(count)]


def test_uniform_selection_is_seeded_and_order_stable() -> None:
    records = _records(100)
    first = select_uniform_records(records, count=20, seed=20260802)
    second = select_uniform_records(records, count=20, seed=20260802)
    different = select_uniform_records(records, count=20, seed=20260803)

    assert first == second
    assert first != different
    assert [row["index"] for row in first] == sorted(row["index"] for row in first)
    assert len({row["index"] for row in first}) == 20


def test_uniform_selection_does_not_mutate_pool_records() -> None:
    records = _records(10)
    selected = select_uniform_records(records, count=4, seed=7)
    selected[0]["index"] = -1

    assert all(row["index"] >= 0 for row in records)


def test_training_eligibility_keeps_negative_and_valid_positive_records() -> None:
    negative = {"faces": []}
    positive = {
        "width": 100,
        "height": 80,
        "faces": [{"bbox": {"xtl": 10, "ytl": 10, "xbr": 40, "ybr": 50}}],
    }

    assert training_eligibility(negative) == (True, [])
    assert training_eligibility(positive) == (True, [])


def test_converter_invalid_positive_is_excluded_with_provenance() -> None:
    invalid = {
        "width": 100,
        "height": 80,
        "faces": [{"bbox": {"xtl": 110, "ytl": 10, "xbr": 120, "ybr": 20}}],
        "image_name": "invalid.jpg",
        "image_stem": "video-a",
        "image_path": "/data/invalid.jpg",
        "source_annotation_zip": "/data/task.zip",
        "content_sha256": "abc",
    }
    valid_negative = {
        "faces": [],
        "image_name": "negative.jpg",
        "image_stem": "video-b",
        "image_path": "/data/negative.jpg",
        "source_annotation_zip": "/data/task.zip",
        "content_sha256": "def",
    }

    eligible, excluded = filter_training_eligible_records(
        "test",
        [invalid, valid_negative],
        {"expected_eligible_frames": 1, "expected_excluded_frames": 1},
    )

    assert eligible == [valid_negative]
    assert excluded[0]["image_name"] == "invalid.jpg"
    assert excluded[0]["reasons"] == ["invalid_clipped_face_bbox"]
