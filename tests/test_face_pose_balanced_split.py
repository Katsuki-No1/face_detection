from __future__ import annotations

from collections import Counter

from scripts_YOLO.convert_annotations_to_yolo_pose import (
    build_split_map,
    split_balance_report,
    split_group,
)


def _shape(width: float = 30.0, height: float = 30.0) -> dict:
    return {
        "bbox": {
            "xtl": 10.0,
            "ytl": 10.0,
            "xbr": 10.0 + width,
            "ybr": 10.0 + height,
            "width": width,
            "height": height,
        }
    }


def _record(
    group: str,
    index: int,
    *,
    head_count: int,
    has_face: bool,
    visible_keypoints: int,
    annotation_kind: str = "task",
) -> dict:
    faces = [_shape(50.0, 50.0)] if has_face else []
    keypoint_names = ("e1", "e2", "n", "m1", "m2")
    skeletons = []
    if has_face:
        skeletons.append(
            {
                "points": [
                    {
                        "label": name,
                        "x": 20.0 + point_index * 4.0,
                        "y": 25.0 + point_index * 3.0,
                        "outside": False,
                        "occluded": point_index == 4,
                    }
                    for point_index, name in enumerate(keypoint_names[:visible_keypoints])
                ]
            }
        )
    return {
        "annotation_kind": annotation_kind,
        "image_stem": group,
        "image_name": f"{group}_frame_{index:05d}.jpg",
        "width": 100,
        "height": 100,
        "heads": [_shape(20.0 + head_index, 25.0) for head_index in range(head_count)],
        "faces": faces,
        "simple_landmarks": [],
        "skeleton_landmarks": skeletons,
    }


def _records() -> list[dict]:
    records = []
    for group_index in range(20):
        for image_index in range(10):
            records.append(
                _record(
                    f"video_{group_index:02d}",
                    image_index,
                    head_count=group_index % 4,
                    has_face=group_index % 5 != 0,
                    visible_keypoints=3 if group_index % 3 == 0 else 5,
                )
            )
    return records


def test_split_group_is_stable_across_annotation_tasks() -> None:
    first = _record(
        "same_source_video",
        1,
        head_count=1,
        has_face=True,
        visible_keypoints=5,
        annotation_kind="task_a",
    )
    second = _record(
        "same_source_video",
        2,
        head_count=2,
        has_face=True,
        visible_keypoints=3,
        annotation_kind="task_b",
    )
    assert split_group(first) == split_group(second) == "same_source_video"


def test_balanced_split_is_deterministic_and_balances_all_signal_families() -> None:
    records = _records()
    kwargs = {
        "seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "test_kind_contains": [],
        "strategy": "balanced",
        "search_restarts": 64,
    }
    first = build_split_map(records, **kwargs)
    second = build_split_map(records, **kwargs)
    assert first == second
    assert Counter(first.values()) == {"train": 16, "val": 2, "test": 2}
    assert all(len({first[split_group(record)] for record in records if split_group(record) == group}) == 1 for group in first)

    report = split_balance_report(
        records,
        first,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        strategy="balanced",
        seed=42,
        test_kind_contains=[],
    )
    for feature in (
        "images",
        "head_instances",
        "face_instances",
        "visible_keypoints",
        "complete_keypoint_faces",
        "incomplete_keypoint_faces",
    ):
        assert feature in report["totals"]
        assert report["totals"][feature] > 0
    assert report["splits"]["train"]["counts"]["images"] == 160
    assert report["splits"]["val"]["counts"]["images"] == 20
    assert report["splits"]["test"]["counts"]["images"] == 20


def test_balanced_objective_improves_over_legacy_random_groups() -> None:
    records = _records()
    reports = {}
    for strategy in ("balanced", "random-groups"):
        split_map = build_split_map(
            records,
            seed=42,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            test_kind_contains=[],
            strategy=strategy,
            search_restarts=64,
        )
        reports[strategy] = split_balance_report(
            records,
            split_map,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            strategy=strategy,
            seed=42,
            test_kind_contains=[],
        )
    assert reports["balanced"]["objective"] < reports["random-groups"]["objective"]


def test_forced_test_marker_keeps_the_entire_video_in_test() -> None:
    records = _records()
    forced_group = "video_00"
    for record in records:
        if record["image_stem"] == forced_group:
            record["annotation_kind"] = "external_test"
    split_map = build_split_map(
        records,
        seed=7,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        test_kind_contains=["external_test"],
        strategy="balanced",
        search_restarts=32,
    )
    assert split_map[forced_group] == "test"
