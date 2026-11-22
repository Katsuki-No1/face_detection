"""Deterministic source-video-preserving split helpers for face annotations.

Migration only: seeds, objective weights, tie behavior and forced-test groups
are unchanged. Importing this module does not rebuild any frozen dataset.
"""

import random
from collections import Counter, defaultdict
from typing import Any

from src.data.face_annotations import (
    KEYPOINT_NAMES, assign_keypoints, infer_image_stem, point_inside_image,
)

SPLITS = ("train", "val", "test")
SPLIT_FEATURE_WEIGHTS = {
    "groups": 1.0,
    "images": 8.0,
    "face_positive_images": 2.0,
    "face_negative_images": 2.0,
    "head_positive_images": 2.0,
    "head_negative_images": 2.0,
    "head_instances": 5.0,
    "face_instances": 4.0,
    "visible_keypoints": 3.0,
    "complete_keypoint_faces": 2.0,
    "incomplete_keypoint_faces": 2.0,
    "occluded_keypoints": 1.0,
    "head_count_0": 1.5,
    "head_count_1": 1.5,
    "head_count_2": 1.5,
    "head_count_3plus": 1.5,
    "face_count_0": 1.0,
    "face_count_1": 1.0,
    "face_count_2": 1.0,
    "face_count_3plus": 1.0,
    "head_area_tiny": 1.5,
    "head_area_small": 1.5,
    "head_area_medium": 1.5,
    "head_area_large": 1.5,
    "face_area_tiny": 1.0,
    "face_area_small": 1.0,
    "face_area_medium": 1.0,
    "face_area_large": 1.0,
    **{f"visible_keypoint_{name}": 1.5 for name in KEYPOINT_NAMES},
    **{f"missing_keypoint_{name}": 1.0 for name in KEYPOINT_NAMES},
    **{f"keypoint_count_{count}": 1.0 for count in range(len(KEYPOINT_NAMES) + 1)},
}


def split_group(record: dict[str, Any]) -> str:
    image_stem = str(record.get("image_stem", "")).strip()
    if image_stem:
        return image_stem
    inferred = infer_image_stem(str(record.get("image_name", ""))).strip()
    if inferred:
        return inferred
    annotation_kind = str(record.get("annotation_kind", "")).strip()
    return f"unknown::{annotation_kind}"


def _ratio_map(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    return {split: value / total for split, value in ratios.items()}


def _count_bin(prefix: str, count: int) -> str:
    suffix = str(count) if count < 3 else "3plus"
    return f"{prefix}_count_{suffix}"


def _area_bin(shape: dict[str, Any], width: int, height: int) -> str:
    bbox = shape.get("bbox", {})
    area = max(0.0, float(bbox.get("width", 0.0))) * max(
        0.0, float(bbox.get("height", 0.0))
    )
    image_area = max(1.0, float(width * height))
    ratio = area / image_area
    if ratio < 0.0025:
        return "tiny"
    if ratio < 0.01:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


def record_split_features(record: dict[str, Any]) -> Counter[str]:
    features: Counter[str] = Counter(images=1)
    heads = list(record.get("heads", []))
    faces = list(record.get("faces", []))
    head_count = len(heads)
    face_count = len(faces)
    features["head_positive_images" if heads else "head_negative_images"] += 1
    features["face_positive_images" if faces else "face_negative_images"] += 1
    features["head_instances"] += head_count
    features["face_instances"] += face_count
    features[_count_bin("head", head_count)] += 1
    features[_count_bin("face", face_count)] += 1

    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    for shape in heads:
        features[f"head_area_{_area_bin(shape, width, height)}"] += 1
    for shape in faces:
        features[f"face_area_{_area_bin(shape, width, height)}"] += 1

    if faces:
        face_bboxes = [face["bbox"] for face in faces]
        for face_index in range(len(faces)):
            keypoints = assign_keypoints(record, face_bboxes, face_index)
            visible = []
            for name, point in keypoints.items():
                if point.get("v", 0) > 0 and point_inside_image(point, width, height):
                    visible.append(point)
                    features[f"visible_keypoint_{name}"] += 1
                else:
                    features[f"missing_keypoint_{name}"] += 1
            features["visible_keypoints"] += len(visible)
            features[f"keypoint_count_{len(visible)}"] += 1
            features["occluded_keypoints"] += sum(
                bool(point.get("occluded", False)) for point in visible
            )
            if len(visible) == len(KEYPOINT_NAMES):
                features["complete_keypoint_faces"] += 1
            else:
                features["incomplete_keypoint_faces"] += 1
    return features


def group_split_features(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[split_group(record)].update(record_split_features(record))
    for features in grouped.values():
        features["groups"] = 1
    return dict(grouped)


def split_objective(
    actual: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
) -> float:
    score = 0.0
    for feature, weight in SPLIT_FEATURE_WEIGHTS.items():
        total = float(totals.get(feature, 0.0))
        if total <= 0:
            continue
        for split in SPLITS:
            target_fraction = ratios[split]
            if target_fraction <= 0:
                score += weight * (float(actual[split].get(feature, 0.0)) / total) ** 2
                continue
            actual_fraction = float(actual[split].get(feature, 0.0)) / total
            relative_error = (actual_fraction - target_fraction) / target_fraction
            score += weight * relative_error * relative_error
    return score


def _assignment_counters(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
) -> dict[str, Counter[str]]:
    actual = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        actual[split].update(group_features[group])
    return actual


def _greedy_balanced_assignment(
    groups: list[str],
    forced_test: set[str],
    group_features: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, Counter[str]], float]:
    assignment = {group: "test" for group in forced_test}
    actual = _assignment_counters(assignment, group_features)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda group: (
            -group_features[group]["images"],
            rng.random(),
        )
    )
    enabled_splits = [split for split in SPLITS if ratios[split] > 0]
    for group in shuffled:
        choices = []
        for split in enabled_splits:
            actual[split].update(group_features[group])
            score = split_objective(actual, totals, ratios)
            actual[split].subtract(group_features[group])
            choices.append((score, rng.random(), split))
        _, _, selected = min(choices)
        assignment[group] = selected
        actual[selected].update(group_features[group])
    return assignment, actual, split_objective(actual, totals, ratios)


def _improve_balanced_assignment(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: dict[str, float],
    forced_test: set[str],
    score: float,
) -> tuple[dict[str, str], dict[str, Counter[str]], float]:
    actual = _assignment_counters(assignment, group_features)
    movable = sorted(set(assignment) - forced_test)
    enabled_splits = [split for split in SPLITS if ratios[split] > 0]

    for _ in range(20):
        changed = False
        group_counts = Counter(assignment.values())
        for group in movable:
            source = assignment[group]
            if group_counts[source] <= 1:
                continue
            best = (score, source)
            for destination in enabled_splits:
                if destination == source:
                    continue
                actual[source].subtract(group_features[group])
                actual[destination].update(group_features[group])
                candidate = split_objective(actual, totals, ratios)
                actual[destination].subtract(group_features[group])
                actual[source].update(group_features[group])
                if candidate < best[0] - 1e-12:
                    best = (candidate, destination)
            if best[1] != source:
                actual[source].subtract(group_features[group])
                actual[best[1]].update(group_features[group])
                assignment[group] = best[1]
                group_counts[source] -= 1
                group_counts[best[1]] += 1
                score = best[0]
                changed = True
        if not changed:
            break

    for _ in range(20):
        best_swap: tuple[float, str, str] | None = None
        for left_index, left in enumerate(movable):
            left_split = assignment[left]
            for right in movable[left_index + 1 :]:
                right_split = assignment[right]
                if left_split == right_split:
                    continue
                actual[left_split].subtract(group_features[left])
                actual[right_split].subtract(group_features[right])
                actual[left_split].update(group_features[right])
                actual[right_split].update(group_features[left])
                candidate = split_objective(actual, totals, ratios)
                actual[right_split].subtract(group_features[left])
                actual[left_split].subtract(group_features[right])
                actual[right_split].update(group_features[right])
                actual[left_split].update(group_features[left])
                if candidate < score - 1e-12 and (
                    best_swap is None or candidate < best_swap[0]
                ):
                    best_swap = (candidate, left, right)
        if best_swap is None:
            break
        score, left, right = best_swap
        left_split = assignment[left]
        right_split = assignment[right]
        actual[left_split].subtract(group_features[left])
        actual[right_split].subtract(group_features[right])
        actual[left_split].update(group_features[right])
        actual[right_split].update(group_features[left])
        assignment[left], assignment[right] = right_split, left_split
    return assignment, actual, score


def _legacy_random_group_assignment(
    groups: list[str],
    forced_test: set[str],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * ratios["train"])
    val_count = round(len(shuffled) * ratios["val"])
    test_count = len(shuffled) - train_count - val_count
    if ratios["val"] > 0 and val_count == 0 and len(shuffled) >= 2:
        val_count = 1
        if test_count > 0:
            test_count -= 1
        else:
            train_count = max(1, train_count - 1)
    if ratios["test"] > 0 and test_count == 0 and len(shuffled) >= 3 and not forced_test:
        test_count = 1
        train_count = max(1, train_count - 1)
    train_count = max(0, min(train_count, len(shuffled)))
    val_count = max(0, min(val_count, len(shuffled) - train_count))
    train_cut = train_count
    val_cut = train_cut + val_count
    assignment = {group: "train" for group in shuffled[:train_cut]}
    assignment.update({group: "val" for group in shuffled[train_cut:val_cut]})
    assignment.update({group: "test" for group in shuffled[val_cut:]})
    assignment.update({group: "test" for group in forced_test})
    return assignment


def build_split_map(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test_kind_contains: list[str],
    strategy: str = "balanced",
    search_restarts: int = 256,
) -> dict[str, str]:
    ratios = _ratio_map(train_ratio, val_ratio, test_ratio)
    forced_test = {
        split_group(record)
        for record in records
        if any(marker in record.get("annotation_kind", "") for marker in test_kind_contains)
    }
    groups = sorted({split_group(record) for record in records} - forced_test)
    if strategy == "random-groups":
        return _legacy_random_group_assignment(groups, forced_test, seed, ratios)
    if strategy != "balanced":
        raise ValueError(f"Unsupported split strategy: {strategy}")
    if search_restarts <= 0:
        raise ValueError("split search restarts must be positive")

    group_features = group_split_features(records)
    totals = Counter()
    for features in group_features.values():
        totals.update(features)
    rng = random.Random(seed)
    best: tuple[float, dict[str, str]] | None = None
    for _ in range(search_restarts):
        assignment, _, score = _greedy_balanced_assignment(
            groups,
            forced_test,
            group_features,
            totals,
            ratios,
            rng,
        )
        if best is None or score < best[0]:
            best = (score, assignment)
    assert best is not None
    assignment, _, _ = _improve_balanced_assignment(
        dict(best[1]),
        group_features,
        totals,
        ratios,
        forced_test,
        best[0],
    )
    return assignment


def split_balance_report(
    records: list[dict[str, Any]],
    split_map: dict[str, str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strategy: str,
    seed: int,
    test_kind_contains: list[str],
) -> dict[str, Any]:
    ratios = _ratio_map(train_ratio, val_ratio, test_ratio)
    group_features = group_split_features(records)
    actual = _assignment_counters(split_map, group_features)
    totals = Counter()
    for features in group_features.values():
        totals.update(features)
    split_rows: dict[str, Any] = {}
    for split in SPLITS:
        counts = {feature: int(actual[split].get(feature, 0)) for feature in SPLIT_FEATURE_WEIGHTS}
        proportions = {
            feature: (
                float(counts[feature]) / float(totals[feature]) if totals.get(feature, 0) else None
            )
            for feature in SPLIT_FEATURE_WEIGHTS
        }
        split_rows[split] = {
            "target_fraction": ratios[split],
            "counts": counts,
            "fractions_of_dataset": proportions,
        }
    forced_test = sorted(
        group
        for group in split_map
        if any(
            split_group(record) == group
            and any(marker in str(record.get("annotation_kind", "")) for marker in test_kind_contains)
            for record in records
        )
    )
    group_concentration: dict[str, Any] = {}
    for feature in SPLIT_FEATURE_WEIGHTS:
        total = int(totals.get(feature, 0))
        ranked = sorted(
            (
                (group, int(features.get(feature, 0)))
                for group, features in group_features.items()
                if features.get(feature, 0) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        group_concentration[feature] = {
            "nonzero_groups": len(ranked),
            "largest_group_fraction": (float(ranked[0][1]) / total if ranked and total else None),
            "top_two_group_fraction": (
                float(sum(count for _, count in ranked[:2])) / total if ranked and total else None
            ),
            "top_groups": [
                {"group": group, "count": count}
                for group, count in ranked[:5]
            ],
        }
    return {
        "strategy": strategy,
        "seed": seed,
        "group_key": "image_stem (annotation_kind is deliberately excluded)",
        "ratios": ratios,
        "objective": split_objective(actual, totals, ratios),
        "feature_weights": SPLIT_FEATURE_WEIGHTS,
        "totals": {feature: int(totals.get(feature, 0)) for feature in SPLIT_FEATURE_WEIGHTS},
        "group_concentration": group_concentration,
        "forced_test_markers": list(test_kind_contains),
        "forced_test_groups": forced_test,
        "splits": split_rows,
    }
