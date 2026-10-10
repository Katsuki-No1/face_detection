"""Opt-in binary-object occlusion policy; legacy material sampling is unchanged."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math

import cv2
import numpy as np

from src.data.landmark_occlusion import ellipse_region_mask

SLOTS = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
ROUTES = ("material_exception", "keypoint_exception", "medium", "strong")


@dataclass(frozen=True)
class ObjectOverlapPolicy:
    version: str = "conditional_v1"
    exempt_material_ids: tuple = ("glasses",)
    route_probabilities: tuple = (.10, .10, .40, .40)
    keypoint_slots: tuple = (0, 1, 2, 3, 4)
    local_radius_face_fraction: float = .08
    minimum_local_coverage: float = .20
    side_face_factors: tuple = (.20, 1.20)
    maximum_attempts: int = 128
    maximum_skip_fraction: float = .05
    maximum_light_fraction: float = .30
    minimum_strong_fraction: float = .25

    @classmethod
    def from_mapping(cls, raw):
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown object overlap policy keys: {sorted(unknown)}")
        values = dict(raw)
        for key in ("exempt_material_ids", "route_probabilities", "keypoint_slots", "side_face_factors"):
            if key in values:
                values[key] = tuple(values[key])
        obj = cls(**values)
        if obj.version != "conditional_v1":
            raise ValueError("Unsupported object overlap policy version")
        if not obj.exempt_material_ids or any(not isinstance(x, str) or not x for x in obj.exempt_material_ids):
            raise ValueError("Explicit exempt material ids are required")
        if len(obj.route_probabilities) != 4 or any(not math.isfinite(x) or x < 0 for x in obj.route_probabilities) or abs(sum(obj.route_probabilities)-1) > 1e-8:
            raise ValueError("Four finite route probabilities must sum to one")
        if not obj.keypoint_slots or any(type(x) is not int or x not in range(5) for x in obj.keypoint_slots):
            raise ValueError("keypoint_slots must reference the five semantic slots")
        if len(set(obj.keypoint_slots)) != len(obj.keypoint_slots):
            raise ValueError("Duplicate keypoint slots")
        for key in ("local_radius_face_fraction", "minimum_local_coverage", "maximum_skip_fraction", "maximum_light_fraction", "minimum_strong_fraction"):
            if not 0 < getattr(obj, key) < 1:
                raise ValueError(f"Invalid {key}")
        if len(obj.side_face_factors) != 2 or not 0 < obj.side_face_factors[0] <= obj.side_face_factors[1] <= 2:
            raise ValueError("Invalid aspect-preserving object size bounds")
        if type(obj.maximum_attempts) is not int or not 1 <= obj.maximum_attempts <= 1024:
            raise ValueError("Invalid placement attempt budget")
        return obj


def keypoint_regions(shape, face_mask, face_ellipse, points, states, policy):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    radius = policy.local_radius_face_fraction * 2 * max(face_ellipse[2:4])
    regions = {}
    for i in policy.keypoint_slots:
        x, y = points[i]
        if states[i] <= 0 or not np.isfinite([x, y]).all() or not (0 <= x < shape[1] and 0 <= y < shape[0]):
            continue
        disk = ((xx-x)**2 + (yy-y)**2 <= radius**2) & face_mask
        if disk.any():
            regions[i] = disk
    return regions


def assess_overlap(mask, face_mask, regions, material_id, policy):
    """The upper cap applies to every route. Transparent pixels never count."""
    pixels = int((mask & face_mask).sum())
    face_pixels = int(face_mask.sum())
    overlap = pixels / max(1, face_pixels)
    coverage = {SLOTS[i]: float((mask & region).sum()/region.sum()) for i, region in regions.items()}
    local = [slot for slot, value in coverage.items() if value >= policy.minimum_local_coverage]
    reason = None
    if pixels > 0 and overlap <= .40:
        if overlap >= .10:
            reason = "standard"
        elif material_id in policy.exempt_material_ids:
            reason = "exempt_material"
        elif local:
            reason = "keypoint_neighborhood"
    return dict(accepted=reason is not None, acceptance_reason=reason, face_overlap=overlap,
                face_pixels=face_pixels, masked_face_pixels=pixels, local_coverage=coverage,
                qualifying_keypoint_slots=local)


def apply_conditional_object(image, *, face_ellipse, keypoints, keypoint_visibility,
                             config, rng, materials, requested_kind):
    policy = ObjectOverlapPolicy.from_mapping(config.object_overlap_policy)
    objects = [(a, rgba) for a, rgba in materials if a["kind"] == "object"]
    ids = {a["id"] for a, _ in objects}
    if not set(policy.exempt_material_ids) <= ids:
        raise ValueError("Exempt material id is missing from locked manifest")
    seed = int.from_bytes(hashlib.sha256(repr(rng.getstate()).encode()).digest()[:8], "big")
    route = rng.choices(ROUTES, weights=policy.route_probabilities, k=1)[0]
    eligible = [(a, rgba) for a, rgba in objects if route != "material_exception" or a["id"] in policy.exempt_material_ids]
    asset, rgba = rng.choice(eligible)
    # Preserve source aspect ratio; arbitrary quarter turns would turn glasses sideways.
    angle = rng.uniform(-20, 20)
    h, w = rgba.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1)
    nw = int(math.ceil(abs(matrix[0, 0])*w + abs(matrix[0, 1])*h))
    nh = int(math.ceil(abs(matrix[0, 0])*h + abs(matrix[0, 1])*w))
    matrix[:, 2] += [(nw-w)/2, (nh-h)/2]
    rgba = cv2.warpAffine(rgba, matrix, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))
    flipped = rng.random() < .5
    if flipped:
        rgba = np.ascontiguousarray(rgba[:, ::-1])
    face_mask = ellipse_region_mask(*image.shape[:2], face_ellipse)
    points, states = np.asarray(keypoints), np.asarray(keypoint_visibility)
    regions = keypoint_regions(image.shape[:2], face_mask, face_ellipse, points, states, policy)
    base = dict(policy_version=policy.version, requested=True, requested_route=route,
                requested_kind=requested_kind, kind="object", appearance="object", shape="object_cutout",
                material_id=asset["id"], material_sha256=asset["sha256"],
                manifest_sha256=config.material_manifest_sha256, material_seed=seed,
                rotation_degrees=angle, material_flip=flipped,
                coordinate_frame="pre_flip_crop_pixels_xy", mask_source="actual_binary_alpha",
                shape_confound=True)
    def skipped(reason, attempts):
        return image.copy(), np.zeros(image.shape[:2], bool), states.copy(), dict(
            base, applied=False, reason=reason, attempts=attempts, face_overlap=0.,
            masked_face_pixels=0, occluded_keypoint_indices=[])
    if not face_mask.any():
        return skipped("empty_face_region", 0)
    if route == "keypoint_exception" and not regions:
        return skipped("no_eligible_keypoint", 0)
    diameter = 2 * max(face_ellipse[2:4])
    for attempt in range(1, policy.maximum_attempts+1):
        side = diameter * rng.uniform(*policy.side_face_factors)
        scale = side/max(rgba.shape[:2])
        oh, ow = max(1, round(rgba.shape[0]*scale)), max(1, round(rgba.shape[1]*scale))
        sized = cv2.resize(rgba, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if regions and (route == "keypoint_exception" or rng.random() < config.landmark_anchor_probability):
            slot = rng.choice(list(regions))
            # Align an actual opaque pixel near the point, avoiding a transparent center.
            sy, sx = np.where(sized[..., 3] == 255)
            if not len(sx):
                continue
            j = rng.randrange(len(sx))
            jitter = policy.local_radius_face_fraction * diameter * .5
            x0 = round(points[slot][0] - sx[j] + rng.uniform(-jitter, jitter))
            y0 = round(points[slot][1] - sy[j] + rng.uniform(-jitter, jitter))
        else:
            slot = None
            x0 = round(face_ellipse[0] + rng.uniform(-.3, .3)*diameter - ow/2)
            y0 = round(face_ellipse[1] + rng.uniform(-.3, .3)*diameter - oh/2)
        x1, y1 = x0+ow, y0+oh
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(image.shape[1], x1), min(image.shape[0], y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue
        canvas = np.zeros((*image.shape[:2], 4), np.uint8)
        canvas[cy0:cy1, cx0:cx1] = sized[cy0-y0:cy1-y0, cx0-x0:cx1-x0]
        mask = canvas[..., 3] == 255
        check = assess_overlap(mask, face_mask, regions, asset["id"], policy)
        overlap = check["face_overlap"]
        ok = check["accepted"]
        if route == "material_exception":
            ok &= 0 < overlap < .10
        elif route == "keypoint_exception":
            ok &= 0 < overlap < .10 and bool(check["qualifying_keypoint_slots"])
            if ok:
                check["acceptance_reason"] = "keypoint_neighborhood"
        elif route == "medium":
            ok &= .10 <= overlap < .20
        else:
            ok &= .20 <= overlap <= .40
        if not ok:
            continue
        out, updated = image.copy(), states.copy()
        out[mask] = canvas[..., :3][mask]
        occluded = []
        for i, (point, state) in enumerate(zip(points, states)):
            if state <= 0 or not np.isfinite(point).all():
                continue
            x, y = np.rint(point).astype(int)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and mask[y, x]:
                updated[i] = 1
                occluded.append(i)
        return out, mask, updated, dict(base, **check, applied=True, attempts=attempt,
            bbox_xyxy=[x0, y0, x1, y1], anchor_keypoint_index=slot,
            support_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            occluded_keypoint_indices=occluded)
    return skipped("placement_budget_exhausted", policy.maximum_attempts)
