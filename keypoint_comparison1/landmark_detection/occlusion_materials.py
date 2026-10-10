"""Manifest-locked independent procedural textures and opaque object cutouts.

These are original synthetic assets, not photographs of natural occluders.
No train, validation or test pixels/annotations are used to make them.
"""
from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.data.landmark_occlusion import apply_landmark_occlusion_numpy, ellipse_region_mask


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def load_materials(manifest_path, expected_sha256):
    path = Path(manifest_path)
    if not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError("Missing or changed occlusion material manifest")
    manifest = json.loads(path.read_text())
    if manifest.get("source_split") != "independent_procedural" or manifest.get("dataset_pixels_used") is not False:
        raise ValueError("Materials must be independently generated without val/test pixels")
    if not manifest.get("usage_terms") or not manifest.get("generator_sha256"):
        raise ValueError("Material provenance/usage terms are missing")
    result = []
    for asset in manifest["assets"]:
        file = (path.parent / asset["path"]).resolve()
        if not file.is_relative_to(path.parent.resolve()) or sha256(file) != asset["sha256"]:
            raise ValueError("Material path or SHA256 mismatch")
        rgba = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or not np.isin(rgba[..., 3], [0, 255]).all():
            raise ValueError("Materials must have binary opaque support, no soft matte")
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        result.append((asset, rgba))
    if {a[0]["kind"] for a in result} != {"texture", "object"}:
        raise ValueError("Manifest needs both texture and object materials")
    return result


def apply_material_occlusion(image, *, face_ellipse, keypoints, keypoint_visibility, config, rng, force_kind):
    """Texture preserves the solid-arm mask RNG; object uses its actual support.

    All transforms are logged in pre-flip crop pixel coordinates. Dataset records
    the subsequent shared flip separately. No fallback to solid fill is allowed.
    """
    materials = load_materials(config.material_manifest, config.material_manifest_sha256)
    if config.object_overlap_policy is not None:
        from .occlusion_policy import apply_conditional_object
        return apply_conditional_object(image, face_ellipse=face_ellipse, keypoints=keypoints,
            keypoint_visibility=keypoint_visibility, config=config, rng=rng,
            materials=materials, requested_kind=force_kind)
    candidates = [(a, rgba) for a, rgba in materials if a["kind"] == config.appearance]
    # Separate stream: choosing/material-transforming a texture cannot alter mask geometry.
    seed = int.from_bytes(hashlib.sha256(repr(rng.getstate()).encode()).digest()[:8], "big")
    material_rng = random.Random(seed)
    asset, rgba = material_rng.choice(candidates)
    quarter_turns = material_rng.randrange(4)
    flip = bool(material_rng.randrange(2))
    rgba = np.rot90(rgba, quarter_turns)
    if flip:
        rgba = rgba[:, ::-1]
    face_mask = ellipse_region_mask(image.shape[0], image.shape[1], face_ellipse)
    for attempt in range(config.maximum_attempts):
        _, mask, _, metadata = apply_landmark_occlusion_numpy(image,
            face_ellipse=face_ellipse, keypoints=keypoints, keypoint_visibility=keypoint_visibility,
            config=config.landmark_mapping(), rng=rng, force=True, force_kind=force_kind)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        resized = cv2.resize(np.ascontiguousarray(rgba), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((*mask.shape, 4), dtype=np.uint8)
        canvas[y0:y1, x0:x1] = resized
        if config.appearance == "object":
            # Cutout alpha support, never its bounding rectangle or the proposed mask.
            mask = canvas[..., 3] == 255
        overlap = float((mask & face_mask).sum()) / max(1, int(face_mask.sum()))
        if not config.face_overlap[0] <= overlap <= config.face_overlap[1]:
            continue
        output = image.copy()
        output[mask] = canvas[..., :3][mask]
        states = np.asarray(keypoint_visibility).copy()
        occluded = []
        for i, point in enumerate(keypoints):
            x, y = np.rint(point).astype(int)
            if states[i] > 0 and 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
                states[i] = 1
                occluded.append(i)
        metadata = {k: v for k, v in metadata.items() if k != "fill_rgb"}
        metadata.update(applied=True, appearance=config.appearance, material_id=asset["id"],
            material_sha256=asset["sha256"], manifest_sha256=config.material_manifest_sha256,
            material_seed=seed, bbox_xyxy=[x0, y0, x1, y1], quarter_turns=quarter_turns,
            material_flip=flip, attempt=attempt, face_overlap=overlap,
            masked_face_pixels=int((mask & face_mask).sum()), occluded_keypoint_indices=occluded,
            support_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            mask_source="actual_binary_alpha" if config.appearance == "object" else "same_as_solid_control",
            shape_confound=config.appearance == "object", coordinate_frame="pre_flip_crop_pixels_xy")
        return output, mask, states, metadata
    raise RuntimeError("Unable to place material with the locked 10-40% overlap; no solid fallback")


def build_materials(output_dir, seed=20261004):
    """Small versioned original collection: 4 textures and 3 object silhouettes."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:128, :128]
    patterns = [(x // 8 + y // 8) % 2, (x // 5) % 2,
                np.sin(x * .2 + np.sin(y * .1)) > 0, rng.random((128, 128)) > .5]
    assets = []
    for index in range(7):
        kind = "texture" if index < 4 else "object"
        name = ["checker", "fabric_stripes", "grain", "noise", "leaf", "bottle", "glasses"][index]
        texture = np.where(patterns[index % 4][..., None], rng.integers(40, 220, 3), rng.integers(20, 190, 3)).astype(np.uint8)
        support = np.full((128, 128), 255, dtype=np.uint8) if kind == "texture" else np.zeros((128, 128), dtype=np.uint8)
        if name == "leaf":
            cv2.ellipse(support, (64, 64), (57, 30), -35, 0, 360, 255, -1)
        elif name == "bottle":
            cv2.rectangle(support, (49, 5), (78, 36), 255, -1)
            cv2.ellipse(support, (64, 65), (37, 42), 0, 180, 360, 255, -1)
            cv2.rectangle(support, (27, 62), (101, 121), 255, -1)
        elif name == "glasses":
            cv2.ellipse(support, (32, 64), (29, 24), 0, 0, 360, 255, 10)
            cv2.ellipse(support, (96, 64), (29, 24), 0, 0, 360, 255, 10)
            cv2.line(support, (59, 59), (69, 59), 255, 8)
        rgba = np.dstack((texture, support))
        path = directory / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        assets.append(dict(id=name, kind=kind, path=path.name, sha256=sha256(path), seed=seed,
                           source="original procedural drawing", matte="binary opaque"))
    manifest = dict(schema_version=1, source_split="independent_procedural", dataset_pixels_used=False,
        usage_terms="Original project-generated assets; may be used, modified and redistributed with this generator.",
        generator="keypoint_comparison1/landmark_detection/occlusion_materials.py", generator_sha256=sha256(__file__),
        seed=seed, limitation="Synthetic silhouettes, not a claim of realistic natural object occlusion", assets=assets)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
