"""Train-only face-only w48/s2 loader; keeps the real-mixed policy independent."""
import numpy as np
from .real_occlusion import RealOcclusionBank, identity
from scripts.prototype_cross_video_face_occlusion import choose_donor, place
from scripts.opaque_boundary_gradient import opaque_boundary_gradient


class RealFaceOcclusionBank(RealOcclusionBank):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data.get("face_only_policy") != "real_face_w48_s2_v1":
            raise ValueError("Face-only training requires a strict context manifest")
        self.contexts = self.data["recipient_contexts"]
        self.strict_faces = self.data["strict_faces"]

    def select(self, sample, rng, profile):
        key = identity(sample["image"]["file_name"], sample["annotation"]["ellipse"])
        target = self.contexts.get(key)
        if target is None:
            return None
        selected = choose_donor(target, self.strict_faces, rng, profile)
        if selected is None:
            return None
        distance, donor = selected
        row = self.images[donor["file"]]
        x0, y0, x1, y1 = donor["box"]
        patch = self.read(row)[y0:y1, x0:x1].copy()
        ellipse = np.asarray(donor["ellipse"], dtype=float).copy()
        ellipse[:2] -= [x0, y0]
        return patch, ellipse, dict(donor_file=donor["file"], donor_video=row["video"],
            recipient_video=target["video"], donor_annotation_id=donor["id"],
            donor_source_box=donor["box"], context_distance=distance,
            context_delta=(np.asarray(donor["descriptor"])-target["descriptor"]).tolist())


def apply_real_face(image, *, sample, face_ellipse, keypoints, keypoint_visibility, bank, rng):
    states = np.asarray(keypoint_visibility, dtype=np.int64).copy()
    points = np.asarray(keypoints, dtype=float).copy()
    inside = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)
    inside &= (points[:, 0] < image.shape[1]-.5) & (points[:, 1] < image.shape[0]-.5)
    states[~inside] = 0
    points[states == 0] = 0
    route = rng.choices(["light", "medium", "strong"], [.2, .4, .4])[0]
    profile = rng.random() < .5
    meta = dict(policy_version="real_face_w48_s2_v2_light_area_only", appearance="real_face",
        requested_family="face", requested_route=route, requested_profile=profile,
        pair_kind="object", kind="object", manifest_sha256=bank.manifest_sha256,
        gradient_width=48, gradient_strength=2, alpha=False,
        coordinate_frame="pre_flip_crop_pixels_xy")
    def skip(reason):
        return image.copy(), np.zeros(image.shape[:2], bool), states, dict(meta, applied=False, reason=reason)
    if not bank.eligible_sample(sample):
        return skip("recipient_not_originally_unoccluded")
    donor = bank.select(sample, rng, profile)
    if donor is None:
        return skip("no_context_matched_donor")
    patch, ellipse, provenance = donor
    # Exactly the adopted 128-attempt placement; no donor/profile/route fallback.
    raw, mask, placement = place((image, face_ellipse, points, states), (patch, ellipse), rng, route, light_area_only=True)
    if not placement["applied"]:
        return skip("placement_budget")
    output = opaque_boundary_gradient(image, raw, mask, width=48, strength=2)
    updated = np.asarray(placement["states"], dtype=np.int64)
    return output, mask, updated, dict(meta, **provenance, **placement,
        recipient_originally_unoccluded=True)
