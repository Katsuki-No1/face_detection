"""Train-only full-image transport of the adopted crop-side face compositor.

This adapter is preparation code, not a gate or a production launcher. It emits
both amodal donor geometry and actual visible support; detector label selection
must explicitly consume the former or latter in its own frozen contract.
"""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from keypoint_comparison1.landmark_detection.real_face_occlusion import RealFaceOcclusionBank
from scripts.prototype_cross_video_face_occlusion import place
from scripts.opaque_boundary_gradient import opaque_boundary_gradient
from src.data.face_keypoint_dataset import _resize_ellipse_params, ellipse_params_to_xyxy
from src.data.landmark_occlusion import ellipse_region_mask


class FullImageFaceBank(RealFaceOcclusionBank):
    """Keep every source FACE for exclusion, including ineligible donors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = json.loads(Path(self.data['adapter_path']).read_text())
        images = {i['id']: i for i in data['images']}
        self.full_face_annotations = defaultdict(list)
        for ann in data['annotations']:
            self.full_face_annotations[images[ann['image_id']]['file_name']].append(ann)


def _ellipse_transform(ellipse, sx, sy, tx=0, ty=0):
    result = _resize_ellipse_params(list(ellipse), sx, sy)
    result[0] += tx
    result[1] += ty
    return result


def _points(annotation):
    return np.asarray(annotation['keypoints'], dtype=float).reshape(5, 3).copy()


def transported_donor_face_pixels(ellipse, source_box, placement_box, crop_size, window_size):
    """Count source FACE pixels contributing to the final clipped pasted image.

    Match the RGB bilinear resize and crop, then the final nearest transport.
    Even fractional contributions count; a face clipped out before transport
    must not reject a paste solely because it was elsewhere in the donor patch.
    """
    dx0, dy0, dx1, dy1 = source_box
    local = np.asarray(ellipse, dtype=float).copy()
    local[:2] -= [dx0, dy0]
    support = ellipse_region_mask(dy1-dy0, dx1-dx0, local).astype(np.float32)
    if not support.any():
        return 0
    px0, py0, px1, py1 = placement_box
    cw, ch = crop_size
    a, b, c, d = max(0, px0), max(0, py0), min(cw, px1), min(ch, py1)
    if c <= a or d <= b:
        return 0
    resized = cv2.resize(support, (px1-px0, py1-py0), interpolation=cv2.INTER_LINEAR)
    clipped = np.zeros((ch, cw), np.uint8)
    clipped[b:d, a:c] = resized[b-py0:d-py0, a-px0:c-px0] > 0
    transported = cv2.resize(clipped, tuple(window_size), interpolation=cv2.INTER_NEAREST)
    return int(np.count_nonzero(transported))


def compose_full_image(rgb, *, image, annotations, target_id, bank, rng):
    """One requested paste; skip without retries to another donor/route/target.

    Inputs are original source RGB and FACE-adapter annotations in source pixels.
    The recipient window is clipped to source bounds, isotropically resized to
    longest-side 256 (rounded axes), then transported back using binary support.
    Pixels outside that support are copied bit-for-bit from the source image.
    All recipient ellipses/boxes remain amodal; covered visible points become
    occluded. A separate donor record carries truncation, geometry and provenance.
    No detector queries are accepted: FOR must extract them from the output RGB.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Original uint8 RGB required')
    if image.get('metadata', {}).get('split') != 'train':
        raise ValueError('Full-image synthesis is train-only')
    name = image['file_name']
    source = bank.images.get(name)
    if source is None or source['video'] != image['metadata']['source_video_id']:
        raise ValueError('Source image absent from authenticated train bank')
    if rgb.shape[:2] != (image['height'], image['width']):
        raise ValueError('Source dimensions changed before composition')
    targets = [a for a in annotations if a['id'] == target_id]
    if len(targets) != 1 or any(a['image_id'] != image['id'] for a in annotations):
        raise ValueError('Ambiguous recipient or annotations from another image')
    target = targets[0]
    meta = dict(schema='real-face-full-image-preparation-v1', split='train',
                recipient_id=target_id, recipient_file=name,
                recipient_video=source['video'], manifest_sha256=bank.manifest_sha256,
                gradient_width_crop_pixels=48, gradient_strength=2,
                alpha=False, post_composition_queries_required=True,
                coordinate_convention='source pixels xy; ellipse radii and radians')

    def skip(reason):
        return rgb.copy(), deepcopy(annotations), np.zeros(rgb.shape[:2], bool), dict(
            meta, applied=False, reason=reason)

    if not bank.eligible_sample(dict(image=image, annotation=target)):
        return skip('recipient_not_originally_unoccluded')
    route = rng.choices(['light', 'medium', 'strong'], [.2, .4, .4])[0]
    profile = rng.random() < .5
    meta.update(requested_route=route, requested_profile=profile)
    selected = bank.select(dict(image=image, annotation=target), rng, profile)
    if selected is None:
        return skip('no_context_matched_donor')
    patch, donor_ellipse, provenance = selected
    meta.update(provenance)
    if provenance['donor_video'] == source['video']:
        raise ValueError('Donor and recipient source videos must differ')
    h, w = rgb.shape[:2]
    cx, cy, a, b, _ = target['ellipse']
    side = 2.5 * max(a, b)
    x0, y0 = max(0, round(cx-side/2)), max(0, round(cy-side/2))
    x1, y1 = min(w, round(cx+side/2)), min(h, round(cy+side/2))
    if x1 <= x0 or y1 <= y0:
        return skip('empty_recipient_window')
    scale = 256 / max(x1-x0, y1-y0)
    cw, ch = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
    sx, sy = cw/(x1-x0), ch/(y1-y0)
    crop = cv2.resize(rgb[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_LINEAR)
    local_e = np.asarray(target['ellipse'], dtype=float).copy()
    local_e[:2] -= [x0, y0]
    local_e = _ellipse_transform(local_e, sx, sy)
    kp = _points(target)
    points = (kp[:, :2] - [x0, y0]) * [sx, sy]
    states = kp[:, 2].astype(int)
    inside = (points >= 0).all(1) & (points[:, 0] < cw-.5) & (points[:, 1] < ch-.5)
    states[~inside] = 0
    points[states == 0] = 0
    raw, mask, placement = place((crop, local_e, points, states),
                                  (patch, donor_ellipse), rng, route, light_area_only=True)
    meta['placement'] = placement
    if not placement['applied']:
        return skip('placement_budget')
    full_mask = np.zeros((h, w), bool)
    full_mask[y0:y1, x0:x1] = cv2.resize(mask.astype(np.uint8), (x1-x0, y1-y0),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
    # Reject any OTHER donor face actually transported into the composite.
    # Testing the uncut source patch rejected faces that were never pasted.
    meta['additional_donor_face_check'] = 'final_transported_bilinear_pixel_support_v2'
    for face in bank.full_face_annotations[provenance['donor_file']]:
        if face['id'] == provenance['donor_annotation_id']:
            continue
        count = transported_donor_face_pixels(face['ellipse'], provenance['donor_source_box'],
            placement['bbox_xyxy'], (cw, ch), (x1-x0, y1-y0))
        if count:
            meta['additional_donor_face'] = dict(annotation_id=face['id'], transported_pixels=count)
            return skip('additional_face_in_pasted_support')
    touched = []
    for ann in annotations:
        face_mask = ellipse_region_mask(h, w, ann['ellipse'])
        if (full_mask & face_mask).any():
            if not bank.eligible_sample(dict(image=image, annotation=ann)):
                return skip('paste_touches_ineligible_recipient')
            touched.append(ann['id'])
    # Rounding on transport must not silently change the requested severity.
    face_mask = ellipse_region_mask(h, w, target['ellipse'])
    fraction = float((full_mask & face_mask).sum() / max(1, face_mask.sum()))
    lo, hi = {'light': (0, .1), 'medium': (.1, .2), 'strong': (.2, .4)}[route]
    if not (lo < fraction < hi if route == 'light' else lo < fraction <= hi):
        return skip('transport_changed_severity')
    px0, py0, px1, py1 = placement['bbox_xyxy']
    donor_local = _ellipse_transform(donor_ellipse, (px1-px0)/patch.shape[1],
                                      (py1-py0)/patch.shape[0], px0, py0)
    donor_full = _ellipse_transform(donor_local, 1/sx, 1/sy, x0, y0)
    donor_support = ellipse_region_mask(h, w, donor_full) & full_mask
    donor_overlap = float((donor_support & face_mask).sum() / max(1, face_mask.sum()))
    if donor_overlap <= 0 or (route != 'light' and donor_overlap < .01):
        return skip('transport_changed_donor_overlap')
    corrected = opaque_boundary_gradient(crop, raw, mask, width=48, strength=2)
    # Resize the composite, then write strictly inside transported support.
    patch_full = cv2.resize(corrected, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
    output = rgb.copy()
    local_mask = full_mask[y0:y1, x0:x1]
    output[y0:y1, x0:x1][local_mask] = patch_full[local_mask]
    updated = deepcopy(annotations)
    for ann in updated:
        pts = _points(ann)
        xy = np.rint(pts[:, :2]).astype(int)
        valid = (pts[:, 2] > 0) & (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
        covered = np.zeros(5, bool)
        covered[valid] = full_mask[xy[valid, 1], xy[valid, 0]]
        pts[covered, 2] = 1
        ann['keypoints'] = pts.reshape(-1).tolist()
        if ann['id'] in touched:
            ann.setdefault('attributes', {})['face_occluded'] = True
            ann['synthetic_face_overlap'] = True
    ys, xs = np.where(donor_support)
    donor_record = dict(source_annotation_id=provenance['donor_annotation_id'],
                        source_file=provenance['donor_file'], ellipse=donor_full,
                        amodal_bbox_xyxy=ellipse_params_to_xyxy(donor_full),
                        visible_support_bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                        support_pixels=int(donor_support.sum()),
                        support_truncated=bool((ellipse_region_mask(h, w, donor_full) & ~full_mask).any()),
                        detector_label_policy='pending_explicit_training_adapter',
                        include_in_recipient_keypoint_supervision=False)
    return output, updated, full_mask, dict(meta, applied=True, touched_recipient_ids=touched,
        overlap=fraction, donor_face_overlap=donor_overlap, donor=donor_record,
        recipient_window_xyxy=[x0, y0, x1, y1], crop_size_wh=[cw, ch],
        mask_sha256=hashlib.sha256(full_mask.tobytes()).hexdigest(),
        rgb_sha256=hashlib.sha256(output.tobytes()).hexdigest())
