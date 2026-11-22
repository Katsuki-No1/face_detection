"""Shared FOR crop preparation and initialization; not production certification.

HEAD-free FACE/ellipse -> Face-crop DINO -> Query Refinement, with only the
predicted-query/geometry conditioning toggled in the paired arm.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from keypoint_comparison1.landmark_detection.dataset import (
    CropAugmentation, TopDownFaceLandmarkDataset,
)
from keypoint_comparison1.landmark_detection.face_overlap import CONTEXT_SCHEMA


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(source, output):
    """Authenticate bounded composites and bind both arms to recipient-only GT."""
    source, output = Path(source).resolve(), Path(output).resolve()
    rows = json.loads(source.read_text())
    if not rows:
        raise ValueError('Empty composite inventory')
    images, annotations, predictions = [], [], {}
    identity = rows[0]['context']['detector_identity']
    dim = rows[0]['context']['query_dim']
    seen_annotations = set()
    for row in rows:
        image, context, composition = row['image'], row['context'], row['composition']
        path = (source.parent / row['composite_file']).resolve()
        if not path.is_relative_to(source.parent):
            raise ValueError('Composite path escapes source directory')
        rgb = np.asarray(Image.open(path).convert('RGB'))
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if (sha(path) != row['png_sha256'] or rgb_sha != composition['rgb_sha256']
                or rgb_sha != context['composite_rgb_sha256']
                or composition['mask_sha256'] != context['composition_mask_sha256']):
            raise ValueError('Composite pixels/context provenance mismatch')
        if (image['metadata']['split'] != 'train' or composition['split'] != 'train'
                or not composition['applied']
                or composition['recipient_video'] == composition['donor_video']
                or context['context_source'] != 'fresh_composite_full_image_inference'
                or context['detector_identity'] != identity or context['query_dim'] != dim
                or rgb.shape != (image['height'], image['width'], 3)):
            raise ValueError('Composite train/context contract mismatch')
        key = str(image['id'])
        if key in predictions:
            raise ValueError('Duplicate composite image ID')
        derived = deepcopy(image)
        derived['file_name'] = str(path)
        derived['metadata'].update(source_file_name=image['file_name'],
                                   source_sha256=image['metadata']['sha256'],
                                   sha256=row['png_sha256'], composite_rgb_sha256=rgb_sha)
        images.append(derived)
        predictions[key] = context['predictions']
        for original in row['recipient_annotations']:
            ann = deepcopy(original)
            if ann['image_id'] != image['id'] or ann['id'] in seen_annotations:
                raise ValueError('Recipient annotation identity mismatch')
            seen_annotations.add(ann['id'])
            # Derived semantic summaries must agree with the post-paste states.
            states = np.asarray(ann['keypoints']).reshape(5, 3)[:, 2]
            names = ['left_eye', 'right_eye', 'nose_tip', 'left_mouth_corner', 'right_mouth_corner']
            attrs = ann.setdefault('attributes', {})
            attrs['keypoint_occluded'] = (states == 1).tolist()
            attrs['occluded_keypoint_count'] = int((states == 1).sum())
            attrs['occluded_keypoint_names'] = [n for n, s in zip(names, states) if s == 1]
            ann['num_keypoints'] = int((states > 0).sum())
            annotations.append(ann)
    output.mkdir(parents=True, exist_ok=False)
    annotation = output/'recipients.json'
    annotation.write_text(json.dumps(dict(images=images, annotations=annotations,
        categories=[dict(id=2, name='face')]), indent=2)+'\n')
    metadata = dict(identity, face_context_schema=CONTEXT_SCHEMA,
        annotation_sha256=sha(annotation), crop_geometry='ellipse',
        source_composites_sha256=sha(source), shared_arms=['for_off', 'for_on'],
        status='bounded_preparation_not_gate', donor_keypoint_supervision=False)
    cache = output/'context.json'
    cache.write_text(json.dumps(dict(metadata=metadata, predictions=predictions), indent=2)+'\n')
    return dict(annotation=annotation, cache=cache, identity=identity, query_dim=dim,
                source_sha256=sha(source), images=len(images), annotations=len(annotations))


def shared_dataset(bundle, data_config, *, training):
    """Both arms use this same crop path, even when model conditioning is off."""
    return TopDownFaceLandmarkDataset(annotation_path=bundle['annotation'],
        image_root=bundle['annotation'].parent, input_size=data_config['input_size'],
        training=training, augmentation=CropAugmentation(**data_config['augmentation']),
        mean=data_config['mean'], std=data_config['std'], category_ids=[2],
        require_visible_keypoint=not data_config.get('include_missing_keypoint_instances', False),
        detector_crop_cache=bundle['cache'], detector_matching_iou=.5, seed=42,
        face_overlap_query_dim=bundle['query_dim'],
        face_overlap_detector_identity=bundle['identity'])


def initialize_matched_arm(control, conditioned):
    """Copy every common parameter/buffer; permit only new zero FOR residuals."""
    off, on = control.state_dict(), conditioned.state_dict()
    extras = set(on) - set(off)
    if set(off) - set(on) or not extras or any('.face_overlap.' not in k for k in extras):
        raise ValueError('FOR pair differs beyond the conditioning block')
    if any(on[k].shape != v.shape for k, v in off.items()):
        raise ValueError('FOR pair shared tensor shape mismatch')
    for name, module in conditioned.named_modules():
        if name.endswith('face_overlap'):
            for projection in (module.query_projection, module.geometry_projection):
                if torch.count_nonzero(projection[-1].weight) or torch.count_nonzero(projection[-1].bias):
                    raise ValueError('FOR residual must start at zero')
    conditioned.load_state_dict({**on, **off}, strict=True)
    assert all(torch.equal(v, conditioned.state_dict()[k]) for k, v in off.items())
    digest = hashlib.sha256()
    for key, value in sorted(off.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(shared_tensor_count=len(off), conditioning_tensor_count=len(extras),
                shared_initial_state_sha256=digest.hexdigest())
