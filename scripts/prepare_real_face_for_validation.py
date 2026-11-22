"""Prepare shared, unaugmented FOR validation with fresh query-bearing FACE crops.

HEAD-free inference; GT Heads are matching/normalization annotations only.
The scheduled frozen detector and score .33 are retained. No test data or
checkpoint/threshold selection is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch

from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation, TopDownFaceLandmarkDataset
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.face_overlap import CONTEXT_SCHEMA
from keypoint_comparison1.landmark_detection.geometry_prior_validation import (
    head_validation_annotations, assign_validation_crops, CONTRACT,
)
from scripts.prepare_real_face_for_epoch import sha, object_sha, atomic_json, fresh_queries, gpu_safety_sample

SOURCE = ROOT/'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json'
SOURCE_SHA = '0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0'
CHECKPOINT_SHA = '3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0'
CONFIG_SHA = 'b8c98e1943fcc3150f2fca3865701c88e1d521cbc2ec7669880cb79d32a6485c'


def verify_record(record, image, contract_sha, image_sha):
    payload = record['payload']
    if (record['payload_sha256'] != object_sha(payload)
            or payload['contract_sha256'] != contract_sha
            or payload['image_id'] != image['id'] or payload['image_sha256'] != image_sha):
        raise ValueError('Validation record provenance mismatch')
    return payload


def validation_dataset(output, data, identity):
    return TopDownFaceLandmarkDataset(annotation_path=output/'val_head_keypoints.json',
        image_root=data['root'], input_size=data['input_size'], training=False,
        augmentation=CropAugmentation(**data['augmentation']), mean=data['mean'], std=data['std'],
        category_ids=[2], require_visible_keypoint=False,
        detector_crop_cache=output/'val_context.json', face_overlap_query_dim=224,
        face_overlap_detector_identity=identity, seed=42)


def run(output, *, resume=False):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    raw = load_landmark_config(ROOT/'keypoint_comparison1/configs/face_overlap_real_face_w48_s2.yaml').raw
    config, checkpoint = map(Path, (raw['detector']['config'], raw['detector']['checkpoint']))
    if sha(SOURCE) != SOURCE_SHA or sha(config) != CONFIG_SHA or sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Scheduled validation source or detector identity changed')
    native = json.loads(SOURCE.read_text())
    if len(native['images']) != 530 or len(native['annotations']) != 882:
        raise ValueError('Canonical validation population changed')
    adapted, audit = head_validation_annotations(native)
    image_root = Path(raw['data']['root'])
    images = sorted(native['images'], key=lambda row: row['id'])
    image_hashes = {str(i['id']): sha(image_root/i['file_name']) for i in images}
    identity = dict(config=str(config), config_sha256=sha(config), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), image_size=960, score_threshold=.33,
        maximum_detections=100, query_dim=224, external_head=False)
    sources = [Path(__file__), ROOT/'scripts/prepare_real_face_for_epoch.py',
        *sorted((ROOT/'keypoint_comparison1/landmark_detection').glob('*.py')),
        ROOT/'src/data/head_anchored_keypoint_cache.py', ROOT/'src/data/face_crop_cache.py',
        ROOT/'tools/evaluate_deim_query_baseline_aligned.py']
    contract = dict(schema='real-face-for-validation-context-v1', source=str(SOURCE),
        source_sha256=SOURCE_SHA, image_sha256=image_hashes, detector_identity=identity,
        data_config=raw['data'], matching_contract=CONTRACT,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        shared_arms=['for_off','for_on'], split='val', training=False, test_used=False,
        threshold_source='fixed scheduled FOR parent score .33; no threshold search')
    output = output.resolve()
    if output.exists():
        if not resume or json.loads((output/'contract.json').read_text()) != contract:
            raise ValueError('Resume requires unchanged validation preparation contract')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'contract.json', contract)
    contract_sha = sha(output/'contract.json')
    (output/'records').mkdir(exist_ok=True)
    detector = None
    safety = []
    predictions = {}
    reused = 0
    for index, image in enumerate(images):
        path = output/'records'/f'{image["id"]}.json'
        image_sha = image_hashes[str(image['id'])]
        if path.exists():
            payload = verify_record(json.loads(path.read_text()), image, contract_sha, image_sha)
            reused += 1
        else:
            rgb = np.asarray(Image.open(image_root/image['file_name']).convert('RGB'))
            if rgb.shape != (image['height'], image['width'], 3):
                raise ValueError('Validation RGB dimensions differ from annotation')
            if detector is None:
                safety.append(gpu_safety_sample())
                detector = DeimFaceDetector(config_path=config, checkpoint_path=checkpoint,
                    image_size=960, score_threshold=.33, maximum_detections=100,
                    crop_geometry='ellipse', return_face_query_features=True)
            rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            payload = dict(image_id=image['id'], image_sha256=image_sha, rgb_sha256=rgb_sha,
                contract_sha256=contract_sha, predictions=fresh_queries(detector, rgb, rgb_sha))
            atomic_json(path, dict(payload=payload, payload_sha256=object_sha(payload)))
            if (index+1)%25 == 0:
                safety.append(gpu_safety_sample())
                print(f'validation context {index+1}/530', flush=True)
        predictions[str(image['id'])] = payload['predictions']
    if detector is not None:
        safety.append(gpu_safety_sample())
    assigned = assign_validation_crops(native, adapted, predictions)
    atomic_json(output/'val_head_keypoints.json', adapted)
    metadata = dict(identity, face_context_schema=CONTEXT_SCHEMA, split='val',
        annotation_sha256=sha(output/'val_head_keypoints.json'), source_annotation_sha256=SOURCE_SHA,
        matching_contract=CONTRACT, crop_geometry='ellipse', contract_sha256=contract_sha,
        shared_arms=['for_off','for_on'], threshold_source=contract['threshold_source'])
    atomic_json(output/'val_predictions.json', dict(metadata=metadata, predictions=predictions))
    atomic_json(output/'val_context.json', dict(metadata=metadata, predictions=assigned))
    dataset = validation_dataset(output, raw['data'], identity)
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        tensor, target = dataset[index]
        if not torch.isfinite(tensor).all():
            raise ValueError('Non-finite validation crop')
        digest.update(tensor.numpy().tobytes())
        for key in ['face_query_features','face_geometry','affine','bbox_xyxy']:
            digest.update(torch.as_tensor(target[key]).numpy().tobytes())
    audit.update(matched_eligible_heads=len(dataset),
        unmatched_eligible_heads=len(adapted['annotations'])-len(dataset))
    report = dict(status='shared_validation_context_complete_not_gate', training='not run', test='not run',
        audit=audit, matcher_report=dataset.detector_crop_report, crop_tensor_geometry_sha256=digest.hexdigest(),
        contract_sha256=contract_sha, reused_records=reused, detector_restarted=detector is not None,
        detector_identity=identity, gpu_safety_samples=safety,
        artifacts={name:sha(output/name) for name in ['val_head_keypoints.json','val_predictions.json','val_context.json']},
        limitations=['Preparation is not a gate pass or production completion',
            'Overlap/attribution protocol and production launcher/resume/preflight/four gates remain required'])
    atomic_json(output/'report.json', report)
    print(json.dumps(audit), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, resume=args.resume)
