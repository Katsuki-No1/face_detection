#!/usr/bin/env python3
"""Frozen parent FACE inference on full canonical validation; no model selection/test."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_validation import head_validation_annotations, assign_validation_crops, CONTRACT


def run(output):
    from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
    from keypoint_comparison1.landmark_detection.evaluation import load_or_create_detector_predictions
    plan = yaml.safe_load((ROOT/'configs/training_plans/messenger_geometry_prior_v1.yaml').read_text())
    raw = load_landmark_config(ROOT/plan['initialization']['parent_training_config']).raw
    source = raw['data']['continuous_source_inventory']['val']
    native_path = Path(source['path'])
    if file_hash(native_path) != source['sha256']:
        raise ValueError('Canonical validation changed')
    for kind in ['checkpoint', 'config']:
        if file_hash(raw['detector'][kind]) != plan['feature_preparation'][f'detector_{kind}_sha256']:
            raise ValueError('Pinned detector changed')
    native = json.loads(native_path.read_text())
    if len(native['images']) != 530 or len(native['annotations']) != 882:
        raise ValueError('Canonical validation inventory changed')
    image_files = [(i['file_name'], file_hash(Path(raw['data']['root'])/i['file_name'])) for i in native['images']]
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    adapted, audit = head_validation_annotations(native)
    annotation = output/'val_head_keypoints.json'
    write(annotation.name, adapted)
    torch.set_num_threads(1)
    detector = DeimFaceDetector(config_path=raw['detector']['config'], checkpoint_path=raw['detector']['checkpoint'],
        score_threshold=.33, maximum_detections=100, image_size=960, device='cuda:0')
    predictions = load_or_create_detector_predictions(detector=detector, annotation_path=native_path,
        image_root=Path(raw['data']['root']), cache_path=output/'val_predictions.json')
    assigned = assign_validation_crops(native, adapted, predictions)
    metadata = dict(crop_geometry='ellipse', split='val', matching_contract=CONTRACT,
        annotation_sha256=file_hash(annotation), source_annotation_sha256=source['sha256'],
        detector_checkpoint_sha256=file_hash(raw['detector']['checkpoint']),
        detector_config_sha256=file_hash(raw['detector']['config']), score_threshold=.33,
        prediction_cap=100, weight_branch='ema', threshold_source='fixed parent ABC-A configuration',
        source_prediction_sha256=file_hash(output/'val_predictions.json'))
    write('val_predicted_crops.json', dict(metadata=metadata, predictions=assigned))
    audit.update(matched_eligible_heads=sum(map(len,assigned.values())),
        unmatched_eligible_heads=len(adapted['annotations'])-sum(map(len,assigned.values())))
    write('preparation_complete.json', dict(status='validation_inputs_prepared_not_model_gate', audit=audit,
        artifacts={p.name:file_hash(p) for p in output.glob('*.json')}, image_files=image_files,
        source_sha256={str(Path(__file__).relative_to(ROOT)):file_hash(__file__),
            'keypoint_comparison1/landmark_detection/geometry_prior_validation.py':file_hash(ROOT/'keypoint_comparison1/landmark_detection/geometry_prior_validation.py')},
        training='not run', test='not run', external_HEAD='none', **metadata))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
