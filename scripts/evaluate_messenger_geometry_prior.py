#!/usr/bin/env python3
"""Common HEAD-free evaluation of authenticated geometry-prior production EMA.

Every retained detector prediction is processed, including unmatched faces.
GT Heads enter only the shared evaluator, never crop generation or inference.
"""
from argparse import ArgumentParser
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import authenticate, verify_gates, write_json
from keypoint_comparison1.landmark_detection.geometry_prior_training import build_prior_model, file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_inputs import predicted_geometry_inputs
from keypoint_comparison1.landmark_detection.geometry import build_crop_transform, warp_crop, normalized_crop_to_original
from keypoint_comparison1.landmark_detection.dataset import image_to_tensor
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.evaluation import load_or_create_detector_predictions
from src.data.face_crop_cache import ellipse_aligned_rectangle, canonicalize_ellipse
from src.evaluation.baseline_aligned import evaluate_normalized_predictions


def prepare_cuda(index=0):
    # Initialize the CUDA context before querying allocator statistics. The
    # installed runtime rejects reset_peak_memory_stats on an uninitialized device.
    torch.cuda.set_device(index)
    torch.cuda.reset_peak_memory_stats(index)
    return torch.device('cuda', index)


def load_model(spec, identity, checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    experiment = state.get('contract', {}).get('experiment', {})
    if (state.get('schema') != 'geometry-prior-runtime-v1'
            or experiment.get('identity') != identity
            or experiment.get('architecture') != spec['architecture']
            or experiment.get('mode') != 'production'):
        raise ValueError('Evaluation requires this authenticated production checkpoint')
    prior = spec['prior']
    model, _ = build_prior_model(spec['parent_config'], prior_path=ROOT/prior['path'],
        prior_sha256=prior['sha256'], fusion_mode='messenger' if spec['arm']=='control' else 'prior')
    model.load_state_dict(state['ema'], strict=True)
    provenance = dict(path=str(checkpoint), sha256=file_hash(checkpoint), epoch=state['epoch'],
        step=state['step'], state='ema', identity=identity, prior=prior)
    del state
    gc.collect()
    return model.to(device).eval(), provenance


def infer(model, coco, predictions, raw, *, device, batch_size):
    data = raw['data']; size = int(data['input_size'])
    normalized = {int(row['id']): dict(heads=[], faces=[], poses=[]) for row in coco['images']}
    pending = []

    def flush():
        if not pending:
            return
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
            output = model(torch.stack([p['image'] for p in pending]).to(device),
                geometry=torch.stack([p['geometry'] for p in pending]).to(device),
                support=torch.stack([p['support'] for p in pending]).to(device))
        points = output['decoded_coordinates'].float().cpu().numpy()
        scores = output['presence_logits'].sigmoid().float().cpu().numpy()
        states = output['state_probabilities'].float().cpu().numpy()
        for row, xy, score, state in zip(pending, points, scores, states, strict=True):
            normalized[row['image_id']]['poses'].append(dict(
                bbox_xyxy=row['face']['bbox_xyxy'], score=row['face']['score'],
                keypoints=normalized_crop_to_original(xy, row['transform']).tolist(),
                point_scores=score.tolist(), state_probabilities=state.tolist()))
        pending.clear()

    for image_info in coco['images']:
        image_id = int(image_info['id'])
        detections = predictions[image_id]
        if len(detections)>100:
            raise ValueError('Detector prediction cap exceeded')
        if not detections:
            continue
        image = np.asarray(Image.open(Path(data['root'])/image_info['file_name']).convert('RGB'))
        for detection in detections:
            ellipse = canonicalize_ellipse(detection['ellipse'])
            box, rotation = ellipse_aligned_rectangle(ellipse)
            transform = build_crop_transform(box, output_size=size,
                scale_factor=float(data['augmentation']['scale_factor']), rotation_deg=rotation,
                warp_bbox_aspect=False)
            geometry = predicted_geometry_inputs(detection, affine=transform.matrix, flipped=False,
                source_size=(image.shape[1],image.shape[0]), crop_size=(size,size), grid_size=(16,16))
            face = dict(bbox_xyxy=detection['bbox_xyxy'], ellipse=ellipse,
                score=float(detection['score']))
            normalized[image_id]['faces'].append(face)
            pending.append(dict(image_id=image_id, face=face, transform=transform,
                image=image_to_tensor(warp_crop(image,transform),mean=data['mean'],std=data['std']),
                **geometry))
            if len(pending)>=batch_size:
                flush()
    flush()
    return normalized


def run(args, *, authenticator=authenticate):
    started = time.perf_counter()
    spec, identity = authenticator(args.config)
    verify_gates(spec,identity)
    output = ROOT/spec['output_root']; raw=spec['parent_config']
    completion = json.loads((output/'training_success.json').read_text())
    if completion.get('status')!='complete' or completion.get('identity')!=identity:
        raise ValueError('Production must complete with the authenticated identity')
    if args.split=='test':
        marker=json.loads((output/'evaluation/val/baseline_aligned/success.json').read_text())
        if marker.get('status')!='success':
            raise ValueError('Test requires completed common validation')
    checkpoint = output/'checkpoints/best.pt'
    if args.split=='test':
        validation=json.loads((output/'evaluation/val/baseline_aligned/metrics.json').read_text())
        locked=validation['provenance']['query_checkpoint']
        if locked.get('identity')!=identity or locked.get('sha256')!=file_hash(checkpoint):
            raise ValueError('Test checkpoint/config must match completed validation')
    annotation = ROOT/f'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/{args.split}.json'
    coco = json.loads(annotation.read_text())
    device=prepare_cuda(0);torch.set_num_threads(1)
    cfg=raw['detector']
    detector=DeimFaceDetector(config_path=cfg['config'],checkpoint_path=cfg['checkpoint'],
        device=str(device),image_size=cfg['image_size'],resize_mode=cfg.get('resize_mode'),
        pad_color=cfg.get('pad_color'),size_divisor=cfg.get('size_divisor'),
        score_threshold=float(cfg['score_threshold']),maximum_detections=cfg['maximum_detections'],
        crop_geometry=cfg['crop_geometry'])
    cache=ROOT/f'outputs/landmark_detection/messenger_geometry_prior/evaluation_cache/{args.split}.json'
    detector_started=time.perf_counter()
    predictions=load_or_create_detector_predictions(detector=detector,annotation_path=annotation,
        image_root=Path(raw['data']['root']),cache_path=cache)
    detector_seconds=time.perf_counter()-detector_started
    if set(predictions)!={int(im['id']) for im in coco['images']}:
        raise ValueError('Detector cache split inventory mismatch')
    del detector;gc.collect();torch.cuda.empty_cache()
    model, provenance=load_model(spec,identity,checkpoint,device)
    batch=16
    torch.cuda.synchronize();inference_started=time.perf_counter()
    while True:
        try:
            normalized=infer(model,coco,predictions,raw,device=device,batch_size=batch)
            break
        except torch.OutOfMemoryError:
            if batch==1:raise
            batch//=2;gc.collect();torch.cuda.empty_cache()
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-inference_started
    common=output/f'evaluation/{args.split}/baseline_aligned'
    write_json(common.parent/'normalized_predictions.json',normalized)
    cache_metadata=json.loads(cache.read_text())['metadata']
    elapsed=time.perf_counter()-started
    return evaluate_normalized_predictions(annotation_path=annotation,predictions_by_image=normalized,
        split=args.split,architecture_id=spec['architecture'],
        comparison_stage='full_image_FACE_ellipse_crop_DINO_geometry_prior_Messenger_Query',output_dir=common,
        provenance=dict(pipeline_contract='two_stage_face_keypoint',seed=42,
            query_checkpoint=provenance,config_path=str(args.config),config_sha256=file_hash(args.config),
            driver_path=__file__,driver_sha256=file_hash(__file__),
            detector_cache=str(cache),detector_cache_sha256=file_hash(cache),detector_cache_metadata=cache_metadata,
            continuous_local_face_contract=dict(checkpoint_sha256=cache_metadata['detector_checkpoint_sha256'],
                config_sha256=cache_metadata['detector_config_sha256'],prediction_cap=100),
            baseline_roles=dict(local_matched_control='Historical A1/A2 detector shared across these crop ablations; not current V3 baseline',
                current_local_baseline='V3 complete pipeline; same-split combined comparison remains a separate pending report',
                global_reference='Locked distributed reference, legacy diagnostic contract only'),
            prediction_scope='All retained detector predictions; no GT crop selection'),
        runtime=dict(total_seconds=elapsed,images_per_second=len(coco['images'])/elapsed,
            model_inference_seconds=inference_seconds,model_inference_milliseconds_per_image=1000*inference_seconds/len(coco['images']),
            detector_cache_or_inference_seconds=detector_seconds,
            inference_scope='Crop model only; detector cache generation/loading timed separately',
            peak_vram_mib=torch.cuda.max_memory_allocated(device)/1024**2,batch_size=batch,input_size=raw['data']['input_size']),
        validation_selected_point_threshold=.70,face_score_candidate_floor=cfg['score_threshold'])


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    run(parser.parse_args())
