"""Materialize one audited FOR epoch with fresh full-image detector context.

Preparation only: no training, checkpoint selection, or gate certification.
Clean and skipped images remain in the inventory. Only changed RGB is cached.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.face_overlap import CONTEXT_SCHEMA
from scripts.real_face_epoch_inputs import SharedEpochInputs
from scripts.real_face_for_dataset import SharedFORCropDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def gpu_safety_sample():
    result = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits','--id=0'], capture_output=True, text=True, check=True)
    used, total, temperature = map(float, result.stdout.strip().split(','))
    if used > .90*total or temperature >= 83:
        raise RuntimeError(f'Preparation GPU safety limit: {used}/{total}MiB, {temperature}C')
    return dict(used_mib=used,total_mib=total,temperature_c=temperature)


def fresh_queries(detector, rgb, expected_rgb_sha, query_dim=224):
    if hashlib.sha256(rgb.tobytes()).hexdigest() != expected_rgb_sha:
        raise ValueError('Actual epoch RGB differs from recipe')
    rows = detector.predict(rgb)
    seen = set()
    for row in rows:
        index = row.get('face_query_index')
        features = np.asarray(row.get('face_query_features', []), dtype=float)
        ellipse = np.asarray(row.get('ellipse', []), dtype=float)
        bbox = np.asarray(row.get('detector_bbox_xyxy', []), dtype=float)
        if (type(index) is not int or index < 0 or index in seen or
            features.shape != (query_dim,) or not np.isfinite(features).all() or
            ellipse.shape != (5,) or not np.isfinite(ellipse).all() or np.any(ellipse[2:4] <= 0) or
            bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2])):
            raise ValueError('Invalid epoch query or geometry')
        seen.add(index)
    return rows


def check_record(record, recipe, contract_sha):
    payload = record['payload']
    if (record['payload_sha256'] != object_sha(payload) or
        payload['contract_sha256'] != contract_sha or
        payload['recipe_sha256'] != object_sha(recipe) or
        payload['image']['id'] != recipe['image_id']):
        raise ValueError('Epoch record provenance mismatch')
    if sha(payload['image']['file_name']) != payload['image_sha256']:
        raise ValueError('Epoch RGB file changed after preparation')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    output = args.output.resolve()
    audit_contract = json.loads((args.audit/'contract.json').read_text())
    audit_report = json.loads((args.audit/'report.json').read_text())
    if (audit_report['contract_sha256'] != sha(args.audit/'contract.json') or
        audit_report['recipes_sha256'] != sha(args.audit/'recipes.jsonl')):
        raise ValueError('Audit artifact identity mismatch')
    for name, expected in audit_contract['code_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Audited synthesis code changed: {name}')
    recipes = [json.loads(line) for line in (args.audit/'recipes.jsonl').read_text().splitlines()]
    recipes = [r for r in recipes if r['seed']==args.seed and r['epoch']==args.epoch]
    inputs = SharedEpochInputs(ROOT/'outputs/dataset_adapters/real_face_occlusion_w48_s2/manifest.json')
    if [r['image_id'] for r in recipes] != [i['id'] for i in inputs.images]:
        raise ValueError('Epoch is not a full authenticated train inventory')
    if args.limit is not None and not 1 <= args.limit <= len(recipes):
        raise ValueError('Invalid preparation limit')
    recipes = recipes[:args.limit]
    images = {i['id']:i for i in inputs.images}
    config = ROOT/'configs/dinov3_deimv2/mal_alpha2_ellipse_from_existing_wider.yaml'
    checkpoint = ROOT/'outputs/experiments/mal_alpha2_ellipse_from_existing_wider/checkpoints/best_val_map.pt'
    if sha(checkpoint) != '3bd6e40517bc6f6e276c47e1dfdf639173e8b9c04d32486adbb6f4b407f006a0':
        raise ValueError('Scheduled frozen detector changed')
    from tools.dinov3_deimv2_common import load_config
    if load_config(config)['model']['decoder']['hidden_dim'] != 224:
        raise ValueError('Unexpected detector query width')
    identity = dict(config=str(config), config_sha256=sha(config), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), image_size=960, score_threshold=.33,
        maximum_detections=100, query_dim=224, external_head=False)
    cfg = load_landmark_config(ROOT/'keypoint_comparison1/configs/face_overlap_real_face_w48_s2.yaml').raw
    sources = [Path(__file__), ROOT/'scripts/real_face_for_dataset.py',
        *sorted((ROOT/'keypoint_comparison1/landmark_detection').glob('*.py'))]
    contract = dict(schema='shared-for-epoch-materialization-v1', seed=args.seed, epoch=args.epoch,
        image_ids=[r['image_id'] for r in recipes], audit_report_sha256=sha(args.audit/'report.json'),
        detector_identity=identity, data_config=cfg['data'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        full_population=len(recipes)==4237, preparation_only=True, maximum_cached_png_bytes=2*1024**3)
    if output.exists():
        if not args.resume or json.loads((output/'contract.json').read_text()) != contract:
            raise ValueError('Resume requires unchanged materialization contract')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'contract.json', contract)
    for name in ['records','images']:
        (output/name).mkdir(exist_ok=True)
    contract_sha = sha(output/'contract.json')
    detector = None
    safety_samples = []
    payloads = []
    used_bytes = sum(p.stat().st_size for p in (output/'images').glob('*.png'))
    for index, recipe in enumerate(recipes):
        path = output/'records'/f'{recipe["image_id"]}.json'
        if path.exists():
            payloads.append(check_record(json.loads(path.read_text()), recipe, contract_sha))
            continue
        rgb, annotations, metadata = inputs.render(images[recipe['image_id']],
            seed=args.seed, epoch=args.epoch, load_unmodified=True)
        if any(metadata.get(k) != v for k,v in recipe.items()):
            raise ValueError('Audited epoch recipe does not reproduce exactly')
        if detector is None:
            safety_samples.append(gpu_safety_sample())
            detector = DeimFaceDetector(config_path=config, checkpoint_path=checkpoint,
                image_size=960, score_threshold=.33, maximum_detections=100,
                crop_geometry='ellipse', return_face_query_features=True)
        predictions = fresh_queries(detector, rgb, metadata['rgb_sha256'])
        image = deepcopy(images[recipe['image_id']])
        if recipe['applied']:
            image_path = output/'images'/f'{image["id"]}.png'
            previous_size = image_path.stat().st_size if image_path.exists() else 0
            temporary = image_path.with_suffix('.tmp.png')
            Image.fromarray(rgb).save(temporary)
            used_bytes += temporary.stat().st_size-previous_size
            if used_bytes > contract['maximum_cached_png_bytes']:
                temporary.unlink()
                raise ValueError('Bounded epoch PNG storage budget exceeded')
            os.replace(temporary, image_path)
        else:
            image_path = inputs.bank.root/image['file_name']
        image['metadata'].update(original_file_name=image['file_name'],
            original_sha256=image['metadata']['sha256'], sha256=sha(image_path),
            epoch_rgb_sha256=metadata['rgb_sha256'], synthetic_applied=recipe['applied'])
        image['file_name'] = str(image_path)
        payload = dict(contract_sha256=contract_sha, recipe_sha256=object_sha(recipe),
            image=image, image_sha256=sha(image_path), annotations=annotations,
            predictions=predictions, rgb_sha256=metadata['rgb_sha256'],
            query_source='fresh_actual_epoch_full_image', donor_keypoint_supervision=False)
        atomic_json(path, dict(payload=payload, payload_sha256=object_sha(payload)))
        payloads.append(payload)
        if (index+1)%25==0:
            safety_samples.append(gpu_safety_sample())
            print(f'prepared {index+1}/{len(recipes)} images',flush=True)
    # Only finalize inventory after every original image has an authenticated record.
    annotations = output/'recipients.json'
    atomic_json(annotations, dict(images=[p['image'] for p in payloads],
        annotations=[a for p in payloads for a in p['annotations']], categories=inputs.coco['categories']))
    cache = output/'context.json'
    atomic_json(cache, dict(metadata=dict(identity, face_context_schema=CONTEXT_SCHEMA,
        annotation_sha256=sha(annotations), contract_sha256=contract_sha, crop_geometry='ellipse',
        shared_arms=['for_off','for_on'], recipe_epoch=args.epoch, seed=args.seed),
        predictions={str(p['image']['id']):p['predictions'] for p in payloads}))
    data = cfg['data']
    dataset = SharedFORCropDataset(recipe_epoch=args.epoch, annotation_path=annotations,
        image_root=output, input_size=data['input_size'], training=True,
        augmentation=CropAugmentation(**data['augmentation']), mean=data['mean'], std=data['std'],
        category_ids=[2], require_visible_keypoint=False, detector_crop_cache=cache,
        face_overlap_query_dim=224, face_overlap_detector_identity=identity, seed=args.seed)
    crop_digest = hashlib.sha256()
    for index in range(len(dataset)):
        tensor, target = dataset[index]
        if not torch.isfinite(tensor).all():
            raise ValueError('Non-finite prepared crop')
        crop_digest.update(tensor.numpy().tobytes())
        for key in ['face_query_features','face_geometry','affine']:
            crop_digest.update(target[key].numpy().tobytes())
    report = dict(status='shared_epoch_materialization_complete_not_gate', training='not run',
        full_population=len(recipes)==4237, images=len(payloads), seed=args.seed, epoch=args.epoch,
        requested=sum(r['requested'] for r in recipes), applied=sum(r['applied'] for r in recipes),
        original_images_retained=sum(not r['applied'] for r in recipes),
        annotations=sum(len(p['annotations']) for p in payloads), predicted_crops=len(dataset),
        matcher_report=dataset.detector_crop_report, crop_tensor_geometry_sha256=crop_digest.hexdigest(),
        annotations_sha256=sha(annotations), context_sha256=sha(cache), contract_sha256=contract_sha,
        cached_png_bytes=used_bytes, detector_identity=identity,
        gpu_safety_samples=safety_samples,
        process_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if detector is not None else None,
        limitations=['one epoch preparation; not a synthesis acceptance or four-gate certification',
            'production launcher/resume/W&B and validation protocol still required'])
    atomic_json(output/'report.json', report)
    print(json.dumps({k:report[k] for k in ['status','images','requested','applied','predicted_crops','cached_png_bytes']},indent=2))


if __name__ == '__main__':
    main()
