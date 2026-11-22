"""Authentication shared by the new natural-occlusion runners only."""
import hashlib
import json
from pathlib import Path

from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset,CropAugmentation
from scripts.evaluate_messenger_geometry_prior import load_model

ROOT=Path(__file__).resolve().parents[1]
TOPOLOGY=['full_image_face_ellipse','face_crop_dino','messenger_prior_fusion','query_refinement']


def authenticate(config):
    cfg=json.loads(Path(config).read_text())
    if cfg['schema']!='messenger-natural-occlusion-v1' or cfg['topology']!=TOPOLOGY:
        raise ValueError('Missing/contradictory HEAD-free topology')
    if cfg['observer']['selection']!='occlusion_f1' or cfg['task']['selection']!='conditional_semantic_macro_f1_nme_0.10':
        raise ValueError('Checkpoint selection must use the declared F1 contracts')
    if cfg['train']['ema'] or cfg['train']['precision']!='fp32':raise ValueError('Initial runtime explicitly uses raw FP32 checkpoints')
    # All implemented policy constants must agree with the declared config.
    if (cfg['discriminator']['patch_size']!=32 or cfg['task']['presence_threshold']!=.7
            or cfg['composition']['bins']!=[[.05,.2],[.2,.4],[.4,.6],[.6,.8]]
            or cfg['composition']['family_fractions']!={'face':.5,'non_face':.5}
            or cfg['composition']['minimum_requests_per_stratum']!=10):
        raise ValueError('Unsupported policy change requires implementation and new gates')
    spec_path=ROOT/cfg['parent_config'];spec=json.loads(spec_path.read_text());production=ROOT/spec['output_root']
    done=json.loads((production/'training_success.json').read_text());parent_identity=done['identity']
    if done['status']!='complete' or parent_identity['config_sha256']!=file_hash(spec_path):
        raise ValueError('Parent production/config identity changed')
    for path,expected in parent_identity['artifacts'].items():
        if file_hash(ROOT/path)!=expected:raise ValueError('Parent artifact changed: '+path)
    checkpoint=production/'checkpoints/best.pt'
    if file_hash(checkpoint)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    drift={p:dict(historical=h,current=file_hash(ROOT/p)) for p,h in parent_identity['sources'].items() if file_hash(ROOT/p)!=h}
    if any(p.endswith(('/geometry_prior_recovery.py','/geometry_prior_inputs.py','/continuous_recovery.py')) for p in drift):
        raise ValueError('Parent inference source changed')
    sources=set(parent_identity['sources'])
    for folder,pattern in [('src/models','messenger*.py'),('src/data','messenger*.py'),('scripts','*messenger*natural*.py')]:
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).glob(pattern))
    sources.update(['scripts/opaque_boundary_gradient.py','src/data/landmark_occlusion.py',
                    'scripts/evaluate_messenger_geometry_prior.py','scripts/train_messenger_natural_occlusion.py'])
    native={};videos={}
    for split in ('train','val'):
        record=spec['parent_config']['data']['continuous_source_inventory'][split]
        if file_hash(ROOT/record['path'])!=record['sha256']:raise ValueError('Dataset identity changed')
        native[split]=record
        videos[split]={im['source_video_id'] for im in json.loads((ROOT/record['path']).read_text())['images']}
    if videos['train'] & videos['val']:raise ValueError('Train/val video overlap')
    material=cfg['material_bank']
    if file_hash(ROOT/material['path'])!=material['sha256']:raise ValueError('Donor bank changed')
    identity=dict(config_sha256=file_hash(config),sources={p:file_hash(ROOT/p) for p in sorted(sources)},
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],parent_artifacts=parent_identity['artifacts'],
        native_annotations=native,material_bank=material,historical_source_drift=drift)
    return cfg,spec,parent_identity,checkpoint,identity


def datasets(spec):
    d=spec['parent_config']['data']
    shared=dict(image_root=d['root'],training=False,input_size=d['input_size'],category_ids=d['category_ids'],
        require_visible_keypoint=False,augmentation=CropAugmentation(**d['augmentation']),
        mean=d['mean'],std=d['std'],geometry_prior_grid_size=(16,16),seed=42)
    train=TopDownFaceLandmarkDataset(annotation_path=d['annotations']['train'],detector_crop_cache=ROOT/spec['train_crops'],**shared)
    val=TopDownFaceLandmarkDataset(annotation_path=ROOT/spec['val_annotations'],detector_crop_cache=ROOT/spec['val_crops'],**shared)
    if len(train)!=6185 or len(val)!=768:raise ValueError('Canonical predicted crop population changed')
    return train,val


def verify_gates(directory, contract):
    """Fresh per-arm gates, exact source/data/parent/optional O/G identities."""
    directory=Path(directory)
    for stage in ('schema','gpu_step','overfit','small_scale'):
        marker=json.loads((directory/stage/'success.json').read_text())
        if marker.get('status')!='passed' or marker.get('stage')!=stage or marker.get('contract')!=contract:
            raise ValueError('Missing/stale independent gate: '+stage)
        evidence=marker.get('evidence',{})
        if not evidence or any(file_hash(ROOT/p)!=h for p,h in evidence.items()):
            raise ValueError('Gate evidence changed or missing')
    plan=json.loads((directory/'gpu_step/batch_plan.json').read_text())
    if plan.get('contract')!=contract or [a['microbatch'] for a in plan['attempts']]!=[4,8,16]:
        raise ValueError('Fresh isolated 4/8/16 preflight required')
    safe=[]
    for a in plan['attempts']:
        if a['status']!='safe':continue
        if (a['optimizer_updates']<2 or a['validation_crops']!=768 or a['peak_vram_fraction']>.90
                or a['maximum_temperature']>=83 or not a['finite'] or a['oom'] or a['projected_seconds']<=0):
            raise ValueError('Unsafe/incomplete preflight evidence')
        safe.append(a)
    if not safe:raise ValueError('No safe preflight candidate')
    fastest=min(a['projected_seconds'] for a in safe)
    chosen=min(a['microbatch'] for a in safe if a['projected_seconds']<=fastest*1.03)
    if plan['selected_microbatch']!=chosen:raise ValueError('Batch choice violates elapsed-time rule')
    return chosen
