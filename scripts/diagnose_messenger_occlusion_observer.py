#!/usr/bin/env python3
"""Read-only k3 EMA diagnostics on fixed validation predicted FACE crops.

No training/threshold selection. CPU by default to coexist with production.
Natural point labels are sparse, never dense segmentation supervision.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from keypoint_comparison1.landmark_detection.dataset import (
    CropAugmentation, TopDownFaceLandmarkDataset, PairedOcclusionConfig, image_to_tensor)
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_validation import ConditionalSemanticCounters
from keypoint_comparison1.landmark_detection.evaluation import predictions_to_original
from scripts.evaluate_messenger_geometry_prior import load_model
from scripts.evaluate_recovery_abc_diagnostics import summarize
from src.data.landmark_occlusion import apply_landmark_occlusion_numpy, transform_ellipse_affine


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binary_metrics(labels, scores, decisions):
    """Tie-correct empirical PR/AP and AUROC; larger score means anomaly."""
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    decisions = np.asarray(decisions, bool)
    tp = int((labels & decisions).sum()); fp = int((~labels & decisions).sum())
    fn = int((labels & ~decisions).sum()); tn = int((~labels & ~decisions).sum())
    order = np.argsort(-scores, kind='stable'); y = labels[order]; s = scores[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1] if len(s) else np.array([], int)
    pos = int(labels.sum()); neg = len(labels)-pos
    tps = np.cumsum(y)[ends]; fps = (ends+1)-tps
    recall = tps/max(1,pos); precision = tps/np.maximum(1,tps+fps)
    ap = float(np.sum(np.diff(np.r_[0,recall])*precision)) if pos else None
    auc = float(np.trapezoid(np.r_[0,tps/pos], np.r_[0,fps/neg])) if pos and neg else None
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/pos if pos else None,visible_false_positive_rate=fp/neg if neg else None,
                f1=2*tp/max(1,2*tp+fp+fn),average_precision=ap,auroc=auc,
                prevalence=pos/max(1,len(labels)),count=len(labels),
                pr_curve=[dict(threshold=float(s[e]),precision=float(p),recall=float(r))
                          for e,p,r in zip(ends,precision,recall)])


def point_cell(xy, size=256, grid=16):
    """Do not clip out-of-crop GT points onto an unrelated boundary patch."""
    x,y=map(float,xy)
    if not (math.isfinite(x) and math.isfinite(y) and 0<=x<size and 0<=y<size):
        return None
    return int(y*grid/size), int(x*grid/size)


def feature_errors(x, z):
    return dict(cosine_distance=(1-torch.nn.functional.cosine_similarity(x.float(),z.float(),dim=1))[0],
                mse=(x.float()-z.float()).square().mean(1)[0])


def run(args):
    started=time.perf_counter(); torch.set_num_threads(args.threads); torch.manual_seed(42)
    np.random.seed(42); random.seed(42)
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): raise ValueError('Refusing to overwrite a completed diagnostic')
    spec=json.loads(args.config.read_text()); production=ROOT/spec['output_root']
    done=json.loads((production/'training_success.json').read_text()); identity=done['identity']
    if done['status']!='complete' or identity['config_sha256']!=file_hash(args.config):
        raise ValueError('Production config/completion mismatch')
    for path,expected in identity['artifacts'].items():
        if file_hash(ROOT/path)!=expected: raise ValueError('Changed input artifact: '+path)
    # Historical gate source inventory includes unrelated subsequently extended modules.
    # Preserve it and expose every drift rather than pretending to rerun production gates.
    drift={p:dict(historical=h,current=file_hash(ROOT/p) if (ROOT/p).is_file() else None) for p,h in identity['sources'].items()
           if not (ROOT/p).is_file() or file_hash(ROOT/p)!=h}
    essential=['geometry_prior_recovery.py','geometry_prior_inputs.py','continuous_recovery.py']
    if any(any(p.endswith('/'+n) for n in essential) for p in drift):
        raise ValueError('Core inference source drift requires review: '+str(drift))
    checkpoint=production/'checkpoints/best.pt'
    prior_eval=json.loads((production/'evaluation/val/baseline_aligned/metrics.json').read_text())
    if file_hash(checkpoint)!=prior_eval['provenance']['query_checkpoint']['sha256']:
        raise ValueError('Checkpoint differs from locked evaluation')
    model,provenance=load_model(spec,identity,checkpoint,torch.device('cpu'))
    data=spec['parent_config']['data']
    dataset=TopDownFaceLandmarkDataset(annotation_path=ROOT/spec['val_annotations'],
        image_root=data['root'],training=False,input_size=data['input_size'],category_ids=data['category_ids'],
        require_visible_keypoint=False,augmentation=CropAugmentation(**data['augmentation']),
        mean=data['mean'],std=data['std'],geometry_prior_grid_size=(16,16),seed=42,
        detector_crop_cache=ROOT/spec['val_crops'])
    if len(dataset)!=768: raise ValueError('Changed conditional validation population')
    channel=model.channel_adoption.sigmoid().detach()[None,:,None,None]
    # Sample by stable random hash, independent of predictions and outcomes.
    clean_indices=[i for i,s in enumerate(dataset.samples)
                   if not s['face_occluded'] and s['visible'].all()]
    clean_indices.sort(key=lambda i:hashlib.sha256(f"42:{dataset.samples[i]['annotation']['id']}:O".encode()).hexdigest())
    artificial_indices=set(clean_indices[:args.artificial_crops])
    cfg=PairedOcclusionConfig.from_mapping(data['augmentation']['paired_occlusion']).landmark_mapping()
    groups=defaultdict(list); artificial=defaultdict(list); synth_labels=defaultdict(lambda:[[],[],[]])
    points=[]; manifests=[]; counters=Counter(); prediction_counts={k:ConditionalSemanticCounters() for k in ['actual','raw','mask_all_support']}
    natural_outcomes=defaultdict(Counter)
    corpus=json.loads((ROOT/spec['val_annotations']).read_text()); matched_ids={s['annotation']['id'] for s in dataset.samples}
    missing=[a for a in corpus['annotations'] if a['id'] not in matched_ids]
    missing_states=Counter(int(v) for a in missing for v in a['keypoints'][2::3])
    native=json.loads((ROOT/data['continuous_source_inventory']['val']['path']).read_text())
    native_by_id={int(a['id']):a for a in native['annotations']}
    video_by_image={int(im['id']):im['source_video_id'] for im in native['images']}
    image_hashes={}
    def record_outcome(mode,output,target):
        pred=predictions_to_original(output['decoded_coordinates'],target['inverse_affine'][None],input_size=256)
        scores=output['presence_logits'].sigmoid().float().numpy()
        prediction_counts[mode].add(pred,scores,[target])
        truth=target['original_keypoints'].numpy(); b=target['bbox_xyxy'].numpy(); side=1.2*max(b[2]-b[0],b[3]-b[1])
        nme=np.linalg.norm(pred[0]-truth,axis=1)/side; present=scores[0]>=.70-1e-7
        states=np.where(target['original_labeled'],np.where(target['original_visible'],2,1),0)
        for st in (1,2):
            g=states==st; c=natural_outcomes[f'{mode}/state{st}']; c['gt']+=int(g.sum());c['emitted']+=int((g&present).sum())
            c['tp010']+=int((g&present&(nme<=.1)).sum());c['tp005']+=int((g&present&(nme<=.05)).sum())
        return scores[0],nme
    with torch.inference_mode(), (out/'points.jsonl').open('w') as stream:
        for i in range(min(args.limit or len(dataset),len(dataset))):
            clean,t=dataset[i]; gp=t['geometry_prior']; geo=gp['geometry'][None];support=gp['support'][None]
            x=clean[None]; result=model(x,geometry=geo,support=support);d=result['prior_diagnostics']
            original=d['original'];candidate=d['candidate'];gate=d['gate'];mask=d['anomalous']
            coef=mask[:,None]*gate*channel; correction=(d['fused']-original).norm(dim=1)/original.norm(dim=1).clamp_min(1e-8)
            margin=model.bank.thresholds[d['selected_center']]-d['score']
            scores,nmes=record_outcome('actual',result,t)
            for mode,feat in [('raw',original),('mask_all_support',original+support[:,None]*gate*channel*(candidate-original))]:
                record_outcome(mode,model.base.head(feat),t)
            native_row=native_by_id[int(t['annotation_id'])]
            xy=t['keypoints'].numpy()*255
            states=np.where(t['original_labeled'],np.where(t['original_visible'],2,1),0)
            for slot,state in enumerate(states):
                if state==0: counters['missing_state0_excluded']+=1;continue
                cell=point_cell(xy[slot]);counters[f'state{state}_all_matched']+=1
                if cell is None: counters[f'state{state}_crop_outside']+=1;continue
                y,z=cell
                row=dict(annotation_id=int(t['annotation_id']),image_id=int(t['image_id']),video=video_by_image[int(t['image_id'])],
                    slot=slot,state=int(state),inside_support=bool(support[0,y,z]),m=bool(mask[0,y,z]),
                    margin=float(margin[0,y,z]) if support[0,y,z] else -3.,raw_margin=float(margin[0,y,z]),
                    g=float(gate[0,0,y,z]),coefficient=float(coef[0,:,y,z].mean()),
                    relative_feature_change=float(correction[0,y,z]),presence=float(scores[slot]),nme=float(nmes[slot]),
                    center=int(d['selected_center'][0]),face_contour_occluded=native_row.get('attributes',{}).get('face_occluded'))
                points.append(row);stream.write(json.dumps(row)+'\n')
                for field in ['g','coefficient','relative_feature_change']:
                    groups[f'state{state}/{field}'].append(row[field])
                counters[f'state{state}_outside_ellipse_support']+=int(not row['inside_support'])
            counters['crops']+=1;counters['anomalous_support_cells']+=int(mask.sum());counters['support_cells']+=int(support.sum())
            counters[f'center{int(d["selected_center"][0])}_crops']+=1
            image_path=Path(data['root'])/t['file_name']
            if str(image_path) not in image_hashes:image_hashes[str(image_path)]=file_hash(image_path)
            if i in artificial_indices:
                pixels=np.rint(np.clip(clean.permute(1,2,0).numpy()*np.asarray(data['std'])+np.asarray(data['mean']),0,1)*255).astype(np.uint8)
                ellipse=transform_ellipse_affine(t['gt_ellipse'].numpy(),t['affine'].numpy())
                teacher=model.observer(x)
                for kind in ['rectangle','irregular','boundary']:
                    seed=int.from_bytes(hashlib.sha256(f"42:{t['annotation_id']}:O:{kind}".encode()).digest()[:8],'big')
                    im,paste,_,meta=apply_landmark_occlusion_numpy(pixels,face_ellipse=ellipse,keypoints=xy,
                        keypoint_visibility=t['keypoint_visibility'].numpy(),config=cfg,rng=random.Random(seed),force=True,force_kind=kind)
                    manifests.append(dict(annotation_id=int(t['annotation_id']),kind=kind,seed=seed,metadata=meta))
                    if not meta['applied']:counters['artificial_skipped']+=1;continue
                    sx=image_to_tensor(im,mean=data['mean'],std=data['std'])[None]
                    sd=model(sx,geometry=geo,support=support)['prior_diagnostics']
                    ratio=torch.from_numpy(paste.copy()).float().reshape(16,16,16,16).mean((1,3))
                    known=(ratio>0)&support[0]; core=(ratio>=.5)&support[0]
                    smargin=model.bank.thresholds[sd['selected_center']]-sd['score']
                    for threshold,label in [(0.,'any_overlap'),(.5,'majority_overlap')]:
                        truth=(ratio>0) if threshold==0 else ratio>=threshold
                        a=synth_labels[kind+'/'+label];a[0].extend(truth[support[0]].tolist());a[1].extend(smargin[0][support[0]].tolist());a[2].extend(sd['anomalous'][0][support[0]].tolist())
                    oracle=sd['original']+known[None,None]*sd['gate']*channel*(sd['candidate']-sd['original'])
                    for region,r in [('occluded',known),('occluded_core',core),('uncovered',support[0]&~known)]:
                        if not r.any():continue
                        for branch,feat in [('X',sd['original']),('R',sd['candidate']),('Y',sd['fused']),('known_mask_Y',oracle)]:
                            for metric,e in feature_errors(feat,teacher).items():artificial[f'{kind}/{region}/{branch}/{metric}'].extend(e[r].tolist())
                        artificial[f'{kind}/{region}/g'].extend(sd['gate'][0,0][r].tolist())
                    counters['artificial_applied']+=1
            if (i+1)%16==0 or i==0:
                stream.flush();write(out/'progress.json',dict(crops=i+1,total=min(args.limit or len(dataset),len(dataset)),elapsed_seconds=time.perf_counter()-started,artificial_applied=counters['artificial_applied']))
                print(json.dumps(dict(crops=i+1,elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    point_metrics={}
    for group,subset in [('all_in_crop',points),('inside_support',[r for r in points if r['inside_support']])]+[(f'slot{s}',[r for r in points if r['slot']==s]) for s in range(5)]+[(f'center{k}',[r for r in points if r['center']==k]) for k in range(3)]+[(f'video/{v}',[r for r in points if r['video']==v]) for v in sorted(set(r['video'] for r in points))]:
        point_metrics[group]=binary_metrics([r['state']==1 for r in subset],[r['margin'] for r in subset],[r['m'] for r in subset])
    metrics=dict(status='completed_diagnostic_not_gate',split='val',training='not run',test_used=False,seed=42,
        topology='HEAD-free full-image FACE/ellipse -> predicted crop DINO + frozen observer -> prior Messenger -> Query',
        scope='Historical k3 EMA on canonical-v2, not current V3 baseline; natural point metrics conditional on matched FACE and in-crop GT',
        thresholds=dict(face=.33,presence=.70,nme=[.05,.10],O='original per-center per-cell calibrated threshold; no reselection'),
        missing_FACE=dict(faces=len(missing),state_counts=dict(missing_states)),cohort_report=dataset.detector_crop_report,
        counts=dict(counters),natural_point_detection=point_metrics,natural_distributions={k:summarize(v) for k,v in groups.items()},
        conditional_five_slot={k:v.compute() for k,v in prediction_counts.items()},natural_location_counts={k:dict(v) for k,v in natural_outcomes.items()},
        artificial_detection={k:binary_metrics(*v) for k,v in synth_labels.items()},
        artificial_features={k:summarize(v) for k,v in artificial.items()},artificial_available_clean_crops=len(clean_indices),
        artificial_requested_crops=len(artificial_indices),channel_adoption=summarize(channel.numpy()),
        provenance=dict(checkpoint=provenance,config_sha256=file_hash(args.config),source_drift=drift,
            current_historical_sources={p:file_hash(ROOT/p) for p in identity['sources']},script_sha256=file_hash(__file__),
            val_annotation_sha256=file_hash(ROOT/spec['val_annotations']),val_crops_sha256=file_hash(ROOT/spec['val_crops']),image_sha256=image_hashes),
        runtime=dict(device='cpu',threads=args.threads,elapsed_seconds=time.perf_counter()-started),
        limitations=['Natural labels are sparse points, not dense masks.','All-support and known-mask interventions are diagnostics, not deployment or retraining.',
                    'Inference is CPU FP32; existing production validation used CUDA autocast.','Same-patch semantic points may be correlated; no point-independence significance claim.',
                    'Artificial sample has all five original points visible; native Head annotations do not certify contour visibility. Target is fixed observer before added occlusion.',
                    'PR score is calibrated threshold minus similarity; outside predicted ellipse support receives -3, below every possible margin.'])
    write(out/'artificial_manifest.json',manifests);write(out/'metrics.json',metrics)
    write(out/'diagnostic_complete.json',dict(status=metrics['status'],metrics_sha256=file_hash(out/'metrics.json'),points_sha256=file_hash(out/'points.jsonl')))
    print('DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/training_plans/messenger_geometry_prior_runs/k3.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--artificial-crops',type=int,default=128);p.add_argument('--limit',type=int)
    run(p.parse_args())
