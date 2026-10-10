#!/usr/bin/env python3
"""Natural occlusion diagnosis on fixed validation, with predicted FACE crops."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.audit_messenger_feature_recovery import branches, sha
from keypoint_comparison1.landmark_detection.config import load_landmark_config
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.evaluation import predictions_to_original
from keypoint_comparison1.landmark_detection.training import make_dataset, load_landmark_checkpoint
from keypoint_comparison1.landmark_detection.model import build_landmark_model
from src.data.face_crop_cache import canonicalize_ellipse


def discrimination(labels, scores):
    """Tie-aware AUROC/AP. No threshold is selected from these diagnostics."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    assert len(labels) == len(scores) and np.isfinite(scores).all()
    pos, neg = int(labels.sum()), int((~labels).sum())
    auc = ap = None
    if pos and neg:
        order = np.argsort(scores, kind='stable')
        values, truth = scores[order], labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
        ends = np.r_[starts[1:], len(values)]
        favorable, negatives_before = 0., 0
        for start, end in zip(starts, ends):
            p = int(truth[start:end].sum()); n = end-start-p
            favorable += p*(negatives_before + .5*n)
            negatives_before += n
        auc = favorable/(pos*neg)
        tp = fp = 0; ap = 0.
        for start, end in zip(starts[::-1], ends[::-1]):
            p = int(truth[start:end].sum()); n = end-start-p
            tp += p; fp += n
            ap += (p/pos)*(tp/(tp+fp))
    thresholds = {}
    for threshold in (.1, .3, .5):
        pred = scores >= threshold
        tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
        fn = pos-tp; tn = neg-fp
        thresholds[str(threshold)] = dict(tp=tp,fp=fp,fn=fn,tn=tn,
            precision=tp/(tp+fp) if tp+fp else 0., recall=tp/pos if pos else None,
            false_positive_rate=fp/neg if neg else None)
    return dict(occluded_points=pos, visible_points=neg, prevalence=pos/max(1,pos+neg),
                auroc=auc, average_precision=ap, thresholds=thresholds,
                occluded_mean=float(scores[labels].mean()) if pos else None,
                visible_mean=float(scores[~labels].mean()) if neg else None,
                occluded_median=float(np.median(scores[labels])) if pos else None,
                visible_median=float(np.median(scores[~labels])) if neg else None)


def keypoint_metrics(rows, mode, state=None):
    rows = [r for r in rows if state is None or r['state']==state]
    n = len(rows)
    fs = []; tp_total = 0
    for slots in [(0,1),(2,),(3,4)]:
        group = [r for r in rows if r['slot'] in slots]
        if not group: continue
        pred = sum(r[mode]['presence'] >= .7 for r in group)
        tp = sum(r[mode]['presence'] >= .7 and r[mode]['nme'] <= .10 for r in group)
        tp_total += tp
        fs.append(2*tp/(len(group)+pred))
    return dict(points=n, nme=float(np.mean([r[mode]['nme'] for r in rows])) if n else None,
                semantic_macro_f1_nme_0_10=float(np.mean(fs)) if fs else None,
                correct_present_points=tp_total, recall=tp_total/n if n else None)


def effect(rows, mode):
    delta = np.asarray([r['raw']['nme']-r[mode]['nme'] for r in rows])
    raw_correct = np.asarray([r['raw']['presence']>=.7 and r['raw']['nme']<=.1 for r in rows])
    after_correct = np.asarray([r[mode]['presence']>=.7 and r[mode]['nme']<=.1 for r in rows])
    return dict(points=len(rows), mean_nme_reduction=float(delta.mean()) if len(rows) else None,
                coordinate_improved=int((delta>1e-6).sum()),coordinate_worsened=int((delta < -1e-6).sum()),
                rescued=int((~raw_correct & after_correct).sum()),lost=int((raw_correct & ~after_correct).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=['v1','direct','gram','option1','local','option3','option3_consistency','combined_option1'])
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.manual_seed(42);np.random.seed(42)
    canonical=ROOT/'outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json'
    assert sha(canonical)=='0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0'
    cd=json.loads(canonical.read_text());image_names={x['id']:x['file_name'] for x in cd['images']}
    heads={}
    for a in cd['annotations']:
        if a.get('attribute_valid') and a.get('ellipse'):
            key=(image_names[a['image_id']],tuple(round(float(x),4) for x in canonicalize_ellipse(a['ellipse'])))
            assert key not in heads
            heads[key]=a
    config_root=ROOT/'keypoint_comparison1/configs'
    direct=load_landmark_config(config_root/'continuous_recovery_direct.yaml')
    cache=direct.output_root/'evaluation/v2_val/cache/deimv2_l_predictions.json'
    cache_data=json.loads(cache.read_text());meta=cache_data['metadata']
    for key in ['detector_config','detector_checkpoint','annotation']:
        assert sha(meta[key])==meta[key+'_sha256']
    selected_path=direct.output_root/'query_refinement/evaluation/val/baseline_aligned/metrics.json'
    locked=json.loads(selected_path.read_text())['primary_local_face_bbox']['validation_selected']['threshold']
    derived=copy.deepcopy(cache_data)
    derived['predictions']={k:[r for r in rows if r['score']>=locked] for k,rows in derived['predictions'].items()}
    derived['metadata']['score_threshold']=locked
    derived['metadata']['source_cache']=str(cache);derived['metadata']['source_cache_sha256']=sha(cache)
    filtered=args.output/'validation_locked_face_cache.json'
    filtered.write_text(json.dumps(derived)+'\n')
    raw=copy.deepcopy(direct.raw);ev=raw['data']['evaluation_sets']['v2_val']
    raw['data']['annotations']['val']=ev['annotation'];raw['data'].setdefault('image_roots',{})['val']=ev['root']
    raw['data']['validation_detector_crop_cache']=str(filtered)
    source=make_dataset(raw,split='val',training=False)
    lookup={};unmatched=0
    for sample in source.samples:
        key=(sample['image']['file_name'],tuple(round(float(x),4) for x in sample['gt_ellipse']))
        head=heads.get(key)
        if head is None:unmatched+=1;continue
        states=np.asarray(head['keypoint_states']);valid=states>0
        assert np.allclose(sample['keypoints'][valid],np.asarray(head['keypoints'])[valid],atol=1e-3)
        lookup[sample['annotation']['id']]=head
    loader=DataLoader(source,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_landmark_batch)
    modes=['raw','recovered','gt_occluded_output','gt_occluded_candidate']
    common=dict(schema_version=1,split='val',seed=42,diagnostic_only=True,device='cpu',threads=args.threads,
        topology='HEAD-free ordered full-image FACE/ellipse -> predicted Face-crop DINO -> Messenger -> Query Refinement',
        scope='conditional on predicted FACE bbox matched to GT FACE; no detector rerun; natural inputs with no artificial occlusion',
        unsupported='natural clean counterfactual features and pixelwise occlusion masks unavailable; scores evaluated at annotated keypoint patches only',
        score_definitions={'alpha':'Messenger occlusion-ratio estimate; not final keypoint state classifier',
                           'adoption':'feature mixing amount, not an occlusion probability'},
        intervention_definitions={'raw':'same checkpoint/DINO, bypass Messenger',
            'recovered':'normal deployed blended features',
            'gt_occluded_output':'diagnostic only: apply normal recovered features only to patches containing in-crop GT state1 keypoints',
            'gt_occluded_candidate':'diagnostic only: replace those GT state1 patches with full candidate; no gate; not deployable'},
        annotation=dict(path=str(canonical),sha256=sha(canonical),adapter=ev['annotation'],adapter_sha256=sha(ev['annotation']),
            canonical_images=len(cd['images']),canonical_heads=len(cd['annotations']),matched_face_crops=len(source),
            valid_canonical_matched_crops=len(lookup),unmatched_or_ambiguous_crops=unmatched),
        detector=dict(source=str(cache),sha256=sha(cache),metadata=meta,derived=str(filtered),derived_sha256=sha(filtered),
            threshold=locked,threshold_source=str(selected_path),crop_matching=source.detector_crop_report),
        coordinate_convention='image x-right/y-down; inverse affine from predicted ellipse crop256 scale1.25; NME denominator1.20*max(canonical GT Head width,height)',
        keypoint_presence_threshold=.7,nme_threshold=.1,occlusion_score_thresholds=[.1,.3,.5],threshold_selection='none; diagnostic fixed points only',
        patch_sampling='floor(normalized_coordinate*(input_size-1)/16); out-of-crop labeled points excluded from score/feature diagnostic but retained as localization denominator',
        checkpoint_selection='existing validation-selected best, EMA preferred; no model or threshold reselection',
        script_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/audit_messenger_feature_recovery.py'))
    results={};input_hash=None
    for name in args.models:
        path=config_root/('dinov3_vits16_face_pose_drive_v2_messenger_recovery.yaml' if name=='v1' else f'continuous_recovery_{name}.yaml')
        cfg=load_landmark_config(path);checkpoint=cfg.output_root/'query_refinement/checkpoints/best_conditional_semantic_macro_f1.pt'
        print('START',name,flush=True)
        model,_=build_landmark_model(cfg.raw,variant='query_refinement',initialize_from_detector=False)
        info=load_landmark_checkpoint(model,checkpoint,prefer_ema=True,strict=True);model.eval()
        rows=[];digest=hashlib.sha256();started=time.perf_counter()
        with torch.inference_mode():
            for bi,(images,targets) in enumerate(loader):
                digest.update(images.numpy().tobytes())
                feature=model.backbone(images);x,proposals,diag=branches(model.messenger_recovery,feature)
                b,c,h,w=feature.shape
                occluded_mask=torch.zeros(b,h*w,dtype=torch.bool)
                valid_info=[]
                for index,target in enumerate(targets):
                    head=lookup.get(target['annotation_id'])
                    if head is None:valid_info.append(None);continue
                    points=target['keypoints'].numpy()*255
                    inside=((points>=0)&(points<256)).all(-1)
                    px=np.floor(points[:,0]/16).astype(int).clip(0,w-1)
                    py=np.floor(points[:,1]/16).astype(int).clip(0,h-1)
                    ids=py*w+px;states=np.asarray(head['keypoint_states'])
                    occluded_mask[index,torch.as_tensor(ids[inside&(states==1)])]=True
                    valid_info.append((head,inside,ids))
                maps={'raw':feature,'recovered':proposals['output'].transpose(1,2).reshape(b,c,h,w)}
                for mode,branch in [('gt_occluded_output','output'),('gt_occluded_candidate','candidate')]:
                    maps[mode]=torch.where(occluded_mask[...,None],proposals[branch],x).transpose(1,2).reshape(b,c,h,w)
                decoded={}
                inverse=torch.stack([t['inverse_affine'] for t in targets])
                for mode,features in maps.items():
                    out=model.head(features)
                    decoded[mode]=(predictions_to_original(out['decoded_coordinates'],inverse,input_size=256),out['presence_logits'].sigmoid().numpy())
                alpha=diag['gate_alpha'].numpy();gate=diag.get('adoption_gate',diag['gate_alpha']).numpy()
                rms=(proposals['output']-x).square().mean(-1).sqrt().numpy()
                relative=((proposals['output']-x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).numpy()
                candidate_relative=((proposals['candidate']-x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).numpy()
                for index,(target,item) in enumerate(zip(targets,valid_info)):
                    if item is None:continue
                    head,inside,ids=item;side=1.2*max(head['bbox'][2:]);truth=np.asarray(head['keypoints'])
                    for slot,state in enumerate(head['keypoint_states']):
                        if state not in (1,2):continue
                        patch=int(ids[slot]);row=dict(image_id=head['image_id'],head_id=head['id'],annotation_id=target['annotation_id'],
                            slot=slot,state=state,in_crop=bool(inside[slot]),patch=patch)
                        if inside[slot]:
                            row.update(alpha=float(alpha[index,patch]),adoption=float(gate[index,patch]),
                                output_change_rms=float(rms[index,patch]),output_relative_change=float(relative[index,patch]),candidate_relative_change=float(candidate_relative[index,patch]))
                        for mode,(prediction,presence) in decoded.items():
                            row[mode]=dict(nme=float(np.linalg.norm(prediction[index,slot]-truth[slot])/side),presence=float(presence[index,slot]))
                        rows.append(row)
                if bi%15==0:print(' crops',min((bi+1)*args.batch_size,len(source)),'/',len(source),flush=True)
        scored=[r for r in rows if r['in_crop']]
        discrimination_results={}
        for label,slots in [('all',range(5)),('eye',(0,1)),('nose',(2,)),('mouth',(3,4))]:
            sample=[r for r in scored if r['slot'] in slots]
            discrimination_results[label]={score:discrimination([r['state']==1 for r in sample],[r[score] for r in sample]) for score in ['alpha','adoption']}
        features={label:{key:float(np.mean([r[key] for r in scored if r['state']==state])) for key in ['alpha','adoption','output_change_rms','output_relative_change','candidate_relative_change']} for label,state in [('occluded',1),('visible',2)]}
        metrics={mode:{label:keypoint_metrics(rows,mode,state) for label,state in [('all',None),('occluded',1),('visible',2)]} for mode in modes}
        effects={mode:{label:effect([r for r in rows if state is None or r['state']==state],mode) for label,state in [('all',None),('occluded',1),('visible',2)]} for mode in modes[1:]}
        assert input_hash in (None,digest.hexdigest());input_hash=digest.hexdigest()
        points_path=args.output/f'{name}_points.jsonl'
        points_path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
        payload=dict(**common,status='complete',model=name,config=dict(path=str(path),sha256=sha(path)),
            checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),epoch=info['epoch']),
            input_sha256=input_hash,point_rows=len(rows),scored_points=len(scored),out_of_crop_points=len(rows)-len(scored),
            score_discrimination=discrimination_results,feature_changes=features,keypoint_metrics=metrics,paired_effects=effects,
            point_records=dict(path=str(points_path),sha256=sha(points_path)),elapsed_seconds=time.perf_counter()-started)
        out=args.output/f'{name}.json';out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        results[name]=dict(path=str(out),sha256=sha(out),score_discrimination=discrimination_results,feature_changes=features,keypoint_metrics=metrics,paired_effects=effects)
        print('DONE',name,'alpha',discrimination_results['all']['alpha'],'effects',effects['recovered']['occluded'],flush=True)
        del model
    comparison=args.output/'comparison.json';comparison.write_text(json.dumps(dict(**common,status='complete',input_sha256=input_hash,models=results),indent=2,allow_nan=False)+'\n')
    (args.output/'success.json').write_text(json.dumps(dict(status='success',meaning='diagnostic execution and finite results, not proof of improvement',comparison_sha256=sha(comparison),models=list(results)),indent=2)+'\n')


if __name__=='__main__':main()
