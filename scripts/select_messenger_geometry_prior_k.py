#!/usr/bin/env python3
"""Freeze k using authenticated completed production validation checkpoints only.

This CPU-only report never reads test artifacts, launches training, or promotes
weights. Missing/incomplete arms fail closed; exact ties choose the smaller k.
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_launch import authenticate,verify_gates,write_json
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash

METRIC='conditional_semantic_macro_f1_nme_0.10'


def choose(rows):
    if sorted(row['k'] for row in rows)!=list(range(1,8)):
        raise ValueError('All seven distinct completed k candidates are required')
    if any(not math.isfinite(row['value']) or not 0<=row['value']<=1 for row in rows):
        raise ValueError('Invalid validation metric')
    return min(rows,key=lambda row:(-row['value'],row['k']))


def collect(config_directory):
    # Check every completion before reading checkpoints or considering a winner.
    specs=[]
    for k in range(1,8):
        spec,identity=authenticate(config_directory/f'k{k}.json')
        verify_gates(spec,identity)
        out=ROOT/spec['output_root']; marker=out/'training_success.json'
        result=json.loads(marker.read_text())
        if result.get('status')!='complete' or result.get('identity')!=identity:
            raise ValueError(f'k{k} has no authenticated completed production')
        specs.append((k,spec,identity,out,marker,result))
    parent=specs[0][1]['parent_config']
    val_inputs=(specs[0][1]['val_annotations'],specs[0][1]['val_crops'])
    rows=[]
    for k,spec,identity,out,marker,result in specs:
        if spec['parent_config']!=parent or (spec['val_annotations'],spec['val_crops'])!=val_inputs:
            raise ValueError('Candidate data, initialization or validation contracts differ')
        summary={}
        for name in ['last','best']:
            path=out/f'checkpoints/{name}.pt'
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            experiment=state['contract']['experiment']
            if (state.get('schema')!='geometry-prior-runtime-v1' or experiment.get('identity')!=identity
                    or experiment.get('architecture')!=spec['architecture'] or experiment.get('mode')!='production'):
                raise ValueError('Checkpoint identity differs from completed production')
            driver=state['epoch_driver'];policy=driver['contract'];val=driver['last_row']['validation']
            if (policy['minimum_epoch']!=10 or policy['patience']!=5 or policy['min_delta']!=1e-4
                    or val.get('eligible_face_crops')!=768 or val.get('point_presence_threshold')!=.70
                    or val.get('nme_threshold')!=.10):
                raise ValueError('Checkpoint selection/validation contract differs')
            if name=='last':
                if state['epoch']!=result['epoch'] or state['step']!=result['step']:
                    raise ValueError('Completion does not match last checkpoint')
                if not (state['epoch']==policy['maximum_epochs'] or
                        (state['epoch']>=policy['minimum_epoch'] and driver['no_improvement']>=policy['patience'])):
                    raise ValueError('Production is not terminal under its stopping contract')
            else:
                if state['epoch']!=result['best_epoch'] or val[METRIC]!=result['best_value']:
                    raise ValueError('Reported best validation metric differs from best checkpoint')
            summary[name]=dict(path=str(path.relative_to(ROOT)),sha256=file_hash(path),epoch=state['epoch'],step=state['step'])
            del state
        rows.append(dict(k=k,value=result['best_value'],best_epoch=result['best_epoch'],
            architecture=spec['architecture'],prior=spec['prior'],identity=identity,
            completion_marker=str(marker.relative_to(ROOT)),completion_sha256=file_hash(marker),
            checkpoints=summary,wandb_url=result['wandb_url']))
    return rows


def freeze(rows,output):
    selected=choose(rows)
    result=dict(schema='geometry-prior-validation-selection-v1',status='success',selected_on_split='val',
        metric=METRIC,point_presence_threshold=.70,nme_threshold=.10,eligible_face_crops=768,
        tie_rule='maximum metric; exact numerical ties choose smaller k',selected_k=selected['k'],
        selected_architecture=selected['architecture'],selected_prior=selected['prior'],candidates=rows,
        test_used=False,adversarial_training='not run',baseline_promotion=False)
    if output.exists():
        if json.loads(output.read_text())!=result:
            raise ValueError('Refuse to overwrite a different frozen k selection')
    else:
        write_json(output,result)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--config-directory',type=Path,default=ROOT/'configs/training_plans/messenger_geometry_prior_runs')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/landmark_detection/messenger_geometry_prior/selection/validation_selected_k.json')
    args=parser.parse_args()
    result=freeze(collect(args.config_directory),args.output)
    print(json.dumps(dict(status=result['status'],selected_k=result['selected_k'],metric=result['metric'])))
