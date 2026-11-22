#!/usr/bin/env python3
"""Read-only initial/best gate comparison for a saved policy smoke, no retraining."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from keypoint_comparison1.landmark_detection.geometry_prior_training import file_hash
from keypoint_comparison1.landmark_detection.geometry_prior_launch import write_json
from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
from keypoint_comparison1.landmark_detection.losses import LandmarkLoss
from keypoint_comparison1.landmark_detection.real_occlusion import RealOcclusionBank
from scripts.evaluate_messenger_geometry_prior import load_model
from scripts.messenger_natural_occlusion_common import authenticate,datasets
from src.data.messenger_policy_occlusion import PolicyOcclusionDataset
from src.models.messenger_occlusion_policy import PolicyMessengerTask
from src.models.messenger_gate_metrics import GateCounters,fixed_gate_probe,gate_deltas
from src.models.messenger_occlusion_training import T_METRIC


def run(args):
    source=args.run.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'contract.json').read_text());cfg=original['config']
    base,spec,parent_identity,checkpoint,identity=authenticate(ROOT/cfg['base_config'])
    for key in ('parent_checkpoint_sha256','parent_artifacts','native_annotations','material_bank'):
        if identity[key]!=original['identity'][key]:raise ValueError('Source/data/parent identity changed: '+key)
    done=json.loads((source/'completion.json').read_text())
    best=source/'checkpoints/best.pt'
    if file_hash(best)!=done['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    saved=torch.load(best,map_location='cpu',weights_only=False)
    if saved['contract']['experiment']!=original or saved['epoch']!=saved['best_epoch']:
        raise ValueError('Checkpoint contract/selection mismatch')
    torch.set_num_threads(2);device=torch.device('cpu')
    parent,_=load_model(spec,parent_identity,checkpoint,device)
    train,val=datasets(spec);data=spec['parent_config']['data'];c=cfg['composition']
    bank=RealOcclusionBank(ROOT/base['material_bank']['path'],base['material_bank']['sha256'],
        annotation_path=data['annotations']['train'],image_root=data['root'])
    dataset=PolicyOcclusionDataset(train,bank,split='train',seed=cfg['seed'],
        **{k:c[k] for k in ('probability','material_candidates','placements_per_material','attempts')})
    fixture=json.loads((source/'fixtures.json').read_text())
    rows=[dataset[i] for i in fixture['indices']]
    if [[c['metadata'] for c in r['candidates']] for r in rows]!=fixture['candidates']:
        raise ValueError('Cannot reproduce saved train candidate pool')
    # Freeze the actual first-update actions for BOTH checkpoints, not new G draws.
    history=[json.loads(s) for s in (source/'history.jsonl').read_text().splitlines()]
    actions={r['index']:r['action'] for u in history[0]['updates'] for r in u['records'] if r['applied']}
    if set(actions)!={r['metadata']['index'] for r in rows if r['candidates']}:
        raise ValueError('Incomplete fixed action inventory')
    task=PolicyMessengerTask(parent,LandmarkLoss(spec['parent_config']['loss']),mean=data['mean'],std=data['std'],
        reconstruction_weight=base['task']['reconstruction_weight'])
    results={}
    for label in ('initial','best'):
        if label=='best':task.load_state_dict(saved['modules']['t'],strict=True)
        gates=GateCounters()
        metrics=validate_prior(parent,val,device=device,batch_size=4,
            diagnostics=lambda result,targets,hw:gates.add(result,targets,input_hw=hw,channel_adoption=parent.channel_adoption))
        probe=fixed_gate_probe(task,rows,actions=actions)
        results[label]=dict(validation=metrics,validation_gates=gates.compute(),fixed_train_probe=probe)
        write_json(output/(label+'.json'),results[label])
        print(json.dumps(dict(checkpoint=label,f1=metrics[T_METRIC],
            natural_g=gates.compute()['groups']['natural_occluded']['g_mean'])),flush=True)
    expected=json.loads((source/'epoch0_validation.json').read_text())[T_METRIC]
    if abs(results['initial']['validation'][T_METRIC]-expected)>1e-9 or abs(results['best']['validation'][T_METRIC]-done['best_f1'])>1e-9:
        raise ValueError('Replayed F1 differs from saved inference result')
    report=dict(status='completed_gate_diagnostic_not_training_gate',training='not run',test_used=False,
        source_run=str(source),initial_checkpoint_sha256=file_hash(checkpoint),best_checkpoint_sha256=file_hash(best),
        best_epoch=saved['epoch'],validation_delta=gate_deltas(results['best']['validation_gates'],results['initial']['validation_gates']),
        artificial_delta=gate_deltas(results['best']['fixed_train_probe']['synthetic'],results['initial']['fixed_train_probe']['synthetic']),
        results=results,provenance=identity,
        historical_source_drift={p:dict(original=h,current=file_hash(ROOT/p)) for p,h in original['identity']['sources'].items() if file_hash(ROOT/p)!=h},
        reporter_sha256=file_hash(__file__),saved_f1_reproduced=True)
    write_json(output/'metrics.json',report)
    lines=['# 採用係数g：初期値と保存済みbestの同一入力比較','',
        'gは遮蔽確率ではなく復元特徴の採用係数。O/mは別の異常判定。新規学習・Test使用なし。',
        '人工遮蔽はtrainの第1更新で実際に選んだ素材・配置を固定。自然遮蔽と可視点は未加工validation。',
        '可視点は「その点が遮蔽されていない」という意味で、顔全体が非遮蔽とは限らない。',
        'state0・crop外は除外。予測presenceで点を除外しない。各点の属するfeature tokenを集計。','',
        '| 条件 | 点数 | 初期g平均 | best g平均 | 差 | best g<0.01 | best実効融合係数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label,scope in [('artificial_occluded','人工遮蔽（train固定）','train'),('natural_occluded','自然遮蔽（val）','val'),('visible','遮蔽なしの可視点（val）','val')]:
        def get(stage):
            r=results[stage]
            return (r['fixed_train_probe']['synthetic'] if scope=='train' else r['validation_gates'])['groups'][key]
        a,b=get('initial'),get('best')
        fmt=lambda v:'N/A' if v is None else f'{v:.9f}'
        delta=None if a['g_mean'] is None else b['g_mean']-a['g_mean']
        lines.append(f"| {label} | {b['count']} | {fmt(a['g_mean'])} | {fmt(b['g_mean'])} | {fmt(delta)} | {fmt(b['fraction_g_lt_0_01'])} | {fmt(b['effective_coefficient_mean'])} |")
    lines+=['','位置誤差（crop正規化xyのSmooth L1）、gの標準偏差・最小/最大、g<0.001割合、O陽性割合、support外件数、遮蔽率別値、人工遮蔽前の同じ点のgはmetrics.jsonに記録。',
        '人工点と自然点は異なる母集団であり、平均値の差を遮蔽種類の因果効果としない。初期→bestの差は各集合内で同一入力を比較したもの。',
        f"保存済みbestはepoch{saved['epoch']}。既存2更新のcheckpoint診断で、収束・品質gate・baseline改善を証明しない。",'']
    (output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
