"""Re-evaluate sampling seeds with uniform candidate-cache precision, no training."""
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_messenger_multiaxis as base
from src.models.messenger_feature_diffusion import FeatureDiffusion
from src.models.natural_feature_gate import NaturalFeatureGate
OUT=ROOT/'outputs/diagnostics/messenger_diffusion_comparison_20261114'


@torch.no_grad()
def main():
    read=lambda p:json.loads(p.read_text())
    assert read(OUT/'status.json')['phase']=='completed'
    plan=read(OUT/'plan.json')
    for name,h in plan['hashes'].items():assert base.sha(ROOT/name)==h
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    manifest=read(OUT/'manifest.json');wanted=set(manifest['roles']['eval']['natural'])
    anns=read(base.ANN);images={i['id']:i for i in anns['images']}
    records=sorted([a for a in anns['annotations'] if a['id'] in wanted],key=lambda a:manifest['roles']['eval']['natural'].index(a['id']))
    ids=np.array([manifest['faces'].index(a['id']) for a in records]);raw=yaml.safe_load(base.CONFIG.read_text())
    model,_=base.build_landmark_model(raw,variant='query_refinement',initialize_from_detector=False)
    base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True);model.cuda().eval().requires_grad_(False)
    frozen=base.tensor_digest(model);ds=base.Crops(records,images,Path(raw['data']['root']))
    mean=torch.tensor(raw['data']['mean'],device='cuda')[None,:,None,None];std=torch.tensor(raw['data']['std'],device='cuda')[None,:,None,None]
    maps=[];xy=[];states=[]
    for batch in base.batches(len(records),8):
        rows=[ds[int(i)] for i in batch]
        x=torch.tensor(np.stack([r['image'] for r in rows]),device='cuda').permute(0,3,1,2).float()/255
        maps.append(torch.stack(model.backbone.forward_intermediate_feature_maps((x-mean)/std,layer_indexes=plan['levels'] if 'levels' in plan else [3,7,11]),1).half().cpu())
        xy.extend([r['xy'] for r in rows]);states.extend([r['state'] for r in rows])
    maps=torch.cat(maps);xy=torch.tensor(np.array(xy));states=np.array(states);result={}
    for arm in ('diff_feature','diff_head'):
        previous=read(OUT/arm/'evaluation.json')
        for name in ('best_R.pt','best_g.pt'):assert base.sha(OUT/arm/name)==previous['weights'][name]
        ck=torch.load(OUT/arm/'best_R.pt',weights_only=True)['state']
        net=FeatureDiffusion(ck['mean'].cpu().flatten(),ck['std'].cpu().flatten()).cuda().eval();net.load_state_dict(ck)
        gate=NaturalFeatureGate(384).cuda().eval();gate.load_state_dict(torch.load(OUT/arm/'best_g.pt',weights_only=True)['state'])
        result[arm]={}
        for seed in [42,1042,2042,3042]:
            outputs=[];max_delta=0.
            for b in base.batches(len(ids)):
                m=base.gpu(maps[b]);r=net.sample(m,ids[b],seed,16);preds=[]
                for candidate in (r.half().float(),r):
                    _,y=gate(m[:,2],candidate,1.);pred=base.run_head(model.head,y)
                    preds.append(torch.stack(((pred['decoded_coordinates']-base.gpu(xy[b])).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
                max_delta=max(max_delta,float((preds[0]-preds[1]).abs().max()));outputs.append(torch.stack(preds,1).cpu().numpy())
            outputs=np.concatenate(outputs);success=(outputs[...,0]<=.1)&(outputs[...,1]>=.7)
            entry=dict(max_absolute_error_or_presence_delta=max_delta,changed_success_points=int(((success[:,0]!=success[:,1])&(states>0)).sum()))
            for k,precision in enumerate(('half_cache','full_precision')):
                entry[precision]={label:dict(correct=int(success[:,k][states==s].sum()),points=int((states==s).sum()),
                    mean_error=float(outputs[:,k,:,0][states==s].mean())) for s,label in [(1,'occluded'),(2,'visible')]}
            if seed==42:
                for label in ('occluded','visible'):
                    ref=previous['natural']['learned_g'][label];new=entry['half_cache'][label]
                    assert ref['correct']==new['correct']
                    assert abs(ref['mean_error']['mean']-new['mean_error'])<2e-5
            result[arm][str(seed)]=entry
        del net,gate
    assert base.tensor_digest(model)==frozen
    base.write(OUT/'sampling_precision_audit.json',dict(conditions=result,main_seed_reproduced=True,training=False,
        reason='Primary natural evaluation uses fp16 R cache; original additional seeds used fp32 R. Re-evaluate all seeds with same fp16 cache precision.'))
    print('sampling precision audit completed')


if __name__=='__main__':
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
