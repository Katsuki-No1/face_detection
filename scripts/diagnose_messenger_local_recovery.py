"""Frozen, evaluation-only local recovery and oracle-region interventions."""
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_messenger_multiaxis as base
from src.models.messenger_multiaxis import MultiAxisRecovery,feature_error
from src.models.natural_feature_gate import NaturalFeatureGate

SOURCE=ROOT/'outputs/diagnostics/messenger_multiaxis_convergence_20261114'
OUT=ROOT/'outputs/diagnostics/messenger_local_recovery_20261114'
ARMS=('scale_C','level_scale_C')
VARIANTS=('raw','full_R','learned_g','oracle_region_R','oracle_region_clean','clean_outside_R_inside','clean')


def read(p):return json.loads(p.read_text())


def boot(values,selected,videos):
    """Point-weighted mean, clustered by source video; paired differences allowed."""
    unique=np.unique(videos);sums=[];counts=[]
    for v in unique:
        m=(videos==v)[:,None]&selected
        sums.append(values[m].sum());counts.append(m.sum())
    sums=np.array(sums);counts=np.array(counts);valid=counts>0
    sums=sums[valid];counts=counts[valid]
    ix=np.random.default_rng(42).integers(0,len(sums),(2000,len(sums)))
    draws=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean=float(sums.sum()/counts.sum()),video_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                points=int(counts.sum()),videos=len(sums))


def point_features(f,xy):
    ij=((xy*255+.5)/16).floor().long().clamp(0,15)
    # Exact containing token; a 3x3 neighborhood has 32px center-to-center extent.
    center=f.permute(0,2,3,1)[torch.arange(len(f),device=f.device)[:,None],ij[:,:,1],ij[:,:,0]]
    offsets=torch.tensor([(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1)],device=f.device)
    nbr=ij[:,:,None,:]+offsets;valid=((nbr>=0)&(nbr<16)).all(-1)
    nbr=nbr.clamp(0,15)
    local=f.permute(0,2,3,1)[torch.arange(len(f),device=f.device)[:,None,None],nbr[...,1],nbr[...,0]]
    return center,local,valid


@torch.no_grad()
def run():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    plan=read(SOURCE/'plan.json');manifest=read(SOURCE/'manifest.json')
    for name,sha in plan['hashes'].items():assert base.sha(ROOT/name)==sha,name
    nets={};gates={};weight_hashes={}
    for arm in ARMS:
        evaluation=read(SOURCE/arm/'evaluation.json')
        for name in ('best_recovery.pt','best_gate.pt'):
            p=SOURCE/arm/name;actual=base.sha(p);assert actual==evaluation['weights'][name]
            weight_hashes[str(p.relative_to(ROOT))]=actual
        net=MultiAxisRecovery(arm.rsplit('_',1)[0]).cuda().eval()
        net.load_state_dict(torch.load(SOURCE/arm/'best_recovery.pt',weights_only=False)['state']);net.requires_grad_(False)
        gate=NaturalFeatureGate(384).cuda().eval()
        gate.load_state_dict(torch.load(SOURCE/arm/'best_gate.pt',weights_only=False)['state']);gate.requires_grad_(False)
        nets[arm]=net;gates[arm]=gate
    config=yaml.safe_load(base.CONFIG.read_text())
    model,_=base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
    base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True)
    model.cuda().eval().requires_grad_(False);before=base.tensor_digest(model)
    corpus=read(base.ANN);images={i['id']:i for i in corpus['images']}
    wanted=set(manifest['roles']['eval']['synthetic'])
    records=sorted([a for a in corpus['annotations'] if a['id'] in wanted],key=lambda a:a['id'])
    assert len(records)==128
    ds=base.Crops(records,images,Path(config['data']['root']))
    mean=torch.tensor(config['data']['mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(config['data']['std'],device='cuda')[None,:,None,None]
    def encode(rgb):
        x=torch.from_numpy(np.stack(rgb)).cuda().permute(0,3,1,2).float()/255
        maps=model.backbone.forward_intermediate_feature_maps((x-mean)/std,layer_indexes=plan['levels'])
        return torch.stack(maps,1).half().float()
    data={arm:dict(head=[],local=[],covered_error=[],outside_error=[],spatial_energy=[]) for arm in ARMS}
    point=[];videos=[];ids=[];counts=[];outside_counts=[]
    for batch in base.batches(len(records),8):
        rows=[ds[int(i)] for i in batch];clean=encode([r['image'] for r in rows]);rgb=[];parents=[];cov=[];ext=[];xy=[]
        for k,(i,row) in enumerate(zip(batch,rows)):
            assert len(row['synthetic'])==3
            for kind,img,c,e,p,fraction in row['synthetic']:
                rgb.append(img);parents.append(k);cov.append(c);ext.append(e);xy.append(row['xy'])
                point.append(p);ids.append((records[int(i)]['id'],kind))
                videos.append(images[records[int(i)]['image_id']]['metadata']['source_video_id'])
        syn=torch.cat([encode([rgb[int(i)] for i in b]) for b in base.batches(len(rgb),8)])
        z=clean[parents,2];x=syn[:,2];xy=torch.tensor(np.stack(xy),device='cuda',dtype=torch.float32)
        c=torch.tensor(np.stack(cov),device='cuda');e=torch.tensor(np.stack(ext),device='cuda')
        counts.extend(c.sum((1,2)).cpu().tolist());outside_counts.extend(e.sum((1,2)).cpu().tolist())
        zc,zl,valid=point_features(z,xy)
        for arm in ARMS:
            r,_,_=nets[arm](syn);_,g=gates[arm](x,r,1.)
            variants=(x,r,g,torch.where(c[:,None],r,x),torch.where(c[:,None],z,x),torch.where(c[:,None],r,z),z)
            head=[]
            for value in variants:
                pred=base.run_head(model.head,value)
                head.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
            data[arm]['head'].append(torch.stack(head,1).cpu().numpy())
            local=[];covered=[];outside=[];energy=[]
            for value in (x,r,z):
                vc,vl,_=point_features(value,xy)
                ce=1-F.cosine_similarity(vc,zc,dim=-1);mse=(vc-zc).square().mean(-1)
                le=1-F.cosine_similarity(vl,zl,dim=-1)+(vl-zl).square().mean(-1)
                local.append(torch.stack((ce,mse,(le*valid).sum(-1)/valid.sum(-1)), -1))
                fe=feature_error(value,z)
                covered.append((fe*c).sum((1,2)));outside.append((fe*e).sum((1,2)))
                # Local contrast about each neighborhood's mean, not across faces.
                avg=(vl*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
                energy.append((((vl-avg[...,None,:]).square().mean(-1))*valid).sum(-1)/valid.sum(-1))
            data[arm]['local'].append(torch.stack(local,1).cpu().numpy())
            data[arm]['covered_error'].append(torch.stack(covered,1).cpu().numpy())
            data[arm]['outside_error'].append(torch.stack(outside,1).cpu().numpy())
            data[arm]['spatial_energy'].append(torch.stack(energy,1).cpu().numpy())
        print('evaluated',int(batch[-1])+1,'/',len(records),flush=True)
    point=np.array(point,dtype=bool);videos=np.array(videos);counts=np.array(counts);outside_counts=np.array(outside_counts)
    assert point.sum()==663 and len(point)==384
    results={}
    for arm,parts in data.items():
        arrays={k:np.concatenate(v) for k,v in parts.items()}
        np.savez_compressed(OUT/f'{arm}.npz',**arrays,point=point,videos=videos,ids=np.array(ids),counts=counts,outside_counts=outside_counts)
        head=arrays['head'];ok=(head[...,0]<=.1)&(head[...,1]>=.7);local=arrays['local']
        result=dict(head={},local={},region_error={})
        for k,name in enumerate(VARIANTS):
            result['head'][name]=dict(correct=int(ok[:,k][point].sum()),points=int(point.sum()),
                mean_error=boot(head[:,k,:,0],point,videos),
                delta_success_vs_raw=boot(ok[:,k].astype(float)-ok[:,0],point,videos),
                unoccluded_correct=int(ok[:,k][~point].sum()),unoccluded_points=int((~point).sum()))
        for k,name in enumerate(('raw','R','clean')):
            result['local'][name]=dict(center_cosine=boot(local[:,k,:,0],point,videos),center_MSE=boot(local[:,k,:,1],point,videos),
                center_combined=boot(local[:,k,:,:2].sum(-1),point,videos),neighborhood_combined=boot(local[:,k,:,2],point,videos),
                local_spatial_energy=boot(arrays['spatial_energy'][:,k],point,videos))
            result['region_error'][name]=dict(covered=float(arrays['covered_error'][:,k].sum()/counts.sum()),
                outside=float(arrays['outside_error'][:,k].sum()/outside_counts.sum()))
        result['local']['R_minus_raw_center']=boot(local[:,1,:,:2].sum(-1)-local[:,0,:,:2].sum(-1),point,videos)
        result['local']['R_minus_raw_neighborhood']=boot(local[:,1,:,2]-local[:,0,:,2],point,videos)
        old=read(SOURCE/arm/'evaluation.json')
        assert abs(result['region_error']['R']['covered']-old['synthetic']['fused'])<2e-5
        for new,key in [('raw','raw'),('full_R','full_recovery'),('learned_g','learned_gate')]:
            assert result['head'][new]['correct']==old['extra']['synthetic_covered_head'][key]['correct'],(arm,new)
            assert abs(result['head'][new]['mean_error']['mean']-old['extra']['synthetic_covered_head'][key]['mean_error'])<2e-5
        results[arm]=result
    assert base.tensor_digest(model)==before
    for p,h in weight_hashes.items():assert base.sha(ROOT/p)==h
    base.write(OUT/'summary.json',dict(conditions=results,variants=VARIANTS,source=str(SOURCE.relative_to(ROOT)),
        weights=weight_hashes,backbone_unchanged=True,historical_metrics_reproduced=True,faces=128,pairs=384,
        masked_points=663,mask='same artificial covered tokens intersected with ellipse support; oracle intervention only',
        training=False,limitations=['reused artificial eval, no natural clean target','local variance is not proof of conditional-mean collapse',
        'oracle region and clean features are diagnostic, unavailable at natural-image inference','single seed, clustered bootstrap unadjusted']))
    print('COMPLETED',OUT,flush=True)


if __name__=='__main__':
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
