"""Explain batch-size roundoff and measure natural fusion use; no clean-head control."""
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_messenger_position_primary as run
base=run.base


@torch.no_grad()
def main():
    folder=run.OUT/'batch_precision_audit';folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=yaml.safe_load(base.CONFIG.read_text());model,_=base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
        base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True);model.cuda().eval().requires_grad_(False)
        frozen=base.tensor_digest(model);cache=base.make_cache(folder,model,config['data'])
        results={}
        for domain in ('natural','artificial'):
            ids=cache['natural_eval' if domain=='natural' else 'syn_eval'];predictions={}
            for size in (4,16):
                rows=[]
                for b in base.batches(len(ids),size):
                    j=ids[b];x=base.gpu(cache['clean'][j,2]) if domain=='natural' else base.gpu(cache['syn'][j,2])
                    xy=base.gpu(cache['xy'][j if domain=='natural' else cache['parent'][j]])
                    p=base.run_head(model.head,x)
                    rows.append(torch.stack(((p['decoded_coordinates']-xy).norm(dim=-1)*1.25,p['presence_logits'].sigmoid()),-1).cpu().numpy())
                predictions[size]=np.concatenate(rows)
            current=np.load(run.OUT/'det_position'/f'{domain}_seed42.npz')
            historical=np.load(run.old.OUT/'det_feature'/f'{domain}_seed42.npz')
            np.testing.assert_allclose(predictions[4],current['rows'][:,0],rtol=0,atol=1e-7)
            np.testing.assert_allclose(predictions[16],historical['rows'][:,0],rtol=0,atol=1e-7)
            diff=abs(predictions[4]-predictions[16]);masks={'occluded':current['state']==1,'visible':current['state']==2} if domain=='natural' else {'covered':current['point'],'uncovered':~current['point']}
            result=dict(batch4_reproduces_current=True,batch16_reproduces_historical=True,max_error_difference=float(diff[...,0].max()),
                        max_presence_difference=float(diff[...,1].max()),threshold_changes={})
            for group,m in masks.items():
                result['threshold_changes'][group]={}
                for t in (.02,.03,.05,.1):
                    a,b=predictions[4][m],predictions[16][m];aj=(a[:,0]<=t)&(a[:,1]>=.7);bj=(b[:,0]<=t)&(b[:,1]>=.7)
                    result['threshold_changes'][group][str(t)]=dict(current=int(aj.sum()),historical=int(bj.sum()),flipped=int((aj!=bj).sum()))
            results[domain]=result
        norm=run.read(run.OUT/'normalization.json');stats=(torch.tensor(norm['mean']),torch.tensor(norm['std']));gates={}
        for arm in run.ARMS:
            net=run.old.make_net(arm,stats);net.load_state_dict(torch.load(run.OUT/arm/'best_R.pt',weights_only=True)['state']);net.eval().requires_grad_(False)
            gate=run.NaturalFeatureGate(384).cuda();gate.load_state_dict(torch.load(run.OUT/arm/'best_g.pt',weights_only=True)['state']);gate.eval().requires_grad_(False)
            values=[];ids=cache['natural_eval']
            for b in base.batches(len(ids),4):
                j=ids[b];maps=base.gpu(cache['clean'][j]);r=run.generate(net,maps,j)
                logits,_=gate(maps[:,2],r,1.);xy=base.gpu(cache['xy'][j]);ij=((xy*255+.5)/16).floor().long().clamp(0,15)
                values.append(logits.sigmoid()[:,0][torch.arange(len(j),device='cuda')[:,None],ij[:,:,1],ij[:,:,0]].cpu().numpy())
            values=np.concatenate(values);state=cache['state'][ids].numpy()
            gates[arm]={name:dict(mean=float(values[state==s].mean()),median=float(np.median(values[state==s])),p90=float(np.quantile(values[state==s],.9))) for s,name in ((1,'occluded'),(2,'visible'))}
            np.savez_compressed(folder/f'{arm}_natural_gate.npz',gate=values,state=state,ids=ids)
            del net,gate
        assert base.tensor_digest(model)==frozen
        base.write(folder/'summary.json',dict(batch_precision=results,gate_use=gates,frozen_parent_unchanged=True,clean_head=False))
        print(json.dumps(dict(batch_precision=results,gate_use=gates),indent=2))


if __name__=='__main__':main()
