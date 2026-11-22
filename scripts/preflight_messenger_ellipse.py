"""Small real-crop gradient/parity check before the ellipse comparison."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import fcntl
import time
import json
import numpy as np
import torch
import yaml
from scripts import run_messenger_ellipse_comparison as run


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    out=ROOT/'outputs/diagnostics/messenger_ellipse_preflight_20261119';out.mkdir(exist_ok=True)
    with (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:print('waiting for GPU lease',flush=True);time.sleep(30)
        reg=yaml.safe_load(run.REGISTRY.read_text());cfg=run.load_config(ROOT/reg['config']['path']);cfg['model'].pop('init_checkpoint',None)
        model=run.build_detector_from_config(cfg);run.load_checkpoint(model,ROOT/reg['checkpoint']['path'],strict=True)
        probe=run.FrozenEllipseProbe(model).cuda().eval();frozen=run.tensor_digest(probe)
        corpus=run.read(run.ANN);images={i['id']:i for i in corpus['images']};annotations={a['id']:a for a in corpus['annotations']}
        predictions=run.read(run.CROPS)['predictions'];crops=[]
        fit_videos=set(run.read(run.PRIOR)['roles']['fit'])
        for group in predictions.values():
            for pred in group:
                a=annotations[pred['assigned_annotation_id']];info=images[a['image_id']]
                if info['metadata']['source_video_id'] not in fit_videos:continue
                if a['keypoints'][2::3]!=[2]*5 or a['attributes']['face_occluded'] is not False:continue
                row=run.make_crop(a,info,pred,Path(cfg['data']['image_root']))
                if len(row['synth'])==3:crops.append(row)
                if len(crops)==4:break
            if len(crops)==4:break
        assert len(crops)==4
        def rgb(rows):return torch.from_numpy(np.stack(rows)).cuda().permute(0,3,1,2).float()/255
        with torch.no_grad():
            x=rgb([r['synth'][0][0] for r in crops]);maps,levels,detail=probe.encode(x)
            clean,_,_=probe.encode(rgb([r['rgb'] for r in crops]))
            direct=probe.detector.decoder(probe.detector.encoder(probe.detector.backbone(x)))
            injected=probe(maps[:,2],levels,detail)
            parity={key:float((direct[key]-injected[key]).abs().max()) for key in ('pred_logits','pred_ellipses')}
            for key in parity:torch.testing.assert_close(direct[key],injected[key],atol=2e-5,rtol=2e-5)
        cache=dict(natural=dict(maps=clean.cpu()),synthetic=dict(maps=maps.cpu(),levels=[l.cpu() for l in levels],detail=detail.cpu()),
            parent=np.arange(4),target=torch.tensor(np.stack([r['target'] for r in crops])),
            covered=torch.tensor(np.stack([r['synth'][0][1] for r in crops])),outside=torch.tensor(np.stack([r['synth'][0][2] for r in crops])))
        stats=clean[:,2].mean((0,2,3)).cpu(),clean[:,2].std((0,2,3),unbiased=False).clamp_min(.01).cpu()
        results={}
        for arm in ('regression','diffusion'):
            torch.manual_seed(42);net=(run.FeatureDiffusion(*stats) if arm=='diffusion' else run.MultiAxisRecovery('scale')).cuda()
            optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);losses=[];shapes=[];tick=time.monotonic()
            with torch.no_grad():raw_pred,raw_score,_=probe.select(injected)
            for step in range(25):
                optimizer.zero_grad(set_to_none=True);loss,parts=run.objective(net,probe,cache,np.arange(4),42)
                if step==0:
                    grad=torch.autograd.grad(parts['shape'],tuple(net.parameters()),retain_graph=True,allow_unused=True)
                    gn=float(sum(g.square().sum() for g in grad if g is not None).sqrt());assert gn>0 and np.isfinite(gn)
                losses.append(float(loss.detach()));shapes.append(float(parts['shape'].detach()))
                if step<24:
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);optimizer.step()
            assert all(p.grad is None for p in probe.parameters()) and run.tensor_digest(probe)==frozen
            results[arm]=dict(initial_loss=losses[0],final_loss=losses[-1],initial_shape=shapes[0],final_shape=shapes[-1],
                best_shape=min(shapes),shape_gradient=gn,seconds=time.monotonic()-tick,losses=losses,shapes=shapes)
            assert min(shapes[1:])<shapes[0],arm
            del net,optimizer,loss,parts,grad;torch.cuda.empty_cache()
        run.write(out/'result.json',dict(parity=parity,arms=results,fixed_parent=frozen,no_clean_head=True,
            source_hashes={p:run.sha(ROOT/p) for p in ('scripts/preflight_messenger_ellipse.py','scripts/run_messenger_ellipse_comparison.py','src/models/messenger_ellipse_probe.py')}))
        print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
