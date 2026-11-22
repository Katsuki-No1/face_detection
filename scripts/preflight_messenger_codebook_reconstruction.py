"""CPU distribution evidence and isolated 4/8/16 GPU batch probes."""
import argparse,json,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,yaml
from src.data import messenger_reconstruction_cache as dynamic
from scripts import run_messenger_codebook_reconstruction as run
OUT=ROOT/'outputs/diagnostics/messenger_codebook_reconstruction_preflight_20261121'

def distribution():
    manifest=run.read(run.BASELINE/'manifest.json');data=yaml.safe_load(run.base.CONFIG.read_text())['data']
    source=dynamic.sources(manifest,data);summaries=[]
    for role,epoch in [('fit',0),('fit',1),('fit',2),('dev',0),('eval',0)]:
        ids=sorted(manifest['roles'][role]['synthetic'])
        rows,rgb,c,e,p,masks,preview=dynamic.render_role(source,ids,role,epoch)
        summary=dynamic.audit_records(rows,masks,ids,OUT/'distribution',role,epoch,preview)
        summaries.append(summary);print(role,epoch,summary['counts'],summary['unique_masks'],flush=True)
    run.base.write(OUT/'distribution_complete.json',dict(passed=True,summaries=summaries))

def probe():
    with run.lease(False):
        run.gpu_idle(False);torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        config=yaml.safe_load(run.base.CONFIG.read_text())
        model,_=run.base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
        run.base.load_landmark_checkpoint(model,run.base.BASE,prefer_ema=True,strict=True)
        model.requires_grad_(False).eval().cuda()
        cache=dynamic.make_cache(OUT/'gpu',model,config['data'])
        manifest=run.read(OUT/'gpu/manifest.json');basis=run.load_basis(manifest).cuda()
        parents=np.unique(cache['parent'][cache['syn_fit']]);mean=cache['clean'][parents,2].float().mean((0,2,3)).cuda()
        # Probe the largest decoder and dictionary; real full-fit centers authenticate VQ behavior.
        centers,audit=run.initial_codes(cache,basis,mean,8192);centers=torch.from_numpy(centers)
        results=[]
        ids=cache['syn_fit'][:64];view=dict(cache,syn_dev=cache['syn_dev'][:64])
        for micro in (4,8,16):
            run.PLAN['microbatch']=micro;run.RUNTIME_MICROBATCH=micro
            net=run.make_recovery('E_d32_k8192',basis.cpu(),mean.cpu(),centers).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=run.PLAN['lr'],weight_decay=run.PLAN['weight_decay'])
            row=dict(microbatch=micro,effective_batch=16)
            try:
                run._train_epoch_once(net,model,cache,basis,'E_d32_k8192',opt,ids[:16])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                for repeat in range(2):run._train_epoch_once(net,model,cache,basis,'E_d32_k8192',opt,ids)
                torch.cuda.synchronize();train_seconds=(time.monotonic()-start)/2
                start=time.monotonic();dev=run.validate(net,model,view,basis,'E_d32_k8192');torch.cuda.synchronize()
                val_seconds=time.monotonic()-start
                import subprocess
                temp=float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
                peak=torch.cuda.max_memory_allocated();capacity=torch.cuda.get_device_properties(0).total_memory
                row.update(safe=peak<=.9*capacity and temp<83,train_seconds=train_seconds,val_seconds=val_seconds,
                    projected_seconds=train_seconds*20+val_seconds*10,peak_bytes=peak,capacity_bytes=capacity,temperature=temp,dev=dev)
            except torch.cuda.OutOfMemoryError:
                row.update(safe=False,error='CUDA OOM')
            finally:
                del net,opt;gc.collect();torch.cuda.empty_cache()
            results.append(row);print(row,flush=True)
        safe=[a for a in results if a['safe']]
        if not safe:raise RuntimeError('No safe batch candidate')
        best=min(a['projected_seconds'] for a in safe)
        selected=min(a['microbatch'] for a in safe if a['projected_seconds']<=best*1.03)
        run.base.write(OUT/'batch_plan.json',dict(passed=True,selected_microbatch=selected,effective_batch=16,candidates=results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['distribution','batch']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    distribution() if args.stage=='distribution' else probe()
