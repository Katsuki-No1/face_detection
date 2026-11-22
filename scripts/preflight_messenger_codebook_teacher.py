"""CPU technical check on two real fit faces; no convergence/quality claim."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_messenger_codebook_teacher as run
from src.models.messenger_codebook_teacher import ARMS, DIMENSIONS, COUNTS, arm_spec, make_recovery
from src.models.messenger_multiaxis import MultiAxisRecovery

OUT=ROOT/'outputs/diagnostics/messenger_codebook_teacher_preflight_20261120'


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    base=run.base
    hashes=run.source_hashes()
    manifest=run.read(run.BASELINE/'manifest.json')
    full_basis=run.load_basis(manifest)
    corpus=run.read(base.ANN)
    images={row['id']:row for row in corpus['images']}
    annotations={row['id']:row for row in corpus['annotations']}
    ids=manifest['roles']['fit']['synthetic'][:2]
    config=yaml.safe_load(base.CONFIG.read_text())
    dataset=base.Crops([annotations[i] for i in ids],images,Path(config['data']['root']))
    rows=[dataset[i] for i in range(2)]
    model,_=base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False)
    base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True)
    model.cpu().eval().requires_grad_(False)
    fixed=base.tensor_digest(model)
    mean_rgb=torch.tensor(config['data']['mean'])[None,:,None,None]
    std_rgb=torch.tensor(config['data']['std'])[None,:,None,None]
    def encode(rgb):
        x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).float()/255
        with torch.no_grad():
            maps=model.backbone.forward_intermediate_feature_maps((x-mean_rgb)/std_rgb,layer_indexes=[3,7,11])
        return torch.stack(maps,1).half().float()
    clean=encode([row['image'] for row in rows])
    synthetic=[row['synthetic'][0] for row in rows]
    maps=encode([row[1] for row in synthetic])
    teacher=clean[:,2]
    mean=teacher.mean((0,2,3))
    xy=torch.tensor(np.stack([row['xy'] for row in rows])).float()
    covered=torch.tensor(np.stack([row[2] for row in synthetic]))
    outside=torch.tensor(np.stack([row[3] for row in synthetic]))
    points=torch.tensor(np.stack([row[4] for row in synthetic]))
    valid=torch.tensor(np.stack([row['state'] for row in rows]))>0
    assert (points&valid).any() and (~points&valid).any()
    centers={}
    for dimension in DIMENSIONS:
        projected=torch.einsum('cr,bchw->bhwr',full_basis[:,:dimension],teacher-mean[None,:,None,None]).reshape(-1,dimension)
        for count in COUNTS:
            centers[f'd{dimension}_k{count}']=projected.repeat((count+len(projected)-1)//len(projected),1)[:count].clone()
    results={}
    initial_outputs={}
    for arm in ARMS:
        tick=time.monotonic()
        spec=arm_spec(arm)
        basis=full_basis[:,:spec['dimension']]
        key=f'd{spec["dimension"]}_k{spec["count"]}'
        net=make_recovery(arm,basis,mean,centers.get(key))
        before=base.tensor_digest(net)
        optimizer=torch.optim.AdamW(net.parameters(),lr=run.PLAN['lr'])
        restored,details=run.generate(net,maps)
        initial_outputs[arm]=restored.detach().clone()
        parts,_=run.loss_parts(model,restored,maps[:,2],teacher,covered,outside,xy,points,valid,basis,arm,details)
        assert all(torch.isfinite(value) for value in parts.values())
        position_code_gradient=None
        if getattr(net,'codebook',None) is not None and net.codebook.embedding.requires_grad:
            gradient=torch.autograd.grad(parts['position'],net.codebook.embedding,retain_graph=True)[0]
            position_code_gradient=float(gradient.norm())
            assert position_code_gradient>0
        parts['objective'].backward()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        optimizer.step()
        if getattr(net,"codebook",None) is not None: net.codebook.end_step()
        assert base.tensor_digest(net)!=before
        code_delta=None
        if getattr(net,'codebook',None) is not None:
            code_delta=float((net.codebook.embedding.detach()-centers[key]).norm())
            assert code_delta>0
            assert (net.codebook.embedding.grad is None) == (spec['update']=='ema')
        results[arm]=dict(objectives={k:float(v.detach()) for k,v in parts.items()},
                         gradient_norm=float(norm),position_only_code_gradient_norm=position_code_gradient,
                         code_change_l2=code_delta,seconds=time.monotonic()-tick,parameters=sum(p.numel() for p in net.parameters()))
    for dimension in DIMENSIONS:
        for count in COUNTS:
            for a,b in (('A','C'),('B','D'),('D','E')):
                torch.testing.assert_close(initial_outputs[f'{a}_d{dimension}_k{count}'],initial_outputs[f'{b}_d{dimension}_k{count}'],rtol=0,atol=0)
    assert base.tensor_digest(model)==fixed and all(p.grad is None for p in model.parameters())
    run.verify_hashes(hashes)
    base.write(OUT/'audit.json',dict(passed=True,device='cpu',fit_face_ids=ids,results=results,hashes=hashes,
        parent_unchanged=True,fixed_pair_initial_output_equal=True,D_E_initial_outputs_equal=True,
        dictionary_fixture='repeated two-face fit projection; not training dictionaries or quality assessment',
        training_run_started=False,persistent_weights_created=False))
    print('CPU real-data preflight passed for all 23 teacher-g arms',flush=True)


if __name__=='__main__':
    main()
