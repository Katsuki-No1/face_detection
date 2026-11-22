"""Teacher-g-only recovery comparison; all controls freshly trained."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_messenger_position_cosine as reference
from scripts import run_messenger_diffusion_comparison as legacy
from scripts.run_messenger_multiaxis_convergence import save, set_lr, signature, Plateau
from src.models.messenger_codebook_teacher import ARMS, DIMENSIONS, COUNTS, arm_spec, make_recovery, teacher_blend

base = reference.base
OUT = ROOT / "outputs/diagnostics/messenger_codebook_teacher_20261120"
BASELINE = ROOT / "outputs/diagnostics/messenger_position_primary_20261118"
DIRECTIONS = ROOT / "outputs/diagnostics/keypoint_direction_conditions_20261119"
PLAN = {
    "arms": list(ARMS), "controls": "fresh regression384 and continuous16; no archived performance controls",
    "seed": 42, "latent_dimensions": list(DIMENSIONS), "arm_specs": {arm: arm_spec(arm) for arm in ARMS},
    "direction_scope": "existing natural fit512, 137 videos; full shared basis, first16/32",
    "data": "existing synthetic fit256/dev128/eval128 x3; natural fit512/dev128/eval512; GT crops",
    "mse_weight": .01, "cosine_weight": .01, "vq_weight": .01, "commitment_weight": .25,
    "code_counts": [4096, 8192], "learned_g": False,
    "initialization": "shared sensitivity projection; zero contextual residual; shared linear-plus-MLP decoder",
    "dictionary": "chunked Lloyd k-means on unoccluded synthetic-fit projected features, all 256 cells; init fixed per K",
    "kmeans": {"chunk_size": 512, "max_iter": 30, "relative_tolerance": 1e-5, "initialization": "seed42 fit samples without replacement", "empty_cluster": "keep previous center"},
    "gradient": "STE encoder; direct selected-code gradient only in A/B; C/D/E codes updated by raw-valued batch-mean EMA",
    "ema": {"decay": .99, "cadence": "once per effective batch after optimizer step", "empty": "retain", "input": "all train encoder tokens, no dev/eval updates"},
    "lookup": "B/D/E normalize for selection only; raw code values go to decoder; same initial codes across A-E",
    "spatial_decoder": "E = D plus width96, heads6, one self-attention+FFN residual, sine position, zero final 384 projection; no raw-feature bypass",
    "pretraining": False,
    "topology": "deployment parent is full-image FACE/ellipse -> face-crop keypoint; this diagnostic replaces crop features with R and fixed teacher g; GT-crop train-internal evaluation only, no HEAD",
    "R_selection": "minimum balanced dev position; position and total objective plateau",
    "teacher_g": {"artificial": "existing generated binary patch mask, identical in R train/dev/eval", "natural": "existing five-point visibility weak teacher map; fixed, no confidence multiplication, not a true segmentation mask"},
    "g_training": "none: no g model, optimizer, BCE, checkpoint or g quality gate",
    "microbatch": 4, "effective_batch": 16, "R_cap": 600,
    "weight_decay": .01, "lr": 1e-4, "R_min_epochs": 40,
    "lr_floors": [1e-6], "lr_reduce_patience": 8, "stop_patience": 15,
    "retention": "delete new R and initial dictionary weights after every numeric comparison and audit is saved",
    "frozen": ["DINO", "keypoint_head"], "production": False,
    "evaluation": "raw, full_R, teacher_g for both domains; natural weak-teacher diagnostic separate from artificial known-mask evaluation",
}


def read(path):
    return json.loads(Path(path).read_text())


def basis_from_fit(grams, dimension=32):
    matrix = np.asarray(grams, dtype=np.float64)
    if matrix.shape != (384, 384) or not np.isfinite(matrix).all():
        raise ValueError("Invalid fit gradient Gram matrix")
    _, basis = np.linalg.eigh(matrix)
    basis = basis[:, ::-1][:, :dimension].copy()
    basis *= np.sign(basis[np.abs(basis).argmax(0), np.arange(dimension)])
    np.testing.assert_allclose(basis.T @ basis, np.eye(dimension), atol=1e-6)
    return torch.tensor(basis, dtype=torch.float32)


def source_hashes():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == ".py" and ".venv" not in path.parts:
                paths.add(path)
    paths.update([base.BASE, base.CONFIG, base.ANN, base.PRIOR,
                  BASELINE / "manifest.json",
                  DIRECTIONS / "conditions.npz", DIRECTIONS / "audit.json",
                  ROOT / "scripts/report_messenger_codebook_teacher.py",
                  ROOT / "scripts/gate_messenger_codebook_teacher.py"])
    return {str(p.relative_to(ROOT)): base.sha(p) for p in sorted(paths)}


def verify_hashes(hashes):
    for path, digest in hashes.items():
        if base.sha(ROOT / path) != digest:
            raise ValueError(f"Input/source changed: {path}")


@contextmanager
def lease(wait=False):
    with (ROOT / "outputs/training_queues/pending_models_tests_then_training/queue.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise RuntimeError("Canonical GPU lease is in use")
                print("Waiting for canonical GPU lease", flush=True)
                time.sleep(30)
        yield


def manifest_contract(current, historical):
    for key in ("roles", "faces", "image_hashes"):
        if current[key] != historical[key]:
            raise ValueError(f"Historical data mismatch: {key}")


def load_basis(manifest):
    audit = read(DIRECTIONS / "audit.json")
    if base.sha(DIRECTIONS / "conditions.npz") != audit["sha256"]:
        raise ValueError("Direction archive changed")
    for path in (base.BASE, base.ANN):
        key = str(path.relative_to(ROOT))
        if base.sha(path) != audit["hashes"][key]:
            raise ValueError(f"Direction teacher changed: {key}")
    with np.load(DIRECTIONS / "conditions.npz") as values:
        np.testing.assert_array_equal(values["fit_face_id"], manifest["roles"]["fit"]["natural"])
        assert set(values["fit_video"]).isdisjoint(values["eval_video"])
        return basis_from_fit(values["fit_grams"][0])


def generate(net, maps):
    outputs = net(maps)
    details = outputs[1] if isinstance(outputs[1], dict) else {}
    return outputs[0], details


def loss_parts(model, restored, original, teacher, covered_map, outside_map, xy, point_covered,
               valid, basis, arm, details):
    prediction = base.run_head(model.head, teacher_blend(original, restored, covered_map))
    errors = (prediction["decoded_coordinates"] - xy).norm(dim=-1) * 1.25
    position = reference.balanced_mean(errors, point_covered & valid, ~point_covered & valid)
    difference = restored - teacher
    feature_map = difference.square().mean(1)
    mse = legacy.region_mean(feature_map, covered_map, outside_map)
    cosine = legacy.region_mean(1 - F.cosine_similarity(restored, teacher, dim=1), covered_map, outside_map)
    code = position.new_zeros(())
    commitment = position.new_zeros(())
    if "code_loss" in details:
        code = legacy.region_mean(details["code_loss"], covered_map, outside_map)
        commitment = legacy.region_mean(details["commitment"], covered_map, outside_map)
    auxiliary = PLAN["mse_weight"] * mse + PLAN["cosine_weight"] * cosine
    auxiliary = auxiliary + PLAN["vq_weight"] * (code + PLAN["commitment_weight"] * commitment)
    return dict(position=position, auxiliary=auxiliary, objective=position+auxiliary,
                mse=mse, cosine=cosine, code=code, commitment=commitment), errors


@torch.no_grad()
def validate(net, model, cache, basis, arm):
    net.eval()
    sums = np.zeros(4)
    aux = []
    for batch in base.batches(len(cache["syn_dev"]), PLAN["microbatch"]):
        ids = cache["syn_dev"][batch]
        maps, teacher, covered, outside = base.syn_batch(cache, ids)
        restored, details = generate(net, maps)
        xy = base.gpu(cache["xy"][cache["parent"][ids]])
        point = base.gpu(cache["point"][ids])
        valid = base.gpu(cache["state"][cache["parent"][ids]]) > 0
        parts, error = loss_parts(model, restored, maps[:, 2], teacher, covered, outside, xy, point, valid, basis, arm, details)
        for k, mask in enumerate((point & valid, ~point & valid)):
            sums[k*2:k*2+2] += [float(error[mask].sum()), int(mask.sum())]
        aux.append([float(parts[key]) * len(ids) for key in ("auxiliary", "mse", "cosine", "code", "commitment")])
    if min(sums[1], sums[3]) <= 0:
        raise ValueError("Validation must include covered and uncovered points")
    position = (sums[0]/sums[1] + sums[2]/sums[3])/2
    averages = np.sum(aux, axis=0)/len(cache["syn_dev"])
    return dict(position=float(position), objective=float(position+averages[0]),
                **{name: float(v) for name,v in zip(("auxiliary","mse","cosine","code","commitment"),averages)})


def initial(net, folder, stage, identity):
    if stage != 'R':
        raise ValueError('Only R is trained in the teacher-g study')
    lr = PLAN['lr']
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=PLAN['weight_decay'])
    controller = Plateau(lr, PLAN['lr_floors'][0], PLAN[f'{stage}_min_epochs'],
                         absolute=1e-5 if stage=='R' else 1e-6,
                         reduce_patience=PLAN['lr_reduce_patience'], stop_patience=PLAN['stop_patience'])
    rng = np.random.default_rng(PLAN['seed'])
    history, best, start, done = [], float('inf'), 0, False
    path = folder/f'last_{stage}.pt'
    if path.exists():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['identity'] != identity:
            raise ValueError('Resume identity mismatch')
        net.load_state_dict(checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        controller = Plateau(**checkpoint['controller'])
        rng.bit_generator.state = checkpoint['rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        history = read(folder/f'{stage}_history.json')[:checkpoint['epoch']+1]
        if not history or history[-1]['epoch'] != checkpoint['epoch']:
            raise ValueError('Resume history does not match checkpoint')
        best, start, done = checkpoint['best'], checkpoint['epoch']+1, checkpoint['converged']
    return optimizer, controller, rng, history, best, start, done


def gpu_idle(wait=False):
    while True:
        output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        others = [line.strip() for line in output.splitlines() if line.strip() and line.strip()!=str(os.getpid())]
        if not others:
            return
        if not wait:
            raise RuntimeError(f'GPU is in use by processes {others}')
        print(f'Waiting for GPU processes {others}',flush=True)
        time.sleep(30)

def train_epoch(net, model, cache, basis, arm, optimizer, order):
    net.train()
    training = []
    for batch in base.batches(len(order), PLAN["effective_batch"]):
        group = order[batch]
        point = cache["point"][group]
        valid = cache["state"][cache["parent"][group]] > 0
        denominators = [int(mask.sum()) for mask in (point & valid, ~point & valid)]
        optimizer.zero_grad(set_to_none=True)
        if getattr(net, "codebook", None) is not None:
            net.codebook.begin_step()
        for microbatch in base.batches(len(group), PLAN["microbatch"]):
            ids = group[microbatch]
            maps, teacher, covered, outside = base.syn_batch(cache, ids)
            restored, details = generate(net, maps)
            xy = base.gpu(cache["xy"][cache["parent"][ids]])
            point = base.gpu(cache["point"][ids])
            valid = base.gpu(cache["state"][cache["parent"][ids]]) > 0
            parts, error = loss_parts(model, restored, maps[:, 2], teacher, covered, outside,
                                      xy, point, valid, basis, arm, details)
            position = sum(error[mask].sum()/max(count,1) for mask,count in
                           zip((point & valid, ~point & valid), denominators))/sum(d>0 for d in denominators)
            loss = position + parts["auxiliary"] * len(ids)/len(group)
            loss.backward()
            training.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if getattr(net, "codebook", None) is not None:
            net.codebook.end_step()
    return training


def train_recovery(folder, arm, model, cache, basis, net, identity):
    optimizer, controller, rng, history, best, start, done = initial(net, folder, "R", identity)
    for epoch in range(start, PLAN["R_cap"]+1) if not done else ():
        tick = time.monotonic()
        training = []
        if epoch:
            training = train_epoch(net, model, cache, basis, arm, optimizer, rng.permutation(cache["syn_fit"]))
        dev = validate(net, model, cache, basis, arm)
        if dev["position"] < best:
            best = dev["position"]
            save(folder/"best_R.pt", dict(state=net.state_dict(), epoch=epoch, dev=dev, identity=identity))
        done = controller.step(epoch, [dev["position"], dev["objective"]])
        set_lr(optimizer, controller.lr)
        row = dict(epoch=epoch, dev=dev, train=float(np.sum(training)), lr=controller.lr,
                   seconds=time.monotonic()-tick, converged=done)
        history.append(row)
        legacy.persist(folder, "R", net, optimizer, controller, rng, history, best, identity, done)
        base.status(OUT, "training", arm=arm, stage="R", **row)
        if done:
            break
    if not done:
        raise RuntimeError(f"{arm} reached its cap before plateau")
    checkpoint = torch.load(folder/"best_R.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["state"])
    net.eval().requires_grad_(False)
    base.write(folder/"R_convergence.json", dict(converged=True, selected_epoch=checkpoint["epoch"], stop_epoch=history[-1]["epoch"]))
    return net


def atomic_array(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as file:
        np.save(file, value)
    temporary.replace(path)


def kmeans_codes(latent, count):
    if latent.ndim != 2 or len(latent) < count or not torch.isfinite(latent).all():
        raise ValueError("Insufficient or invalid fit patches for initial dictionary")
    generator = torch.Generator(device='cpu').manual_seed(PLAN['seed'])
    indices = torch.randperm(len(latent), generator=generator)[:count].to(latent.device)
    centers = latent[indices].clone()
    def assign():
        labels = []
        for part in latent.split(PLAN['kmeans']['chunk_size']):
            distances = part.square().sum(1,keepdim=True) + centers.square().sum(1) - 2*part@centers.T
            labels.append(distances.argmin(1))
        return torch.cat(labels)
    previous = None
    for iteration in range(1, PLAN['kmeans']['max_iter']+1):
        assignments = assign()
        counts = torch.bincount(assignments, minlength=count)
        # Sorted segments avoid nondeterministic CUDA scatter atomics.
        order = assignments.argsort(stable=True)
        sums = torch.segment_reduce(latent[order], 'sum', lengths=counts)
        populated = counts > 0
        centers[populated] = sums[populated]/counts[populated,None]
        inertia = float((latent-centers[assignments]).square().sum())
        if previous is not None and abs(previous-inertia) <= PLAN['kmeans']['relative_tolerance']*max(previous,1e-12):
            break
        previous = inertia
    assignments = assign()
    counts = torch.bincount(assignments, minlength=count)
    return centers, counts, float((latent-centers[assignments]).square().sum()), iteration


@torch.no_grad()
def initial_codes(cache, basis, mean, count):
    fit = np.unique(cache['parent'][cache['syn_fit']])
    features = cache['clean'][fit, 2].float().to(basis.device)
    latent = torch.einsum('cr,bchw->bhwr', basis, features-mean[None,:,None,None]).reshape(-1,basis.shape[1])
    tick = time.monotonic()
    centers, counts, inertia, iterations = kmeans_codes(latent, count)
    return centers.cpu().numpy(), dict(
        fit_faces=len(fit), fit_patches=len(latent), code_count=count, used_codes=int((counts>0).sum()),
        cluster_counts=counts.cpu().tolist(), inertia=inertia, iterations=iterations,
        seconds=time.monotonic()-tick)

@torch.no_grad()
def usage(net, cache, name):
    if getattr(net, "codebook", None) is None:
        return None
    counts = np.zeros(len(net.codebook.embedding), np.int64)
    errors = []
    ids = cache["natural_eval"] if name == "natural" else cache["syn_eval"]
    for batch in base.batches(len(ids), 4):
        selected = ids[batch]
        maps = base.gpu(cache["clean"][selected] if name == "natural" else cache["syn"][selected])
        _, details = generate(net, maps)
        counts += np.bincount(details["indices"].cpu().numpy().ravel(), minlength=len(counts))
        errors.extend(details["quantization_error"].cpu().numpy().ravel().tolist())
    probability = counts[counts>0]/counts.sum()
    return dict(counts=counts.tolist(), used_codes=int((counts>0).sum()),
                perplexity=float(np.exp(-(probability*np.log(probability)).sum())),
                quantization_mse=float(np.mean(errors)))


def train(resume=False, wait=False):
    global OUT
    with lease(wait):
        if (OUT/"numeric_completion.json").exists():
            from scripts.report_messenger_codebook_teacher import cleanup
            cleanup(OUT)
            print("Numerical comparison already complete", flush=True)
            return
        if OUT.exists() and not resume:
            raise ValueError("Output exists; inspect it and use --resume for the same run")
        if resume and not (OUT/"plan.json").exists():
            raise ValueError("No resumable plan")
        from scripts.gate_messenger_codebook_teacher import authenticate
        gate_proof = authenticate()
        gpu_idle(wait)
        OUT.mkdir(exist_ok=True, parents=True)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        plan = dict(PLAN, hashes=source_hashes(), runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__))
        if resume and read(OUT/"plan.json") != plan:
            raise ValueError("Plan, data or source identity changed")
        if not resume:
            base.write(OUT/"plan.json", plan)
            base.write(OUT/"gate_authentication.json", gate_proof)
        try:
            config = yaml.safe_load(base.CONFIG.read_text())
            model, _ = base.build_landmark_model(config, variant="query_refinement", initialize_from_detector=False)
            base.load_landmark_checkpoint(model, base.BASE, prefer_ema=True, strict=True)
            model.requires_grad_(False).eval().cuda()
            frozen = base.tensor_digest(model)
            cache = base.make_cache(OUT, model, config["data"])
            manifest = read(OUT/"manifest.json")
            manifest_contract(manifest, read(BASELINE/"manifest.json"))
            full_basis = load_basis(manifest).cuda()
            fit = np.unique(cache["parent"][cache["syn_fit"]])
            mean = cache["clean"][fit,2].float().mean((0,2,3)).cuda()
            base.write(OUT/"teacher_audit.json", legacy.teacher_cache(cache,manifest))
            base.write(OUT/"normalization.json", dict(mean=mean.cpu().tolist(),basis=full_basis.cpu().tolist()))
            codes = {}
            for dimension in DIMENSIONS:
                for count in COUNTS:
                    key = f"d{dimension}_k{count}"
                    path = OUT/f"initial_codes_{key}.npy"
                    audit_path = OUT/f"initial_codes_{key}.json"
                    if not audit_path.exists():
                        centers, audit = initial_codes(cache,full_basis[:,:dimension],mean,count)
                        atomic_array(path,centers)
                        base.write(audit_path,dict(audit,sha256=base.sha(path),plan_identity=signature(plan)))
                    audit = read(audit_path)
                    if audit["plan_identity"] != signature(plan) or base.sha(path) != audit["sha256"]:
                        raise ValueError("Initial dictionary identity changed")
                    codes[key] = torch.from_numpy(np.load(path))
            authenticate(manifest=manifest, code_hashes={k:base.sha(OUT/f"initial_codes_{k}.npy") for k in codes})
            results = {}
            for arm in ARMS:
                spec = arm_spec(arm)
                basis = full_basis[:,:spec["dimension"]]
                key = f'd{spec["dimension"]}_k{spec["count"]}'
                folder = OUT/arm
                folder.mkdir(exist_ok=True)
                identity = signature(dict(plan=plan,manifest=manifest,arm=arm,
                    normalization=read(OUT/"normalization.json"),
                    initial_codes={str(k):base.sha(OUT/f"initial_codes_{k}.npy") for k in codes}))
                if (folder/"evaluation.json").exists():
                    evaluation = read(folder/"evaluation.json")
                    if evaluation["identity"] != identity:
                        raise ValueError("Evaluation identity mismatch")
                    for file,digest in evaluation["artifacts"].items():
                        if base.sha(folder/file) != digest:
                            raise ValueError("Saved evaluation/checkpoint changed")
                else:
                    net = make_recovery(arm,basis.cpu(),mean.cpu(),codes.get(key)).cuda()
                    initial_hash = base.tensor_digest(net)
                    base.write(folder/"initialization.json",dict(model_sha256=initial_hash,
                         codebook_trainable=spec["update"] == "gradient",parameters=sum(p.numel() for p in net.parameters())))
                    torch.cuda.reset_peak_memory_stats()
                    net = train_recovery(folder,arm,model,cache,basis,net,identity)
                    tick = time.monotonic()
                    evaluation = evaluate(folder,net,model,cache)
                    evaluation.update(identity=identity,R=read(folder/"R_convergence.json"),
                        code_usage={name:usage(net,cache,name) for name in ("natural","artificial")},
                        inference_seconds=time.monotonic()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        trainable_codebook=spec["update"] == "gradient", arm_spec=spec)
                    if getattr(net,"codebook",None) is not None:
                        initial = codes[key].to(net.codebook.embedding.device)
                        difference = float((net.codebook.embedding-initial).norm())
                        evaluation["codebook_change_l2"] = difference
                        evaluation["ema_steps"] = int(net.codebook.ema_steps)
                    evaluation["artifacts"] = {p.name:base.sha(p) for p in folder.iterdir() if p.suffix in (".pt",".npz") or p.name in ("R_history.json","R_convergence.json","initialization.json")}
                    base.write(folder/"evaluation.json",evaluation)
                    del net
                    gc.collect()
                    torch.cuda.empty_cache()
                results[arm] = evaluation
                base.write(OUT/"comparison.json",results)
                if base.tensor_digest(model) != frozen or any(p.grad is not None for p in model.parameters()):
                    raise ValueError("Fixed parent changed")
            verify_hashes(plan["hashes"])
            base.write(OUT/"training_completion.json",dict(all_converged=True,source_verified=True,
                       fixed_parent=frozen,arms=list(ARMS),fixed_parent_retrained=False,controls_retrained=True,learned_g=False))
            from scripts.report_messenger_codebook_teacher import report, cleanup
            report(OUT)
            cleanup(OUT)
        except Exception as error:
            base.status(OUT,"failed",error=repr(error))
            raise


def teacher_statistics(values):
    values=np.asarray(values,dtype=np.float64)
    if not values.size:
        return dict(mean=None,median=None,p90=None)
    return dict(mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)))


@torch.no_grad()
def evaluate(folder,net,model,cache,seed=42):
    result={}
    for domain in ('natural','artificial'):
        rows=[];ids=cache['natural_eval' if domain=='natural' else 'syn_eval'];errs=[];counts=[];locals_=[];energies=[]
        norm_maps=[];point_norms=[];gates=[];teacher_maps=[];cover_maps=[];outside_maps=[];components=[]
        for b in base.batches(len(ids),4):
            j=ids[b]
            if domain=='natural':
                maps=base.gpu(cache['clean'][j]);xy=base.gpu(cache['xy'][j]);x=maps[:,2]
            else:
                maps,z,c,e=base.syn_batch(cache,j);x=maps[:,2];xy=base.gpu(cache['xy'][cache['parent'][j]])
            r,_=generate(net,maps)
            teacher_g=base.gpu(cache['teacher'][j]) if domain=='natural' else c
            y=teacher_blend(x,r,teacher_g)
            teacher_maps.append(teacher_g.cpu().numpy())
            norms=torch.stack([v.norm(dim=1) for v in ((x,r,y) if domain=='natural' else (x,r,z,y))],1)
            norm_maps.append(norms.cpu().numpy())
            ij=((xy*255+.5)/16).floor().long().clamp(0,15)
            pn=norms.permute(0,2,3,1)[torch.arange(len(j),device='cuda')[:,None],ij[:,:,1],ij[:,:,0]].transpose(1,2)
            point_norms.append(pn.cpu().numpy())
            gates.append(teacher_g[torch.arange(len(j),device='cuda')[:,None],ij[:,:,1],ij[:,:,0]].cpu().numpy())
            values=(x,r,y)
            pr=[]
            for value in values:
                pred=base.run_head(model.head,value)
                pr.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1))
            rows.append(torch.stack(pr,1).cpu().numpy())
            if domain=='artificial':
                cover_maps.append(c.cpu().numpy());outside_maps.append(e.cpu().numpy())
                components.append(torch.stack([torch.stack([torch.stack([(metric*m).sum((1,2)) for metric in
                    ((v-z).square().mean(1),1-F.cosine_similarity(v,z,dim=1))],1) for m in (c,e)],1) for v in (x,r)],1).cpu().numpy())
                errs.append(torch.stack([(legacy.feature_error(v,z)*m).sum((1,2)) for v in (x,r) for m in (c,e)],1).cpu().numpy())
                counts.append(torch.stack((c.sum((1,2)),e.sum((1,2))),1).cpu().numpy())
                zc,zl,valid=legacy.point_features(z,xy);ll=[];ee=[]
                for v in (x,r,z):
                    vc,vl,_=legacy.point_features(v,xy)
                    ce=1-F.cosine_similarity(vc,zc,dim=-1)+(vc-zc).square().mean(-1)
                    le=1-F.cosine_similarity(vl,zl,dim=-1)+(vl-zl).square().mean(-1)
                    ll.append(torch.stack((ce,(le*valid).sum(-1)/valid.sum(-1)),-1))
                    avg=(vl*valid[...,None]).sum(-2)/valid.sum(-1)[...,None]
                    ee.append((((vl-avg[...,None,:]).square().mean(-1))*valid).sum(-1)/valid.sum(-1))
                locals_.append(torch.stack(ll,1).cpu().numpy());energies.append(torch.stack(ee,1).cpu().numpy())
        rows=np.concatenate(rows)
        norms=np.concatenate(norm_maps);pn=np.concatenate(point_norms);gs=np.concatenate(gates)
        assert np.isfinite(norms).all() and (norms>=0).all()
        if domain=='natural':
            state=cache['state'][ids].numpy();videos=cache['videos'][ids];masks={'occluded':state==1,'visible':state==2}
            np.savez_compressed(folder/f'natural_seed{seed}.npz',rows=rows,state=state,ids=ids,videos=videos,
                                norms=norms,point_norms=pn,gate=gs,norm_order=np.array(['X','R','fused']),teacher_g=np.concatenate(teacher_maps),face_occluded=cache['contour'][ids])
        else:
            point=cache['point'][ids].numpy();videos=cache['videos'][cache['parent'][ids]]
            masks={'covered':point,'uncovered':~point};er=np.concatenate(errs);ct=np.concatenate(counts)
            np.savez_compressed(folder/f'artificial_seed{seed}.npz',rows=rows,point=point,ids=ids,videos=videos,error_sums=er,counts=ct,
                                local=np.concatenate(locals_),energy=np.concatenate(energies),norms=norms,point_norms=pn,gate=gs,
                                covered=np.concatenate(cover_maps),outside=np.concatenate(outside_maps),components=np.concatenate(components),
                                norm_order=np.array(['X','R','Z','fused']),teacher_g=np.concatenate(teacher_maps))
            result['feature']={'raw':float(er[:,0].sum()/ct[:,0].sum()),'R':float(er[:,2].sum()/ct[:,0].sum())}
            comps=np.concatenate(components)
            result['feature_components']={name:{region:{metric:float(comps[:,i,k,l].sum()/ct[:,k].sum()) for l,metric in enumerate(('mse','cosine'))}
                for k,region in enumerate(('covered','outside'))} for i,name in enumerate(('raw','R'))}
        result[domain+'_teacher_g']={label:teacher_statistics(gs[mask]) for label,mask in masks.items()}
        names=('raw','full_R','teacher_g')
        result[domain]={name:{label:reference.metrics(rows[:,i],mask,videos) for label,mask in masks.items()} for i,name in enumerate(names)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=("plan","train","report","cleanup"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--wait",action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(PLAN,ensure_ascii=False,indent=2))
    elif args.command == "train":
        train(args.resume,args.wait)
    else:
        from scripts.report_messenger_codebook_teacher import report, cleanup
        with lease(args.wait):
            if args.command == "report":
                report(OUT)
            cleanup(OUT)


if __name__ == "__main__":
    main()