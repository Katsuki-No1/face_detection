"""Six new keypoint-recovery arms, with the saved position-primary regression as control."""
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
from src.models.messenger_codebook import ARMS, directional_mse, make_recovery
from src.models.natural_feature_gate import NaturalFeatureGate

base = reference.base
OUT = ROOT / "outputs/diagnostics/messenger_codebook_20261119"
BASELINE = ROOT / "outputs/diagnostics/messenger_position_primary_20261118"
DIRECTIONS = ROOT / "outputs/diagnostics/keypoint_direction_conditions_20261119"
PLAN = {
    "arms": list(ARMS), "saved_control": str(BASELINE.relative_to(ROOT) / "det_position"),
    "seed": 42, "latent_dimension": 16, "direction_weight": 4.,
    "direction_scope": "existing natural fit512, 137 videos; full shared basis, first16",
    "data": "existing synthetic fit256/dev128/eval128 x3; natural fit512/dev128/eval512; GT crops",
    "mse_weight": .01, "cosine_weight": .01, "vq_weight": .01, "commitment_weight": .25,
    "code_counts": [4096, 8192], "gate_bce_weight": .01,
    "initialization": "shared sensitivity projection; zero contextual residual; shared linear-plus-MLP decoder",
    "dictionary": "chunked Lloyd k-means on unoccluded synthetic-fit projected features, all 256 cells; init fixed per K",
    "kmeans": {"chunk_size": 512, "max_iter": 30, "relative_tolerance": 1e-5, "initialization": "seed42 fit samples without replacement", "empty_cluster": "keep previous center"},
    "gradient": "hard nearest code forward; straight-through encoder gradient and direct selected-code gradient",
    "R_selection": "minimum balanced dev position; position and total objective plateau",
    "g_selection": "minimum balanced natural dev position; old weak teacher plus 0.01 BCE",
    "microbatch": 4, "effective_batch": 16, "R_cap": 600, "g_cap": 300,
    "weight_decay": .01, "lr": 1e-4, "gate_lr": 5e-4, "R_min_epochs": 40, "g_min_epochs": 20,
    "lr_floors": [1e-6, 5e-6], "lr_reduce_patience": 8, "stop_patience": 15,
    "retention": "delete new R/g and initial dictionary weights after every numeric comparison and audit is saved",
    "frozen": ["DINO", "keypoint_head"], "production": False,
    "evaluation": "full_R and learned_g, artificial mask_R; coordinate and presence metrics separately",
}


def read(path):
    return json.loads(Path(path).read_text())


def basis_from_fit(grams):
    matrix = np.asarray(grams, dtype=np.float64)
    if matrix.shape != (384, 384) or not np.isfinite(matrix).all():
        raise ValueError("Invalid fit gradient Gram matrix")
    _, basis = np.linalg.eigh(matrix)
    basis = basis[:, ::-1][:, :16].copy()
    basis *= np.sign(basis[np.abs(basis).argmax(0), np.arange(16)])
    np.testing.assert_allclose(basis.T @ basis, np.eye(16), atol=1e-6)
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
                  BASELINE / "manifest.json", BASELINE / "numeric_completion.json",
                  BASELINE / "det_position/natural_seed42.npz", BASELINE / "det_position/artificial_seed42.npz",
                  DIRECTIONS / "conditions.npz", DIRECTIONS / "audit.json",
                  ROOT / "scripts/report_messenger_codebook.py",
                  ROOT / "scripts/gate_messenger_codebook.py"])
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
    prediction = base.run_head(model.head, torch.where(covered_map[:, None], restored, original))
    errors = (prediction["decoded_coordinates"] - xy).norm(dim=-1) * 1.25
    position = reference.balanced_mean(errors, point_covered & valid, ~point_covered & valid)
    difference = restored - teacher
    feature_map = (directional_mse(difference, basis, PLAN["direction_weight"]) if arm == "weighted16"
                   else difference.square().mean(1))
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
    lr = PLAN['lr'] if stage == 'R' else PLAN['gate_lr']
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=PLAN['weight_decay'])
    controller = Plateau(lr, PLAN['lr_floors'][0 if stage=='R' else 1], PLAN[f'{stage}_min_epochs'],
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


def train_gate(folder, arm, model, cache, restored, identity):
    torch.manual_seed(42)
    gate = NaturalFeatureGate(384).cuda()
    optimizer, controller, rng, history, best, start, done = initial(gate, folder, "g", identity)
    for epoch in range(start, PLAN["g_cap"]+1) if not done else ():
        tick = time.monotonic()
        train = reference.gate_pass(gate, model, cache, restored, rng.permutation(cache["natural_fit"]), optimizer) if epoch else None
        dev = reference.gate_pass(gate, model, cache, restored, cache["natural_dev"])
        if dev["position"] < best:
            best = dev["position"]
            save(folder/"best_g.pt", dict(state=gate.state_dict(), epoch=epoch, dev=dev, identity=identity))
        done = controller.step(epoch, [dev["position"], dev["objective"]])
        set_lr(optimizer, controller.lr)
        row = dict(epoch=epoch, dev=dev, train=train, lr=controller.lr, seconds=time.monotonic()-tick, converged=done)
        history.append(row)
        legacy.persist(folder, "g", gate, optimizer, controller, rng, history, best, identity, done)
        base.status(OUT, "training", arm=arm, stage="g", **row)
        if done:
            break
    if not done:
        raise RuntimeError(f"{arm} gate reached its cap before plateau")
    checkpoint = torch.load(folder/"best_g.pt", map_location="cpu", weights_only=True)
    gate.load_state_dict(checkpoint["state"])
    gate.eval().requires_grad_(False)
    base.write(folder/"g_convergence.json", dict(converged=True, selected_epoch=checkpoint["epoch"], stop_epoch=history[-1]["epoch"]))
    return gate


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
    latent = torch.einsum('cr,bchw->bhwr', basis, features-mean[None,:,None,None]).reshape(-1,16)
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
            from scripts.report_messenger_codebook import cleanup
            cleanup(OUT)
            print("Numerical comparison already complete", flush=True)
            return
        if OUT.exists() and not resume:
            raise ValueError("Output exists; inspect it and use --resume for the same run")
        if resume and not (OUT/"plan.json").exists():
            raise ValueError("No resumable plan")
        from scripts.gate_messenger_codebook import authenticate
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
        if not read(BASELINE/"numeric_completion.json")["all_numeric_evaluations_complete"]:
            raise ValueError("Saved control has incomplete evaluation")
        try:
            config = yaml.safe_load(base.CONFIG.read_text())
            model, _ = base.build_landmark_model(config, variant="query_refinement", initialize_from_detector=False)
            base.load_landmark_checkpoint(model, base.BASE, prefer_ema=True, strict=True)
            model.requires_grad_(False).eval().cuda()
            frozen = base.tensor_digest(model)
            cache = base.make_cache(OUT, model, config["data"])
            manifest = read(OUT/"manifest.json")
            manifest_contract(manifest, read(BASELINE/"manifest.json"))
            basis = load_basis(manifest).cuda()
            fit = np.unique(cache["parent"][cache["syn_fit"]])
            mean = cache["clean"][fit,2].float().mean((0,2,3)).cuda()
            base.write(OUT/"teacher_audit.json", legacy.teacher_cache(cache,manifest))
            base.write(OUT/"normalization.json", dict(mean=mean.cpu().tolist(),basis=basis.cpu().tolist()))
            codes = {}
            for count in PLAN["code_counts"]:
                path = OUT/f"initial_codes_{count}.npy"
                audit_path = OUT/f"initial_codes_{count}.json"
                if not audit_path.exists():
                    centers, audit = initial_codes(cache,basis,mean,count)
                    atomic_array(path,centers)
                    base.write(audit_path,dict(audit,sha256=base.sha(path),plan_identity=signature(plan)))
                audit = read(audit_path)
                if audit["plan_identity"] != signature(plan) or base.sha(path) != audit["sha256"]:
                    raise ValueError("Initial dictionary identity changed")
                codes[count] = torch.from_numpy(np.load(path))
            authenticate(manifest=manifest, code_hashes={str(k):base.sha(OUT/f"initial_codes_{k}.npy") for k in codes})
            results = {}
            for arm in ARMS:
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
                    count = 4096 if arm.startswith("vq4096") else 8192
                    net = make_recovery(arm,basis.cpu(),mean.cpu(),codes.get(count)).cuda()
                    initial_hash = base.tensor_digest(net)
                    base.write(folder/"initialization.json",dict(model_sha256=initial_hash,
                         codebook_trainable=arm.endswith("_trainable"),parameters=sum(p.numel() for p in net.parameters())))
                    torch.cuda.reset_peak_memory_stats()
                    net = train_recovery(folder,arm,model,cache,basis,net,identity)
                    restored = reference.candidates(net,cache)
                    gate = train_gate(folder,arm,model,cache,restored,identity)
                    del restored
                    tick = time.monotonic()
                    evaluation = reference.evaluate(folder,net,gate,model,cache)
                    evaluation.update(identity=identity,R=read(folder/"R_convergence.json"),g=read(folder/"g_convergence.json"),
                        code_usage={name:usage(net,cache,name) for name in ("natural","artificial")},
                        inference_seconds=time.monotonic()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        trainable_codebook=arm.endswith("_trainable"))
                    if getattr(net,"codebook",None) is not None:
                        initial = codes[count].to(net.codebook.embedding.device)
                        difference = float((net.codebook.embedding-initial).norm())
                        if arm.endswith("_fixed") and difference != 0:
                            raise ValueError("Fixed dictionary changed")
                        evaluation["codebook_change_l2"] = difference
                    evaluation["artifacts"] = {p.name:base.sha(p) for p in folder.iterdir() if p.suffix in (".pt",".npz") or p.name in ("R_history.json","g_history.json","R_convergence.json","g_convergence.json","initialization.json")}
                    base.write(folder/"evaluation.json",evaluation)
                    del net,gate
                    gc.collect()
                    torch.cuda.empty_cache()
                results[arm] = evaluation
                base.write(OUT/"comparison.json",results)
                if base.tensor_digest(model) != frozen or any(p.grad is not None for p in model.parameters()):
                    raise ValueError("Fixed parent changed")
            verify_hashes(plan["hashes"])
            base.write(OUT/"training_completion.json",dict(all_converged=True,source_verified=True,
                       fixed_parent=frozen,arms=list(ARMS),baseline_retrained=False))
            from scripts.report_messenger_codebook import report, cleanup
            report(OUT)
            cleanup(OUT)
        except Exception as error:
            base.status(OUT,"failed",error=repr(error))
            raise


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
        from scripts.report_messenger_codebook import report, cleanup
        with lease(args.wait):
            if args.command == "report":
                report(OUT)
            cleanup(OUT)


if __name__ == "__main__":
    main()