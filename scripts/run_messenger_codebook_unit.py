"""Repeat teacher-g B/D/E using a consistent unit-vector quantization space."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_messenger_codebook_teacher as previous
from scripts.gate_messenger_codebook_teacher import array_sha
from scripts.run_messenger_codebook_teacher import (
    atomic_array, base, evaluate, gpu_idle, initial, lease, legacy, load_basis,
    manifest_contract, read, save, set_lr, signature, train_epoch, usage, validate, verify_hashes,
)
from src.models.messenger_codebook_factorial import COUNTS, DIMENSIONS, arm_spec
from src.models.messenger_codebook_unit import ARMS, make_recovery, unit_centers

OUT = ROOT / 'outputs/diagnostics/messenger_codebook_unit_20261120'
BASELINE = previous.OUT
PLAN = dict(previous.PLAN,
    arms=list(ARMS),
    arm_specs={arm: dict(arm_spec(arm), normalized_space='assignment, fitting, EMA and decoding') for arm in ARMS},
    controls='saved teacher-g A/C and continuous16/32 controls, same data and evaluation; matched B/D/E before versus after',
    dictionary='same seed42 raw Lloyd centers as previous B/D/E, then normalize each center; raw initialization SHA checked',
    gradient='STE through normalized encoder; B gradients to selected unit codes and reproject after AdamW',
    ema=dict(decay=.99, cadence='once per effective batch after optimizer step', empty='retain',
             input='unit-normalized encoder output, unit-normalized mean and updated codes'),
    lookup='nearest cosine code; unit vector goes to decoder; normalized code/commitment loss',
    quantization_metrics='unit-vector MSE; prior raw-scale quantization MSE is not directly comparable',
    telemetry_name='Messenger codebook - consistent normalization - 2026-11-20',
    telemetry_group='messenger_codebook_unit_20261120',
)


def source_hashes():
    hashes = previous.source_hashes()
    controls = (*ARMS, 'continuous16', 'continuous32', 'regression384',
                *(f'{mode}_d{d}_k{k}' for d in DIMENSIONS for k in COUNTS for mode in 'AC'))
    paths = [Path(__file__), ROOT/'scripts/gate_messenger_codebook_unit.py', ROOT/'scripts/report_messenger_codebook_unit.py',
             ROOT/'scripts/evaluate_messenger_success_thresholds.py', BASELINE/'manifest.json', BASELINE/'numeric_completion.json']
    paths.extend(BASELINE/f'initial_codes_d{d}_k{k}.json' for d in DIMENSIONS for k in COUNTS)
    for arm in controls:
        paths.extend(BASELINE/arm/name for name in ('natural_seed42.npz','artificial_seed42.npz','evaluation.json'))
    hashes.update({str(path.relative_to(ROOT)): base.sha(path) for path in paths})
    return dict(sorted(hashes.items()))


def authenticated_raw_codes(cache, basis, mean, count):
    centers, audit = previous.initial_codes(cache, basis, mean, count)
    historical = read(BASELINE / f'initial_codes_d{basis.shape[1]}_k{count}.json')
    digest = array_sha(centers)
    if digest != historical['sha256']:
        raise ValueError('Raw Lloyd initialization differs from the prior comparison')
    norms = np.linalg.norm(centers, axis=1)
    return centers, dict(audit, prior_raw_sha256=digest, stored_dtype='raw initial values; normalized by constructor',
                         raw_norm_min=float(norms.min()), raw_norm_max=float(norms.max()))


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


def train(resume=False, wait=False):
    global OUT
    with lease(wait):
        if (OUT/"numeric_completion.json").exists():
            from scripts.report_messenger_codebook_unit import cleanup
            cleanup(OUT)
            print("Numerical comparison already complete", flush=True)
            return
        if OUT.exists() and not resume:
            raise ValueError("Output exists; inspect it and use --resume for the same run")
        if resume and not (OUT/"plan.json").exists():
            raise ValueError("No resumable plan")
        from scripts.gate_messenger_codebook_unit import authenticate
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
                        centers, audit = authenticated_raw_codes(cache,full_basis[:,:dimension],mean,count)
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
                        initial = unit_centers(codes[key].to(net.codebook.embedding.device))
                        difference = float((net.codebook.embedding-initial).norm())
                        evaluation["codebook_change_l2"] = difference
                        evaluation["ema_steps"] = int(net.codebook.ema_steps)
                        evaluation["unit_code_norm_max_error"] = float((net.codebook.embedding.norm(dim=1)-1).abs().max())
                        if evaluation["unit_code_norm_max_error"] > 2e-6 or difference <= 0:
                            raise ValueError('Selected dictionary did not update on the unit sphere')
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
                       fixed_parent=frozen,arms=list(ARMS),fixed_parent_retrained=False,controls_retrained=False,learned_g=False))
            from scripts.report_messenger_codebook_unit import report, cleanup
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
        from scripts.report_messenger_codebook_unit import report, cleanup
        with lease(args.wait):
            if args.command == "report":
                report(OUT)
            cleanup(OUT)


if __name__ == "__main__":
    main()