"""Audit the factorial conditions, then remove only this comparison's new weights."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_messenger_codebook_factorial as experiment
from scripts.evaluate_messenger_success_thresholds import paired_interval
from src.models.messenger_codebook_factorial import ARMS, DIMENSIONS, COUNTS, arm_spec

base = experiment.base


def read(path):
    return json.loads(Path(path).read_text())


def verify_artifacts(out, hashes):
    for name, digest in hashes.items():
        if base.sha(out/name) != digest:
            raise ValueError(f"Artifact changed: {name}")


def summarize(rows, mask, videos, baseline):
    if not mask.any():
        return dict(points=0, mean_error=None, reason="No eligible GT points", thresholds={})
    native = rows[mask]
    old_native = baseline[mask]
    current = native.astype(np.float64)
    before = baseline[mask].astype(np.float64)
    video = np.broadcast_to(videos[:,None],mask.shape)[mask]
    metrics = dict(points=len(current),mean_error=float(current[:,0].mean()),
                   median_error=float(np.median(current[:,0])),p90_error=float(np.quantile(current[:,0],.9)),
                   presence_pass=int((native[:,1]>=.7).sum()),
                   versus_saved_control=paired_interval(current[:,0]-before[:,0],video),
                   thresholds={})
    for threshold in (.02,.03,.05,.1):
        location = native[:,0]<=threshold
        success = location & (native[:,1]>=.7)
        old_success = (old_native[:,0]<=threshold) & (old_native[:,1]>=.7)
        metrics["thresholds"][str(threshold)] = dict(location=int(location.sum()),joint=int(success.sum()),
            versus_saved_control=paired_interval(success.astype(float)-old_success,video))
    return metrics


def semantic_metrics(rows, mask):
    """GT-present, same-face slots; detector misses/state-0 false positives excluded."""
    slots=('left_eye','right_eye','nose','left_mouth','right_mouth')
    weights=np.array([2,2,1,1,1],dtype=float)
    per_slot={}
    for index,name in enumerate(slots):
        values=rows[:,index][mask[:,index]]
        n=len(values)
        present=values[:,1]>=.7
        emitted=int(present.sum())
        per_slot[name]=dict(eligible=n,emitted=emitted,missed=n-emitted,
            emission_rate=emitted/n if n else None,
            mean_error=float(values[:,0].mean()) if n else None,
            emitted_mean_error=float(values[present,0].mean()) if emitted else None,thresholds={})
        for threshold in (.05,.1):
            tp=int((present & (values[:,0]<=threshold)).sum())
            fp,fn=emitted-tp,n-tp
            per_slot[name]['thresholds'][str(threshold)]=dict(tp=tp,fp=fp,fn=fn,
                precision=tp/emitted if emitted else None,recall=tp/n if n else None,
                f1=2*tp/(n+emitted) if n else None)
    macro={}
    for threshold in ('0.05','0.1'):
        f1=[per_slot[name]['thresholds'][threshold]['f1'] for name in slots]
        supported=all(value is not None for value in f1)
        macro[threshold]=dict(ordinary=float(np.mean(f1)) if supported else None,
            eye_weighted=float(np.dot(weights,f1)/weights.sum()) if supported else None,
            per_slot_f1=f1,weights=weights.tolist(),weight_sum=float(weights.sum()),
            unsupported_slots=[name for name,value in zip(slots,f1) if value is None])
    means=[per_slot[name]['mean_error'] for name in slots]
    emitted_mask=mask & (rows[...,1]>=.7)
    return dict(per_slot=per_slot,macro_f1=macro,
        point_weighted_mean=float(rows[...,0][mask].mean()) if mask.any() else None,
        equal_semantic_mean=float(np.mean(means)) if all(v is not None for v in means) else None,
        emitted_points=int(emitted_mask.sum()),
        emitted_mean=float(rows[...,0][emitted_mask].mean()) if emitted_mask.any() else None,
        scope='GT-present same-face conditional slots; NME normalized by GT face ellipse major diameter; no detector misses or state-0 FP',
        undefined='empty GT slots are null; macro is null if any of the five slots is unsupported')

def report(out):
    out=Path(out)
    complete=out/"numeric_completion.json"
    if complete.exists():
        verify_artifacts(out,read(complete)["artifacts"])
        return
    training=read(out/"training_completion.json")
    if not training["all_converged"] or not training["source_verified"] or training["baseline_retrained"]:
        raise ValueError("Training is incomplete")
    plan=read(out/"plan.json")
    experiment.verify_hashes(plan["hashes"])
    result={"conditions":{},"raw_parity":{},"capacity_pairs":{},"dictionary_update_pairs":{},"lookup_pairs":{},"dimension_pairs":{},"attention_pairs":{},"vq_vs_continuous":{}}
    files=[out/"plan.json",out/"manifest.json",out/"training_completion.json",out/"comparison.json",
           out/"normalization.json",out/"teacher_audit.json",out/"gate_authentication.json"]
    manifest=read(out/"manifest.json")
    experiment.manifest_contract(manifest,read(experiment.BASELINE/"manifest.json"))
    baseline={}
    for domain in ("natural","artificial"):
        path=experiment.BASELINE/"det_position"/f"{domain}_seed42.npz"
        with np.load(path) as archive:
            baseline[domain]={key:archive[key] for key in ("rows","videos","ids","state" if domain=="natural" else "point")}
    arrays={}
    for arm in ARMS:
        folder=out/arm
        published=read(folder/"evaluation.json")
        verify_artifacts(folder,published["artifacts"])
        for stage in ("R","g"):
            history=read(folder/f"{stage}_history.json")
            convergence=read(folder/f"{stage}_convergence.json")
            if not history[-1]["converged"] or history[-1]["epoch"]!=convergence["stop_epoch"]:
                raise ValueError("Convergence record mismatch")
            if min(history,key=lambda row:row["dev"]["position"])["epoch"]!=convergence["selected_epoch"]:
                raise ValueError("Selected checkpoint does not minimize dev position")
            for name in ("best","last"):
                checkpoint=torch.load(folder/f"{name}_{stage}.pt",map_location="cpu",weights_only=True)
                if checkpoint["identity"]!=published["identity"]:
                    raise ValueError("Checkpoint identity mismatch")
                if name=="last" and not checkpoint["converged"]:
                    raise ValueError("Last checkpoint is incomplete")
        methods={}
        arrays[arm]={}
        for domain in ("natural","artificial"):
            path=folder/f"{domain}_seed42.npz"
            with np.load(path) as archive:
                data={key:archive[key] for key in archive.files}
            before=baseline[domain]
            group_key="state" if domain=="natural" else "point"
            for key in ("ids","videos",group_key):
                np.testing.assert_array_equal(data[key],before[key])
            for key in ("rows","norms","point_norms","gate"):
                if not np.isfinite(data[key]).all():
                    raise ValueError("Nonfinite evaluation")
            raw_difference=float(np.max(np.abs(data["rows"][:,0]-before["rows"][:,0])))
            np.testing.assert_allclose(data["rows"][:,0],before["rows"][:,0],rtol=0,atol=1e-6)
            for threshold in (.02,.03,.05,.1):
                for coordinate in (0,1):
                    boundary=threshold if coordinate==0 else .7
                    new = data["rows"][:,0,:,coordinate]
                    old = before["rows"][:,0,:,coordinate]
                    np.testing.assert_array_equal(new<=boundary if coordinate==0 else new>=boundary,
                                                  old<=boundary if coordinate==0 else old>=boundary)
            result["raw_parity"][arm+"_"+domain]=raw_difference
            masks={"occluded":data["state"]==1,"visible":data["state"]==2} if domain=="natural" else {"covered":data["point"],"uncovered":~data["point"]}
            names=("raw","full_R","learned_g") if domain=="natural" else ("raw","full_R","learned_g","mask_R")
            methods[domain]={name:{group:summarize(data["rows"][:,i],mask,data["videos"],before["rows"][:,i])
                for group,mask in masks.items()} for i,name in enumerate(names)}
            methods[domain+"_semantic"]={name:{group:semantic_metrics(data['rows'][:,i],mask)
                for group,mask in masks.items()} for i,name in enumerate(names)}
            for name in names:
                for group in masks:
                    measured=methods[domain][name][group]
                    saved=published[domain][name][group]
                    if measured["points"]!=saved["points"] or abs(measured["mean_error"]-saved["mean_error"]["mean"])>1e-6:
                        raise ValueError("Published metric mismatch")
                    for threshold in measured["thresholds"]:
                        for metric in ("joint","location"):
                            if measured["thresholds"][threshold][metric]!=saved["thresholds"][threshold][metric]:
                                raise ValueError("Published threshold mismatch")
            arrays[arm][domain]=data
        methods.update(R=published["R"],g=published["g"],code_usage=published["code_usage"],
                       peak_gpu_bytes=published["peak_gpu_bytes"],inference_seconds=published["inference_seconds"])
        result["conditions"][arm]=methods
        files.extend(p for p in folder.iterdir() if p.suffix in (".json",".npz"))
    historical = experiment.ROOT/'outputs/diagnostics/messenger_codebook_20261119'
    historic_completion=read(historical/'numeric_completion.json')
    verify_artifacts(historical,historic_completion['artifacts'])
    archived={}
    for domain in ('natural','artificial'):
        with np.load(historical/'continuous16'/f'{domain}_seed42.npz') as archive:
            archived[domain]={key:archive[key] for key in archive.files}
        for key in ('ids','videos','state' if domain=='natural' else 'point'):
            np.testing.assert_array_equal(archived[domain][key],baseline[domain][key])
    arrays['continuous16_saved']=archived
    result['dimension_pairs']['continuous32_minus_saved16']=pair_comparison(arrays['continuous32'],archived)
    for dimension in DIMENSIONS:
        for count in COUNTS:
            names={mode:f'{mode}_d{dimension}_k{count}' for mode in 'ABCDE'}
            initial=[read(out/names[mode]/'initialization.json')['model_sha256'] for mode in 'ABCD']
            if len(set(initial)) != 1:
                raise ValueError('A-D initial weights differ')
            key=f'd{dimension}_k{count}'
            files.append(out/f'initial_codes_{key}.json')
            for label,a,b in (('B_minus_A','B','A'),('D_minus_C','D','C')):
                result['lookup_pairs'][key+'_'+label]=pair_comparison(arrays[names[a]],arrays[names[b]])
            for label,a,b in (('C_minus_A','C','A'),('D_minus_B','D','B')):
                result['dictionary_update_pairs'][key+'_'+label]=pair_comparison(arrays[names[a]],arrays[names[b]])
            result['attention_pairs'][key+'_E_minus_D']=pair_comparison(arrays[names['E']],arrays[names['D']])
            for mode in 'ABCDE':
                control='continuous32' if dimension == 32 else 'continuous16_saved'
                result['vq_vs_continuous'][names[mode]]=pair_comparison(arrays[names[mode]],arrays[control])
        for mode in 'ABCDE':
            result['capacity_pairs'][f'{mode}_d{dimension}']=pair_comparison(arrays[f'{mode}_d{dimension}_k8192'],arrays[f'{mode}_d{dimension}_k4096'])
    for mode in 'ABCDE':
        for count in COUNTS:
            result['dimension_pairs'][f'{mode}_k{count}']=pair_comparison(arrays[f'{mode}_d32_k{count}'],arrays[f'{mode}_d16_k{count}'])
    result['historical_control']=dict(numeric_completion_sha256=base.sha(historical/'numeric_completion.json'),
        continuous16={domain:base.sha(historical/'continuous16'/f'{domain}_seed42.npz') for domain in ('natural','artificial')})
    result["limitations"]=[
        "Small historical train-internal split with backbone exposure, single seed42.",
        "Saved control is position-primary regression with 0.01 MSE and 0.01 cosine; it is not retrained.",
        "Compressed architecture includes a learned projection and contextual residual; comparison to the 384D model also changes architecture.",
        "Directions start from the measured shared fit basis; the compressed encoder can rotate them during training.",
        "Hard nearest-code inference with a straight-through training estimator.",
        "No clean teacher features are passed to the keypoint head."
    ]
    base.write(out/"comparison_audit.json",result)
    lines=["# Codebook factorial comparison: dimensions, lookup, update, spatial decoder","",
           "21 fresh arms; saved 384D and continuous16 controls; identical historical train-internal video split.","",
           "| 条件 | 自然遮蔽平均誤差 | 5%座標成功 | 5%presence込み成功 | 可視平均誤差 |",
           "|---|---:|---:|---:|---:|"]
    saved=baseline["natural"]
    mask=saved["state"]==1
    visible=saved["state"]==2
    control=summarize(saved["rows"][:,2],mask,saved["videos"],saved["rows"][:,2])
    control_visible=summarize(saved["rows"][:,2],visible,saved["videos"],saved["rows"][:,2])
    table=[("saved_control",control,control_visible)]
    old=archived['natural']
    table.append(('continuous16_saved',summarize(old['rows'][:,2],old['state']==1,old['videos'],old['rows'][:,2]),
                  summarize(old['rows'][:,2],old['state']==2,old['videos'],old['rows'][:,2])))
    for arm in ARMS:
        groups=result["conditions"][arm]["natural"]["learned_g"]
        table.append((arm,groups["occluded"],groups["visible"]))
    for name,occluded,visible in table:
        threshold=occluded["thresholds"]["0.05"]
        lines.append(f'| {name} | {occluded["mean_error"]*100:.5f}% | {threshold["location"]} | {threshold["joint"]} | {visible["mean_error"]*100:.5f}% |')
    lines.extend(["","誤差の正規化は既存比較と同じGT楕円長径。g融合後の表を掲載。",
                  "full_R・人工mask_R・動画対応95%区間・コード使用率・メモリはcomparison_audit.json。",
                  "全数値評価の完了後、今回新規生成した重みと初期辞書を削除し、SHAと削除記録を保存。"])
    (out/"report.md").write_text("\n".join(lines)+"\n")
    files.extend([out/"comparison_audit.json",out/"report.md"])
    artifacts={str(p.relative_to(out)):base.sha(p) for p in files}
    weights=[]
    for path in weight_paths(out):
        if not path.is_file():
            raise ValueError(f"Expected checkpoint missing before numerical completion: {path}")
        weights.append(dict(path=str(path.relative_to(out)),absolute_path=str(path.resolve()),sha256=base.sha(path),bytes=path.stat().st_size))
    base.write(complete,dict(all_numeric_evaluations_complete=True,artifacts=artifacts,weights=weights,
                           baseline_retrained=False,weight_state_at_numeric_completion="retained",no_clean_head_comparison=True))


def pair_comparison(current,before):
    result={}
    for domain in ("natural","artificial"):
        a,b=current[domain],before[domain]
        masks={"occluded":a["state"]==1,"visible":a["state"]==2} if domain=="natural" else {"covered":a["point"],"uncovered":~a["point"]}
        result[domain]={}
        for group,mask in masks.items():
            videos=np.broadcast_to(a["videos"][:,None],mask.shape)[mask]
            result[domain][group]={}
            for index,name in ((1,"full_R"),(2,"learned_g")) + (((3,"mask_R"),) if domain == "artificial" else ()):
                result[domain][group][name]=paired_interval((a["rows"][:,index,:,0]-b["rows"][:,index,:,0])[mask].astype(float),videos)
    return result


def weight_paths(out):
    root=Path(out).resolve()
    if Path(out).is_symlink() or Path(out).absolute()!=root:
        raise ValueError("Output root is a symlink")
    paths=[root/arm/f"{which}_{stage}.pt" for arm in ARMS for stage in ("R","g") for which in ("best","last")]
    paths.extend(root/f"initial_codes_d{dim}_k{count}.npy" for dim in DIMENSIONS for count in COUNTS)
    for path in paths:
        if path.is_symlink() or path.parent.resolve()!=path.parent or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe checkpoint path")
    return paths


def assert_not_in_use(out):
    targets={str(path) for path in weight_paths(out)}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            args=(entry/"cmdline").read_bytes().split(b"\0")
            if args and b"python" in args[0] and any(b"run_messenger_codebook_factorial.py" in arg for arg in args):
                raise RuntimeError(f"Another comparison process is alive: {entry.name}")
            for descriptor in (entry/"fd").iterdir():
                try:
                    if os.readlink(descriptor) in targets:
                        raise RuntimeError(f"Process {entry.name} has a generated weight open")
                except (FileNotFoundError,PermissionError,OSError):
                    continue
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue


def cleanup(out):
    """Call while holding the canonical lease. Supports recovery after partial deletion."""
    out=Path(out)
    completion=read(out/"numeric_completion.json")
    if not completion["all_numeric_evaluations_complete"]:
        raise ValueError("Numeric evaluation is incomplete")
    verify_artifacts(out,completion["artifacts"])
    expected={str(p.relative_to(out.resolve())):p for p in weight_paths(out)}
    saved={row["path"]:row for row in completion["weights"]}
    if set(expected)!=set(saved):
        raise ValueError("Unexpected deletion scope")
    deletion=out/"checkpoint_deletion.json"
    prepared=deletion.exists()
    if prepared and read(deletion)["weights"]!=completion["weights"]:
        raise ValueError("Deletion ledger changed")
    assert_not_in_use(out)
    for name,path in expected.items():
        if saved[name].get("absolute_path",str(path)) != str(path):
            raise ValueError("Absolute checkpoint scope changed")
        if path.exists():
            if base.sha(path)!=saved[name]["sha256"] or path.stat().st_size!=saved[name]["bytes"]:
                raise ValueError("Checkpoint changed before deletion")
        elif not prepared:
            raise ValueError("Checkpoint disappeared before deletion was prepared")
    base.write(deletion,dict(phase="prepared",output_root=str(out.resolve()),weights=completion["weights"],process_audit_pid=os.getpid()))
    for path in expected.values():
        path.unlink(missing_ok=True)
    if any(path.exists() for path in expected.values()):
        raise RuntimeError("Some generated weights remain")
    base.write(deletion,dict(phase="completed",output_root=str(out.resolve()),weights=completion["weights"],
               deleted_bytes=sum(row["bytes"] for row in saved.values()),process_audit_pid=os.getpid()))
    base.status(out,"completed",all_numeric_evaluations_complete=True,new_weights_remaining=0)
