from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from script.script_experiment.run_fdp_series_eval import (  # noqa: E402
    default_checkpoints,
    evaluate_keypoints,
    read_json,
    run_pose,
    select_detections,
    training_runtime_rows,
    write_json,
    xywh_to_xyxy,
)


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E0_baseline": {
        "purpose": "No train-time augmentation.",
        "geometry": False,
        "flip": 0.0,
        "photo": None,
        "mosaic": 0.0,
        "blur": None,
    },
    "E1_photo_light": {
        "purpose": "Light photometric augmentation.",
        "geometry": False,
        "flip": 0.0,
        "photo": {"hue_delta": 3, "saturation_delta": 25, "value_delta": 20, "brightness_delta": 16, "contrast_range": (0.8, 1.2)},
        "mosaic": 0.0,
        "blur": None,
    },
    "E2_photo_mid": {
        "purpose": "Medium photometric augmentation, matching the useful YOLO11s direction.",
        "geometry": False,
        "flip": 0.0,
        "photo": {"hue_delta": 5, "saturation_delta": 45, "value_delta": 35, "brightness_delta": 24, "contrast_range": (0.7, 1.3)},
        "mosaic": 0.0,
        "blur": None,
    },
    "E3_geo_mid": {
        "purpose": "Medium geometric augmentation.",
        "geometry": True,
        "flip": 0.5,
        "photo": None,
        "mosaic": 0.0,
        "blur": None,
    },
    "E4_mosaic_mid": {
        "purpose": "Mosaic augmentation only.",
        "geometry": False,
        "flip": 0.0,
        "photo": None,
        "mosaic": 0.4,
        "blur": None,
    },
    "E5_combined_best": {
        "purpose": "Combined medium photometric, geometry, and mosaic augmentation.",
        "geometry": True,
        "flip": 0.5,
        "photo": {"hue_delta": 5, "saturation_delta": 45, "value_delta": 35, "brightness_delta": 24, "contrast_range": (0.7, 1.3)},
        "mosaic": 0.3,
        "blur": None,
    },
    "E6_blur_weak": {
        "purpose": "Weak blur-only augmentation.",
        "geometry": False,
        "flip": 0.0,
        "photo": None,
        "mosaic": 0.0,
        "blur": {"prob": 0.15, "radius_min": 0.2, "radius_max": 0.8},
    },
    "E7_blur_medium": {
        "purpose": "Medium blur-only augmentation.",
        "geometry": False,
        "flip": 0.0,
        "photo": None,
        "mosaic": 0.0,
        "blur": {"prob": 0.30, "radius_min": 0.3, "radius_max": 1.5},
    },
    "E8_blur_strong": {
        "purpose": "Strong blur-only augmentation.",
        "geometry": False,
        "flip": 0.0,
        "photo": None,
        "mosaic": 0.0,
        "blur": {"prob": 0.50, "radius_min": 1.0, "radius_max": 3.0},
    },
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pose_pipeline(exp: dict[str, Any]) -> list[dict[str, Any]]:
    pipeline: list[dict[str, Any]] = [dict(type="LoadImage", backend_args=dict(backend="local"))]
    if float(exp.get("mosaic", 0.0)) > 0.0:
        pipeline.append(dict(type="EnsureMosaicArrayFields"))
        pipeline.append(
            dict(
                type="Mosaic",
                img_scale=(256, 256),
                center_range=(0.75, 1.25),
                pad_val=114.0,
                prob=float(exp["mosaic"]),
                pre_transform=[dict(type="LoadImage", backend_args=dict(backend="local")), dict(type="EnsureMosaicArrayFields")],
            )
        )
        pipeline.append(dict(type="SelectSingleInstance", index=0))
    pipeline.append(dict(type="GetBBoxCenterScale"))
    if float(exp.get("flip", 0.0)) > 0.0:
        pipeline.append(dict(type="RandomFlip", prob=float(exp["flip"]), direction="horizontal"))
    if exp.get("geometry"):
        pipeline.append(
            dict(
                type="RandomBBoxTransform",
                shift_factor=0.08,
                shift_prob=1.0,
                scale_factor=(0.70, 1.30),
                scale_prob=1.0,
                rotate_factor=10.0,
                rotate_prob=0.6,
            )
        )
    pipeline.append(dict(type="TopdownAffine", input_size="codec['input_size']"))
    photo = exp.get("photo")
    if photo:
        pipeline.append(
            dict(
                type="RandomSimplePhotometric",
                prob=1.0,
                hsv_h=int(photo["hue_delta"]),
                hsv_s=int(photo["saturation_delta"]),
                hsv_v=int(photo["value_delta"]),
                brightness_delta=int(photo["brightness_delta"]),
                contrast_delta=max(abs(1.0 - float(photo["contrast_range"][0])), abs(float(photo["contrast_range"][1]) - 1.0)),
                gamma_delta=0.10,
                noise_std=3.0 if int(photo["value_delta"]) >= 35 else 1.5,
            )
        )
    blur = exp.get("blur")
    if blur:
        pipeline.append(dict(type="RandomGaussianBlur", **blur))
    pipeline.extend(
        [
            dict(type="GenerateTarget", encoder="codec"),
            dict(type="PackPoseInputs"),
        ]
    )
    return pipeline


def format_pipeline(pipeline: list[dict[str, Any]]) -> str:
    text = pformat(pipeline, width=110)
    text = text.replace("'codec[\\'input_size\\']'", "codec['input_size']")
    text = text.replace('"codec[\\\'input_size\\\']"', "codec['input_size']")
    text = text.replace("'codec'", "codec")
    text = text.replace("'encoder': 'codec'", "'encoder': codec")
    text = text.replace("'input_size': \"codec['input_size']\"", "'input_size': codec['input_size']")
    text = text.replace('"input_size": "codec[\'input_size\']"', "\"input_size\": codec['input_size']")
    return text


def generate_config(output_dir: Path, aug_dir: Path, name: str, exp: dict[str, Any], epochs: int) -> Path:
    data_root = (output_dir / "data" / "mmpose_head").resolve().as_posix() + "/"
    work_dir = (aug_dir / "work_dirs" / name).resolve().as_posix()
    meta_file = (output_dir / "configs" / "mmpose" / "face5_dataset_meta.py").resolve().as_posix()
    cspnext_checkpoint = (
        REPO_ROOT
        / "output_experiment"
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "cspnext-s_imagenet_600e-ea671761.pth"
    ).resolve().as_posix()
    train_pipeline = format_pipeline(pose_pipeline(exp))
    text = f"""_base_ = ['mmpose::_base_/default_runtime.py']
custom_imports = dict(imports=['mmdet', 'src.data.mmpose_blur_transform'], allow_failed_imports=False)
default_scope = 'mmpose'

experiment_name = '{name}'
experiment_purpose = {exp['purpose']!r}
num_keypoints = 5
input_size = (256, 256)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

max_epochs = {int(epochs)}
base_lr = 4e-3
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=42, deterministic=False)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.005, begin=30, end=max_epochs, T_max=max_epochs - 30, by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=512)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='{cspnext_checkpoint}')),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0., drop_path=0., act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        loss=dict(type='KLDiscretLoss', use_target_weight=True, beta=10., label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '{data_root}'
metainfo = dict(from_file='{meta_file}')
backend_args = dict(backend='local')

train_pipeline = {train_pipeline}
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = dict(type='NME', norm_mode='use_norm_item', norm_item='bbox_size')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less', max_keep_ckpts=2, interval=1), logger=dict(interval=50))
work_dir = '{work_dir}'
"""
    config_path = aug_dir / "configs" / f"{name}.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def write_experiment_plan(aug_dir: Path, configs: dict[str, Path], epochs: int) -> None:
    rows = []
    selected_experiments = {name: EXPERIMENTS[name] for name in configs}
    for name, exp in selected_experiments.items():
        row = {"experiment": name, "config": configs[name].as_posix(), "epochs": epochs, **exp}
        row["photo"] = json.dumps(row["photo"], ensure_ascii=False) if row.get("photo") else ""
        row["blur"] = json.dumps(row["blur"], ensure_ascii=False) if row.get("blur") else ""
        rows.append(row)
    write_csv(aug_dir / "experiment_plan.csv", rows)
    write_json(aug_dir / "experiment_plan.json", {"created_at": now_text(), "epochs": epochs, "experiments": selected_experiments})


def smoke_config(config_path: Path) -> dict[str, Any]:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS

    init_default_scope("mmpose")
    cfg = Config.fromfile(config_path.as_posix())
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    inputs = sample.get("inputs")
    shape = tuple(inputs.shape) if hasattr(inputs, "shape") else None
    return {"config": config_path.as_posix(), "status": "ok", "dataset_len": len(dataset), "input_shape": shape}


def run_smoke(configs: dict[str, Path], aug_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name, config in configs.items():
        try:
            rows.append({"experiment": name, **smoke_config(config)})
        except Exception as exc:
            rows.append({"experiment": name, "config": config.as_posix(), "status": "failed", "error": repr(exc)})
    write_csv(aug_dir / "smoke" / "config_smoke.csv", rows)
    write_json(aug_dir / "smoke" / "config_smoke.json", {"created_at": now_text(), "rows": rows})
    return rows


def train_one(config: Path, output_dir: Path, aug_dir: Path, name: str, epochs: int, device: str) -> dict[str, Any]:
    work_dir = aug_dir / "work_dirs" / name
    completed_existing = sorted(work_dir.glob(f"epoch_{int(epochs)}.pth"))
    if completed_existing:
        best_existing = sorted(work_dir.glob("best*.pth"), key=lambda path: path.stat().st_mtime)
        return {
            "experiment": name,
            "status": "skipped_existing",
            "config": config.as_posix(),
            "checkpoint": (best_existing[-1] if best_existing else completed_existing[-1]).as_posix(),
        }

    log_path = aug_dir / "logs" / f"train_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from pathlib import Path\n"
        "from script.script_experiment.rtmdet_rtmpose_experiment import train_with_mmengine\n"
        f"raise SystemExit(train_with_mmengine(Path({config.as_posix()!r}), Path({output_dir.as_posix()!r}), "
        f"{('train_hdp_aug_' + name)!r}, ['mmpose', 'mmengine'], max_epochs={int(epochs)}))\n"
    )
    env = os.environ.copy()
    if device.startswith("cuda"):
        env.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1] if ":" in device else "0")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    checkpoints = sorted(work_dir.glob("best*.pth"), key=lambda path: path.stat().st_mtime)
    row = {
        "experiment": name,
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "config": config.as_posix(),
        "log": log_path.as_posix(),
        "elapsed_seconds": round(elapsed, 3),
        "checkpoint": checkpoints[-1].as_posix() if checkpoints else "",
    }
    write_json(aug_dir / "logs" / f"train_{name}_runtime.json", row)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {name}, see {log_path}")
    return row


def checkpoint_for(aug_dir: Path, name: str) -> Path:
    work_dir = aug_dir / "work_dirs" / name
    checkpoints = sorted(work_dir.glob("best*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        checkpoints = sorted(work_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"checkpoint not found for {name}: {work_dir}")
    return checkpoints[-1]


def load_detector_seconds(output_dir: Path) -> float:
    path = output_dir / "reports" / "runtime_summary.json"
    if not path.exists():
        return 0.0
    data = read_json(path)
    for row in data.get("runtime", []):
        if row.get("name") == "full_detector_shared":
            try:
                return float(row.get("elapsed_seconds", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def augmentation_training_runtime_rows(aug_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((aug_dir / "logs").glob("train_*_runtime.json")):
        try:
            data = read_json(path)
        except Exception as exc:
            rows.append(
                {
                    "name": path.stem.removeprefix("train_").removesuffix("_runtime"),
                    "kind": "train_hdp_aug",
                    "status": "read_failed",
                    "runtime_json": path.as_posix(),
                    "error": repr(exc),
                }
            )
            continue
        name = data.get("experiment") or path.stem.removeprefix("train_").removesuffix("_runtime")
        rows.append(
            {
                "name": name,
                "experiment": name,
                "kind": "train_hdp_aug",
                "status": data.get("status"),
                "returncode": data.get("returncode"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "checkpoint": data.get("checkpoint"),
                "config": data.get("config"),
                "log": data.get("log"),
                "runtime_json": path.as_posix(),
            }
        )
    return rows


def evaluate_hdp(output_dir: Path, aug_dir: Path, name: str, config: Path, checkpoint: Path, device: str, head_score_thr: float, topk: int) -> dict[str, Any]:
    from mmpose.apis import init_model
    from script.script_experiment.rtmdet_rtmpose_experiment import local_checkpoint_load_context

    detector_predictions_path = output_dir / "predictions" / "detector" / "predictions.json"
    if not detector_predictions_path.exists():
        raise FileNotFoundError(f"detector predictions not found: {detector_predictions_path}")
    detector_predictions = read_json(detector_predictions_path)
    head_coco = read_json(output_dir / "data" / "mmpose_head" / "test.json")
    dataset_root = output_dir / "data" / "mmdet_hf"

    with local_checkpoint_load_context(checkpoint, output_dir):
        pose_model = init_model(config.as_posix(), checkpoint.as_posix(), device=device)

    started = time.perf_counter()
    rows = []
    detections_by_image = {int(row["image_id"]): row.get("detections", []) for row in detector_predictions.get("images", [])}
    for image in head_coco.get("images", []):
        detections = select_detections(detections_by_image.get(int(image["id"]), []), 1, head_score_thr, topk)
        boxes = [xywh_to_xyxy(det["bbox"]) for det in detections]
        instances = run_pose(pose_model, dataset_root / image["file_name"], boxes)
        for instance, det in zip(instances, detections):
            instance["detector_score"] = det.get("score", 0.0)
            instance["detector_category_id"] = det.get("category_id")
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "instances": instances})
    pose_seconds = time.perf_counter() - started

    eval_dir = aug_dir / "evaluations" / name
    predictions = {"created_at": now_text(), "status": "completed", "stage": "HDP", "split": "test", "images": rows}
    write_json(eval_dir / "predictions" / "HDP_predictions.json", predictions)
    metrics = evaluate_keypoints(head_coco, predictions, "HDP", eval_dir)
    detector_seconds = load_detector_seconds(output_dir)
    runtime = {
        "experiment": name,
        "kind": "inference",
        "pose_only_seconds": round(pose_seconds, 6),
        "estimated_end_to_end_seconds": round(detector_seconds + pose_seconds, 6),
        "detector_shared_seconds": detector_seconds,
        "images": len(head_coco.get("images", [])),
        "checkpoint": checkpoint.as_posix(),
    }
    write_json(eval_dir / "runtime.json", runtime)
    return {"experiment": name, **metrics, **runtime}


def scalar_path(aug_dir: Path, name: str) -> Path | None:
    candidates = sorted((aug_dir / "work_dirs" / name).glob("*/vis_data/scalars.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def scalar_paths(aug_dir: Path, name: str) -> list[Path]:
    return sorted((aug_dir / "work_dirs" / name).glob("*/vis_data/scalars.json"), key=lambda path: path.stat().st_mtime)


def plot_curves(aug_dir: Path, names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    curve_dir = aug_dir / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        paths = scalar_paths(aug_dir, name)
        if not paths:
            continue
        train_x: list[float] = []
        train_loss: list[float] = []
        val_x: list[float] = []
        val_nme: list[float] = []
        for path in paths:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    if "epoch" in row and "loss" in row:
                        train_x.append(float(row["epoch"]))
                        train_loss.append(float(row["loss"]))
                        rows.append({"experiment": name, "phase": "train", "epoch": row.get("epoch"), "loss": row.get("loss"), "loss_kpt": row.get("loss_kpt"), "lr": row.get("lr")})
                    elif "NME" in row:
                        val_x.append(float(row.get("step", len(val_x) + 1)))
                        val_nme.append(float(row["NME"]))
                        rows.append({"experiment": name, "phase": "val", "epoch": row.get("step"), "NME": row.get("NME")})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
        axes[0].plot(train_x, train_loss, label="loss")
        axes[0].set_title(f"{name} train loss")
        axes[0].set_xlabel("epoch")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(val_x, val_nme, label="NME")
        axes[1].set_title(f"{name} validation NME")
        axes[1].set_xlabel("epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(curve_dir / f"{name}_curves.png")
        plt.close(fig)
    write_csv(curve_dir / "hdp_aug_training_curves.csv", rows)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    aug_dir = (output_dir / "augmentation_hdp").resolve()
    aug_dir.mkdir(parents=True, exist_ok=True)
    selected = args.experiments or list(EXPERIMENTS)
    configs = {name: generate_config(output_dir, aug_dir, name, EXPERIMENTS[name], args.epochs) for name in selected}
    write_experiment_plan(aug_dir, configs, args.epochs)
    smoke_rows = run_smoke(configs, aug_dir)
    failed = [row for row in smoke_rows if row.get("status") != "ok"]
    if failed:
        print(json.dumps({"status": "smoke_failed", "failed": failed}, ensure_ascii=False, indent=2))
        return 2
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "aug_dir": aug_dir.as_posix(), "experiments": selected}, ensure_ascii=False, indent=2))
        return 0

    train_rows = []
    if not args.eval_only:
        for name in selected:
            train_rows.append(train_one(configs[name], output_dir, aug_dir, name, args.epochs, args.device))
        write_csv(aug_dir / "training_runtime.csv", augmentation_training_runtime_rows(aug_dir))

    plot_curves(aug_dir, selected)

    eval_rows = []
    for name in selected:
        checkpoint = checkpoint_for(aug_dir, name)
        eval_rows.append(evaluate_hdp(output_dir, aug_dir, name, configs[name], checkpoint, args.device, args.head_score_thr, args.topk))
    write_csv(aug_dir / "hdp_aug_eval_summary.csv", eval_rows)

    runtime_rows = training_runtime_rows(output_dir) + augmentation_training_runtime_rows(aug_dir)
    runtime_rows.extend(
        {
            "name": row["experiment"],
            "kind": "inference_hdp_aug",
            "elapsed_seconds": row.get("pose_only_seconds"),
            "estimated_end_to_end_seconds": row.get("estimated_end_to_end_seconds"),
            "images": row.get("images"),
        }
        for row in eval_rows
    )
    write_csv(aug_dir / "runtime_summary.csv", runtime_rows)
    write_json(aug_dir / "run_summary.json", {"created_at": now_text(), "experiments": selected, "training": train_rows, "evaluation": eval_rows})
    print(json.dumps({"status": "completed", "summary": (aug_dir / "hdp_aug_eval_summary.csv").as_posix()}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HDP RTMPose-head augmentation ablation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outsput/experience"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-score-thr", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
