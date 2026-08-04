from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the production RTMDet-HF training run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rtmdet/rtmdet_s_hf_full_960_wandb.py",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wandb_init_kwargs(cfg: Config) -> dict[str, Any]:
    for backend in cfg.vis_backends:
        if str(backend.get("type")) == "WandbVisBackend":
            return dict(backend["init_kwargs"])
    raise RuntimeError("RTMDet-HF production config requires WandbVisBackend")


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = Config.fromfile(config_path)
    data_root = (REPO_ROOT / str(cfg.data_root)).resolve()
    annotation_root = Path(str(cfg.get("annotation_root", "")))
    contract = cfg.dataset_contract
    split_paths: dict[str, set[str]] = {}
    dataset_report: dict[str, Any] = {}
    for split in SPLITS:
        annotation = data_root / annotation_root / f"{split}.json"
        data = json.loads(annotation.read_text(encoding="utf-8"))
        files = [str(row["file_name"]) for row in data["images"]]
        missing = [name for name in files if not (data_root / name).is_file()]
        actual_sha = sha256(annotation)
        if len(files) != int(contract.images[split]):
            raise RuntimeError(f"{split} image count changed")
        if len(data["annotations"]) != int(contract.annotations[split]):
            raise RuntimeError(f"{split} annotation count changed")
        if actual_sha != str(contract.coco_sha256[split]):
            raise RuntimeError(f"{split} COCO SHA-256 changed")
        if {str(row["name"]) for row in data["categories"]} != {"head", "face"}:
            raise RuntimeError(f"{split} category schema changed")
        if missing:
            raise RuntimeError(f"{split} has {len(missing)} missing images")
        if len(set(files)) != len(files):
            raise RuntimeError(f"{split} contains duplicate image paths")
        split_paths[split] = set(files)
        dataset_report[split] = {
            "images": len(files),
            "annotations": len(data["annotations"]),
            "coco_sha256": actual_sha,
            "missing_images": 0,
        }
    overlaps = {
        f"{left}_{right}": len(split_paths[left] & split_paths[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise RuntimeError(f"RTMDet-HF image paths overlap across splits: {overlaps}")

    checkpoint = (REPO_ROOT / str(cfg.load_from)).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing official RTMDet-S pretrained checkpoint: {checkpoint}")

    wandb_kwargs = _wandb_init_kwargs(cfg)
    wandb_config = dict(wandb_kwargs.get("config", {}))
    conflicting_config_keys = sorted(
        key for key, value in wandb_config.items()
        if key in cfg and cfg[key] != value
    )
    if conflicting_config_keys:
        raise RuntimeError(
            "W&B init config keys collide with MMEngine's full config upload: "
            f"{conflicting_config_keys}"
        )
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required") from error
    api = wandb.Api(timeout=30)
    if not api.api_key:
        raise RuntimeError("W&B API key is not configured")
    project = api.project(str(wandb_kwargs["project"]), entity=str(wandb_kwargs["entity"]))
    if project is None:
        raise RuntimeError("W&B project is not accessible")
    if not torch.cuda.is_available():
        raise RuntimeError("RTMDet-HF production training requires CUDA")

    report = {
        "status": "pass",
        "config": str(config_path),
        "work_dir": str((REPO_ROOT / str(cfg.work_dir)).resolve()),
        "dataset": dataset_report,
        "data_root": str(data_root),
        "annotation_root": str(annotation_root),
        "cross_split_path_overlap": overlaps,
        "classes": list(cfg.classes),
        "image_size": list(cfg.img_scale),
        "epochs": int(cfg.train_cfg.max_epochs),
        "batch_size": int(cfg.train_dataloader.batch_size),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": sha256(checkpoint),
        "wandb": {
            "version": wandb.__version__,
            "entity": wandb_kwargs["entity"],
            "project": wandb_kwargs["project"],
            "group": wandb_kwargs["group"],
            "run_name": wandb_kwargs["name"],
            "api_key_configured": True,
            "project_accessible": True,
        },
        "cuda": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    output = REPO_ROOT / str(cfg.work_dir) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
