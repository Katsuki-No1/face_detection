from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.coco_split_audit import audit_coco_splits
from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import image_to_tensor, load_config


LOGGER = logging.getLogger("preflight_deimv2_l")
SPLITS = ("train", "val", "test")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: expected={expected!r}, actual={actual!r}")


def validate_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    backbone = model["backbone"]
    adapter = model["adapter"]
    encoder = model["encoder"]
    decoder = model["decoder"]
    require_equal("model.type", model["type"], "dinov3_deimv2")
    require_equal("model.num_classes", int(model["num_classes"]), 1)
    require_equal("backbone.name", backbone["name"], "dinov3_vits16")
    require_equal("backbone.patch_size", int(backbone["patch_size"]), 16)
    require_equal("backbone.embed_dim", int(backbone["embed_dim"]), 384)
    backbone_policy = str(config["verification"].get("backbone_policy", "full_unfreeze"))
    if backbone_policy not in {"freeze", "full_unfreeze"}:
        raise RuntimeError(f"Unsupported verification.backbone_policy: {backbone_policy!r}")
    expected_freeze = backbone_policy == "freeze"
    require_equal("backbone.freeze", bool(backbone["freeze"]), expected_freeze)
    require_equal("train.backbone_policy", str(config["train"]["backbone_policy"]), backbone_policy)
    require_equal("adapter.interaction_indexes", list(adapter["interaction_indexes"]), [5, 8, 11])
    require_equal("adapter.hidden_dim", int(adapter["hidden_dim"]), 224)
    require_equal("adapter.conv_inplane", int(adapter["conv_inplane"]), 32)
    require_equal("encoder.hidden_dim", int(encoder["hidden_dim"]), 224)
    require_equal("encoder.dim_feedforward", int(encoder["dim_feedforward"]), 896)
    require_equal("decoder.hidden_dim", int(decoder["hidden_dim"]), 224)
    require_equal("decoder.dim_feedforward", int(decoder["dim_feedforward"]), 1792)
    require_equal("decoder.num_layers", int(decoder["num_layers"]), 4)
    require_equal("train.production_mode", bool(config["train"]["production_mode"]), True)
    require_equal("train.ema.enabled", bool(config["train"]["ema"]["enabled"]), True)
    return {
        "status": "pass",
        "variant": "DEIMv2-L",
        "backbone": "DINOv3 ViT-S/16",
        "interaction_indexes": [5, 8, 11],
        "sta_hidden_dim": 224,
        "decoder_layers": 4,
        "backbone_policy": backbone_policy,
        "full_backbone_finetuning": not expected_freeze,
        "ema": True,
    }


def validate_assets(config: Mapping[str, Any]) -> dict[str, Any]:
    assets = config["assets"]
    source_cfg = assets["dinov3_source"]
    checkpoint_cfg = assets["deimv2_checkpoint"]
    source_path = resolve_path(source_cfg["path"])
    checkpoint_path = resolve_path(checkpoint_cfg["path"])
    if not (source_path / ".git").is_dir():
        raise RuntimeError(f"Pinned DINOv3 source is missing: {source_path}; run the asset preparation command")
    result = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not read DINOv3 source revision: {result.stderr}")
    actual_commit = result.stdout.strip()
    require_equal("DINOv3 source commit", actual_commit, str(source_cfg["commit"]))
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Official DEIMv2-L checkpoint is missing: {checkpoint_path}")
    actual_sha = sha256_file(checkpoint_path)
    require_equal("DEIMv2-L checkpoint SHA-256", actual_sha, str(checkpoint_cfg["sha256"]).lower())
    return {
        "status": "pass",
        "dinov3_source": str(source_path),
        "dinov3_commit": actual_commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }


def validate_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    verification = config["verification"]
    data_root = resolve_path(data_cfg["data_root"])
    configured = data_cfg.get("annotation_paths")
    annotations = {
        split: (
            resolve_path(configured[split])
            if isinstance(configured, Mapping) and split in configured
            else data_root / f"{split}.json"
        )
        for split in SPLITS
    }
    hashes: dict[str, str] = {}
    for split, path in annotations.items():
        if not path.is_file():
            raise RuntimeError(f"Missing fixed COCO annotation: {path}")
        hashes[split] = sha256_file(path)
        require_equal(
            f"{split} COCO SHA-256",
            hashes[split],
            str(verification["expected_coco_sha256"][split]),
        )
    audit = audit_coco_splits(
        data_root,
        annotation_paths=annotations,
        expected_category_id=int(data_cfg["category_ids"][0]),
    )
    for split in SPLITS:
        require_equal(
            f"{split} image count",
            int(audit["splits"][split]["images"]),
            int(verification["expected_images"][split]),
        )
    return {"status": "pass", "coco_sha256": hashes, "audit": audit}


def checkpoint_state_numel(path: Path) -> int:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required; install requirements.txt") from error
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        return sum(int(stream.get_tensor(key).numel()) for key in stream.keys())


def parameter_report(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    verification = config["verification"]
    dino = model.backbone.dino
    dino_parameters = sum(parameter.numel() for parameter in dino.parameters())
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in model.decoder.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    official_parameters = int(verification["official_model_zoo_parameters"])
    official_num_classes = int(verification.get("official_num_classes", 80))
    local_num_classes = int(config["model"]["num_classes"])
    decoder_hidden_dim = int(config["model"]["decoder"]["hidden_dim"])
    decoder_layers = int(config["model"]["decoder"]["num_layers"])
    parameters_per_added_class = decoder_hidden_dim + (decoder_layers + 1) * (decoder_hidden_dim + 1)
    official_equivalent_parameters = total_parameters + (
        official_num_classes - local_num_classes
    ) * parameters_per_added_class
    require_equal(
        "Reconstructed local 80-class parameter count",
        official_equivalent_parameters,
        int(verification["reconstructed_official_80_class_parameters"]),
    )
    relative_difference = abs(official_equivalent_parameters - official_parameters) / official_parameters
    tolerance = float(verification["parameter_relative_tolerance"])
    if relative_difference > tolerance:
        raise RuntimeError(
            f"Local official-equivalent parameter count is anomalous: local={official_equivalent_parameters}, "
            f"official={official_parameters}, relative_difference={relative_difference:.6f}, tolerance={tolerance}"
        )
    checkpoint_path = resolve_path(config["assets"]["deimv2_checkpoint"]["path"])
    source_state_numel = checkpoint_state_numel(checkpoint_path)
    require_equal(
        "Official Hugging Face checkpoint state numel",
        source_state_numel,
        int(verification["official_huggingface_state_numel"]),
    )
    transfer = dict(getattr(model, "pretrained_load_info", {}))
    if int(transfer.get("skipped_shape", -1)) != 0:
        raise RuntimeError(f"Official checkpoint has shape mismatches: {transfer}")
    if int(transfer.get("remapped_keys", 0)) <= 0:
        raise RuntimeError(f"Expected DINOv3 wrapper key remapping was not used: {transfer}")
    expected_unexpected = {"decoder.anchors", "decoder.valid_mask"}
    if set(transfer.get("unexpected_keys", [])) != expected_unexpected:
        raise RuntimeError(f"Unexpected official checkpoint-only tensors changed: {transfer}")
    if int(transfer.get("skipped_class_heads", -1)) != 11:
        raise RuntimeError(f"Expected the 11 COCO class-head tensors to be reinitialized: {transfer}")
    if float(transfer.get("loaded_state_numel_fraction", 0.0)) < 0.9999:
        raise RuntimeError(f"Official checkpoint transfer coverage is too low: {transfer}")
    return {
        "status": "pass",
        "dino_vits16_parameters": dino_parameters,
        "sta_and_projections_parameters": backbone_parameters - dino_parameters,
        "hybrid_encoder_parameters": encoder_parameters,
        "deim_decoder_parameters": decoder_parameters,
        "face_model_total_parameters": total_parameters,
        "parameters_per_added_class": parameters_per_added_class,
        "official_equivalent_80_class_parameters": official_equivalent_parameters,
        "trainable_parameters": trainable_parameters,
        "official_model_zoo_parameters": official_parameters,
        "official_parameter_counting_difference": official_equivalent_parameters - official_parameters,
        "relative_difference_to_official": relative_difference,
        "tolerance": tolerance,
        "official_checkpoint_state_numel": source_state_numel,
        "checkpoint_transfer": transfer,
    }


def real_image_forward(model: torch.nn.Module, config: Mapping[str, Any], device: str) -> dict[str, Any]:
    data_root = resolve_path(config["data"]["data_root"])
    train = json.loads((data_root / "train.json").read_text(encoding="utf-8"))
    image_record = train["images"][0]
    image_path = data_root / str(image_record["file_name"])
    image = Image.open(image_path).convert("RGB")
    image_size = int(config["data"]["image_size"])
    tensor, input_hw, original_hw = image_to_tensor(image, image_size)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight was requested but CUDA is unavailable")
    model = model.to(device).eval()
    if hasattr(model, "set_epoch"):
        model.set_epoch(0)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
        detections = model(tensor.to(device))[0]
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    boxes = detections["boxes"]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise RuntimeError(f"Unexpected detection boxes shape: {tuple(boxes.shape)}")
    return {
        "status": "pass",
        "image": str(image_path),
        "input_hw": list(input_hw),
        "original_hw": list(original_hw),
        "device": device,
        "detections_at_score_min": int(boxes.shape[0]),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))) if device.startswith("cuda") else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the production DEIMv2-L DINOv3 ViT-S/16 face detector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.monotonic()
    config = load_config(args.config)
    LOGGER.info("Validating official DEIMv2-L architecture and pinned assets")
    architecture = validate_architecture(config)
    assets = validate_assets(config)
    LOGGER.info(
        "Auditing the complete fixed %d-frame split",
        sum(int(value) for value in config["verification"]["expected_images"].values()),
    )
    dataset = validate_dataset(config)
    LOGGER.info("Building the full DINOv3 ViT-S/16 + DEIMv2-L face model")
    model = build_detector_from_config(config)
    parameters = parameter_report(model, config)
    forward = {"status": "skipped"} if args.skip_forward else real_image_forward(model, config, args.device)
    report = {
        "status": "pass",
        "training_started": False,
        "config": str(args.config.resolve()),
        "architecture": architecture,
        "assets": assets,
        "dataset": dataset,
        "parameters": parameters,
        "real_image_forward": forward,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_value = args.output or Path(config["verification"]["output"]) / "preflight_report.json"
    output = resolve_path(output_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Preflight passed; report=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
