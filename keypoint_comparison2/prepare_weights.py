#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"
)
RTMPOSE_DEST = REPO_ROOT / "checkpoint/RTMPose/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400.pth"
DEIMV2_URL = "https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO/resolve/main/model.safetensors"
DEIMV2_DEST = REPO_ROOT / "checkpoint/DEIMv2/deimv2_dinov3_l_coco.safetensors"
DINOV3_DEST = REPO_ROOT / "checkpoint/DINOv3/dinov3_vits16_face_pose_drive_v2_deimv2_backbone.pth"
DINOV3_DETECTOR_SOURCE = (
    REPO_ROOT
    / "outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full/checkpoints/best_val_map.pt"
)
PRODUCTION_CONFIG = REPO_ROOT / "keypoint_comparison2/configs/production.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    print(f"saved: {destination} sha256={_sha256(destination)}")


def _extract_dinov3_s_from_detector(source: Path, destination: Path) -> None:
    import torch

    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Detector checkpoint is not a mapping: {source}")
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError(f"Detector checkpoint has no EMA/model state: {source}")
    if isinstance(state.get("shadow"), dict):
        state = state["shadow"]
    prefixes = ("backbone.dino.backbone.", "module.backbone.dino.backbone.")
    extracted = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in prefixes:
            if str(key).startswith(prefix):
                extracted[str(key)[len(prefix) :]] = value.detach().cpu()
                break
    if len(extracted) != 188:
        raise RuntimeError(f"Expected 188 DINOv3-S tensors, extracted {len(extracted)} from {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(extracted, temporary)
    temporary.replace(destination)
    print(
        f"extracted: {destination} tensors={len(extracted)} "
        f"elements={sum(value.numel() for value in extracted.values())} sha256={_sha256(destination)}"
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _ensure_ellipse_error_calibration(detector_checkpoint: Path) -> None:
    """Preserve the prerequisite for queue runners started before it was added.

    The active queue process holds its original in-memory job list. Its next
    command is this weight extraction, so calibration is enforced here as well
    as by the explicit queue job.
    """

    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    configured_checkpoint = _project_path(raw["detector"]["checkpoint"])
    if configured_checkpoint != detector_checkpoint:
        return
    data = raw["data"]
    model_path = _project_path(data["ellipse_error_model"])
    calibration_annotation = _project_path(
        data["crop_error_calibration"]["annotation"]
    )
    current = False
    if model_path.is_file():
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
            source = model["source"]
            current = (
                source.get("detector_checkpoint_sha256")
                == _sha256(detector_checkpoint)
                and source.get("calibration_annotation_sha256")
                == _sha256(calibration_annotation)
                and source.get("test_split_used") is False
                and source.get("keypoint_selection_split_used") is False
            )
        except (OSError, ValueError, KeyError, TypeError):
            current = False
    if current:
        print(f"ellipse error calibration is current: {model_path}")
        return
    print(
        "ellipse error calibration is missing or stale; calibrating before "
        "DINOv3 extraction",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "keypoint_comparison2/calibrate_ellipse_error.py"),
            "--config",
            str(PRODUCTION_CONFIG),
            "--device",
            "cuda:0",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    if not model_path.is_file():
        raise RuntimeError(f"Ellipse error calibration did not produce {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare production pretrained weights for keypoint comparison 2")
    parser.add_argument("--rtmpose", action="store_true", help="Download the public official RTMPose-M Face6 checkpoint")
    parser.add_argument("--deimv2", action="store_true", help="Download the public official DEIMv2-L COCO checkpoint")
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        help="Copy a DINOv3 ViT-S/16 checkpoint into the expected local path",
    )
    parser.add_argument(
        "--dinov3-url",
        help="Download from the temporary URL supplied by Meta after DINOv3 access approval",
    )
    parser.add_argument(
        "--extract-dinov3-s-from-detector",
        action="store_true",
        help="Strictly extract the DINOv3-S EMA backbone from the local fine-tuned DEIMv2-L checkpoint",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=DINOV3_DETECTOR_SOURCE,
        help="Source checkpoint for --extract-dinov3-s-from-detector",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DINOV3_DEST,
        help="Destination for copied, downloaded, or extracted DINOv3-S weights",
    )
    parser.add_argument("--all-public", action="store_true", help="Download RTMPose and DEIMv2 public weights")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    if args.all_public or args.rtmpose:
        _download(RTMPOSE_URL, RTMPOSE_DEST)
    if args.all_public or args.deimv2:
        _download(DEIMV2_URL, DEIMV2_DEST)
    if args.dinov3_weights and args.dinov3_url:
        raise ValueError("Use either --dinov3-weights or --dinov3-url, not both")
    if args.extract_dinov3_s_from_detector and (args.dinov3_weights or args.dinov3_url):
        raise ValueError(
            "Use --extract-dinov3-s-from-detector, --dinov3-weights, or --dinov3-url, not more than one"
        )
    if args.dinov3_weights:
        source = args.dinov3_weights.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied: {destination} sha256={_sha256(destination)}")
    elif args.dinov3_url:
        _download(args.dinov3_url, destination)
    elif args.extract_dinov3_s_from_detector:
        detector_checkpoint = args.detector_checkpoint.expanduser().resolve()
        _ensure_ellipse_error_calibration(detector_checkpoint)
        _extract_dinov3_s_from_detector(
            detector_checkpoint,
            destination,
        )
    if not any(
        (
            args.all_public,
            args.rtmpose,
            args.deimv2,
            args.dinov3_weights,
            args.dinov3_url,
            args.extract_dinov3_s_from_detector,
        )
    ):
        print(
            "Prepare DINOv3 ViT-S/16 with --extract-dinov3-s-from-detector, "
            "--dinov3-weights FILE, or --dinov3-url URL.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
