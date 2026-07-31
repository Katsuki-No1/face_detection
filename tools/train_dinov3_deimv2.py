from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dino as train_impl  # noqa: E402

try:  # Support both `python tools/...py` and package-style imports in tests.
    from .dinov3_deimv2_common import (
        load_config,
        resolve_data_paths,
        run_configured_leakage_audit,
        save_run_context,
    )
except ImportError:
    from dinov3_deimv2_common import (  # type: ignore[no-redef]
        load_config,
        resolve_data_paths,
        run_configured_leakage_audit,
        save_run_context,
    )


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and self.streams[0].isatty())

    def fileno(self) -> int:
        if not self.streams:
            raise OSError("Tee has no underlying stream")
        return self.streams[0].fileno()

    @property
    def encoding(self) -> str | None:
        return self.streams[0].encoding if self.streams else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 + official-style DEIMv2 head/face detector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None, help="Run validation every N epochs; 0 disables it.")
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--val-fixed-threshold", type=float, default=None)
    parser.add_argument("--val-score-min", type=float, default=None)
    parser.add_argument("--val-max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--allow-dummy-backbone", action="store_true")
    train_impl.add_training_wandb_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config, force_synthetic_from_paths = resolve_data_paths(config, args.data_root, split="train")
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = int(args.seed)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = int(args.num_workers)
    if args.val_every is not None:
        val_cfg = config.setdefault("train", {}).setdefault("validation", {})
        val_cfg["enabled"] = int(args.val_every) > 0
        val_cfg["every"] = int(args.val_every)
    if args.val_split is not None:
        val_cfg = config.setdefault("train", {}).setdefault("validation", {})
        val_cfg["split"] = str(args.val_split)
    if args.val_fixed_threshold is not None:
        val_cfg = config.setdefault("train", {}).setdefault("validation", {})
        val_cfg["fixed_threshold"] = float(args.val_fixed_threshold)
    if args.val_score_min is not None:
        val_cfg = config.setdefault("train", {}).setdefault("validation", {})
        val_cfg["score_min"] = float(args.val_score_min)
    if args.val_max_images is not None:
        val_cfg = config.setdefault("train", {}).setdefault("validation", {})
        val_cfg["max_images"] = int(args.val_max_images)
    if args.tiny_model:
        config = train_impl.apply_tiny_model_overrides(config)

    force_synthetic = bool(args.synthetic or force_synthetic_from_paths)
    leakage_audit = None
    if not force_synthetic:
        leakage_audit = run_configured_leakage_audit(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_context(args.output_dir, config)
    if leakage_audit is not None:
        (args.output_dir / "data_leakage_audit.json").write_text(
            json.dumps(leakage_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    log_path = args.output_dir / "train.log"
    train_cfg = config.get("train", {})
    allow_cpu_training = bool(train_cfg.get("allow_cpu_training", False))
    allow_dummy_backbone = bool(args.allow_dummy_backbone or train_cfg.get("allow_dummy_backbone", False))

    with log_path.open("a", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        try:
            device = train_impl.resolve_device(
                config,
                args.device,
                training=not args.dry_run,
                allow_cpu_training=allow_cpu_training,
            )
            speed_settings = train_impl.configure_torch_speed(device, config, tf32_override=None)
            print(f"speed_settings: {speed_settings}")
            print(f"force_synthetic: {force_synthetic}")
            if args.dry_run:
                return train_impl.dry_run(config, device=str(device), force_synthetic=force_synthetic)
            wandb_run = train_impl.init_training_wandb(config, args, output_dir=args.output_dir)
            try:
                result = train_impl.train(
                    config,
                    device=str(device),
                    force_synthetic=force_synthetic,
                    output_dir=args.output_dir,
                    resume_from=args.resume_from,
                    save_every=int(args.save_every if args.save_every is not None else config.get("train", {}).get("checkpoint_every", 10)),
                    use_amp=train_impl.resolve_amp(config, args.amp) and torch.cuda.is_available(),
                    max_train_batches=args.max_train_batches,
                    allow_dummy_backbone=allow_dummy_backbone,
                    wandb_run=wandb_run,
                )
            except BaseException:
                if wandb_run is not None:
                    wandb_run.finish(exit_code=1)
                raise
            if wandb_run is not None:
                wandb_run.finish(exit_code=result)
            summary_path = args.output_dir / "summary.md"
            if not summary_path.exists():
                summary_path.write_text(
                    "\n".join(
                        [
                            "# DINOv3 Official-Style DEIMv2 Training Summary",
                            "",
                            "## Purpose",
                            "Train a DINOv3 backbone with official-style DEIMv2 bbox components for head/face detection.",
                            "",
                            "## Result",
                            f"Training command completed with exit code `{result}`.",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            return result
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    raise SystemExit(main())
