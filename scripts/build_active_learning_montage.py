#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.active_learning.config import load_pipeline_config
from src.active_learning.pipeline import (
    prepare_output,
    run_export,
    run_dense,
    run_inventory,
    run_report,
    run_propose,
    run_scan,
    run_score,
    run_select,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and export an original-resolution active-learning frame dataset from unlabelled videos."
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline YAML config")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="One subdirectory name under input_root; the same name is used under output_root",
    )
    parser.add_argument("--stage", choices=["inventory", "scan", "propose", "dense", "score", "select", "export", "report", "all"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Regenerate outputs for the requested stage(s)")
    mode.add_argument("--resume", action="store_true", help="Resume per-video/per-clip work that is not complete")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def main() -> int:
    args = parse_args()
    config = load_pipeline_config(args.config, dataset_name=args.dataset_name)
    configure_logging(config.output_dir)
    logger = logging.getLogger("active_learning")
    started = time.perf_counter()
    prepare_output(config, force=args.force, requested_stage=args.stage)
    stages = ["inventory", "scan", "propose", "dense", "score", "select", "export", "report"] if args.stage == "all" else [args.stage]
    inventory = None
    scan_metrics = None
    candidates = None
    selected = None
    manifest = None
    proposals = None
    try:
        for stage in stages:
            if stage == "inventory":
                inventory = run_inventory(config, force=args.force)
            elif stage == "scan":
                inventory = inventory or run_inventory(config, force=False)
                scan_metrics = run_scan(config, inventory, force=args.force, resume=args.resume)
            elif stage == "score":
                candidates = run_score(config, force=args.force)
            elif stage == "propose":
                proposals = run_propose(config, force=args.force)
            elif stage == "dense":
                inventory = inventory or run_inventory(config, force=False)
                proposals = proposals or run_propose(config, force=False)
                run_dense(config, inventory, proposals, force=args.force)
            elif stage == "select":
                candidates = candidates or run_score(config, force=False)
                selected = run_select(config, candidates, force=args.force)
            elif stage == "export":
                inventory = inventory or run_inventory(config, force=False)
                candidates = candidates or run_score(config, force=False)
                selected = selected or run_select(config, candidates, force=False)
                manifest = run_export(config, inventory, selected, force=args.force, resume=args.resume)
            elif stage == "report":
                manifest_path = config.output_dir / "selection_manifest.csv"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Run --stage export first: {manifest_path}")
                import csv
                with manifest_path.open(encoding="utf-8", newline="") as source:
                    manifest = list(csv.DictReader(source))
                metrics_path = config.output_dir / "scans" / "scan_metrics.json"
                if metrics_path.exists():
                    import json
                    scan_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                run_report(config, manifest, scan_metrics, force=args.force)
    except Exception:
        logger.exception("Pipeline failed at stage %s", stage)
        return 1
    logger.info("Pipeline completed stage=%s total_elapsed_sec=%.2f", args.stage, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
