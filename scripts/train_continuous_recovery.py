#!/usr/bin/env python3
"""Production launcher with config-bound batch preflight and bounded OOM fallback."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.train_multiscale_query_refinement import parse_args
from keypoint_comparison1.landmark_detection.config import load_landmark_config


def verify_abc_gates(raw, digest, root=PROJECT_ROOT):
    """Fail closed even when invoked outside the queue; a report is not a gate."""
    from keypoint_comparison1.landmark_detection.recovery_acceptance import implementation_hashes
    architecture = raw["evaluation"]["architecture_id"]
    if "continuous_abc_" not in architecture:
        return
    for stage in ("schema", "gpu_step", "overfit", "small_scale"):
        marker = root / "outputs/training_gates" / architecture / stage / "success.json"
        value = json.loads(marker.read_text())
        if (value.get("status") != "passed" or value.get("architecture") != architecture
                or value.get("gate_stage") != stage or value.get("production_config_sha256") != digest
                or value.get("implementation_sha256") != implementation_hashes()):
            raise RuntimeError(f"Missing/stale ABC success marker: {marker}")
    # Co-DINO is scheduled AFTER Messenger by the 2026-10-08 user correction.
    # Its evaluation is not a model/gate prerequisite; ordering belongs to the queue.


def main():
    args = parse_args()
    config = load_landmark_config(args.config)
    raw = config.raw
    plan = json.loads(args.batch_plan.read_text())
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
    verify_abc_gates(raw, digest)
    if plan.get("status") != "passed" or plan.get("candidates") != [4, 8, 16] or plan.get("production_config_sha256") != digest:
        raise RuntimeError("Missing/stale production-config-specific batch preflight")
    batch = int(plan["selected_batch_size"])
    if not any(attempt.get("status") == "safe" and attempt.get("batch_size") == batch for attempt in plan["attempts"]):
        raise RuntimeError("Selected batch was not measured safe")
    effective = int(plan["effective_batch_size"])
    root = config.output_root / "query_refinement"
    root.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    attempts = []
    while batch >= 1:
        command = [str(PROJECT_ROOT / ".venv/bin/python"), str(PROJECT_ROOT / "keypoint_comparison1/scripts/run_landmark_experiments.py"),
                   "--config", str(args.config.resolve()), "--stage", "train", "--variants", "query_refinement", "--device", args.device,
                   "--batch-size", str(batch), "--gradient-accumulation-steps", str(max(1, effective // batch)),
                   "--batch-plan", str(args.batch_plan.resolve()), "--resume" if resume else "--no-resume", "--wandb" if args.wandb else "--no-wandb"]
        log_path = root / f"runtime_batch_{batch}.log"
        oom = False
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                oom |= "CUDA out of memory" in line or "torch.OutOfMemoryError" in line
            code = process.wait()
        attempts.append(dict(batch_size=batch, accumulation=max(1, effective // batch), returncode=code, cuda_oom=oom, log=str(log_path)))
        (root / "runtime_batch_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if code == 0 or not oom or batch == 1:
            return code
        batch //= 2
        resume = (root / "checkpoints/last.pt").is_file()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
