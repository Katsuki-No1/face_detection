"""Mirror scalar history for the running teacher-g experiment to W&B."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
from pathlib import Path

import wandb


TRAINING_FIELDS = ("train", "lr", "seconds")
VALIDATION_FIELDS = ("position", "objective", "auxiliary", "mse", "cosine", "code", "commitment")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def metrics(arm, row):
    prefix = f"conditions/{arm}"
    values = {f"{prefix}/epoch": row["epoch"]}
    for field in TRAINING_FIELDS:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[f"{prefix}/train/{field}"] = value
    for field in VALIDATION_FIELDS:
        value = row["dev"].get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[f"{prefix}/dev/{field}"] = value
    values["progress/current_condition"] = arm
    values["progress/current_epoch"] = row["epoch"]
    return values


def sync(output, sync_dir, entity, poll_interval):
    plan = read(output / "plan.json")
    arms = plan["arms"]
    valid_original = len(arms) == 23 and arms[0] == "continuous32"
    valid_unit = len(arms) == 12 and plan.get("telemetry_group") == "messenger_codebook_unit_20261120"
    if plan.get("learned_g") is not False or not (valid_original or valid_unit or (len(arms)==15 and plan.get("telemetry_group")=="messenger_codebook_reconstruction_20261121")):
        raise ValueError("Unsupported teacher-g experiment")
    sync_dir.mkdir(parents=True, exist_ok=True)
    with (sync_dir / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding_path = sync_dir / "wandb.json"
        state_path = sync_dir / "progress.json"
        state = read(state_path) if state_path.exists() else {"epochs": {}}
        if state.get("phase") in ("completed", "failed"):
            return
        previous = read(binding_path) if binding_path.exists() else None
        if previous and previous["entity"] != entity:
            raise ValueError("The existing W&B binding uses a different entity")
        run_id = previous["id"] if previous else wandb.util.generate_id()
        write(binding_path, {"id": run_id, "entity": entity, "project": "face_detection",
                             "url": f"https://wandb.ai/{entity}/face_detection/runs/{run_id}"})
        settings = wandb.Settings(console="off", disable_git=True, disable_code=True,
                                  x_disable_stats=True, x_disable_machine_info=True,
                                  x_save_requirements=False)
        with wandb.init(entity=entity, project="face_detection", id=run_id, resume="allow",
                        name=plan.get("telemetry_name", "Messenger codebook - fixed teacher g - 2026-11-20"),
                        group=plan.get("telemetry_group", "messenger_codebook_teacher_20261120"), job_type="metrics_sync",
                        mode="online", dir=str(sync_dir), settings=settings, save_code=False,
                        config={"learned_g": False, "conditions": arms, "seed": plan["seed"],
                                "microbatch": plan["microbatch"], "effective_batch": plan["effective_batch"],
                                "selection": plan["R_selection"], "source": "local epoch history", "architecture": plan.get("architecture"), "augmentation": plan.get("augmentation"), "feature_region": plan.get("feature_region")}) as run:
            write(binding_path, {"id": run.id, "entity": run.entity, "project": run.project, "url": run.url})
            for arm in arms:
                run.define_metric(f"conditions/{arm}/epoch", hidden=True)
                run.define_metric(f"conditions/{arm}/dev/*", step_metric=f"conditions/{arm}/epoch")
                run.define_metric(f"conditions/{arm}/train/*", step_metric=f"conditions/{arm}/epoch")
            while True:
                status = read(output / "status.json")
                for arm in arms:
                    history = output / arm / "R_history.json"
                    if not history.exists():
                        continue
                    for row in read(history):
                        epoch = row["epoch"]
                        last = state["epochs"].get(arm, -1)
                        if epoch <= last:
                            continue
                        if epoch != last + 1:
                            raise ValueError(f"Missing history for {arm}, expected epoch {last + 1}")
                        run.log(metrics(arm, row))
                        state["epochs"][arm] = epoch
                evaluations = sum((output / arm / "evaluation.json").exists() for arm in arms)
                phase = status["phase"]
                run.summary["evaluated_conditions"] = evaluations
                run.summary["training_phase"] = phase
                state.update(phase=phase, evaluated_conditions=evaluations)
                write(state_path, state)
                if phase == "completed":
                    return
                if phase == "failed":
                    raise RuntimeError("Training stopped; available epoch history was synced")
                time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sync-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--poll", type=float, default=15)
    args = parser.parse_args()
    if args.poll <= 0:
        parser.error("--poll must be positive")
    sync(args.output.resolve(), args.sync_dir.resolve(), args.entity, args.poll)


if __name__ == "__main__":
    main()
