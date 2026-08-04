from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_wandb_run_id(identity: Mapping[str, Any]) -> str:
    """Return a reproducible W&B run id for one training condition."""

    payload = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_wandb_access(
    wandb_module: Any,
    *,
    entity: str | None,
    project: str,
    mode: str,
) -> dict[str, Any]:
    """Fail before training when an online W&B destination is unusable."""

    report: dict[str, Any] = {
        "status": "pass",
        "version": str(wandb_module.__version__),
        "entity": entity,
        "project": project,
        "mode": mode,
        "api_key_configured": None,
        "project_accessible": None,
    }
    if mode != "online":
        return report
    try:
        api = wandb_module.Api(timeout=30)
        report["api_key_configured"] = bool(api.api_key)
        if not report["api_key_configured"]:
            raise RuntimeError(
                "W&B API key is not configured; run `wandb login` first"
            )
        resolved_entity = entity or api.default_entity
        if not resolved_entity:
            raise RuntimeError("W&B entity could not be resolved")
        report["entity"] = resolved_entity
        report["project_accessible"] = bool(
            api.project(project, entity=resolved_entity)
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            "W&B authentication/project preflight failed "
            f"({type(error).__name__}): {error}"
        ) from error
    if not report["project_accessible"]:
        raise RuntimeError(
            f"W&B project is not accessible: {report['entity']}/{project}"
        )
    return report


def flatten_wandb_summary(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, int | float | str | bool]:
    """Flatten finite scalar JSON values while leaving bulky arrays in artifacts."""

    flattened: dict[str, int | float | str | bool] = {}
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).replace(".", "_")
            child_prefix = f"{prefix}/{key}" if prefix else key
            flattened.update(flatten_wandb_summary(child, prefix=child_prefix))
        return flattened
    if isinstance(value, bool):
        flattened[prefix] = value
    elif isinstance(value, int):
        flattened[prefix] = value
    elif isinstance(value, float):
        if math.isfinite(value):
            flattened[prefix] = value
    elif isinstance(value, str) and len(value) <= 512:
        flattened[prefix] = value
    return flattened


def sync_evaluation_to_wandb(
    wandb_module: Any,
    *,
    state_path: Path,
    split: str,
    metric_sources: Mapping[str, Path],
    artifact_files: Sequence[Path] = (),
    headline: Mapping[str, int | float | str | bool] | None = None,
    report_path: Path,
) -> dict[str, Any]:
    """Attach full evaluation summaries and source artifacts to a training run."""

    if not state_path.is_file():
        raise FileNotFoundError(f"W&B training state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entity = str(state["entity"])
    project = str(state["project"])
    run_id = str(state["run_id"])
    mode = str(state.get("mode", "online"))
    if mode == "disabled":
        return {
            "status": "disabled",
            "run_id": run_id,
            "split": split,
        }

    summaries: dict[str, int | float | str | bool] = {}
    resolved_sources: dict[str, str] = {}
    for namespace, source_path in metric_sources.items():
        source = source_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation metrics are missing: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        summaries.update(
            flatten_wandb_summary(
                payload,
                prefix=f"evaluation/{split}/{namespace}",
            )
        )
        resolved_sources[str(namespace)] = str(source)
    if headline:
        summaries.update(
            {
                f"evaluation/{split}/headline/{key}": value
                for key, value in headline.items()
            }
        )
    summaries[f"evaluation/{split}/status"] = "complete"

    local_dir = state_path.parent / "wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    run = wandb_module.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(local_dir),
        job_type="evaluation",
    )
    logged_files: list[str] = []
    try:
        run.summary.update(summaries)
        artifact = wandb_module.Artifact(
            name=f"{run_id}-{split}-evaluation",
            type="evaluation",
            metadata={
                "run_id": run_id,
                "split": split,
                "metric_sources": resolved_sources,
            },
        )
        candidates = [*metric_sources.values(), *artifact_files]
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.expanduser().resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            logical_name = f"{resolved.parent.name}_{resolved.name}"
            artifact.add_file(str(resolved), name=logical_name)
            logged_files.append(str(resolved))
        run.log_artifact(artifact, aliases=[split, "latest"])
    finally:
        run.finish()

    synced_at = datetime.now(timezone.utc).isoformat()
    report = {
        "status": "synced",
        "run_id": run_id,
        "run_url": state.get("url"),
        "entity": entity,
        "project": project,
        "split": split,
        "metric_sources": resolved_sources,
        "summary_key_count": len(summaries),
        "artifact_files": logged_files,
        "synced_at_utc": synced_at,
    }
    write_json_atomic(report_path, report)
    evaluation_sync = dict(state.get("evaluation_sync", {}))
    evaluation_sync[split] = {
        "status": "synced",
        "report": str(report_path.resolve()),
        "synced_at_utc": synced_at,
    }
    state["evaluation_sync"] = evaluation_sync
    write_json_atomic(state_path, state)
    return report
