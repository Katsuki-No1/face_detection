"""Bounded promotion of WIDER pretraining checkpoints into reusable artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_wider_checkpoint(
    *,
    component: str,
    source: Path,
    destination: Path,
    config: Path,
    dataset_manifest: Path,
    initialization: str,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    config = config.resolve()
    dataset_manifest = dataset_manifest.resolve()
    for required in (source, config, dataset_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "component": component,
        "checkpoint": str(destination),
        "checkpoint_sha256": _sha256(destination),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256(source),
        "config": str(config),
        "config_sha256": _sha256(config),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "initialization": initialization,
        "retention_contract": (
            "one promoted best checkpoint for this component; COCO source remains separate"
        ),
    }
    manifest = destination.with_name(destination.stem + "_manifest.json")
    manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_tmp.replace(manifest)
    return payload
