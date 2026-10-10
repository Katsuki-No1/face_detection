"""Authenticated raw-weight initialization for isolated continuation gates.

Production additionally restores optimizer/EMA/epoch from its own copied last.
No behavior changes when train.continuation_source is absent.
"""
import hashlib
from pathlib import Path

import torch


def load_continuation_source(config):
    source = config.get("train", {}).get("continuation_source")
    if not source:
        return None
    path = Path(source["path"])
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise RuntimeError("Continuation source checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("epoch") != source["epoch"] or payload.get("step") != source["step"]:
        raise RuntimeError("Continuation source epoch/step mismatch")
    if not all(key in payload for key in ("model", "optimizer", "ema", "ema_updates", "amp_runtime", "config")):
        raise RuntimeError("Continuation source lacks full training state")
    original = payload["config"]["model"]
    current = config["model"]
    for key in ("type", "architecture_version", "neck", "query_head", "o2m_head", "ellipse_head"):
        if original.get(key) != current.get(key):
            raise RuntimeError(f"Continuation model contract changed: {key}")
    if current.get("init_checkpoint"):
        raise RuntimeError("Continuation cannot also use transfer initialization")
    return payload
