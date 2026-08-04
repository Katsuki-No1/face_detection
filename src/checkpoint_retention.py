from __future__ import annotations

import re
from pathlib import Path


_PERIODIC_CHECKPOINT = re.compile(r"^epoch_\d+\.(?:pt|pth)$")


def prune_periodic_checkpoints(directory: Path, *, keep: int = 1) -> list[Path]:
    """Bound local epoch snapshots while preserving last/best checkpoints."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    if not directory.is_dir():
        return []
    snapshots = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and _PERIODIC_CHECKPOINT.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    removed = snapshots[keep:]
    for path in removed:
        path.unlink()
    return removed
