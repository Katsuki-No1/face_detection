from pathlib import Path

import pytest

from src.checkpoint_retention import prune_periodic_checkpoints


def test_periodic_checkpoint_retention_preserves_last_and_best(tmp_path: Path) -> None:
    for name in (
        "epoch_0005.pt",
        "epoch_0010.pt",
        "epoch_0015.pt",
        "last.pt",
        "best_val_loss.pt",
    ):
        (tmp_path / name).write_bytes(name.encode())

    removed = prune_periodic_checkpoints(tmp_path, keep=1)

    assert {path.name for path in removed} == {"epoch_0005.pt", "epoch_0010.pt"}
    assert {path.name for path in tmp_path.iterdir()} == {
        "epoch_0015.pt",
        "last.pt",
        "best_val_loss.pt",
    }


def test_periodic_checkpoint_retention_can_remove_all_snapshots(
    tmp_path: Path,
) -> None:
    (tmp_path / "epoch_0005.pth").write_bytes(b"checkpoint")
    (tmp_path / "last.pth").write_bytes(b"last")

    prune_periodic_checkpoints(tmp_path, keep=0)

    assert not (tmp_path / "epoch_0005.pth").exists()
    assert (tmp_path / "last.pth").is_file()


def test_periodic_checkpoint_retention_rejects_negative_keep(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prune_periodic_checkpoints(tmp_path, keep=-1)
