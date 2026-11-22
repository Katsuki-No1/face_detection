"""IO-only recovery adapter after observed WSL reboots and zero-length unsynced files."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_messenger_adversarial_comparison as runner


def sync_file(path):
    path = Path(path)
    with path.open('rb') as f:
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def install():
    original_save = runner.save
    original_write = runner.base.write

    def durable_save(path, obj):
        original_save(path, obj)
        sync_file(path)

    def durable_write(path, obj):
        original_write(path, obj)
        sync_file(path)

    runner.save = runner.primary.save = runner.old.save = durable_save
    runner.base.write = durable_write


if __name__ == '__main__':
    install()
    runner.main(resume=True, wait=True)
