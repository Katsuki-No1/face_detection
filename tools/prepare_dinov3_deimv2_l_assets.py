from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dinov3_deimv2_common import load_config


LOGGER = logging.getLogger("prepare_deimv2_l")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def prepare_source(config: dict[str, Any]) -> dict[str, Any]:
    source = dict(config["assets"]["dinov3_source"])
    path = resolve_path(source["path"])
    repository = str(source["repository"])
    expected_commit = str(source["commit"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Cloning official DINOv3 source into %s", path)
        run(["git", "clone", "--filter=blob:none", repository, str(path)])
        run(["git", "-C", str(path), "checkout", "--detach", expected_commit])
    if not (path / ".git").is_dir():
        raise RuntimeError(f"DINOv3 source path exists but is not a Git checkout: {path}")
    actual_commit = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"DINOv3 source revision mismatch at {path}: expected={expected_commit}, actual={actual_commit}. "
            "The preparation step will not overwrite an existing checkout."
        )
    return {"path": str(path), "repository": repository, "commit": actual_commit, "status": "ready"}


def prepare_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = dict(config["assets"]["deimv2_checkpoint"])
    path = resolve_path(checkpoint["path"])
    expected_sha = str(checkpoint["sha256"]).lower()
    if path.exists():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Checkpoint hash mismatch at {path}: expected={expected_sha}, actual={actual_sha}. "
                "The preparation step will not overwrite it."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading official DEIMv2-L checkpoint to %s", path)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".part", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
        try:
            urllib.request.urlretrieve(str(checkpoint["url"]), temporary_path)
            actual_sha = sha256_file(temporary_path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Downloaded checkpoint hash mismatch: expected={expected_sha}, actual={actual_sha}"
                )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return {
        "path": str(path),
        "url": str(checkpoint["url"]),
        "sha256": expected_sha,
        "bytes": path.stat().st_size,
        "status": "ready",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pinned DINOv3 source and official DEIMv2-L weights.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/dinov3_deimv2_l_vits16_assets.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    report = {
        "status": "ready",
        "config": str(args.config.resolve()),
        "dinov3_source": prepare_source(config),
        "deimv2_checkpoint": prepare_checkpoint(config),
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Assets ready; report=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
