from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.normal_performance.pipeline import load_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a representative normal-operation 10-minute interval and run YOLO11s Pose inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/normal_performance_10m_20260718.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse completed anchor scans and outputs where possible.")
    mode.add_argument("--force", action="store_true", help="Replace this pipeline's existing per-video output directories.")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    file_handler = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for handler in handlers:
        handler.setFormatter(formatter)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.output_dir)
    summary = run_pipeline(config, force=args.force, resume=args.resume)
    print(f"output: {config.output_dir}")
    print(f"completed videos: {summary['videos_completed']}/{summary['videos_discovered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
