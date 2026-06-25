from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_features import analyze_features
from .config import apply_cli_overrides, load_config
from .extract_features import extract_features
from .make_report import make_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Phase 1 frozen-DINO feature analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/dino_phase1_feature_analysis.yaml"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--mock-features", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only create crops and extract DINO features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        max_images=args.max_images,
        output_dir_override=args.output_dir,
        mock_features=True if args.mock_features else None,
        split=args.split,
    )
    extract_summary = extract_features(config, max_images=args.max_images)
    if args.extract_only:
        print(json.dumps({"extract": extract_summary}, ensure_ascii=False, indent=2))
        return 0
    analysis_summary = analyze_features(config)
    report_summary = make_report(config)
    print(
        json.dumps(
            {
                "extract": extract_summary,
                "analysis": analysis_summary,
                "report": report_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
