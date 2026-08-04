#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.head_codino.checkpoint import load_checkpoint_exact
from src.models.head_codino.runtime import (
    build_model_from_runtime_config,
    cuda_operator_preflight,
    load_runtime_config,
    resolve_path,
    save_json,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native-resolution Head Co-DINO")
    parser.add_argument("--config", default="configs/head_codino/production.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-random", action="store_true")
    parser.add_argument("--resolution", action="append", help="HxW; may be repeated")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.checkpoint and not args.allow_random:
        raise SystemExit("--checkpoint is required unless --allow-random is explicitly set")

    config = load_runtime_config(args.config)
    if args.checkpoint:
        config = copy.deepcopy(config)
        config["model"]["backbone_weights"] = None
    benchmark_cfg = dict(config["benchmark"])
    resolutions = (
        [tuple(int(part) for part in item.lower().split("x", 1)) for item in args.resolution]
        if args.resolution
        else [tuple(value) for value in benchmark_cfg["resolutions"]]
    )
    batch_size = args.batch_size or int(benchmark_cfg["batch_size"])
    warmup = args.warmup if args.warmup is not None else int(benchmark_cfg["warmup"])
    iterations = (
        args.iterations if args.iterations is not None else int(benchmark_cfg["iterations"])
    )
    device = torch.device("cuda")
    preflight = cuda_operator_preflight(device)
    model = build_model_from_runtime_config(config).to(device).eval()
    checkpoint_sha256 = None
    if args.checkpoint:
        checkpoint_sha256 = load_checkpoint_exact(model, args.checkpoint)["sha256"]

    results = []
    for height, width in resolutions:
        images = [
            torch.randint(0, 256, (3, height, width), dtype=torch.uint8, device=device)
            for _ in range(batch_size)
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(warmup):
                model.predict(images)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            latencies = []
            head_count_total = 0
            head_count = 0
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                predictions = model.predict(images)
                end.record()
                end.synchronize()
                latencies.append(float(start.elapsed_time(end)))
                head_count = sum(len(prediction["scores"]) for prediction in predictions)
                head_count_total += head_count
        mean_ms = statistics.fmean(latencies)
        record = {
            "resolution": [height, width],
            "batch_size": batch_size,
            "gpu": torch.cuda.get_device_name(device),
            "warmup": warmup,
            "iterations": iterations,
            "latency_ms_mean": mean_ms,
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "images_per_second": batch_size * 1000.0 / mean_ms,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "head_count": head_count,
            "mean_heads_per_image": head_count_total / (iterations * batch_size),
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    report = {
        "preflight": preflight,
        "checkpoint_sha256": checkpoint_sha256,
        "random_weights": args.checkpoint is None,
        "results": results,
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else resolve_path(benchmark_cfg["output"], config=config)
    )
    save_json(report, output)


if __name__ == "__main__":
    main()
