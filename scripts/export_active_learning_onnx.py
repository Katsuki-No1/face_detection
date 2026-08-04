#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.active_learning.onnx_export import (  # noqa: E402
    DETECTOR_INPUT_NAME,
    DETECTOR_OUTPUT_NAMES,
    QUERY_INPUT_NAME,
    QUERY_OUTPUT_NAMES,
    ExportedModel,
    compare_pytorch_onnxruntime,
    export_onnx_model,
    load_detector_wrapper,
    load_query_wrapper,
    sha256_file,
    validate_onnx_structure,
    write_json,
)


DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "configs/model_registry/active_learning_strongest_models.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/active_learning_tensorrt/deimv2_l_query_refinement/onnx"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the production DEIMv2-L ellipse detector and Query "
            "Refinement model to dynamic-batch ONNX on CPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--model-ref",
        default="deimv2_l_query_refinement",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--component",
        choices=("all", "detector", "query"),
        default="all",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-numerical-check",
        action="store_true",
        help="Only export and structurally validate the ONNX graphs",
    )
    return parser.parse_args(argv)


def _load_model_entry(
    registry_path: Path,
    model_ref: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    models = raw.get("models", {})
    if model_ref not in models:
        raise KeyError(f"Unknown model_ref={model_ref!r}: {registry_path}")
    entry = dict(models[model_ref])
    assets = {
        str(name): (
            Path(value).expanduser().resolve()
            if Path(value).expanduser().is_absolute()
            else (PROJECT_ROOT / str(value)).resolve()
        )
        for name, value in dict(entry.get("assets", {})).items()
    }
    for name in ("detector_config", "detector_checkpoint", "query_checkpoint"):
        if name not in assets or not assets[name].is_file():
            raise FileNotFoundError(f"Missing {model_ref}.{name}: {assets.get(name)}")
    return entry, assets


def _artifact_manifest(
    artifact: ExportedModel,
    validation: dict[str, Any],
    numerical: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "path": artifact.path.resolve().as_posix(),
        "sha256": sha256_file(artifact.path),
        "input_name": artifact.input_name,
        "input_shape": list(artifact.input_shape),
        "dynamic_axes": {
            artifact.input_name: {"0": "batch"},
            **{
                output: {"0": "batch"}
                for output in artifact.output_names
            },
        },
        "output_names": list(artifact.output_names),
        "source_assets": {
            name: {
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in artifact.source_assets.items()
        },
        "structure_validation": validation,
        "numerical_validation": numerical,
    }


def _run_component(
    *,
    artifact: ExportedModel,
    wrapper: Any,
    opset: int,
    force: bool,
    numerical_check: bool,
    report_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if force or not artifact.path.is_file():
        print(f"[export] {artifact.name}: {artifact.path}", flush=True)
        export_onnx_model(
            wrapper,
            output_path=artifact.path,
            input_name=artifact.input_name,
            output_names=artifact.output_names,
            input_shape=artifact.input_shape,
            opset=opset,
        )
    else:
        print(f"[reuse] {artifact.name}: {artifact.path}", flush=True)
    print(f"[validate] {artifact.name}: ONNX checker + shape inference", flush=True)
    validation = validate_onnx_structure(
        artifact.path,
        input_name=artifact.input_name,
        output_names=artifact.output_names,
        fixed_input_shape=artifact.input_shape,
    )
    write_json(
        report_dir / f"{artifact.name}_structure.json",
        validation,
    )
    numerical = None
    if numerical_check:
        print(
            f"[compare] {artifact.name}: PyTorch CPU vs ONNX Runtime CPU",
            flush=True,
        )
        numerical = compare_pytorch_onnxruntime(
            wrapper,
            onnx_path=artifact.path,
            input_name=artifact.input_name,
            output_names=artifact.output_names,
            input_shape=artifact.input_shape,
            align_query_axis=artifact.name == "deimv2_l_ellipse",
            minimum_close_fraction=(
                0.995 if artifact.name == "deimv2_l_ellipse" else 1.0
            ),
            sparse_mismatch_mean_abs_limit=(
                0.01 if artifact.name == "deimv2_l_ellipse" else 0.0
            ),
        )
        write_json(
            report_dir / f"{artifact.name}_pytorch_vs_onnxruntime.json",
            numerical,
        )
        if not numerical["passed"]:
            raise RuntimeError(
                f"Numerical agreement failed for {artifact.name}; "
                f"see {report_dir}"
            )
    payload = _artifact_manifest(artifact, validation, numerical)
    payload["elapsed_sec"] = time.perf_counter() - started
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.opset < 16:
        raise ValueError("opset must be >=16 because Query Refinement uses GridSample")
    registry_path = args.registry.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = output_dir / "validation"
    entry, assets = _load_model_entry(registry_path, args.model_ref)
    options = dict(entry.get("options", {}))
    detector_size = int(options.get("detector_input_size", 960))
    query_size = int(options.get("query_input_size", 256))
    selected = (
        ("detector", "query")
        if args.component == "all"
        else (args.component,)
    )

    import onnx
    import onnxruntime
    import torch

    manifest_path = output_dir / "export_manifest.json"
    existing_artifacts: dict[str, Any] = {}
    if manifest_path.is_file():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            existing_manifest.get("model_ref") == args.model_ref
            and int(existing_manifest.get("opset", -1)) == args.opset
        ):
            existing_artifacts = dict(
                existing_manifest.get("artifacts", {})
            )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_ref": args.model_ref,
        "export_device": "cpu",
        "opset": args.opset,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "artifacts": existing_artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if "detector" in selected:
        wrapper = load_detector_wrapper(
            detector_config=assets["detector_config"],
            detector_checkpoint=assets["detector_checkpoint"],
            input_size=detector_size,
        )
        artifact = ExportedModel(
            name="deimv2_l_ellipse",
            path=output_dir / "deimv2_l_ellipse.onnx",
            input_name=DETECTOR_INPUT_NAME,
            output_names=DETECTOR_OUTPUT_NAMES,
            input_shape=(1, 3, detector_size, detector_size),
            source_assets={
                "detector_config": assets["detector_config"],
                "detector_checkpoint": assets["detector_checkpoint"],
            },
        )
        manifest["artifacts"]["detector"] = _run_component(
            artifact=artifact,
            wrapper=wrapper,
            opset=args.opset,
            force=args.force,
            numerical_check=not args.skip_numerical_check,
            report_dir=report_dir,
        )
        del wrapper
        gc.collect()
    if "query" in selected:
        wrapper, query_config = load_query_wrapper(
            query_checkpoint=assets["query_checkpoint"],
            input_size=query_size,
        )
        artifact = ExportedModel(
            name="query_refinement",
            path=output_dir / "query_refinement.onnx",
            input_name=QUERY_INPUT_NAME,
            output_names=QUERY_OUTPUT_NAMES,
            input_shape=(1, 3, query_size, query_size),
            source_assets={"query_checkpoint": assets["query_checkpoint"]},
        )
        manifest["artifacts"]["query"] = _run_component(
            artifact=artifact,
            wrapper=wrapper,
            opset=args.opset,
            force=args.force,
            numerical_check=not args.skip_numerical_check,
            report_dir=report_dir,
        )
        manifest["artifacts"]["query"]["normalization"] = {
            "mean": [
                float(value) for value in query_config["data"]["mean"]
            ],
            "std": [
                float(value) for value in query_config["data"]["std"]
            ],
        }
        del wrapper
        gc.collect()

    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": manifest_path.as_posix(),
                "artifacts": {
                    name: row["path"]
                    for name, row in manifest["artifacts"].items()
                },
                "numerical_checks_passed": all(
                    row["numerical_validation"] is None
                    or row["numerical_validation"]["passed"]
                    for row in manifest["artifacts"].values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
