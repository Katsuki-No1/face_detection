#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.active_learning.onnx_export import (  # noqa: E402
    DETECTOR_OUTPUT_NAMES,
    QUERY_OUTPUT_NAMES,
    load_detector_wrapper,
    load_query_wrapper,
    sha256_file,
    write_json,
)
from src.active_learning.tensorrt_backend import (  # noqa: E402
    TensorRTEngineRunner,
    compare_pytorch_tensorrt,
)
from src.active_learning.tensorrt_profile import (  # noqa: E402
    TensorRTEngineBuild,
    load_tensorrt_build_profile,
)


DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "configs/tensorrt/active_learning_deimv2_l_query_refinement_fp16.yaml"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/active_learning_tensorrt/deimv2_l_query_refinement/onnx"
    / "export_manifest.json"
)
DEFAULT_VALIDATION_VIDEO_DIR = (
    PROJECT_ROOT
    / "data/input/active_learning"
    / "active_learning(dataset=3000frame)7-17"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-GPU FP16 TensorRT engines and compare all dynamic "
            "profile points with the source PyTorch models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--validation-video-dir",
        type=Path,
        default=DEFAULT_VALIDATION_VIDEO_DIR,
        help=(
            "Real videos used for the PyTorch/TensorRT numerical gate; "
            "deterministic midpoint frames are decoded"
        ),
    )
    parser.add_argument(
        "--detector-confidence-floor",
        type=float,
        default=0.05,
        help=(
            "Detector confidence floor used by the operational numerical "
            "agreement gate"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and dynamic profiles without importing TensorRT or using CUDA",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Generate and validate TensorRT 11 strongly-typed FP16 ONNX "
            "graphs on CPU, without building engines"
        ),
    )
    parser.add_argument(
        "--skip-numerical-check",
        action="store_true",
        help="Build engines without the PyTorch/TensorRT agreement gate",
    )
    return parser.parse_args(argv)


def _build_engine(
    spec: TensorRTEngineBuild,
    *,
    profile: Any,
    tensorrt_major: int,
    force: bool,
) -> dict[str, Any]:
    import tensorrt as trt

    if spec.engine_path.is_file() and not force:
        return {
            "name": spec.name,
            "status": "reused",
            "path": spec.engine_path.as_posix(),
            "sha256": sha256_file(spec.engine_path),
            "size_bytes": spec.engine_path.stat().st_size,
        }
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if tensorrt_major >= 11:
        network = builder.create_network()
        build_onnx_path = spec.mixed_precision_onnx_path
    else:
        explicit_batch = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        network = builder.create_network(explicit_batch)
        build_onnx_path = spec.onnx_path
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(build_onnx_path.as_posix()):
        errors = [
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        ]
        raise RuntimeError(
            f"TensorRT could not parse {build_onnx_path}:\n"
            + "\n".join(errors)
        )
    expected_inputs = set(spec.inputs)
    actual_inputs = {
        network.get_input(index).name
        for index in range(network.num_inputs)
    }
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            f"TensorRT profile/network input mismatch for {spec.name}: "
            f"profile={sorted(expected_inputs)}, "
            f"network={sorted(actual_inputs)}"
        )
    config = builder.create_builder_config()
    if tensorrt_major < 11:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(profile.workspace_mib) * 1024 * 1024,
    )
    if profile.hardware_compatibility == "ampere_plus":
        config.hardware_compatibility_level = (
            trt.HardwareCompatibilityLevel.AMPERE_PLUS
        )
    optimization = builder.create_optimization_profile()
    for input_name, shape in spec.inputs.items():
        optimization.set_shape(
            input_name,
            shape.minimum,
            shape.optimum,
            shape.maximum,
        )
        configured_shapes = tuple(
            tuple(int(value) for value in configured)
            for configured in optimization.get_shape(input_name)
        )
        expected_shapes = (
            shape.minimum,
            shape.optimum,
            shape.maximum,
        )
        if configured_shapes != expected_shapes:
            raise RuntimeError(
                f"TensorRT optimization shape mismatch for "
                f"{spec.name}.{input_name}: "
                f"{configured_shapes} != {expected_shapes}"
            )
    if config.add_optimization_profile(optimization) < 0:
        raise RuntimeError(
            f"TensorRT rejected optimization profile for {spec.name}"
        )
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed: {spec.name}")
    spec.engine_path.parent.mkdir(parents=True, exist_ok=True)
    spec.engine_path.write_bytes(bytes(serialized))
    return {
        "name": spec.name,
        "status": "built",
        "build_onnx": build_onnx_path.as_posix(),
        "build_onnx_sha256": sha256_file(build_onnx_path),
        "path": spec.engine_path.as_posix(),
        "sha256": sha256_file(spec.engine_path),
        "size_bytes": spec.engine_path.stat().st_size,
        "elapsed_sec": time.perf_counter() - started,
    }


def _prepare_mixed_precision_onnx(
    spec: TensorRTEngineBuild,
    *,
    force: bool,
) -> dict[str, Any]:
    import modelopt
    import modelopt.onnx.autocast as autocast
    import onnx

    output_path = spec.mixed_precision_onnx_path
    started = time.perf_counter()
    if force or not output_path.is_file():
        source_model = onnx.load(
            spec.onnx_path.as_posix(),
            load_external_data=False,
        )
        control_flow_nodes = [
            node.name
            for node in source_model.graph.node
            if node.op_type == "If"
        ]
        if control_flow_nodes:
            raise RuntimeError(
                f"{spec.name} contains batch-sensitive If nodes; rerun the "
                "ONNX exporter with its fixed-RoPE/exportable-attention "
                f"boundary. First nodes: {control_flow_nodes[:5]}"
            )
        print(
            f"[autocast] {spec.name}: FP32 -> mixed FP16 "
            f"{output_path}",
            flush=True,
        )
        model = autocast.convert_to_mixed_precision(
            onnx_path=spec.onnx_path.as_posix(),
            low_precision_type="fp16",
            keep_io_types=True,
            providers=["cpu"],
            data_max=4096,
            opset=18,
        )
        # ModelOpt retains source value_info entries for some mixed-precision
        # boundaries. Recompute them from the converted graph so ONNX strict
        # shape/type inference sees the actual FP16 tensor types.
        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, output_path.as_posix())
    else:
        print(f"[reuse] mixed FP16 {spec.name}: {output_path}", flush=True)
    model = onnx.load(output_path.as_posix(), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    float16_initializers = sum(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in model.graph.initializer
    )
    float32_initializers = sum(
        initializer.data_type == onnx.TensorProto.FLOAT
        for initializer in model.graph.initializer
    )
    cast_nodes = sum(node.op_type == "Cast" for node in model.graph.node)
    if float16_initializers <= 0:
        raise RuntimeError(
            f"ModelOpt did not create FP16 initializers: {output_path}"
        )
    io_types = {
        value.name: int(value.type.tensor_type.elem_type)
        for value in (*model.graph.input, *model.graph.output)
    }
    if any(
        value != onnx.TensorProto.FLOAT for value in io_types.values()
    ):
        raise RuntimeError(
            f"Mixed-precision ONNX must retain FP32 I/O: {io_types}"
        )
    report = {
        "name": spec.name,
        "source_onnx": spec.onnx_path.as_posix(),
        "source_sha256": sha256_file(spec.onnx_path),
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "modelopt_version": modelopt.__version__,
        "opset_imports": {
            (item.domain or "ai.onnx"): int(item.version)
            for item in model.opset_import
        },
        "float16_initializers": float16_initializers,
        "float32_initializers": float32_initializers,
        "cast_nodes": cast_nodes,
        "io_tensor_types": io_types,
        "onnx_checker": "passed",
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = (
        output_path.parent
        / "validation"
        / f"{spec.name}_mixed_fp16_structure.json"
    )
    write_json(report_path, report)
    report["report"] = report_path.as_posix()
    return report


def _profile_shapes(spec: TensorRTEngineBuild) -> list[tuple[int, ...]]:
    if len(spec.inputs) != 1:
        raise ValueError(
            "Numerical validation currently requires one input per engine"
        )
    shape = next(iter(spec.inputs.values()))
    result: list[tuple[int, ...]] = []
    for value in (shape.minimum, shape.optimum, shape.maximum):
        if value not in result:
            result.append(value)
    return result


def _decode_validation_frames(
    video_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    import cv2

    from src.active_learning.media import pixel_sha256

    directory = video_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Numerical-validation video directory does not exist: {directory}"
        )
    videos = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    )
    if not videos:
        raise FileNotFoundError(
            f"No validation video found directly below {directory}"
        )
    frames_rgb: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for path in videos:
        capture = cv2.VideoCapture(path.as_posix())
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open validation video: {path}")
        try:
            frame_count = int(
                round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            )
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_index = max(0, frame_count // 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            okay, frame_bgr = capture.read()
            if not okay or frame_bgr is None:
                raise RuntimeError(
                    f"Cannot decode midpoint validation frame: {path}"
                )
        finally:
            capture.release()
        stat = path.stat()
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        provenance.append(
            {
                "source_video": path.as_posix(),
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "frame_index": frame_index,
                "timestamp_sec": (
                    frame_index / fps if fps > 0 else None
                ),
                "decoded_pixel_sha256": pixel_sha256(frame_bgr),
            }
        )
    return frames_rgb, provenance


def _real_validation_batches(
    spec: TensorRTEngineBuild,
    *,
    video_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    import cv2
    import torch
    from PIL import Image

    frames_rgb, frame_provenance = _decode_validation_frames(video_dir)
    shapes = _profile_shapes(spec)
    maximum_batch = max(shape[0] for shape in shapes)
    input_size = int(shapes[0][-1])
    samples: list[Any] = []
    sample_provenance: list[dict[str, Any]] = []
    if spec.name == "detector":
        from tools.dinov3_deimv2_common import image_to_tensor

        if len(frames_rgb) < maximum_batch:
            raise RuntimeError(
                f"Detector profile needs {maximum_batch} distinct real "
                f"frames, but only {len(frames_rgb)} validation videos exist"
            )
        for frame_rgb, provenance in zip(
            frames_rgb[:maximum_batch],
            frame_provenance[:maximum_batch],
            strict=True,
        ):
            samples.append(
                image_to_tensor(
                    Image.fromarray(frame_rgb),
                    input_size,
                )[0].squeeze(0)
            )
            sample_provenance.append(provenance)
    elif spec.name == "query":
        from keypoint_comparison1.landmark_detection.dataset import (
            image_to_tensor,
        )

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for sample_index in range(maximum_batch):
            source_index = sample_index % len(frames_rgb)
            frame_rgb = frames_rgb[source_index]
            height, width = frame_rgb.shape[:2]
            crop_size = max(1, int(round(min(height, width) * 0.50)))
            grid_index = sample_index // len(frames_rgb)
            x_ratio = (grid_index % 4) / 3.0
            y_ratio = ((grid_index // 4) % 4) / 3.0
            x1 = int(round((width - crop_size) * x_ratio))
            y1 = int(round((height - crop_size) * y_ratio))
            crop = frame_rgb[
                y1 : y1 + crop_size,
                x1 : x1 + crop_size,
            ]
            crop = cv2.resize(
                crop,
                (input_size, input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            samples.append(image_to_tensor(crop, mean=mean, std=std))
            sample_provenance.append(
                {
                    **frame_provenance[source_index],
                    "crop_xywh": [x1, y1, crop_size, crop_size],
                }
            )
    else:
        raise ValueError(f"No real validation input contract for {spec.name!r}")
    batches: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for shape in shapes:
        batch_size = int(shape[0])
        batches.append(torch.stack(samples[:batch_size]))
        provenance.append(
            {
                "input_source": "real-video-midpoint-frames",
                "samples": sample_provenance[:batch_size],
            }
        )
    return batches, provenance


def _validate_engine(
    spec: TensorRTEngineBuild,
    *,
    export_manifest: dict[str, Any],
    device: str,
    validation_video_dir: Path,
    detector_confidence_floor: float,
) -> dict[str, Any]:
    artifact = dict(export_manifest["artifacts"][spec.name])
    source_assets = {
        name: Path(dict(value)["path"])
        for name, value in dict(artifact["source_assets"]).items()
    }
    runner = TensorRTEngineRunner(spec.engine_path, device=device)
    input_name = next(iter(spec.inputs))
    if spec.name == "detector":
        input_size = int(spec.inputs[input_name].minimum[-1])
        wrapper = load_detector_wrapper(
            detector_config=source_assets["detector_config"],
            detector_checkpoint=source_assets["detector_checkpoint"],
            input_size=input_size,
        )
        output_names = DETECTOR_OUTPUT_NAMES
        max_tolerance = {
            "pred_logits": 1.0,
            "pred_boxes": 0.8,
            # An ellipse orientation may wrap by pi while representing the
            # same undirected major axis.
            "pred_ellipses": 3.2,
        }
        mean_tolerance = {
            "pred_logits": 0.05,
            "pred_boxes": 0.01,
            "pred_ellipses": 0.03,
        }
        align_query_axis = True
        operational_confidence_floor = detector_confidence_floor
    elif spec.name == "query":
        input_size = int(spec.inputs[input_name].minimum[-1])
        wrapper, _ = load_query_wrapper(
            query_checkpoint=source_assets["query_checkpoint"],
            input_size=input_size,
        )
        output_names = QUERY_OUTPUT_NAMES
        max_tolerance = {
            "decoded_coordinates": 0.01,
            "keypoint_logits": 0.12,
        }
        mean_tolerance = {
            "decoded_coordinates": 0.002,
            "keypoint_logits": 0.03,
        }
        align_query_axis = False
        operational_confidence_floor = None
        query_probability_threshold = 0.50
    else:
        raise ValueError(f"No validation contract for engine {spec.name!r}")
    if spec.name == "detector":
        query_probability_threshold = None
    input_batches, input_provenance = _real_validation_batches(
        spec,
        video_dir=validation_video_dir,
    )
    result = compare_pytorch_tensorrt(
        wrapper,
        runner=runner,
        input_name=input_name,
        output_names=output_names,
        input_shapes=_profile_shapes(spec),
        max_abs_tolerance=max_tolerance,
        mean_abs_tolerance=mean_tolerance,
        align_query_axis=align_query_axis,
        input_batches=input_batches,
        input_provenance=input_provenance,
        detector_confidence_floor=operational_confidence_floor,
        query_probability_threshold=query_probability_threshold,
    )
    del wrapper, runner
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.detector_confidence_floor < 1.0:
        raise ValueError("--detector-confidence-floor must be between 0 and 1")
    profile = load_tensorrt_build_profile(
        args.profile,
        require_onnx=True,
    )
    summary = {
        "profile": profile.path.as_posix(),
        "precision": profile.precision,
        "workspace_mib": profile.workspace_mib,
        "hardware_compatibility": profile.hardware_compatibility,
        "numerical_validation": {
            "input": "real-video-midpoint-frames",
            "video_dir": args.validation_video_dir.expanduser().resolve().as_posix(),
            "detector_confidence_floor": args.detector_confidence_floor,
        },
        "engines": [
            {
                "name": spec.name,
                "onnx": spec.onnx_path.as_posix(),
                "mixed_precision_onnx": (
                    spec.mixed_precision_onnx_path.as_posix()
                ),
                "engine": spec.engine_path.as_posix(),
                "inputs": {
                    name: {
                        "min": list(shape.minimum),
                        "opt": list(shape.optimum),
                        "max": list(shape.maximum),
                    }
                    for name, shape in spec.inputs.items()
                },
            }
            for spec in profile.engines
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    precision_preparation = {
        spec.name: _prepare_mixed_precision_onnx(
            spec,
            force=args.force,
        )
        for spec in profile.engines
    }
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "precision": "mixed-fp16",
                    "artifacts": precision_preparation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is not installed. Run this command after installing "
            "the TensorRT package matching this CUDA host."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TensorRT engines cannot be built")
    tensorrt_major = int(str(trt.__version__).split(".", 1)[0])
    export_manifest_path = args.export_manifest.expanduser().resolve()
    export_manifest = json.loads(
        export_manifest_path.read_text(encoding="utf-8")
    )
    device = "cuda:0"
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": summary,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorrt": trt.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "export_manifest": {
            "path": export_manifest_path.as_posix(),
            "sha256": sha256_file(export_manifest_path),
        },
        "precision_preparation": precision_preparation,
        "builds": {},
        "numerical_validation": {},
    }
    for spec in profile.engines:
        print(f"[build] {spec.name}: {spec.onnx_path}", flush=True)
        report["builds"][spec.name] = _build_engine(
            spec,
            profile=profile,
            tensorrt_major=tensorrt_major,
            force=args.force,
        )
        if not args.skip_numerical_check:
            print(
                f"[compare] {spec.name}: PyTorch FP32 vs TensorRT FP16 "
                "at min/opt/max",
                flush=True,
            )
            result = _validate_engine(
                spec,
                export_manifest=export_manifest,
                device=device,
                validation_video_dir=args.validation_video_dir,
                detector_confidence_floor=args.detector_confidence_floor,
            )
            report["numerical_validation"][spec.name] = result
            if not result["passed"]:
                report_path = (
                    spec.engine_path.parent / "build_validation_report.json"
                )
                write_json(report_path, report)
                raise RuntimeError(
                    f"TensorRT numerical agreement failed for {spec.name}; "
                    f"see {report_path}"
                )
    report_path = (
        profile.engines[0].engine_path.parent
        / "build_validation_report.json"
    )
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "passed",
                "report": report_path.as_posix(),
                "engines": {
                    name: value["path"]
                    for name, value in report["builds"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
