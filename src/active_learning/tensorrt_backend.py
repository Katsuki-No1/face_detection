from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .strongest_types import FaceObservation, StrongestModelAdapter


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TensorRTEngineRunner:
    """TensorRT 10+ runner using PyTorch CUDA tensors as engine buffers."""

    def __init__(self, engine_path: Path, *, device: str = "cuda:0") -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise RuntimeError(
                "TensorRT runtime is not installed. Install the TensorRT "
                "Python package matching the host CUDA/TensorRT libraries."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires an available CUDA GPU")
        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT device must be CUDA")
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize TensorRT engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"Could not create TensorRT context: {self.engine_path}"
            )
        self.stream = torch.cuda.Stream(device=self.device)
        self.input_names: tuple[str, ...] = tuple(
            name
            for name in self._tensor_names()
            if self.engine.get_tensor_mode(name)
            == trt.TensorIOMode.INPUT
        )
        self.output_names: tuple[str, ...] = tuple(
            name
            for name in self._tensor_names()
            if self.engine.get_tensor_mode(name)
            == trt.TensorIOMode.OUTPUT
        )

    def _tensor_names(self) -> tuple[str, ...]:
        return tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        )

    def _torch_dtype(self, name: str) -> Any:
        import numpy as np

        numpy_dtype = np.dtype(
            self.trt.nptype(self.engine.get_tensor_dtype(name))
        )
        mapping = {
            np.dtype(np.float16): self.torch.float16,
            np.dtype(np.float32): self.torch.float32,
            np.dtype(np.int8): self.torch.int8,
            np.dtype(np.int32): self.torch.int32,
            np.dtype(np.int64): self.torch.int64,
            np.dtype(np.bool_): self.torch.bool,
        }
        if numpy_dtype not in mapping:
            raise TypeError(
                f"Unsupported TensorRT tensor dtype {numpy_dtype}: {name}"
            )
        return mapping[numpy_dtype]

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        missing = set(self.input_names) - set(inputs)
        extra = set(inputs) - set(self.input_names)
        if missing or extra:
            raise ValueError(
                f"TensorRT input mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        bound_inputs: dict[str, Any] = {}
        for name in self.input_names:
            tensor = inputs[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"TensorRT input {name!r} must be torch.Tensor")
            tensor = tensor.to(
                device=self.device,
                dtype=self._torch_dtype(name),
                non_blocking=True,
            ).contiguous()
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(
                    f"Input shape is outside the TensorRT profile: "
                    f"{name}={tuple(tensor.shape)}"
                )
            bound_inputs[name] = tensor
        outputs: dict[str, Any] = {}
        for name, tensor in bound_inputs.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT input: {name}")
        for name in self.output_names:
            shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            if not shape or any(value < 0 for value in shape):
                raise RuntimeError(
                    f"TensorRT did not resolve output shape {name}: {shape}"
                )
            tensor = torch.empty(
                shape,
                dtype=self._torch_dtype(name),
                device=self.device,
            )
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT output: {name}")
            outputs[name] = tensor
        current_stream = torch.cuda.current_stream(self.device)
        uses_runner_stream = (
            current_stream.cuda_stream == self.stream.cuda_stream
        )
        if not uses_runner_stream:
            self.stream.wait_stream(current_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(
                f"TensorRT execution failed: {self.engine_path}"
            )
        if not uses_runner_stream:
            current_stream.wait_stream(self.stream)
        return outputs


def _detector_operational_agreement(
    reference_values: Sequence[Any],
    actual_values: Mapping[str, Any],
    *,
    confidence_floor: float,
) -> dict[str, Any]:
    """Compare only detections that can reach active-learning postprocessing."""
    import torch
    from scipy.optimize import linear_sum_assignment

    limits = {
        "confidence": {"maximum": 0.01, "mean": 0.003},
        "box_xywh_normalized": {"maximum": 0.005, "mean": 0.002},
        "ellipse_cxcyab_normalized": {"maximum": 0.01, "mean": 0.003},
        "ellipse_angle_rad_mod_pi": {"maximum": 0.05, "mean": 0.05},
    }
    reference_logits, reference_boxes, reference_ellipses = reference_values
    actual_logits = actual_values["pred_logits"]
    actual_boxes = actual_values["pred_boxes"]
    actual_ellipses = actual_values["pred_ellipses"]
    items: list[dict[str, Any]] = []
    passed = True
    for item_index in range(int(reference_logits.shape[0])):
        reference_scores = reference_logits[item_index].sigmoid().flatten()
        actual_scores = actual_logits[item_index].sigmoid().flatten()
        reference_indices = torch.nonzero(
            reference_scores >= confidence_floor,
            as_tuple=False,
        ).flatten()
        actual_indices = torch.nonzero(
            actual_scores >= confidence_floor,
            as_tuple=False,
        ).flatten()
        reference_count = int(reference_indices.numel())
        actual_count = int(actual_indices.numel())
        count_matches = reference_count == actual_count
        metrics: dict[str, dict[str, Any]] = {}
        if reference_indices.numel() and actual_indices.numel():
            cost = torch.cdist(
                reference_boxes[item_index, reference_indices].float(),
                actual_boxes[item_index, actual_indices].float(),
                p=1,
            )
            rows, columns = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )
            reference_indices = reference_indices[
                torch.as_tensor(
                    rows,
                    dtype=torch.long,
                    device=reference_indices.device,
                )
            ]
            actual_indices = actual_indices[
                torch.as_tensor(
                    columns,
                    dtype=torch.long,
                    device=actual_indices.device,
                )
            ]
            deltas = {
                "confidence": (
                    reference_scores[reference_indices]
                    - actual_scores[actual_indices]
                ).abs(),
                "box_xywh_normalized": (
                    reference_boxes[item_index, reference_indices].float()
                    - actual_boxes[item_index, actual_indices].float()
                ).abs(),
                "ellipse_cxcyab_normalized": (
                    reference_ellipses[
                        item_index,
                        reference_indices,
                        :4,
                    ].float()
                    - actual_ellipses[
                        item_index,
                        actual_indices,
                        :4,
                    ].float()
                ).abs(),
            }
            angle_delta = (
                reference_ellipses[
                    item_index,
                    reference_indices,
                    4,
                ].float()
                - actual_ellipses[
                    item_index,
                    actual_indices,
                    4,
                ].float()
            ).abs().remainder(torch.pi)
            deltas["ellipse_angle_rad_mod_pi"] = torch.minimum(
                angle_delta,
                torch.pi - angle_delta,
            )
            for name, delta in deltas.items():
                maximum = float(delta.max().item()) if delta.numel() else 0.0
                mean = float(delta.mean().item()) if delta.numel() else 0.0
                metric_passed = (
                    maximum <= limits[name]["maximum"]
                    and mean <= limits[name]["mean"]
                )
                metrics[name] = {
                    "max_abs_error": maximum,
                    "mean_abs_error": mean,
                    "max_abs_tolerance": limits[name]["maximum"],
                    "mean_abs_tolerance": limits[name]["mean"],
                    "passed": metric_passed,
                }
        elif not reference_indices.numel() and not actual_indices.numel():
            for name, tolerance in limits.items():
                metrics[name] = {
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "max_abs_tolerance": tolerance["maximum"],
                    "mean_abs_tolerance": tolerance["mean"],
                    "passed": True,
                }
        else:
            for name, tolerance in limits.items():
                metrics[name] = {
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "max_abs_tolerance": tolerance["maximum"],
                    "mean_abs_tolerance": tolerance["mean"],
                    "passed": False,
                }
        item_passed = count_matches and all(
            value["passed"] for value in metrics.values()
        )
        passed = passed and item_passed
        items.append(
            {
                "item_index": item_index,
                "reference_detection_count": reference_count,
                "tensorrt_detection_count": actual_count,
                "detection_count_matches": count_matches,
                "metrics": metrics,
                "passed": item_passed,
            }
        )
    return {
        "contract": "detections_at_or_above_inference_confidence_floor",
        "confidence_floor": confidence_floor,
        "angle_convention": "undirected ellipse angle modulo pi",
        "items": items,
        "passed": passed,
    }


def _query_operational_agreement(
    reference_values: Sequence[Any],
    actual_values: Mapping[str, Any],
    *,
    keypoint_probability_threshold: float,
) -> dict[str, Any]:
    import torch

    reference_coordinates, reference_logits = reference_values
    actual_coordinates = actual_values["decoded_coordinates"]
    actual_logits = actual_values["keypoint_logits"]
    coordinate_delta = (
        reference_coordinates.detach().float()
        - actual_coordinates.detach().float()
    ).abs()
    probability_delta = (
        reference_logits.detach().float().sigmoid()
        - actual_logits.detach().float().sigmoid()
    ).abs()
    reference_presence = (
        reference_logits.detach().float().sigmoid()
        >= keypoint_probability_threshold
    )
    actual_presence = (
        actual_logits.detach().float().sigmoid()
        >= keypoint_probability_threshold
    )
    decisions_match = bool(torch.equal(reference_presence, actual_presence))
    metrics: dict[str, dict[str, Any]] = {}
    for name, delta, maximum_limit, mean_limit in (
        ("decoded_coordinates", coordinate_delta, 0.01, 0.002),
        ("keypoint_probabilities", probability_delta, 0.03, 0.005),
    ):
        maximum = float(delta.max().item()) if delta.numel() else 0.0
        mean = float(delta.mean().item()) if delta.numel() else 0.0
        metrics[name] = {
            "max_abs_error": maximum,
            "mean_abs_error": mean,
            "max_abs_tolerance": maximum_limit,
            "mean_abs_tolerance": mean_limit,
            "passed": maximum <= maximum_limit and mean <= mean_limit,
        }
    passed = decisions_match and all(
        value["passed"] for value in metrics.values()
    )
    return {
        "contract": "keypoint_coordinates_probabilities_and_presence_decisions",
        "keypoint_probability_threshold": keypoint_probability_threshold,
        "presence_decisions_matching": int(
            (reference_presence == actual_presence).sum().item()
        ),
        "presence_decisions_total": int(reference_presence.numel()),
        "presence_decisions_match": decisions_match,
        "metrics": metrics,
        "passed": passed,
    }


def compare_pytorch_tensorrt(
    wrapper: Any,
    *,
    runner: TensorRTEngineRunner,
    input_name: str,
    output_names: Sequence[str],
    input_shapes: Sequence[Sequence[int]],
    seed: int = 20260731,
    max_abs_tolerance: float | Mapping[str, float] = 0.15,
    mean_abs_tolerance: float | Mapping[str, float] = 0.01,
    align_query_axis: bool = False,
    input_batches: Sequence[Any] | None = None,
    input_provenance: Sequence[Mapping[str, Any]] | None = None,
    detector_confidence_floor: float | None = None,
    query_probability_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare FP32 PyTorch outputs with FP16 TensorRT at every profile point."""
    import torch

    if input_batches is not None and len(input_batches) != len(input_shapes):
        raise ValueError("input_batches must match the number of input_shapes")
    if input_provenance is not None and len(input_provenance) != len(input_shapes):
        raise ValueError(
            "input_provenance must match the number of input_shapes"
        )
    if (
        detector_confidence_floor is not None
        and query_probability_threshold is not None
    ):
        raise ValueError(
            "Detector and query operational gates are mutually exclusive"
        )
    wrapper = wrapper.to(runner.device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches: list[dict[str, Any]] = []
    passed = True
    raw_query_diagnostics_passed = True
    for shape_index, shape_value in enumerate(input_shapes):
        shape = tuple(int(value) for value in shape_value)
        if input_batches is None:
            input_tensor = torch.rand(
                shape,
                generator=generator,
            ).to(runner.device)
        else:
            input_tensor = input_batches[shape_index].to(runner.device)
            if tuple(input_tensor.shape) != shape:
                raise ValueError(
                    f"Validation input shape mismatch: "
                    f"{tuple(input_tensor.shape)} != {shape}"
                )
        with torch.inference_mode():
            reference_values = wrapper(input_tensor)
        if isinstance(reference_values, torch.Tensor):
            reference_values = (reference_values,)
        actual_values = runner.run({input_name: input_tensor})
        torch.cuda.synchronize(runner.device)
        if align_query_axis:
            from scipy.optimize import linear_sum_assignment

            boxes_index = tuple(output_names).index("pred_boxes")
            reference_boxes = reference_values[boxes_index].detach().float()
            actual_boxes = actual_values["pred_boxes"].detach().float()
            assignments: list[Any] = []
            for item_index in range(reference_boxes.shape[0]):
                cost = torch.cdist(
                    reference_boxes[item_index],
                    actual_boxes[item_index],
                    p=1,
                )
                rows, columns = linear_sum_assignment(
                    cost.detach().cpu().numpy()
                )
                assignments.append(
                    (
                        torch.as_tensor(
                            rows,
                            dtype=torch.long,
                            device=runner.device,
                        ),
                        torch.as_tensor(
                            columns,
                            dtype=torch.long,
                            device=runner.device,
                        ),
                    )
                )
            for name in output_names:
                value = torch.empty_like(actual_values[name])
                for item_index, (rows, columns) in enumerate(assignments):
                    value[item_index, rows] = actual_values[name][
                        item_index,
                        columns,
                    ]
                actual_values[name] = value
        outputs: list[dict[str, Any]] = []
        for name, reference in zip(
            output_names,
            reference_values,
            strict=True,
        ):
            actual = actual_values[name]
            same_shape = tuple(reference.shape) == tuple(actual.shape)
            if same_shape:
                delta = (
                    reference.detach().float() - actual.detach().float()
                ).abs()
                maximum = float(delta.max().item()) if delta.numel() else 0.0
                mean = float(delta.mean().item()) if delta.numel() else 0.0
                output_max_tolerance = float(
                    max_abs_tolerance.get(name)
                    if isinstance(max_abs_tolerance, Mapping)
                    else max_abs_tolerance
                )
                output_mean_tolerance = float(
                    mean_abs_tolerance.get(name)
                    if isinstance(mean_abs_tolerance, Mapping)
                    else mean_abs_tolerance
                )
                output_passed = (
                    maximum <= output_max_tolerance
                    and mean <= output_mean_tolerance
                )
            else:
                maximum = None
                mean = None
                output_max_tolerance = float("nan")
                output_mean_tolerance = float("nan")
                output_passed = False
            raw_query_diagnostics_passed = (
                raw_query_diagnostics_passed and output_passed
            )
            outputs.append(
                {
                    "name": name,
                    "pytorch_shape": list(reference.shape),
                    "tensorrt_shape": list(actual.shape),
                    "max_abs_error": maximum,
                    "mean_abs_error": mean,
                    "max_abs_tolerance": output_max_tolerance,
                    "mean_abs_tolerance": output_mean_tolerance,
                    "passed": output_passed,
                }
            )
        row: dict[str, Any] = {
            "input_shape": list(shape),
            "raw_query_diagnostics": outputs,
        }
        if input_provenance is not None:
            row["input_provenance"] = dict(input_provenance[shape_index])
        if detector_confidence_floor is not None:
            operational = _detector_operational_agreement(
                reference_values,
                actual_values,
                confidence_floor=detector_confidence_floor,
            )
            row["operational_detection_agreement"] = operational
            passed = passed and operational["passed"]
        elif query_probability_threshold is not None:
            operational = _query_operational_agreement(
                reference_values,
                actual_values,
                keypoint_probability_threshold=query_probability_threshold,
            )
            row["operational_query_agreement"] = operational
            passed = passed and operational["passed"]
        else:
            raw_passed = all(value["passed"] for value in outputs)
            row["raw_output_agreement_passed"] = raw_passed
            passed = passed and raw_passed
        batches.append(row)
    return {
        "backend": "tensorrt-fp16",
        "reference": "pytorch-fp32-cuda",
        "seed": seed,
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "query_axis_aligned": align_query_axis,
        "agreement_gate": (
            "operational-detections"
            if detector_confidence_floor is not None
            else (
                "operational-query"
                if query_probability_threshold is not None
                else "raw-outputs"
            )
        ),
        "raw_query_diagnostics_passed": raw_query_diagnostics_passed,
        "profile_shapes": batches,
        "passed": passed,
    }


class TensorRTDeimv2QueryRefinementAdapter(StrongestModelAdapter):
    """FP16 TensorRT implementation of the strongest-model adapter contract."""

    capabilities = frozenset({"face", "ellipse", "keypoints"})

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        options: Mapping[str, Any],
    ) -> None:
        import torch

        self.model_id = str(model_id)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The TensorRT active-learning adapter requires CUDA")
        self.detector_engine = Path(options["detector_engine"]).resolve()
        self.query_engine = Path(options["query_engine"]).resolve()
        self.export_manifest = Path(options["export_manifest"]).resolve()
        self.profile_path = Path(options["tensorrt_profile"]).resolve()
        self.detector_input_size = int(options.get("detector_input_size", 960))
        self.detector_batch_size = int(options.get("detector_batch_size", 2))
        self.query_input_size = int(options.get("query_input_size", 256))
        self.query_batch_size = int(options.get("query_batch_size", 32))
        self.pipeline_parallel = bool(options.get("pipeline_parallel", False))
        self.pipeline_stage_size = int(options.get("pipeline_stage_size", 4))
        self._keypoint_executor: ThreadPoolExecutor | None = None
        self.crop_scale = float(options.get("crop_scale", 1.25))
        self.maximum_detections = int(options.get("maximum_detections", 100))
        self.inference_confidence_floor = float(
            options["inference_confidence_floor"]
        )
        self.detector_num_classes = int(
            options.get("detector_num_classes", 1)
        )
        self.detector_topk = int(options.get("detector_topk", 300))
        self.mean = tuple(
            float(value)
            for value in options.get("query_mean", (0.485, 0.456, 0.406))
        )
        self.std = tuple(
            float(value)
            for value in options.get("query_std", (0.229, 0.224, 0.225))
        )
        if self.detector_batch_size <= 0 or self.query_batch_size <= 0:
            raise ValueError("Detector and query batch sizes must be positive")
        if self.pipeline_stage_size <= 0:
            raise ValueError("Pipeline stage size must be positive")
        self.detector_runner = TensorRTEngineRunner(
            self.detector_engine,
            device=str(self.device),
        )
        self.query_runner = TensorRTEngineRunner(
            self.query_engine,
            device=str(self.device),
        )
        if self.detector_runner.input_names != ("images",):
            raise RuntimeError(
                f"Unexpected detector bindings: "
                f"{self.detector_runner.input_names}"
            )
        if self.query_runner.input_names != ("crops",):
            raise RuntimeError(
                f"Unexpected query bindings: {self.query_runner.input_names}"
            )

    def _postprocess_detector(
        self,
        raw: Mapping[str, Any],
        original_shapes: Sequence[tuple[int, int]],
    ) -> list[list[dict[str, Any]]]:
        import torch

        from src.models.dinov3_deimv2_ellipse import ellipse_to_aabb
        from tools.dinov3_deimv2_common import scale_detections_to_original

        logits = raw["pred_logits"].float()
        normalized_boxes = raw["pred_boxes"].float()
        normalized_ellipses = raw["pred_ellipses"].float()
        flat_scores = logits.sigmoid().flatten(1)
        topk = min(self.detector_topk, int(flat_scores.shape[1]))
        scores, flat_indices = torch.topk(flat_scores, topk, dim=-1)
        labels = flat_indices % self.detector_num_classes
        query_indices = flat_indices // self.detector_num_classes
        boxes = normalized_boxes.gather(
            1,
            query_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        ellipses = normalized_ellipses.gather(
            1,
            query_indices.unsqueeze(-1).expand(-1, -1, 5),
        )
        cx, cy, width, height = boxes.unbind(-1)
        boxes = torch.stack(
            (
                cx - width * 0.5,
                cy - height * 0.5,
                cx + width * 0.5,
                cy + height * 0.5,
            ),
            dim=-1,
        )
        boxes *= float(self.detector_input_size)
        ellipses = ellipses.clone()
        ellipses[..., :4] *= float(self.detector_input_size)

        all_rows: list[list[dict[str, Any]]] = []
        for index, (original_h, original_w) in enumerate(original_shapes):
            item = {
                "boxes": boxes[index],
                "scores": scores[index],
                "labels": labels[index],
                "ellipses": ellipses[index],
                "ellipse_boxes": ellipse_to_aabb(ellipses[index]),
            }
            item = scale_detections_to_original(
                item,
                input_hw=(
                    self.detector_input_size,
                    self.detector_input_size,
                ),
                original_hw=(original_h, original_w),
            )
            order = torch.argsort(item["scores"], descending=True)
            rows: list[dict[str, Any]] = []
            for detection_index in order.tolist():
                score = float(item["scores"][detection_index])
                if score < self.inference_confidence_floor:
                    continue
                box = [
                    float(value)
                    for value in item["boxes"][detection_index].tolist()
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                rows.append(
                    {
                        "bbox_xyxy": box,
                        "detector_bbox_xyxy": box,
                        "score": score,
                        "label": int(item["labels"][detection_index]),
                        "crop_geometry": "ellipse",
                        "ellipse": [
                            float(value)
                            for value in item["ellipses"][
                                detection_index
                            ].tolist()
                        ],
                        "ellipse_bbox_xyxy": [
                            float(value)
                            for value in item["ellipse_boxes"][
                                detection_index
                            ].tolist()
                        ],
                    }
                )
                if len(rows) >= self.maximum_detections:
                    break
            all_rows.append(rows)
        return all_rows

    def _predict_detector(
        self,
        frames_rgb: Sequence[np.ndarray],
    ) -> list[list[dict[str, Any]]]:
        import torch
        from PIL import Image

        from tools.dinov3_deimv2_common import image_to_tensor

        outputs: list[list[dict[str, Any]]] = []
        device_context = (
            torch.cuda.device(self.device)
            if self.pipeline_parallel
            else nullcontext()
        )
        stream_context = (
            torch.cuda.stream(self.detector_runner.stream)
            if self.pipeline_parallel
            else nullcontext()
        )
        with device_context, stream_context:
            for start in range(0, len(frames_rgb), self.detector_batch_size):
                images = frames_rgb[start : start + self.detector_batch_size]
                tensors = [
                    image_to_tensor(
                        Image.fromarray(image),
                        self.detector_input_size,
                    )[0]
                    for image in images
                ]
                batch = torch.cat(tensors, dim=0).to(
                    self.device,
                    non_blocking=True,
                )
                raw = self.detector_runner.run({"images": batch})
                outputs.extend(
                    self._postprocess_detector(
                        raw,
                        [
                            (int(image.shape[0]), int(image.shape[1]))
                            for image in images
                        ],
                    )
                )
        return outputs

    def _predict_keypoints(
        self,
        frames_rgb: Sequence[np.ndarray],
        detections_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[list[FaceObservation]]:
        import torch

        from keypoint_comparison1.landmark_detection.dataset import image_to_tensor
        from keypoint_comparison1.landmark_detection.geometry import (
            build_crop_transform,
            normalized_crop_to_original,
            warp_crop,
        )
        from src.data.face_crop_cache import crop_geometry_from_detection

        crops: list[Any] = []
        metadata: list[tuple[int, Mapping[str, Any], Any]] = []
        for frame_index, (image_rgb, detections) in enumerate(
            zip(frames_rgb, detections_by_frame, strict=True)
        ):
            for detection in detections:
                crop_box, rotation = crop_geometry_from_detection(
                    detection,
                    require_ellipse=True,
                )
                transform = build_crop_transform(
                    crop_box,
                    output_size=self.query_input_size,
                    scale_factor=self.crop_scale,
                    rotation_deg=rotation,
                    warp_bbox_aspect=False,
                )
                crop = warp_crop(image_rgb, transform)
                crops.append(
                    image_to_tensor(crop, mean=self.mean, std=self.std)
                )
                metadata.append((frame_index, detection, transform))
        output: list[list[FaceObservation]] = [
            [] for _ in detections_by_frame
        ]
        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        device_context = (
            torch.cuda.device(self.device)
            if self.pipeline_parallel
            else nullcontext()
        )
        stream_context = (
            torch.cuda.stream(self.query_runner.stream)
            if self.pipeline_parallel
            else nullcontext()
        )
        with device_context, stream_context:
            for start in range(0, len(crops), self.query_batch_size):
                batch = torch.stack(
                    crops[start : start + self.query_batch_size]
                ).to(self.device, non_blocking=True)
                raw = self.query_runner.run({"crops": batch})
                coordinates = (
                    raw["decoded_coordinates"].detach().float().cpu().numpy()
                )
                scores = (
                    raw["keypoint_logits"]
                    .sigmoid()
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                decoded.extend(zip(coordinates, scores, strict=True))
        for (frame_index, detection, transform), (
            coordinates,
            point_scores,
        ) in zip(metadata, decoded, strict=True):
            points = normalized_crop_to_original(coordinates, transform)
            output[frame_index].append(
                FaceObservation(
                    confidence=float(detection["score"]),
                    bbox_xyxy=tuple(
                        float(value)
                        for value in detection["detector_bbox_xyxy"]
                    ),
                    ellipse=tuple(
                        float(value) for value in detection["ellipse"]
                    ),
                    keypoints=tuple(
                        (float(point[0]), float(point[1]))
                        for point in points
                    ),
                    keypoint_scores=tuple(
                        float(value) for value in point_scores
                    ),
                )
            )
        return output

    def predict_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
    ) -> list[list[FaceObservation]]:
        frames_rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames_bgr
        ]
        if self.pipeline_parallel:
            return self._predict_pipeline_parallel(frames_rgb)
        detections = self._predict_detector(frames_rgb)
        return self._predict_keypoints(frames_rgb, detections)

    def _predict_pipeline_parallel(
        self,
        frames_rgb: Sequence[np.ndarray],
    ) -> list[list[FaceObservation]]:
        if not frames_rgb:
            return []
        stages = [
            frames_rgb[start : start + self.pipeline_stage_size]
            for start in range(0, len(frames_rgb), self.pipeline_stage_size)
        ]
        if self._keypoint_executor is None:
            self._keypoint_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="tensorrt-keypoints",
            )

        output: list[list[FaceObservation]] = []
        previous_frames = stages[0]
        previous_detections = self._predict_detector(previous_frames)
        for next_frames in stages[1:]:
            keypoint_future = self._keypoint_executor.submit(
                self._predict_keypoints,
                previous_frames,
                previous_detections,
            )
            next_detections = self._predict_detector(next_frames)
            output.extend(keypoint_future.result())
            previous_frames = next_frames
            previous_detections = next_detections
        output.extend(
            self._predict_keypoints(previous_frames, previous_detections)
        )
        return output

    def close(self) -> None:
        if self._keypoint_executor is not None:
            self._keypoint_executor.shutdown(wait=True)
            self._keypoint_executor = None

    def provenance(self) -> dict[str, Any]:
        assets = {
            "detector_engine": self.detector_engine,
            "query_engine": self.query_engine,
            "export_manifest": self.export_manifest,
            "tensorrt_profile": self.profile_path,
        }
        try:
            import tensorrt as trt

            tensorrt_version = trt.__version__
        except ImportError:
            tensorrt_version = "unavailable"
        return {
            "model_id": self.model_id,
            "adapter": f"{type(self).__module__}:{type(self).__qualname__}",
            "device": str(self.device),
            "capabilities": sorted(self.capabilities),
            "tensorrt_version": tensorrt_version,
            "precision": "fp16",
            "assets": {
                name: {
                    "path": path.as_posix(),
                    "sha256": _sha256_file(path),
                }
                for name, path in assets.items()
            },
            "detector_batch_size": self.detector_batch_size,
            "query_batch_size": self.query_batch_size,
            "pipeline_parallel": self.pipeline_parallel,
            "pipeline_stage_size": self.pipeline_stage_size,
            "detector_input_size": self.detector_input_size,
            "query_input_size": self.query_input_size,
            "crop_scale": self.crop_scale,
        }
