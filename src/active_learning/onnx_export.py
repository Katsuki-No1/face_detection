from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


DETECTOR_INPUT_NAME = "images"
DETECTOR_OUTPUT_NAMES = ("pred_logits", "pred_boxes", "pred_ellipses")
QUERY_INPUT_NAME = "crops"
QUERY_OUTPUT_NAMES = ("decoded_coordinates", "keypoint_logits")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Deimv2EllipseOnnxWrapper(nn.Module):
    """Stable ONNX boundary around the raw DEIMv2-L ellipse predictions."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.model.forward_raw(images)
        return (
            prediction["pred_logits"],
            prediction["pred_boxes"],
            prediction["pred_ellipses"],
        )


class QueryRefinementOnnxWrapper(nn.Module):
    """Stable ONNX boundary around Query Refinement landmark predictions."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        crops: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(crops)
        logits = prediction.get(
            "presence_logits",
            prediction["visibility_logits"],
        )
        return prediction["decoded_coordinates"], logits


class _FixedRopePositionEmbedding(nn.Module):
    def __init__(self, sin: torch.Tensor, cos: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("sin", sin.detach().clone(), persistent=False)
        self.register_buffer("cos", cos.detach().clone(), persistent=False)

    def forward(
        self,
        *,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del H, W
        return self.sin, self.cos


class _ExportableMultiheadAttention(nn.Module):
    """ONNX-friendly equivalent for batch-first self/cross attention."""

    def __init__(self, source: nn.MultiheadAttention) -> None:
        super().__init__()
        if not source.batch_first or not source._qkv_same_embed_dim:
            raise ValueError(
                "ONNX attention replacement requires batch_first QKV "
                "with a common embedding dimension"
            )
        self.embed_dim = int(source.embed_dim)
        self.num_heads = int(source.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(source.dropout)
        self.in_proj_weight = source.in_proj_weight
        self.in_proj_bias = source.in_proj_bias
        self.out_proj = source.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch.nn.functional as functional

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = functional.linear(query, q_weight, q_bias)
        k = functional.linear(key, k_weight, k_bias)
        v = functional.linear(value, v_weight, v_bias)
        batch = q.shape[0]
        query_length = q.shape[1]
        key_length = k.shape[1]
        q = q.reshape(
            batch,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.reshape(
            batch,
            key_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.reshape(
            batch,
            key_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        attention = torch.matmul(
            q * (self.head_dim**-0.5),
            k.transpose(-2, -1),
        ).softmax(dim=-1)
        if self.training and self.dropout:
            attention = functional.dropout(
                attention,
                p=self.dropout,
                training=True,
            )
        result = torch.matmul(attention, v)
        result = result.transpose(1, 2).reshape(
            batch,
            query_length,
            self.embed_dim,
        )
        result = self.out_proj(result)
        if not need_weights:
            return result, None
        weights = (
            attention.mean(dim=1)
            if average_attn_weights
            else attention
        )
        return result, weights


def _fixed_patch_grid(model: nn.Module, input_size: int) -> tuple[int, int]:
    patch_sizes: set[tuple[int, int]] = set()
    for module in model.modules():
        patch_size = getattr(module, "patch_size", None)
        if isinstance(patch_size, int):
            patch_sizes.add((patch_size, patch_size))
        elif (
            isinstance(patch_size, (tuple, list))
            and len(patch_size) == 2
        ):
            patch_sizes.add((int(patch_size[0]), int(patch_size[1])))
    if not patch_sizes:
        raise RuntimeError("Could not find the DINO patch size for ONNX export")
    if len(patch_sizes) != 1:
        raise RuntimeError(
            f"Ambiguous DINO patch sizes for ONNX export: {patch_sizes}"
        )
    patch_h, patch_w = next(iter(patch_sizes))
    if input_size % patch_h or input_size % patch_w:
        raise ValueError(
            f"input_size={input_size} is not divisible by "
            f"patch_size={(patch_h, patch_w)}"
        )
    return input_size // patch_h, input_size // patch_w


def _prepare_model_for_onnx(model: nn.Module, input_size: int) -> None:
    grid_h, grid_w = _fixed_patch_grid(model, input_size)
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if type(child).__name__ == "RopePositionEmbedding":
                with torch.inference_mode():
                    sin, cos = child(H=grid_h, W=grid_w)
                replacements.append(
                    (
                        parent,
                        child_name,
                        _FixedRopePositionEmbedding(sin, cos),
                    )
                )
            elif isinstance(child, nn.MultiheadAttention):
                replacements.append(
                    (
                        parent,
                        child_name,
                        _ExportableMultiheadAttention(child),
                    )
                )
    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)


@dataclass(frozen=True, slots=True)
class ExportedModel:
    name: str
    path: Path
    input_name: str
    output_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    source_assets: dict[str, Path]


def load_detector_wrapper(
    *,
    detector_config: Path,
    detector_checkpoint: Path,
    input_size: int,
) -> Deimv2EllipseOnnxWrapper:
    from keypoint_comparison1.landmark_detection.detector import (
        DeimFaceDetector,
    )

    detector = DeimFaceDetector(
        config_path=detector_config,
        checkpoint_path=detector_checkpoint,
        device="cpu",
        image_size=input_size,
        score_threshold=0.001,
        maximum_detections=100,
        crop_geometry="ellipse",
    )
    _prepare_model_for_onnx(detector.model, input_size)
    return Deimv2EllipseOnnxWrapper(detector.model.eval())


def load_query_wrapper(
    *,
    query_checkpoint: Path,
    input_size: int,
) -> tuple[QueryRefinementOnnxWrapper, dict[str, Any]]:
    from keypoint_comparison1.landmark_detection.model import (
        build_landmark_model,
    )

    checkpoint = torch.load(
        query_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("config"),
        dict,
    ):
        raise RuntimeError(
            f"Query checkpoint has no embedded immutable config: "
            f"{query_checkpoint}"
        )
    raw_config = dict(checkpoint["config"])
    embedded_size = int(raw_config["data"]["input_size"])
    if embedded_size != input_size:
        raise RuntimeError(
            f"Query input_size mismatch: requested={input_size}, "
            f"checkpoint={embedded_size}"
        )
    model, _ = build_landmark_model(
        raw_config,
        variant="query_refinement",
        initialize_from_detector=False,
    )
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(
            f"Query checkpoint has no EMA/model state: {query_checkpoint}"
        )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Query checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model = model.cpu().eval()
    _prepare_model_for_onnx(model, input_size)
    del checkpoint, state
    return QueryRefinementOnnxWrapper(model), raw_config


def export_onnx_model(
    wrapper: nn.Module,
    *,
    output_path: Path,
    input_name: str,
    output_names: Sequence[str],
    input_shape: Sequence[int],
    opset: int = 18,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(tuple(int(value) for value in input_shape))
    dynamic_axes = {
        input_name: {0: "batch"},
        **{name: {0: "batch"} for name in output_names},
    }
    # The legacy exporter cannot lower PyTorch's inference-only fused
    # _native_multi_head_attention operator. Disabling this export-time
    # fastpath expands the same evaluation graph into standard ONNX ops.
    mha_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper.cpu().eval(),
                (dummy,),
                output_path.as_posix(),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                dynamo=False,
                external_data=True,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath)


def _tensor_shape(value_info: Any) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append(str(dimension.dim_param))
        elif dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append(None)
    return dimensions


def validate_onnx_structure(
    path: Path,
    *,
    input_name: str,
    output_names: Sequence[str],
    fixed_input_shape: Sequence[int],
) -> dict[str, Any]:
    import onnx

    model = onnx.load(path.as_posix(), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    graph_inputs = {
        value.name: _tensor_shape(value) for value in inferred.graph.input
    }
    graph_outputs = {
        value.name: _tensor_shape(value) for value in inferred.graph.output
    }
    if set(graph_inputs) != {input_name}:
        raise RuntimeError(
            f"Unexpected ONNX inputs for {path}: {sorted(graph_inputs)}"
        )
    if tuple(graph_outputs) != tuple(output_names):
        raise RuntimeError(
            f"Unexpected ONNX outputs for {path}: "
            f"{tuple(graph_outputs)} != {tuple(output_names)}"
        )
    actual_input_shape = graph_inputs[input_name]
    if not actual_input_shape or actual_input_shape[0] != "batch":
        raise RuntimeError(
            f"ONNX batch axis is not dynamic: {actual_input_shape}"
        )
    expected_tail = [int(value) for value in fixed_input_shape[1:]]
    if actual_input_shape[1:] != expected_tail:
        raise RuntimeError(
            f"ONNX input shape mismatch: {actual_input_shape[1:]} "
            f"!= {expected_tail}"
        )
    if any(
        not shape or shape[0] != "batch"
        for shape in graph_outputs.values()
    ):
        raise RuntimeError(
            f"One or more ONNX outputs lack a dynamic batch axis: "
            f"{graph_outputs}"
        )
    domains: dict[str, int] = {}
    operators: dict[str, int] = {}
    for node in model.graph.node:
        domain = node.domain or "ai.onnx"
        domains[domain] = domains.get(domain, 0) + 1
        key = f"{domain}::{node.op_type}"
        operators[key] = operators.get(key, 0) + 1
    unsupported_domains = sorted(
        domain
        for domain in domains
        if domain not in {"ai.onnx"}
    )
    if unsupported_domains:
        raise RuntimeError(
            f"ONNX contains custom operator domains: {unsupported_domains}"
        )
    external_initializers = sum(
        int(initializer.data_location == onnx.TensorProto.EXTERNAL)
        for initializer in model.graph.initializer
    )
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "ir_version": int(model.ir_version),
        "opset_imports": {
            (item.domain or "ai.onnx"): int(item.version)
            for item in model.opset_import
        },
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "external_initializers": external_initializers,
        "operator_domains": domains,
        "operators": dict(sorted(operators.items())),
        "onnx_checker": "passed",
        "shape_inference": "passed",
        "custom_operator_domains": [],
    }


def compare_pytorch_onnxruntime(
    wrapper: nn.Module,
    *,
    onnx_path: Path,
    input_name: str,
    output_names: Sequence[str],
    input_shape: Sequence[int],
    batches: Sequence[int] = (1, 2),
    seed: int = 20260731,
    atol: float = 2e-4,
    rtol: float = 2e-3,
    align_query_axis: bool = False,
    minimum_close_fraction: float = 1.0,
    sparse_mismatch_mean_abs_limit: float = 0.0,
) -> dict[str, Any]:
    import onnxruntime as ort

    wrapper = wrapper.cpu().eval()
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    results: list[dict[str, Any]] = []
    passed = True
    for batch in batches:
        shape = (int(batch), *[int(value) for value in input_shape[1:]])
        input_tensor = torch.rand(shape, generator=generator)
        with torch.inference_mode():
            torch_values = wrapper(input_tensor)
        if isinstance(torch_values, torch.Tensor):
            torch_values = (torch_values,)
        reference = [
            value.detach().cpu().float().numpy()
            for value in torch_values
        ]
        actual = session.run(
            list(output_names),
            {input_name: input_tensor.numpy()},
        )
        if align_query_axis:
            from scipy.optimize import linear_sum_assignment

            if "pred_boxes" not in output_names:
                raise ValueError(
                    "align_query_axis requires a pred_boxes output"
                )
            boxes_index = tuple(output_names).index("pred_boxes")
            expected_boxes = reference[boxes_index]
            observed_boxes = actual[boxes_index]
            assignments: list[tuple[np.ndarray, np.ndarray]] = []
            for item_index in range(expected_boxes.shape[0]):
                cost = np.abs(
                    expected_boxes[item_index, :, None, :]
                    - observed_boxes[item_index, None, :, :]
                ).mean(axis=-1)
                assignments.append(linear_sum_assignment(cost))
            aligned: list[np.ndarray] = []
            for observed in actual:
                value = np.empty_like(observed)
                for item_index, (rows, columns) in enumerate(assignments):
                    value[item_index, rows] = observed[item_index, columns]
                aligned.append(value)
            actual = aligned
        outputs: list[dict[str, Any]] = []
        for name, expected, observed in zip(
            output_names,
            reference,
            actual,
            strict=True,
        ):
            if expected.shape != observed.shape:
                output_passed = False
                maximum_absolute_error = None
                mean_absolute_error = None
                maximum_relative_error = None
            else:
                delta = np.abs(expected - observed)
                maximum_absolute_error = float(delta.max(initial=0.0))
                mean_absolute_error = float(delta.mean())
                denominator = np.maximum(np.abs(expected), 1e-6)
                maximum_relative_error = float(
                    (delta / denominator).max(initial=0.0)
                )
                close = np.isclose(
                    expected,
                    observed,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=False,
                )
                close_fraction = float(close.mean())
                strict_passed = bool(close.all())
                sparse_passed = bool(
                    minimum_close_fraction < 1.0
                    and close_fraction >= minimum_close_fraction
                    and mean_absolute_error
                    <= sparse_mismatch_mean_abs_limit
                )
                output_passed = strict_passed or sparse_passed
            passed = passed and output_passed
            outputs.append(
                {
                    "name": name,
                    "pytorch_shape": list(expected.shape),
                    "onnxruntime_shape": list(observed.shape),
                    "max_abs_error": maximum_absolute_error,
                    "mean_abs_error": mean_absolute_error,
                    "max_relative_error": maximum_relative_error,
                    "close_fraction": (
                        close_fraction
                        if expected.shape == observed.shape
                        else 0.0
                    ),
                    "strict_allclose": (
                        strict_passed
                        if expected.shape == observed.shape
                        else False
                    ),
                    "passed": output_passed,
                }
            )
        results.append(
            {
                "batch": int(batch),
                "input_shape": list(shape),
                "outputs": outputs,
            }
        )
    return {
        "backend": "onnxruntime-cpu",
        "reference": "pytorch-cpu",
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "query_axis_aligned": align_query_axis,
        "minimum_close_fraction": minimum_close_fraction,
        "sparse_mismatch_mean_abs_limit": sparse_mismatch_mean_abs_limit,
        "batches": results,
        "passed": passed,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
