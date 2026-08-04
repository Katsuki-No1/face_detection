from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest
import torch
from torch import nn

from src.active_learning.onnx_export import (
    Deimv2EllipseOnnxWrapper,
    QueryRefinementOnnxWrapper,
    _ExportableMultiheadAttention,
    compare_pytorch_onnxruntime,
    export_onnx_model,
    validate_onnx_structure,
)
from src.active_learning.tensorrt_profile import (
    load_tensorrt_build_profile,
)
from src.active_learning.tensorrt_backend import (
    TensorRTDeimv2QueryRefinementAdapter,
)


class _TinyDetector(nn.Module):
    def forward_raw(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = images.mean(dim=(-2, -1))
        logits = pooled[:, :1].unsqueeze(1).repeat(1, 3, 1)
        boxes = pooled[:, :1].unsqueeze(1).repeat(1, 3, 4).sigmoid()
        ellipses = pooled[:, :1].unsqueeze(1).repeat(1, 3, 5).sigmoid()
        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
            "pred_ellipses": ellipses,
        }


class _TinyQuery(nn.Module):
    def forward(self, crops: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = crops.mean(dim=(-2, -1))
        coordinates = pooled[:, :2].unsqueeze(1).repeat(1, 5, 1).sigmoid()
        presence = pooled[:, :1].repeat(1, 5)
        return {
            "decoded_coordinates": coordinates,
            "visibility_logits": presence - 1.0,
            "presence_logits": presence,
        }


def test_stable_onnx_wrappers_use_expected_outputs() -> None:
    detector = Deimv2EllipseOnnxWrapper(_TinyDetector())
    detector_outputs = detector(torch.rand(2, 3, 8, 8))
    assert [tuple(value.shape) for value in detector_outputs] == [
        (2, 3, 1),
        (2, 3, 4),
        (2, 3, 5),
    ]

    query = QueryRefinementOnnxWrapper(_TinyQuery())
    coordinates, logits = query(torch.rand(2, 3, 8, 8))
    assert tuple(coordinates.shape) == (2, 5, 2)
    assert tuple(logits.shape) == (2, 5)
    assert torch.all(logits >= 0)


def test_exportable_attention_matches_pytorch_for_dynamic_batches() -> None:
    torch.manual_seed(7)
    source = nn.MultiheadAttention(
        32,
        4,
        dropout=0.0,
        batch_first=True,
    ).eval()
    exported = _ExportableMultiheadAttention(source).eval()
    for batch in (1, 3):
        query = torch.rand(batch, 5, 32)
        memory = torch.rand(batch, 17, 32)
        with torch.inference_mode():
            expected, _ = source(
                query,
                memory,
                memory,
                need_weights=False,
            )
            actual, _ = exported(
                query,
                memory,
                memory,
                need_weights=False,
            )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_dynamic_onnx_structure_and_numerical_agreement(
    tmp_path: Path,
) -> None:
    wrapper = QueryRefinementOnnxWrapper(_TinyQuery()).eval()
    path = tmp_path / "query.onnx"
    export_onnx_model(
        wrapper,
        output_path=path,
        input_name="crops",
        output_names=("decoded_coordinates", "keypoint_logits"),
        input_shape=(1, 3, 8, 8),
        opset=18,
    )
    structure = validate_onnx_structure(
        path,
        input_name="crops",
        output_names=("decoded_coordinates", "keypoint_logits"),
        fixed_input_shape=(1, 3, 8, 8),
    )
    assert structure["onnx_checker"] == "passed"
    assert structure["inputs"]["crops"] == ["batch", 3, 8, 8]
    numerical = compare_pytorch_onnxruntime(
        wrapper,
        onnx_path=path,
        input_name="crops",
        output_names=("decoded_coordinates", "keypoint_logits"),
        input_shape=(1, 3, 8, 8),
        batches=(1, 3),
    )
    assert numerical["passed"]


def test_fp16_dynamic_batch_profile_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_tensorrt_build_profile(
        root
        / "configs/tensorrt/active_learning_deimv2_l_query_refinement_fp16.yaml",
        require_onnx=False,
    )
    assert profile.precision == "fp16"
    engines = {engine.name: engine for engine in profile.engines}
    assert engines["detector"].inputs["images"].minimum == (1, 3, 960, 960)
    assert engines["detector"].inputs["images"].optimum == (2, 3, 960, 960)
    assert engines["detector"].inputs["images"].maximum == (4, 3, 960, 960)
    assert engines["detector"].mixed_precision_onnx_path.name == (
        "deimv2_l_ellipse_fp16.onnx"
    )
    assert engines["query"].inputs["crops"].minimum == (1, 3, 256, 256)
    assert engines["query"].inputs["crops"].optimum == (16, 3, 256, 256)
    assert engines["query"].inputs["crops"].maximum == (32, 3, 256, 256)


def test_invalid_dynamic_profile_is_rejected(tmp_path: Path) -> None:
    profile_path = tmp_path / "invalid.yaml"
    profile_path.write_text(
        """
schema_version: 1
project_root: .
precision: fp16
workspace_mib: 1024
engines:
  detector:
    onnx: detector.onnx
    mixed_precision_onnx: detector_fp16.onnx
    engine: detector.engine
    inputs:
      images:
        min: [4, 3, 8, 8]
        opt: [2, 3, 8, 8]
        max: [1, 3, 8, 8]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min <= opt <= max"):
        load_tensorrt_build_profile(
            profile_path,
            require_onnx=False,
        )


def test_six_video_config_pins_exact_requested_prefixes() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (
            root / "configs/active_learning_tensorrt_6videos.yaml"
        ).read_text(encoding="utf-8")
    )
    assert raw["input_roots"] == ["data/input/active_learning"]
    assert raw["output_root"] == "data/output/active_learning"
    assert raw["video_include_globs"] == [
        "MFYD-167*.mp4",
        "MIDA-700*.mp4",
        "MIMK-138*.mp4",
        "MUKD-578*.mp4",
        "OFJE-505*.mp4",
        "PRIAN-047*.mp4",
    ]
    assert raw["output"]["write_evidence_overlay"] is False
    assert raw["output"]["jpeg_quality"] == 85
    assert raw["inference"]["pipeline_batch_size"] == 32
    assert raw["inference"]["video_workers"] == 2


def test_generic_tensorrt_config_uses_canonical_io_without_video_pins() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (root / "configs/active_learning_tensorrt.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert raw["model_registry"] == (
        "configs/model_registry/active_learning_tensorrt_models.yaml"
    )
    assert raw["input_roots"] == ["data/input/active_learning"]
    assert raw["output_root"] == "data/output/active_learning"
    assert raw["video_include_globs"] == []
    assert raw["selection"]["max_frames_per_video"] == 50
    assert raw["selection"]["min_interval_sec"] == 10.0
    assert raw["output"]["write_evidence_overlay"] is False
    assert raw["inference"]["pipeline_batch_size"] == 32
    assert raw["inference"]["video_workers"] == 2


def test_tensorrt_registry_enables_validated_pipeline_parallelism() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (
            root
            / "configs/model_registry/active_learning_tensorrt_models.yaml"
        ).read_text(encoding="utf-8")
    )
    options = raw["models"][raw["active_model"]]["options"]
    assert options["pipeline_parallel"] is True
    assert options["pipeline_stage_size"] == 4


def _adapter_without_tensorrt(
    *,
    pipeline_parallel: bool,
    pipeline_stage_size: int = 2,
) -> TensorRTDeimv2QueryRefinementAdapter:
    adapter = object.__new__(TensorRTDeimv2QueryRefinementAdapter)
    adapter.pipeline_parallel = pipeline_parallel
    adapter.pipeline_stage_size = pipeline_stage_size
    adapter._keypoint_executor = None
    return adapter


def test_disabled_pipeline_parallelism_preserves_sequential_call_path() -> None:
    adapter = _adapter_without_tensorrt(pipeline_parallel=False)
    calls: list[tuple[str, list[int]]] = []

    def predict_detector(frames: list[np.ndarray]) -> list[list[int]]:
        identifiers = [int(frame[0, 0, 0]) for frame in frames]
        calls.append(("detector", identifiers))
        return [[identifier] for identifier in identifiers]

    def predict_keypoints(
        frames: list[np.ndarray],
        detections: list[list[int]],
    ) -> list[list[str]]:
        identifiers = [int(frame[0, 0, 0]) for frame in frames]
        calls.append(("keypoints", identifiers))
        assert detections == [[identifier] for identifier in identifiers]
        return [[f"result-{identifier}"] for identifier in identifiers]

    adapter._predict_detector = predict_detector
    adapter._predict_keypoints = predict_keypoints
    frames = [
        np.full((2, 2, 3), identifier, dtype=np.uint8)
        for identifier in range(4)
    ]

    assert adapter.predict_batch(frames) == [
        ["result-0"],
        ["result-1"],
        ["result-2"],
        ["result-3"],
    ]
    assert calls == [
        ("detector", [0, 1, 2, 3]),
        ("keypoints", [0, 1, 2, 3]),
    ]


def test_pipeline_parallelism_overlaps_stages_and_preserves_order() -> None:
    adapter = _adapter_without_tensorrt(pipeline_parallel=True)
    first_keypoints_started = threading.Event()
    second_detector_started = threading.Event()

    def predict_detector(frames: list[np.ndarray]) -> list[list[int]]:
        identifiers = [int(frame[0, 0, 0]) for frame in frames]
        if identifiers == [2, 3]:
            second_detector_started.set()
            assert first_keypoints_started.wait(timeout=2.0)
        return [[identifier] for identifier in identifiers]

    def predict_keypoints(
        frames: list[np.ndarray],
        detections: list[list[int]],
    ) -> list[list[str]]:
        identifiers = [int(frame[0, 0, 0]) for frame in frames]
        assert detections == [[identifier] for identifier in identifiers]
        if identifiers == [0, 1]:
            first_keypoints_started.set()
            assert second_detector_started.wait(timeout=2.0)
        return [[f"result-{identifier}"] for identifier in identifiers]

    adapter._predict_detector = predict_detector
    adapter._predict_keypoints = predict_keypoints
    frames = [
        np.full((2, 2, 3), identifier, dtype=np.uint8)
        for identifier in range(5)
    ]

    try:
        assert adapter.predict_batch(frames) == [
            ["result-0"],
            ["result-1"],
            ["result-2"],
            ["result-3"],
            ["result-4"],
        ]
    finally:
        adapter.close()
    assert adapter._keypoint_executor is None
