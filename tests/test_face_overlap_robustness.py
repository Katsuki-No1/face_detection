"""CPU contract tests, not accuracy evidence or production gate markers."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from keypoint_comparison1.landmark_detection.face_overlap import (
    CONTEXT_SCHEMA, crop_face_context, face_context_kwargs, file_sha256, validate_context_cache,
)
from keypoint_comparison1.landmark_detection.geometry import build_crop_transform
from keypoint_comparison1.landmark_detection.heads import LandmarkQueryHead, LandmarkQueryRefinementHead
from keypoint_comparison1.landmark_detection.model import DinoFaceLandmarkModel
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation, TopDownFaceLandmarkDataset
from keypoint_comparison1.landmark_detection.detector import DeimFaceDetector
from keypoint_comparison1.landmark_detection.config import load_landmark_config, validate_landmark_config
from src.models.dinov3_deimv2_ellipse.official import EllipsePostProcessor, DINOv3DEIMv2EllipseDetector, EllipseDEIMTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def detection(index=7):
    return dict(bbox_xyxy=[20., 25., 60., 65.], detector_bbox_xyxy=[20., 25., 60., 65.],
                ellipse=[40., 45., 20., 10., 0.3], score=0.9, crop_geometry="ellipse",
                face_query_index=index, face_query_features=[.2, -.5, 1., .7])


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("angle", [0., 37., -80.])
def test_affine_geometry_matches_transformed_contour(angle, flip):
    row = detection()
    transform = build_crop_transform([10, 10, 80, 90], output_size=65, rotation_deg=angle)
    context = crop_face_context(row, transform, flipped=flip)
    g = context["face_geometry"].numpy()
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    cx, cy, a, b, t = row["ellipse"]
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    points = np.stack([a * np.cos(phi), b * np.sin(phi)], axis=1) @ r.T + [cx, cy]
    warped = transform.original_to_crop(points) / 64
    if flip:
        warped[:, 0] = 1 - warped[:, 0]
    theta = np.arctan2(g[9], g[8]) / 2
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = (warped - g[4:6]) @ rot
    np.testing.assert_allclose(((local / g[6:8]) ** 2).sum(axis=1), 1, atol=3e-6)
    assert g[6] >= g[7] > 0
    assert context["face_query_index"].item() == 7
    assert torch.equal(context["face_query_features"], torch.tensor(row["face_query_features"]))


def head(enabled):
    return LandmarkQueryHead(8, dimension=16, decoder_layers=2, heads=4, feedforward=32,
                             face_query_dim=4 if enabled else None)


def test_disabled_contract_and_zero_initialized_warm_start():
    torch.manual_seed(42)
    legacy = head(False).eval()
    torch.manual_seed(42)
    candidate = head(True).eval()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    assert not any("face_overlap" in key for key in legacy.state_dict())
    features = torch.randn(2, 8, 4, 4)
    old = legacy(features)
    new = candidate(features, face_query_features=torch.randn(2, 4), face_geometry=torch.randn(2, 10))
    for key in ("coordinates", "visibility_logits", "query_features"):
        assert torch.equal(old[key], new[key])
    with pytest.raises(ValueError, match="requires both"):
        candidate(features)
    with pytest.raises(ValueError, match="disabled"):
        legacy(features, face_geometry=torch.zeros(2, 10))


def test_context_changes_target_queries_and_both_projections_learn():
    torch.manual_seed(42)
    model = head(True)
    memory = torch.randn(2, 8, 4, 4)
    f, g = torch.randn(2, 4), torch.randn(2, 10)
    optimizer = torch.optim.SGD(model.face_overlap.parameters(), lr=0.1)
    output = model(memory, face_query_features=f, face_geometry=g)
    loss = (output["visibility_logits"] - torch.tensor([[1.], [-1.]])).square().mean()
    loss.backward()
    for branch in (model.face_overlap.query_projection, model.face_overlap.geometry_projection):
        assert torch.isfinite(branch[-1].weight.grad).all()
        assert branch[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    after = model(memory, face_query_features=f, face_geometry=g)
    swapped = model(memory, face_query_features=f.flip(0), face_geometry=g.flip(0))
    assert not torch.equal(after["query_features"], swapped["query_features"])
    # IDs are routing metadata; arbitrary renumbering never enters the network.
    targets = [dict(face_query_features=f[i], face_geometry=g[i], face_query_index=70+i) for i in range(2)]
    wrapped = DinoFaceLandmarkModel(nn.Identity(), model, variant="query")
    kwargs = face_context_kwargs(wrapped, targets, torch.device("cpu"))
    assert set(kwargs) == {"face_query_features", "face_geometry"}


def test_refinement_wrapper_transports_context_and_missing_context_fails():
    refinement = LandmarkQueryRefinementHead(head(True), stages=[dict(radius_in_feature_cells=1,
                                                     maximum_delta=.04, samples_per_axis=3)])
    model = DinoFaceLandmarkModel(nn.Conv2d(3, 8, 1), refinement, variant="query_refinement")
    x = torch.randn(2, 3, 8, 8)
    out = model(x, face_query_features=torch.randn(2, 4), face_geometry=torch.randn(2, 10))
    assert out["decoded_coordinates"].shape == (2, 5, 2)
    with pytest.raises(ValueError, match="requires both"):
        model(x)
    with pytest.raises(ValueError, match="batch"):
        model(x, face_query_features=torch.randn(1, 4), face_geometry=torch.randn(1, 10))
    with pytest.raises(ValueError, match="finite"):
        model(x, face_query_features=torch.full((2, 4), float("nan")), face_geometry=torch.randn(2, 10))


def test_postprocess_query_order_filter_cap_and_serialization():
    processor = EllipsePostProcessor(num_classes=1, use_focal_loss=True, num_top_queries=3)
    raw = dict(pred_logits=torch.tensor([[[0.], [3.], [2.]]]),
               pred_boxes=torch.tensor([[[.2, .2, .2, .2], [.5, .5, .2, .2], [.8, .8, .2, .2]]]),
               pred_ellipses=torch.tensor([[[.2, .2, .1, .05, .3], [.5, .5, .1, .05, .3], [.8, .8, .1, .05, .3]]]))
    original = processor(raw, torch.tensor([[100., 100.]]))[0]
    raw["face_query_features"] = torch.arange(12.).reshape(1, 3, 4)
    result = processor(raw, torch.tensor([[100., 100.]]))[0]
    for key in original:
        assert torch.equal(original[key], result[key])
    assert result["face_query_index"].tolist() == [1, 2, 0]
    # Drop the top-scoring invalid ellipse, then enforce cap after filtering.
    result["ellipses"][0, 2] = -1
    detector = object.__new__(DINOv3DEIMv2EllipseDetector)
    nn.Module.__init__(detector)
    detector.score_threshold, detector.detections_per_image = .1, 1
    filtered = detector._clip_and_filter(result, (100, 100))
    assert filtered["face_query_index"].tolist() == [2]
    assert filtered["face_query_features"].tolist() == [[8., 9., 10., 11.]]
    adapter = object.__new__(DeimFaceDetector)
    adapter.crop_geometry, adapter.score_threshold, adapter.maximum_detections = "ellipse", .1, 100
    adapter.return_face_query_features = True
    rows = adapter._serialize_detections(filtered, input_hw=(100, 100), original_hw=(200, 200))
    assert rows[0]["face_query_index"] == 2
    assert rows[0]["face_query_features"] == [8., 9., 10., 11.]
    assert rows[0]["detector_bbox_xyxy"][0] == pytest.approx(140.)


def fixture_cache(tmp_path):
    Image.new("RGB", (100, 100), (100, 50, 30)).save(tmp_path / "a.png")
    coco = dict(images=[dict(id=1, file_name="a.png", width=100, height=100)],
                annotations=[dict(id=10, image_id=1, category_id=2, bbox=[20, 25, 40, 40],
                                  ellipse=[40, 45, 20, 10, .3], keypoints=[35, 40, 2]*5)],
                categories=[dict(id=2, name="face")])
    annotation = tmp_path / "train.json"
    annotation.write_text(json.dumps(coco))
    metadata = dict(crop_geometry="ellipse", face_context_schema=CONTEXT_SCHEMA,
                    annotation_sha256=file_sha256(annotation))
    cache = tmp_path / "context.json"
    cache.write_text(json.dumps(dict(metadata=metadata, predictions={"1": [detection()]})))
    return annotation, cache, metadata


def test_dataset_uses_predicted_geometry_and_flip_not_gt(tmp_path):
    annotation, cache, metadata = fixture_cache(tmp_path)
    augmentation = CropAugmentation(horizontal_flip_probability=1., scale_jitter=0.,
                                    center_jitter=0., rotation_deg=0., occlusion_probability=0.)
    dataset = TopDownFaceLandmarkDataset(annotation_path=annotation, image_root=tmp_path,
        input_size=64, training=True, augmentation=augmentation, category_ids=[2],
        detector_crop_cache=cache, face_overlap_query_dim=4)
    _, target = dataset[0]
    assert target["flipped"]
    assert target["face_query_index"].item() == 7
    original_geometry = target["face_geometry"].clone()
    dataset.samples[0]["gt_ellipse"] = [50, 50, 30, 25, 1.]
    _, changed = dataset[0]
    assert torch.equal(original_geometry, changed["face_geometry"])
    with pytest.raises(ValueError, match="crop-only synthetic occlusion"):
        TopDownFaceLandmarkDataset(annotation_path=annotation, image_root=tmp_path,
            training=True, category_ids=[2], detector_crop_cache=cache, face_overlap_query_dim=4)


def test_cache_rejects_stale_split_identity_and_invalid_features(tmp_path):
    annotation, cache, metadata = fixture_cache(tmp_path)
    kwargs = dict(annotation_path=annotation, image_ids={1}, query_dim=4)
    validate_context_cache(metadata, {1: [detection()]}, **kwargs)
    with pytest.raises(ValueError, match="provenance"):
        validate_context_cache({**metadata, "annotation_sha256": "stale"}, {1: [detection()]}, **kwargs)
    with pytest.raises(ValueError, match="provenance"):
        validate_context_cache(metadata, {1: [detection()]}, detector_identity={"detector_checkpoint_sha256": "different"}, **kwargs)
    with pytest.raises(ValueError, match="inventory"):
        validate_context_cache(metadata, {2: [detection()]}, **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        validate_context_cache(metadata, {1: [detection(), detection()]}, **kwargs)
    bad = detection()
    bad["face_query_features"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        validate_context_cache(metadata, {1: [bad]}, **kwargs)


def test_config_is_separate_head_free_and_disallows_stale_occlusion():
    root = Path(__file__).resolve().parents[1]
    config = load_landmark_config(root / "keypoint_comparison1/configs/face_overlap_robustness_v1.yaml", project_root=root)
    assert config.raw["evaluation"]["predicted_head"] is None
    assert config.raw["implementation_status"] == "implementation_only_not_scheduled"
    assert config.raw["model"]["face_overlap_robustness"]["enabled"]
    bad = copy.deepcopy(config.raw)
    bad["data"]["augmentation"]["occlusion_probability"] = .3
    with pytest.raises(ValueError, match="crop-only"):
        validate_landmark_config(bad)


def test_decoder_features_are_actual_final_ellipse_queries():
    decoder = EllipseDEIMTransformer(num_classes=1, hidden_dim=32, num_queries=5,
        feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
        num_points=[1, 1, 1], nhead=4, num_layers=1, num_denoising=0, reg_max=4, eval_idx=0)
    decoder.initialize_ellipse_head(hidden_dim=32)
    decoder.eval()
    seen = []
    hook = decoder.ellipse_head.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    features = [torch.randn(1, 32, side, side) for side in (8, 4, 2)]
    with torch.no_grad():
        old = decoder(features)
        state_keys = set(decoder.state_dict())
        decoder.return_face_query_features = True
        new = decoder(features)
    hook.remove()
    assert "face_query_features" not in old
    assert state_keys == set(decoder.state_dict())
    assert new["face_query_features"].shape == (1, 5, 32)
    assert torch.equal(new["face_query_features"], seen[-1])
    for key in old:
        assert torch.equal(old[key], new[key])


def test_empty_predictions_keep_empty_context():
    processor = EllipsePostProcessor(num_classes=1, use_focal_loss=True, num_top_queries=3)
    raw = dict(pred_logits=torch.empty(1, 0, 1), pred_boxes=torch.empty(1, 0, 4),
               pred_ellipses=torch.empty(1, 0, 5), face_query_features=torch.empty(1, 0, 4))
    result = processor(raw, torch.tensor([[100., 100.]]))[0]
    assert result["face_query_features"].shape == (0, 4)
    assert result["face_query_index"].shape == (0,)


def test_common_inference_receives_the_matching_face_context(tmp_path):
    from tools.evaluate_deim_query_baseline_aligned import infer_query_predictions
    annotation, _, _ = fixture_cache(tmp_path)
    class Spy(nn.Module):
        face_overlap_enabled = True
        def forward(self, images, *, face_query_features, face_geometry):
            assert torch.allclose(face_query_features, torch.tensor([detection()["face_query_features"]]))
            assert face_geometry.shape == (1, 10)
            return dict(decoded_coordinates=torch.full((1, 5, 2), .5), visibility_logits=torch.zeros(1, 5))
    rows = infer_query_predictions(model=Spy(), device=torch.device("cpu"),
        work_items=[{**detection(), "file_name": "a.png", "annotation_id": 10}],
        image_root=tmp_path, batch_size=2, input_size=64, crop_scale=1.25, mean=[0,0,0], std=[1,1,1])
    assert np.allclose(rows[10]["keypoints"], [[40, 45]] * 5, atol=1e-5)


def test_make_dataset_routes_train_and_val_context_cache(tmp_path):
    from keypoint_comparison1.landmark_detection.training import make_dataset
    annotation, cache, metadata = fixture_cache(tmp_path)
    detector_config, checkpoint = tmp_path / "detector.yaml", tmp_path / "detector.pt"
    detector_config.write_text("toy fixture")
    checkpoint.write_bytes(b"toy fixture, not model weights")
    payload = json.loads(cache.read_text())
    payload["metadata"].update(detector_config_sha256=file_sha256(detector_config),
                               detector_checkpoint_sha256=file_sha256(checkpoint),
                               score_threshold=.33, maximum_detections=100)
    cache.write_text(json.dumps(payload))
    config = dict(model=dict(face_overlap_robustness=dict(enabled=True, query_dim=4)),
        detector=dict(config=str(detector_config), checkpoint=str(checkpoint), score_threshold=.33, maximum_detections=100),
        data=dict(root=str(tmp_path), annotations=dict(train=str(annotation), val=str(annotation)),
                  input_size=64, category_ids=[2], face_context_caches=dict(train=str(cache), val=str(cache)),
                  augmentation=dict(occlusion_probability=0.)))
    for split, training in [("train", True), ("val", False)]:
        dataset = make_dataset(config, split=split, training=training)
        assert dataset[0][1]["face_query_index"].item() == 7
    with pytest.raises(ValueError, match="train only"):
        make_dataset(config, split="val", training=True)


def test_builder_and_legacy_checkpoint_contract():
    from keypoint_comparison1.landmark_detection.model import build_landmark_model
    cfg = {"model": {"num_landmarks": 5,
        "backbone": {"name": "dummy", "dummy": True, "patch_size": 16, "embed_dim": 32,
                     "token_key": "x_norm_patchtokens", "num_prefix_tokens": 0, "freeze": False},
        "variants": {"query": {"dimension": 32, "decoder_layers": 1, "heads": 4, "feedforward": 64},
                     "query_refinement": {"stages": [{"radius_in_feature_cells": 1., "maximum_delta": .04, "samples_per_axis": 3}]}}}}
    legacy, _ = build_landmark_model(cfg, variant="query_refinement")
    enabled = copy.deepcopy(cfg)
    enabled["model"]["face_overlap_robustness"] = dict(enabled=True, query_dim=4)
    candidate, _ = build_landmark_model(enabled, variant="query_refinement")
    missing, unexpected = candidate.load_state_dict(legacy.state_dict(), strict=False)
    assert missing and all(key.startswith("head.query_head.face_overlap.") for key in missing)
    assert not unexpected
    x = torch.randn(1, 3, 64, 64)
    legacy.eval()
    candidate.eval()
    with torch.no_grad():
        a = legacy(x)
        b = candidate(x, face_query_features=torch.randn(1, 4), face_geometry=torch.randn(1, 10))
    for key in ("decoded_coordinates", "visibility_logits"):
        assert torch.equal(a[key], b[key])
    disabled = copy.deepcopy(cfg)
    disabled["model"]["face_overlap_robustness"] = dict(enabled=False, query_dim=4)
    off, _ = build_landmark_model(disabled, variant="query_refinement")
    off.load_state_dict(legacy.state_dict(), strict=True)
