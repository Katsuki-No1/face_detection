from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stable_seed(value: str, base_seed: int = 0) -> int:
    digest = hashlib.sha1(f"{base_seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def l2_normalize(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norm, eps)


class MockDinoExtractor:
    def __init__(self, feature_dim: int = 64, patch_grid: tuple[int, int] = (16, 16), seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.patch_grid = patch_grid
        self.seed = seed
        self.device = "mock"
        self.model_name = "mock_dino"

    def _base_vector(self, key: str) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(key, self.seed))
        vector = rng.normal(0.0, 0.25, size=(self.feature_dim,)).astype(np.float32)
        return vector

    def crop_feature(self, image: Image.Image, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        vector = self._base_vector(str(record.get("crop_id", "")))
        if record.get("label") == "face_like":
            vector[0] += 2.0
            if record.get("crop_type") == "eye_region":
                vector[1] += 0.8
            if record.get("crop_type") == "head":
                vector[2] += 0.6
        else:
            vector[0] -= 2.0
            if record.get("crop_type") == "hard_negative":
                vector[0] += 0.8
                vector[3] += 0.5
        vector = l2_normalize(vector.astype(np.float32))
        patch_tokens = np.tile(vector.reshape(1, -1), (self.patch_grid[0] * self.patch_grid[1], 1))
        return vector.astype(np.float32), patch_tokens.astype(np.float32)

    def image_patch_tokens(self, image: Image.Image, image_record: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
        rng = np.random.default_rng(stable_seed(str(image_record.get("image_id", "")), self.seed))
        grid_h, grid_w = self.patch_grid
        tokens = rng.normal(0.0, 0.25, size=(grid_h, grid_w, self.feature_dim)).astype(np.float32)
        tokens[..., 0] -= 0.75
        width = max(1, int(image_record.get("width", image.width)))
        height = max(1, int(image_record.get("height", image.height)))
        for box in image_record.get("face_bboxes", []):
            x1, y1, x2, y2 = [float(value) for value in box]
            gx1 = max(0, min(grid_w - 1, int(np.floor(x1 / width * grid_w))))
            gy1 = max(0, min(grid_h - 1, int(np.floor(y1 / height * grid_h))))
            gx2 = max(gx1 + 1, min(grid_w, int(np.ceil(x2 / width * grid_w))))
            gy2 = max(gy1 + 1, min(grid_h, int(np.ceil(y2 / height * grid_h))))
            tokens[gy1:gy2, gx1:gx2, 0] += 2.5
        return l2_normalize(tokens, axis=-1).astype(np.float32), (grid_h, grid_w)


class DinoV2Extractor:
    def __init__(
        self,
        *,
        model_name: str,
        input_size: int,
        device: str,
        use_cls_token: bool,
        use_patch_tokens: bool,
    ) -> None:
        import torch
        from torchvision import transforms

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.torch = torch
        self.model_name = model_name
        self.input_size = int(input_size)
        self.device = device
        self.use_cls_token = use_cls_token
        self.use_patch_tokens = use_patch_tokens
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _forward(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                output = self.model.forward_features(tensor)
                cls_token = output.get("x_norm_clstoken")
                patch_tokens = output.get("x_norm_patchtokens")
                if cls_token is None:
                    cls_token = self.model(tensor)
            else:
                cls_token = self.model(tensor)
                patch_tokens = None
        cls_np = cls_token.detach().cpu().float().numpy()[0]
        if patch_tokens is None:
            patch_np = np.empty((0, cls_np.shape[-1]), dtype=np.float32)
        else:
            patch_np = patch_tokens.detach().cpu().float().numpy()[0]
        return cls_np.astype(np.float32), patch_np.astype(np.float32)

    def crop_feature(self, image: Image.Image, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        cls_token, patch_tokens = self._forward(image)
        if self.use_cls_token:
            feature = cls_token
        elif len(patch_tokens):
            feature = patch_tokens.mean(axis=0)
        else:
            feature = cls_token
        return l2_normalize(feature.astype(np.float32)), l2_normalize(patch_tokens.astype(np.float32), axis=-1)

    def image_patch_tokens(self, image: Image.Image, image_record: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
        _, patch_tokens = self._forward(image)
        if patch_tokens.size == 0:
            raise RuntimeError("DINO model did not return patch tokens.")
        grid_h, grid_w = patch_grid_from_count(patch_tokens.shape[0])
        patch_grid = patch_tokens.reshape(grid_h, grid_w, patch_tokens.shape[-1])
        return l2_normalize(patch_grid.astype(np.float32), axis=-1), (grid_h, grid_w)


def patch_grid_from_count(token_count: int) -> tuple[int, int]:
    side = int(round(token_count ** 0.5))
    if side * side == token_count:
        return side, side
    return 1, token_count


def build_extractor(config: dict[str, Any]) -> MockDinoExtractor | DinoV2Extractor:
    dino_cfg = config.get("dino", {})
    if dino_cfg.get("mock_features", False):
        return MockDinoExtractor(
            feature_dim=int(dino_cfg.get("mock_feature_dim", 64)),
            seed=int(config.get("data", {}).get("seed", 42)),
        )
    return DinoV2Extractor(
        model_name=str(dino_cfg.get("model_name", "dinov2_vits14")),
        input_size=int(dino_cfg.get("input_size", 518)),
        device=str(dino_cfg.get("device", "auto")),
        use_cls_token=bool(dino_cfg.get("use_cls_token", True)),
        use_patch_tokens=bool(dino_cfg.get("use_patch_tokens", True)),
    )


def save_feature_info(path: Path, extractor: MockDinoExtractor | DinoV2Extractor, config: dict[str, Any], count: int) -> None:
    info = {
        "model_name": extractor.model_name,
        "device": extractor.device,
        "frozen_backbone": True,
        "fine_tuned": False,
        "detection_head_trained": False,
        "keypoint_head_trained": False,
        "crop_feature_count": count,
        "config_path": config.get("_config_path"),
        "dino_config": config.get("dino", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
