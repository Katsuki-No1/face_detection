from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnsureMosaicArrayFields:
    """Normalize scalar topdown annotation fields for MMPose Mosaic."""

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        for key in ("area", "bbox_score", "category_id"):
            if key in results:
                value = np.asarray(results[key])
                if value.ndim == 0:
                    value = value.reshape(1)
                results[key] = value
        return results


@TRANSFORMS.register_module()
class SelectSingleInstance:
    """Keep one instance after mix-image augmentation for topdown pose."""

    def __init__(self, index: int = 0) -> None:
        self.index = int(index)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if "img_shape" in results:
            shape = tuple(results["img_shape"])
            if len(shape) > 2:
                results["img_shape"] = shape[:2]
        for key in ("bbox", "bbox_score", "category_id", "keypoints", "keypoints_visible", "area"):
            if key not in results:
                continue
            value = np.asarray(results[key])
            if value.ndim == 0:
                results[key] = value.reshape(1)
                continue
            if len(value) == 0:
                results[key] = value
                continue
            index = min(max(self.index, 0), len(value) - 1)
            results[key] = value[index : index + 1]
        return results


@TRANSFORMS.register_module()
class RandomSimplePhotometric:
    """Apply lightweight image-only photometric augmentation."""

    def __init__(
        self,
        prob: float = 1.0,
        hsv_h: int = 0,
        hsv_s: int = 0,
        hsv_v: int = 0,
        brightness_delta: float = 0.0,
        contrast_delta: float = 0.0,
        gamma_delta: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.prob = float(prob)
        self.hsv_h = int(hsv_h)
        self.hsv_s = int(hsv_s)
        self.hsv_v = int(hsv_v)
        self.brightness_delta = float(brightness_delta)
        self.contrast_delta = float(contrast_delta)
        self.gamma_delta = float(gamma_delta)
        self.noise_std = float(noise_std)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return results
        image = results.get("img")
        if image is None:
            return results
        img = np.ascontiguousarray(np.asarray(image)).astype(np.float32)

        if self.brightness_delta > 0:
            img += random.uniform(-self.brightness_delta, self.brightness_delta)
        if self.contrast_delta > 0:
            img = (img - 127.5) * random.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta) + 127.5
        if self.gamma_delta > 0:
            gamma = random.uniform(1.0 - self.gamma_delta, 1.0 + self.gamma_delta)
            gamma = max(0.1, gamma)
            img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)
        if self.noise_std > 0:
            img += np.random.normal(0.0, self.noise_std, img.shape).astype(np.float32)

        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.hsv_h or self.hsv_s or self.hsv_v:
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.int16)
            if self.hsv_h:
                hsv[..., 0] = (hsv[..., 0] + random.randint(-self.hsv_h, self.hsv_h)) % 180
            if self.hsv_s:
                hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-self.hsv_s, self.hsv_s), 0, 255)
            if self.hsv_v:
                hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-self.hsv_v, self.hsv_v), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        results["img"] = np.ascontiguousarray(img)
        return results


@TRANSFORMS.register_module()
class RandomGaussianBlur:
    """Apply image-only Gaussian blur for MMPose pipelines."""

    def __init__(self, prob: float = 0.0, radius_min: float = 0.3, radius_max: float = 1.5) -> None:
        self.prob = float(prob)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return results
        image = results.get("img")
        if image is None:
            return results
        radius = random.uniform(self.radius_min, self.radius_max)
        sigma = max(0.01, float(radius))
        ksize = max(3, int(round(sigma * 4.0)) * 2 + 1)
        blurred = cv2.GaussianBlur(np.asarray(image), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        results["img"] = blurred
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob={self.prob}, radius_min={self.radius_min}, radius_max={self.radius_max})"
