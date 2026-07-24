from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .losses import LandmarkLoss, stack_targets


def attention_diagnostics(output: dict[str, Any]) -> dict[str, Any] | None:
    self_layers = list(output.get("self_attention", []))
    cross_layers = list(output.get("cross_attention", []))
    if not self_layers:
        return None
    self_matrices = []
    for weights in self_layers:
        matrix = weights.detach().float().mean(dim=(0, 1)).cpu().numpy()
        self_matrices.append(matrix.tolist())
    cross_entropy = []
    for weights in cross_layers:
        probabilities = weights.detach().float().clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        normalizer = np.log(max(2, probabilities.shape[-1]))
        cross_entropy.append((entropy.mean(dim=(0, 1)) / normalizer).cpu().tolist())
    return {
        "self_attention_layer_mean": self_matrices,
        "cross_attention_normalized_entropy": cross_entropy,
    }


def gradient_interference_matrix(
    model: torch.nn.Module,
    criterion: LandmarkLoss,
    output: dict[str, Any],
    targets: Sequence[dict[str, Any]],
    *,
    maximum_parameters: int = 2_000_000,
) -> dict[str, Any]:
    """Cosine similarity of per-landmark gradients on shared head parameters."""

    coordinates = stack_targets(targets, "keypoints", output["coordinates"].device)
    visible = stack_targets(targets, "visible", output["coordinates"].device).bool()
    candidates = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "backbone" not in name and "landmark_queries" not in name
    ]
    selected: list[torch.nn.Parameter] = []
    count = 0
    for parameter in reversed(candidates):
        if count + parameter.numel() > maximum_parameters and selected:
            continue
        selected.append(parameter)
        count += parameter.numel()
        if count >= maximum_parameters:
            break
    if not selected:
        raise RuntimeError("No shared head parameters were available for interference diagnostics")
    per_landmark = criterion.per_landmark_coordinate_losses(output["coordinates"], coordinates, visible)
    vectors: list[torch.Tensor] = []
    valid: list[bool] = []
    for index, loss in enumerate(per_landmark):
        if not bool(visible[:, index].any()):
            vectors.append(torch.zeros((count,), device=loss.device))
            valid.append(False)
            continue
        gradients = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
        flattened = [
            torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(selected, gradients, strict=True)
        ]
        vectors.append(torch.cat(flattened))
        valid.append(True)
    matrix = torch.zeros((len(vectors), len(vectors)), device=vectors[0].device)
    for left, left_vector in enumerate(vectors):
        for right, right_vector in enumerate(vectors):
            denominator = left_vector.norm() * right_vector.norm()
            matrix[left, right] = (left_vector @ right_vector) / denominator.clamp_min(1e-12)
    return {
        "matrix": matrix.detach().cpu().tolist(),
        "valid_landmarks": valid,
        "sampled_shared_parameters": count,
        "interpretation": "negative off-diagonal cosine indicates conflicting landmark gradients",
    }
