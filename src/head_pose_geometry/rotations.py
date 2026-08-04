from __future__ import annotations

import torch


def _safe_normalize(
    value: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1.0e-8,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    normalized = value / norm.clamp_min(eps)
    if fallback is None:
        return normalized
    fallback = fallback.to(device=value.device, dtype=value.dtype)
    while fallback.ndim < value.ndim:
        fallback = fallback.unsqueeze(0)
    fallback = fallback.expand_as(value)
    return torch.where(norm > eps, normalized, fallback)


def rotation_6d_to_matrix(
    rotation_6d: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Convert two 3-vectors to SO(3) using stable Gram--Schmidt.

    The two input vectors occupy ``[..., 0:3]`` and ``[..., 3:6]``. The
    returned orthonormal vectors are matrix columns, matching 6DRepNet360.
    Degenerate inputs use a deterministic least-aligned canonical axis.
    """

    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected a final dimension of 6, got {tuple(rotation_6d.shape)}"
        )
    first_raw, second_raw = rotation_6d[..., :3], rotation_6d[..., 3:]
    first_fallback = first_raw.new_tensor((1.0, 0.0, 0.0))
    first = _safe_normalize(
        first_raw,
        eps=eps,
        fallback=first_fallback,
    )

    second_orthogonal = second_raw - (
        first * (first * second_raw).sum(dim=-1, keepdim=True)
    )
    least_aligned = first.abs().argmin(dim=-1)
    basis = torch.nn.functional.one_hot(least_aligned, num_classes=3).to(
        dtype=first.dtype
    )
    fallback_second = basis - (first * (first * basis).sum(dim=-1, keepdim=True))
    fallback_second = _safe_normalize(
        fallback_second,
        eps=eps,
        fallback=first.new_tensor((0.0, 1.0, 0.0)),
    )
    second = _safe_normalize(
        second_orthogonal,
        eps=eps,
        fallback=fallback_second,
    )
    third = _safe_normalize(
        torch.linalg.cross(first, second, dim=-1),
        eps=eps,
        fallback=first.new_tensor((0.0, 0.0, 1.0)),
    )
    # Recompute the second vector so accumulated numerical error cannot make
    # the returned basis left-handed or non-orthogonal.
    second = _safe_normalize(
        torch.linalg.cross(third, first, dim=-1),
        eps=eps,
        fallback=fallback_second,
    )
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix columns in 6DRepNet ordering."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrices ending in (3,3), got {rotation.shape}"
        )
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Project a near-rotation matrix to the closest proper rotation."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(matrix.shape)}")
    left, _, right_h = torch.linalg.svd(matrix)
    candidate = left @ right_h
    determinant = torch.linalg.det(candidate)
    correction = torch.ones(
        (*matrix.shape[:-2], 3),
        device=matrix.device,
        dtype=matrix.dtype,
    )
    correction[..., -1] = torch.where(
        determinant < 0.0,
        correction.new_tensor(-1.0),
        correction.new_tensor(1.0),
    )
    return left @ torch.diag_embed(correction) @ right_h


def geodesic_rotation_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """SO(3) geodesic angle in radians, stable at identity and pi."""

    if predicted.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError("Rotation tensors must end in (3,3)")
    relative = predicted.transpose(-1, -2) @ target
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
    return torch.atan2(sine, cosine.clamp(min=-1.0, max=1.0))


def geodesic_rotation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    distance = geodesic_rotation_distance(predicted, target)
    effective = torch.ones_like(distance)
    if weight is not None:
        effective = effective * weight.to(distance)
    if mask is not None:
        effective = effective * mask.to(distance)
    denominator = effective.sum().clamp_min(1.0)
    return (distance * effective).sum() / denominator


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Diagnostic Euler conversion using ``Rz @ Ry @ Rx``.

    Input order is ``[pitch_x, yaw_y, roll_z]`` in radians.
    """

    if euler.shape[-1] != 3:
        raise ValueError(f"Expected [...,3], got {tuple(euler.shape)}")
    pitch, yaw, roll = euler.unbind(-1)
    one = torch.ones_like(pitch)
    zero = torch.zeros_like(pitch)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)
    rx = torch.stack(
        (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, cp, -sp), dim=-1),
            torch.stack((zero, sp, cp), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cy, zero, sy), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((-sy, zero, cy), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cr, -sr, zero), dim=-1),
            torch.stack((sr, cr, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    return rz @ ry @ rx


def matrix_to_euler_xyz(rotation: torch.Tensor) -> torch.Tensor:
    """Diagnostic inverse of :func:`euler_xyz_to_matrix`.

    At gimbal lock, roll is set to zero and pitch absorbs the observable
    in-plane rotation, following the official 6DRepNet360 reporting code.
    """

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(rotation.shape)}")
    sy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    singular = sy < 1.0e-6
    pitch_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    yaw = torch.atan2(-rotation[..., 2, 0], sy)
    roll_regular = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    pitch_singular = torch.atan2(
        -rotation[..., 1, 2],
        rotation[..., 1, 1],
    )
    pitch = torch.where(singular, pitch_singular, pitch_regular)
    roll = torch.where(singular, torch.zeros_like(roll_regular), roll_regular)
    return torch.stack((pitch, yaw, roll), dim=-1)


def flip_rotation_horizontal(rotation: torch.Tensor) -> torch.Tensor:
    """Transform a rotation label for a semantic horizontal image flip."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(rotation.shape)}")
    reflection = torch.diag(rotation.new_tensor((-1.0, 1.0, 1.0)))
    return reflection @ rotation @ reflection


def doubled_angle_vector(theta: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(2.0 * theta), torch.sin(2.0 * theta)), dim=-1)


def doubled_angle_distance(
    predicted_vector: torch.Tensor,
    target_theta: torch.Tensor,
) -> torch.Tensor:
    predicted = torch.nn.functional.normalize(
        predicted_vector,
        dim=-1,
        eps=1.0e-8,
    )
    target = doubled_angle_vector(target_theta).to(predicted)
    return 1.0 - (predicted * target).sum(dim=-1)


__all__ = [
    "doubled_angle_distance",
    "doubled_angle_vector",
    "euler_xyz_to_matrix",
    "flip_rotation_horizontal",
    "geodesic_rotation_distance",
    "geodesic_rotation_loss",
    "matrix_to_euler_xyz",
    "matrix_to_rotation_6d",
    "project_to_so3",
    "rotation_6d_to_matrix",
]
