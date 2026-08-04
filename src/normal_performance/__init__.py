"""Normal-operation video segment selection and inference."""

from .scoring import (
    build_window_starts,
    invalid_penalty,
    normal_band_points,
    presence_points,
    stable_percentiles,
)

__all__ = [
    "build_window_starts",
    "invalid_penalty",
    "normal_band_points",
    "presence_points",
    "stable_percentiles",
]
