from __future__ import annotations

from .official import DINOv3STABackbone


class DINOv3SpatialAdapter(DINOv3STABackbone):
    """Official DEIMv2-style DINOv3 STA adapter."""


DINOv3STA = DINOv3SpatialAdapter

__all__ = ["DINOv3STA", "DINOv3SpatialAdapter"]
