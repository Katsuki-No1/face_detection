"""Full-range head pose and lightweight canonical geometry extension."""

from .cache import (
    POSE_CACHE_SCHEMA_VERSION,
    PseudoPoseFilterConfig,
    PseudoPoseRecord,
    filter_pseudo_pose,
    load_pose_cache,
    pose_cache_payload,
    write_pose_cache,
)
from .crops import HeadCropAugmentation, HeadCropBatch, HeadCropBuilder
from .extension import HeadPoseGeometryExtension
from .fusion import FusionEvidence, UncertaintyGatedGeometryFusion
from .geometry import (
    CanonicalHeadGeometryAdapter,
    differentiable_covariance_ellipse,
)
from .heads import (
    CachedHeadDetector,
    CallableHeadDetector,
    ExternalHeadDetector,
    HeadDetection,
)
from .losses import GeometryLossWeights, HeadPoseGeometryLoss
from .pose_encoder import (
    FusedSixDRepNet360,
    OfficialSixDRepNet360,
    PoseEncoderAdapter,
    build_pose_encoder,
    load_fused_onnx_checkpoint,
    load_official_pytorch_checkpoint,
)
from .rotations import (
    euler_xyz_to_matrix,
    flip_rotation_horizontal,
    geodesic_rotation_distance,
    matrix_to_euler_xyz,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

__all__ = [
    "POSE_CACHE_SCHEMA_VERSION",
    "CachedHeadDetector",
    "CallableHeadDetector",
    "CanonicalHeadGeometryAdapter",
    "ExternalHeadDetector",
    "FusedSixDRepNet360",
    "FusionEvidence",
    "GeometryLossWeights",
    "HeadCropAugmentation",
    "HeadCropBatch",
    "HeadCropBuilder",
    "HeadDetection",
    "HeadPoseGeometryExtension",
    "HeadPoseGeometryLoss",
    "OfficialSixDRepNet360",
    "PoseEncoderAdapter",
    "PseudoPoseFilterConfig",
    "PseudoPoseRecord",
    "UncertaintyGatedGeometryFusion",
    "build_pose_encoder",
    "differentiable_covariance_ellipse",
    "euler_xyz_to_matrix",
    "filter_pseudo_pose",
    "flip_rotation_horizontal",
    "geodesic_rotation_distance",
    "load_fused_onnx_checkpoint",
    "load_official_pytorch_checkpoint",
    "load_pose_cache",
    "matrix_to_euler_xyz",
    "matrix_to_rotation_6d",
    "pose_cache_payload",
    "rotation_6d_to_matrix",
    "write_pose_cache",
]
