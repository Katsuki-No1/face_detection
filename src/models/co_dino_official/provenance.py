"""Pinned reference contract; changes require a new experiment identity."""
REFERENCE_COMMIT = "26653521e47157fdd819e764b4bb949057ea9f3b"
DETECTOR_CONFIG = "projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py"
SFP_SOURCE = "mmdet/models/necks/sfp.py"
ARCHITECTURE_VERSION = "codino_official_sfp_v1"
INTENTIONAL_DEVIATIONS = (
    "DINOv3 ViT-S/16 (384 channels) and Co-DETR ViT SFP replace R50/ChannelMapper",
    "300 matching queries instead of 900; one FACE class instead of COCO 80",
    "Existing project augmentation, optimizer schedule, EMA and fixed split retained",
    "Optional project ellipse branch/loss on final matching decoder queries",
    "MMDetection 3 tensor/data APIs, batch-first attention; torch checkpoint API",
    "No inference NMS, project cap 100, common validation threshold contract",
    "Insufficient auxiliary negatives get masked invalid padding, never fabricated positives",
    "Full-canvas positional/proposal/reference coordinates; geometric masks only exclude attention values/proposals",
)
