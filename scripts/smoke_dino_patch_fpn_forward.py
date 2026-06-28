from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dino_patch_fpn import DinoPatchFPNFaceKeypointModel, FakeTokenBackbone


def main() -> int:
    torch.manual_seed(0)
    patch_size = 14
    embed_dim = 32
    hidden_dim = 32
    images = torch.zeros(2, 3, 448, 448)
    fake_backbone = FakeTokenBackbone(
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_prefix_tokens=1,
        output_key="tokens",
    )
    model = DinoPatchFPNFaceKeypointModel(
        fake_backbone,
        patch_size=patch_size,
        dino_embed_dim=embed_dim,
        num_prefix_tokens=1,
        freeze_backbone=True,
        adapter_out_channels=hidden_dim,
        fpn_out_channels=hidden_dim,
        num_classes=2,
        num_keypoints=5,
        keypoint_dims=3,
        use_ellipse=False,
    )
    model.eval()

    with torch.no_grad():
        features = model.forward_features(images)
        outputs = model(images)

    token_map = features["token_map"]
    assert isinstance(token_map, torch.Tensor)
    assert tuple(token_map.shape) == (2, embed_dim, 32, 32), tuple(token_map.shape)

    fpn = features["fpn"]
    assert isinstance(fpn, dict)
    expected_feature_shapes = {
        "p2": (2, hidden_dim, 64, 64),
        "p3": (2, hidden_dim, 32, 32),
        "p4": (2, hidden_dim, 16, 16),
        "p5": (2, hidden_dim, 8, 8),
    }
    for level, expected_shape in expected_feature_shapes.items():
        assert tuple(fpn[level].shape) == expected_shape, (level, tuple(fpn[level].shape))
        prediction = outputs[level]
        height, width = expected_shape[-2:]
        assert tuple(prediction["cls_logits"].shape) == (2, 2, height, width)
        assert tuple(prediction["bbox_reg"].shape) == (2, 4, height, width)
        assert tuple(prediction["centerness"].shape) == (2, 1, height, width)
        assert tuple(prediction["keypoints"].shape) == (2, 15, height, width)

    print("DINO patch FPN smoke test passed")
    print(f"token_map: {tuple(token_map.shape)}")
    for level in ("p2", "p3", "p4", "p5"):
        print(f"{level}: feature={tuple(fpn[level].shape)} cls={tuple(outputs[level]['cls_logits'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
