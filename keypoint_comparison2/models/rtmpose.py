from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .simcc import SimCCLabel, SimCCLoss, SimCCSpec


FACE106_TO_FACE5 = (
    tuple(range(66, 75)),  # left eye contour -> eye centre
    tuple(range(75, 84)),  # right eye contour -> eye centre
    (54,),                 # nose tip
    (84,),                 # left mouth corner
    (90,),                 # right mouth corner
)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "ema", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported RTMPose checkpoint: {path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = str(key)
        for prefix in ("module.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        state[clean] = value
    return state


def _semantic_face5_tensor(source: torch.Tensor) -> torch.Tensor:
    if source.shape[0] != 106:
        raise ValueError(f"Expected a 106-landmark RTMPose tensor, got {tuple(source.shape)}")
    return torch.stack([source[list(indices)].mean(0) for indices in FACE106_TO_FACE5], dim=0)


class RTMPoseFace5(nn.Module):
    """Official CSPNeXt-M + RTMCC, semantically initialized from Face6/106 weights."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        try:
            import mmdet  # noqa: F401
            import mmpose  # noqa: F401
            from mmdet.registry import MODELS as MMDET_MODELS
            from mmpose.registry import MODELS as MMPOSE_MODELS
        except ImportError as error:
            raise RuntimeError(
                "RTMPose-Face-5 requires the production OpenMMLab stack: "
                "mmpose, mmdet, mmcv and mmengine."
            ) from error

        input_size = int(cfg.get("input_size", 256))
        backbone_cfg = dict(
            type="CSPNeXt",
            arch="P5",
            expand_ratio=float(cfg.get("expand_ratio", 0.5)),
            deepen_factor=float(cfg.get("deepen_factor", 0.67)),
            widen_factor=float(cfg.get("widen_factor", 0.75)),
            out_indices=(4,),
            channel_attention=True,
            norm_cfg=dict(type=str(cfg.get("norm", "BN"))),
            act_cfg=dict(type="SiLU"),
        )
        self.backbone = MMDET_MODELS.build(backbone_cfg)
        in_channels = int(cfg.get("in_channels", 768))
        head_cfg = dict(cfg.get("head", {}))
        codec = dict(
            type="SimCCLabel",
            input_size=(input_size, input_size),
            sigma=tuple(head_cfg.get("sigma", (5.66, 5.66))),
            simcc_split_ratio=float(head_cfg.get("simcc_split_ratio", 2.0)),
            normalize=False,
            use_dark=False,
        )
        self.head = MMPOSE_MODELS.build(
            dict(
                type="RTMCCHead",
                in_channels=in_channels,
                out_channels=5,
                input_size=(input_size, input_size),
                in_featuremap_size=(input_size // 32, input_size // 32),
                simcc_split_ratio=codec["simcc_split_ratio"],
                final_layer_kernel_size=int(head_cfg.get("final_kernel_size", 7)),
                gau_cfg=dict(
                    hidden_dims=int(head_cfg.get("hidden_dimension", 256)),
                    s=int(head_cfg.get("gau_attention_dimension", 128)),
                    expansion_factor=float(head_cfg.get("gau_expansion_factor", 2.0)),
                    dropout_rate=float(head_cfg.get("dropout", 0.0)),
                    drop_path=float(head_cfg.get("drop_path", 0.0)),
                    act_fn="SiLU",
                    use_rel_bias=False,
                    pos_enc=False,
                ),
                loss=dict(
                    type="KLDiscretLoss",
                    use_target_weight=True,
                    beta=float(head_cfg.get("kl_beta", 10.0)),
                    label_softmax=True,
                ),
                decoder=codec,
            )
        )
        self.visibility = nn.Linear(in_channels, 5)
        self.spec = SimCCSpec(
            input_size=(input_size, input_size),
            split_ratio=codec["simcc_split_ratio"],
            sigma=codec["sigma"],
        )
        self.codec = SimCCLabel(self.spec)
        self.loss_module = SimCCLoss(
            self.codec,
            beta=float(head_cfg.get("kl_beta", 10.0)),
            visibility_weight=float(head_cfg.get("visibility_loss_weight", 0.2)),
        )
        self.register_buffer(
            "mean",
            torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.pretrained_report = self.load_pretrained(cfg)

    def load_pretrained(self, config: Mapping[str, Any]) -> dict[str, Any]:
        five_path = config.get("five_keypoint_checkpoint")
        face106_path = config.get("weights")
        selected = five_path or face106_path
        if not selected or not Path(str(selected)).is_file():
            if bool(config.get("require_pretrained", True)):
                raise FileNotFoundError(
                    "RTMPose Face6/106 pretrained checkpoint is required. "
                    "Run `python keypoint_comparison2/prepare_weights.py --rtmpose`."
                )
            return {"loaded": 0, "source": None}
        path = Path(str(selected))
        source = _checkpoint_state(path)
        target = self.state_dict()
        loadable: dict[str, torch.Tensor] = {}
        remapped: list[str] = []
        skipped: list[str] = []
        for key, value in source.items():
            candidate = key
            if candidate in target and tuple(target[candidate].shape) == tuple(value.shape):
                loadable[candidate] = value
                continue
            if not five_path and candidate in {"head.final_layer.weight", "head.final_layer.bias"}:
                remapped_value = _semantic_face5_tensor(value)
                if candidate in target and tuple(target[candidate].shape) == tuple(remapped_value.shape):
                    loadable[candidate] = remapped_value
                    remapped.append(candidate)
                    continue
            skipped.append(candidate)
        missing, unexpected = self.load_state_dict(loadable, strict=False)
        backbone_keys = [key for key in target if key.startswith("backbone.")]
        loaded_backbone = sum(key in loadable for key in backbone_keys)
        if bool(config.get("require_pretrained", True)) and loaded_backbone < int(0.95 * len(backbone_keys)):
            raise RuntimeError(
                f"RTMPose pretrained load covered only {loaded_backbone}/{len(backbone_keys)} backbone tensors"
            )
        return {
            "source": str(path),
            "loaded": len(loadable),
            "remapped": remapped,
            "skipped": skipped,
            "missing": list(missing),
            "unexpected": list(unexpected),
            "loaded_backbone": loaded_backbone,
            "total_backbone": len(backbone_keys),
        }

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        # OpenMMLab means/stds are in the 0..255 image domain.
        return (images * 255.0 - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)

    def forward(
        self,
        images: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(self._normalize(images))
        if isinstance(features, torch.Tensor):
            features = (features,)
        if not isinstance(features, Sequence):
            raise TypeError(f"CSPNeXt must return a feature sequence, got {type(features)!r}")
        logits_x, logits_y = self.head.forward(tuple(features))
        coordinates, confidence = self.codec.decode(logits_x, logits_y)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        output = {
            "simcc_x": logits_x,
            "simcc_y": logits_y,
            "coordinates": coordinates,
            "simcc_confidence": confidence,
            "visibility_logits": self.visibility(pooled),
        }
        if targets is not None:
            output.update(
                self.loss_module(
                    output,
                    targets["coordinates"],
                    targets["visible"],
                    targets.get("labeled"),
                )
            )
        return output
