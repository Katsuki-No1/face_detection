from __future__ import annotations

from typing import Any, Sequence

import torch
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models.dense_heads import DINOHead
from mmdet.models.layers import (
    CdnQueryGenerator,
    DeformableDetrTransformerEncoder,
    DinoTransformerDecoder,
    SinePositionalEncoding,
)
from mmdet.utils import register_all_modules
from torch import nn
from torch.nn.init import normal_


EXPECTED_QUERY_HEAD_PARAMETERS = 4_541_844
EXPECTED_ENCODER_DOWNSAMPLE_PARAMETERS = 590_592
EXPECTED_COMPACT_CODINO_PARAMETERS = 5_132_436


def _valid_ratio(mask: torch.Tensor) -> torch.Tensor:
    """Return valid width/height ratios for a right/bottom padding mask."""

    height, width = mask.shape[-2:]
    valid_height = (~mask[:, :, 0]).sum(1)
    valid_width = (~mask[:, 0, :]).sum(1)
    return torch.stack(
        (valid_width.to(torch.float32) / width, valid_height.to(torch.float32) / height),
        dim=-1,
    )


class CompactCoDINO(nn.Module):
    """Compact one-class Co-DINO query path on current MMDetection components.

    The transformer and prediction path is the DINO query head used by Co-DINO.
    ``encoded_p6`` is retained for checkpoint/downstream compatibility, but is
    deliberately excluded from decoder inputs.
    """

    embed_dims = 256
    num_feature_levels = 3
    num_queries = 100
    num_decoder_layers = 3

    def __init__(self) -> None:
        super().__init__()
        register_all_modules(init_default_scope=False)

        self.positional_encoding = SinePositionalEncoding(
            num_feats=128,
            temperature=20,
            normalize=True,
            offset=0.0,
        )
        self.encoder = DeformableDetrTransformerEncoder(
            num_layers=1,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=self.embed_dims,
                    num_heads=8,
                    num_levels=self.num_feature_levels,
                    num_points=4,
                    dropout=0.0,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=self.embed_dims,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
        )
        self.decoder = DinoTransformerDecoder(
            num_layers=self.num_decoder_layers,
            return_intermediate=True,
            post_norm_cfg=None,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=self.embed_dims,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=self.embed_dims,
                    num_heads=8,
                    num_levels=self.num_feature_levels,
                    num_points=4,
                    dropout=0.0,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=self.embed_dims,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type="ReLU", inplace=True),
                ),
            ),
        )
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        self.level_embed = nn.Parameter(torch.empty(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        self.bbox_head = DINOHead(
            num_classes=1,
            embed_dims=self.embed_dims,
            num_reg_fcs=2,
            sync_cls_avg_factor=True,
            share_pred_layer=False,
            num_pred_layer=self.num_decoder_layers + 1,
            as_two_stage=True,
            loss_cls=dict(
                type="QualityFocalLoss",
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner",
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                        dict(type="IoUCost", iou_mode="giou", weight=2.0),
                    ],
                )
            ),
            test_cfg=dict(max_per_img=self.num_queries),
        )
        self.dn_query_generator = CdnQueryGenerator(
            num_classes=1,
            embed_dims=self.embed_dims,
            num_matching_queries=self.num_queries,
            label_noise_scale=0.5,
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_dn_queries=50),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(32, self.embed_dims),
        )

        self._init_weights()
        query_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("downsample.")
        )
        downsample_count = sum(parameter.numel() for parameter in self.downsample.parameters())
        total_count = sum(parameter.numel() for parameter in self.parameters())
        expected = (
            (EXPECTED_QUERY_HEAD_PARAMETERS, query_count, "query head"),
            (EXPECTED_ENCODER_DOWNSAMPLE_PARAMETERS, downsample_count, "encoded P6 downsample"),
            (EXPECTED_COMPACT_CODINO_PARAMETERS, total_count, "compact Co-DINO"),
        )
        for expected_count, actual_count, component in expected:
            if actual_count != expected_count:
                raise RuntimeError(
                    f"Unexpected {component} parameter count: "
                    f"expected={expected_count}, actual={actual_count}"
                )

    def _init_weights(self) -> None:
        for coder in (self.encoder, self.decoder):
            for parameter in coder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, MultiScaleDeformableAttention):
                module.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.zeros_(self.memory_trans_fc.bias)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)
        self.bbox_head.init_weights()

    @staticmethod
    def _validate_inputs(
        features: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
    ) -> None:
        if len(features) != 3 or len(masks) != 3:
            raise ValueError("Compact Co-DINO requires P3/P4/P5 and exactly three masks")
        batch_size = features[0].shape[0]
        for level, (feature, mask) in enumerate(zip(features, masks, strict=True), start=3):
            if feature.ndim != 4 or feature.shape[:2] != (batch_size, 256):
                raise ValueError(f"P{level} must have shape [B,256,H,W], got {tuple(feature.shape)}")
            if mask.shape != (batch_size, *feature.shape[-2:]) or mask.dtype != torch.bool:
                raise ValueError(
                    f"P{level} mask must be bool [B,H,W], got {tuple(mask.shape)} {mask.dtype}"
                )

    def _flatten_features(
        self,
        features: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flattened_features = []
        flattened_positions = []
        flattened_masks = []
        spatial_shapes: list[tuple[int, int]] = []
        for level, (feature, mask) in enumerate(zip(features, masks, strict=True)):
            batch_size, channels, height, width = feature.shape
            position = self.positional_encoding(mask)
            flattened_features.append(feature.flatten(2).transpose(1, 2))
            flattened_positions.append(
                position.flatten(2).transpose(1, 2)
                + self.level_embed[level].view(1, 1, self.embed_dims)
            )
            flattened_masks.append(mask.flatten(1))
            spatial_shapes.append((height, width))

        spatial_shapes_tensor = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=features[0].device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios = torch.stack([_valid_ratio(mask) for mask in masks], dim=1)
        return (
            torch.cat(flattened_features, dim=1),
            torch.cat(flattened_positions, dim=1),
            torch.cat(flattened_masks, dim=1),
            spatial_shapes_tensor,
            level_start_index,
            valid_ratios,
        )

    def _encoder_proposals(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.shape[0]
        proposals = []
        cursor = 0
        for level, (height, width) in enumerate(spatial_shapes):
            count = height * width
            level_mask = memory_mask[:, cursor : cursor + count].view(
                batch_size, height, width, 1
            )
            valid_height = (~level_mask[:, :, 0, 0]).sum(1).unsqueeze(-1)
            valid_width = (~level_mask[:, 0, :, 0]).sum(1).unsqueeze(-1)
            scale = torch.cat((valid_width, valid_height), dim=1).view(batch_size, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=memory.device),
                torch.arange(width, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            grid = (grid.unsqueeze(0) + 0.5) / scale
            box_size = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposals.append(torch.cat((grid, box_size), dim=-1).flatten(1, 2))
            cursor += count

        output_proposals = torch.cat(proposals, dim=1)
        proposal_valid = (
            ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(-1, keepdim=True)
            == 4
        )
        output_proposals = torch.log(output_proposals / (1.0 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_mask.unsqueeze(-1) | ~proposal_valid,
            float("inf"),
        )
        output_memory = memory.masked_fill(
            memory_mask.unsqueeze(-1) | ~proposal_valid,
            0.0,
        )
        return self.memory_trans_norm(self.memory_trans_fc(output_memory)), output_proposals

    def forward(
        self,
        features: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        *,
        batch_data_samples: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(features, masks)
        (
            flattened_features,
            flattened_positions,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self._flatten_features(features, masks)
        memory = self.encoder(
            query=flattened_features,
            query_pos=flattened_positions,
            key_padding_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        output_memory, output_proposals = self._encoder_proposals(
            memory,
            memory_mask,
            [tuple(feature.shape[-2:]) for feature in features],
        )
        encoder_branch = self.num_decoder_layers
        encoder_logits = self.bbox_head.cls_branches[encoder_branch](output_memory)
        encoder_boxes_unactivated = (
            self.bbox_head.reg_branches[encoder_branch](output_memory) + output_proposals
        )
        if encoder_logits.shape[1] < self.num_queries:
            raise ValueError(
                f"Feature maps provide only {encoder_logits.shape[1]} proposals; "
                f"at least {self.num_queries} are required"
            )
        selectable_logits = encoder_logits.masked_fill(
            ~torch.isfinite(output_proposals).all(dim=-1, keepdim=True),
            float("-inf"),
        )
        topk_indices = torch.topk(
            selectable_logits.max(dim=-1).values,
            k=self.num_queries,
            dim=1,
        ).indices
        topk_logits = torch.gather(
            selectable_logits,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, encoder_logits.shape[-1]),
        )
        topk_boxes_unactivated = torch.gather(
            encoder_boxes_unactivated,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        topk_boxes = topk_boxes_unactivated.sigmoid()

        batch_size = memory.shape[0]
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if self.training:
            if batch_data_samples is None:
                raise ValueError("Training forward requires batch_data_samples for CDN")
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat((dn_label_query, query), dim=1)
            reference_points = torch.cat(
                (dn_bbox_query, topk_boxes_unactivated.detach()),
                dim=1,
            ).sigmoid()
        else:
            dn_mask = None
            dn_meta = None
            reference_points = topk_boxes_unactivated.detach().sigmoid()

        hidden_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
        )
        references = list(references)
        if query.shape[1] == self.num_queries:
            hidden_states[0] = (
                hidden_states[0] + self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )
        all_logits, all_boxes = self.bbox_head(hidden_states, references)
        if dn_meta is None:
            matching_logits, matching_boxes = all_logits, all_boxes
            dn_logits = dn_boxes = None
        else:
            (
                matching_logits,
                matching_boxes,
                dn_logits,
                dn_boxes,
            ) = self.bbox_head.split_outputs(all_logits, all_boxes, dn_meta)

        p5_height, p5_width = features[-1].shape[-2:]
        p5_start = int(sum(feature.shape[-2] * feature.shape[-1] for feature in features[:-1]))
        encoded_p5 = memory[:, p5_start:].transpose(1, 2).reshape(
            batch_size, self.embed_dims, p5_height, p5_width
        )
        encoded_p6 = self.downsample(encoded_p5)
        # Keep this compatibility branch connected for DDP even though it must
        # not affect detector predictions.
        zero_dependency = encoded_p6.sum() * 0.0
        matching_logits = matching_logits + zero_dependency

        result: dict[str, Any] = {
            "pred_logits": matching_logits[-1],
            "pred_boxes": matching_boxes[-1],
            "aux_outputs": [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(
                    matching_logits[:-1], matching_boxes[:-1], strict=True
                )
            ],
            "enc_outputs": {
                "pred_logits": topk_logits,
                "pred_boxes": topk_boxes,
            },
            "encoded_p6": encoded_p6,
            "_all_logits": all_logits + zero_dependency,
            "_all_boxes": all_boxes,
            "_hidden_states": hidden_states,
            "_references": references,
            "_enc_logits": topk_logits,
            "_enc_boxes": topk_boxes,
        }
        if self.training:
            result["dn_outputs"] = {
                "pred_logits": dn_logits,
                "pred_boxes": dn_boxes,
            }
            result["dn_meta"] = dn_meta
            result["_dn_mask"] = dn_mask
        return result

    def loss(
        self,
        raw_outputs: dict[str, Any],
        batch_data_samples: Sequence[Any],
    ) -> dict[str, torch.Tensor]:
        batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        # MMDetection 3.3's QFL empty-target path mixes fp32 target scores with
        # fp16 IoU tensors. Compute all loss inputs in fp32 under AMP.
        losses = self.bbox_head.loss_by_feat(
            raw_outputs["_all_logits"].float(),
            raw_outputs["_all_boxes"].float(),
            raw_outputs["_enc_logits"].float(),
            raw_outputs["_enc_boxes"].float(),
            batch_gt_instances,
            batch_img_metas,
            raw_outputs.get("dn_meta"),
        )
        named_losses = {
            name.replace("loss_iou", "loss_giou"): value for name, value in losses.items()
        }
        named_losses["loss_total"] = sum(named_losses.values())
        return named_losses


__all__ = [
    "CompactCoDINO",
    "EXPECTED_COMPACT_CODINO_PARAMETERS",
    "EXPECTED_ENCODER_DOWNSAMPLE_PARAMETERS",
    "EXPECTED_QUERY_HEAD_PARAMETERS",
]
