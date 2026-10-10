"""Pinned Co-DINO query path using MMDetection 3 attention/loss primitives.

Encoder proposals, Hungarian and CDN target plumbing reuse audited project
adapters. Reference refinement, head epsilon and auxiliary decoding are explicit
here; no old cooperative forward or old auxiliary loss is executed.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmdet.models.dense_heads import DINOHead
from mmdet.models.layers import DeformableDetrTransformerEncoder, DinoTransformerDecoder
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, inverse_sigmoid
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import reduce_mean

from src.models.co_dino.query_head import CoDINOQueryHead


class OfficialEncoder(DeformableDetrTransformerEncoder):
    checkpoint_layers = 4

    def forward(self, query, query_pos, key_padding_mask, spatial_shapes,
                level_start_index, valid_ratios, **kwargs):
        references = self.get_encoder_reference_points(spatial_shapes, valid_ratios, query.device)
        inputs = dict(query_pos=query_pos, key_padding_mask=key_padding_mask,
                      spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                      valid_ratios=valid_ratios, reference_points=references, **kwargs)
        for index, layer in enumerate(self.layers):
            if self.training and index < self.checkpoint_layers:
                query = checkpoint(layer, query=query, use_reentrant=False, **inputs)
            else:
                query = layer(query=query, **inputs)
        return query


class OfficialDecoder(DinoTransformerDecoder):
    def forward(self, query, value, key_padding_mask, self_attn_mask,
                reference_points, spatial_shapes, level_start_index,
                valid_ratios, reg_branches, **kwargs):
        hidden, references = [], [reference_points]
        for index, layer in enumerate(self.layers):
            ref_input = reference_points[:, :, None] * torch.cat(
                (valid_ratios, valid_ratios), -1)[:, None]
            query_pos = self.ref_point_head(coordinate_to_encoding(
                ref_input[:, :, 0], num_feats=self.embed_dims // 2))
            query = layer(query, query_pos=query_pos, value=value,
                key_padding_mask=key_padding_mask, self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                valid_ratios=valid_ratios, reference_points=ref_input, **kwargs)
            updated = (reg_branches[index](query) + inverse_sigmoid(reference_points, eps=1e-3)).sigmoid()
            reference_points = updated.detach()
            hidden.append(self.norm(query))
            references.append(updated)  # Look-forward-twice: do NOT detach this output.
        return torch.stack(hidden), torch.stack(references)


class OfficialPredictionHead(DINOHead):
    def forward(self, hidden_states, references):
        logits, boxes = [], []
        for index, hidden in enumerate(hidden_states):
            logits.append(self.cls_branches[index](hidden))
            boxes.append((self.reg_branches[index](hidden) +
                          inverse_sigmoid(references[index], eps=1e-3)).sigmoid())
        return torch.stack(logits), torch.stack(boxes)


def proposal_position_embedding(coords, embed_dims):
    # Upstream get_proposal_pos_embed is x,y,w,h, unlike decoder y,x,w,h.
    coords = inverse_sigmoid(coords).sigmoid()
    n = embed_dims // 2
    dim = torch.arange(n, device=coords.device, dtype=torch.float32)
    dim = 10000 ** (2 * (dim // 2) / n)
    position = coords[..., None] * (2 * math.pi) / dim
    return torch.stack((position[..., 0::2].sin(), position[..., 1::2].cos()), -1).flatten(2)


class OfficialQueryHead(CoDINOQueryHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.auxiliary_query_embedding
        old = self.encoder
        self.encoder = OfficialEncoder(num_layers=self.num_encoder_layers, layer_cfg=old.layer_cfg)
        self.encoder.load_state_dict(old.state_dict())
        old = self.decoder
        self.decoder = OfficialDecoder(num_layers=self.num_decoder_layers,
            return_intermediate=True, post_norm_cfg=None, layer_cfg=old.layer_cfg)
        self.decoder.load_state_dict(old.state_dict())
        self.bbox_head = OfficialPredictionHead(num_classes=self.num_classes,
            embed_dims=self.embed_dims, num_reg_fcs=2, sync_cls_avg_factor=True,
            share_pred_layer=False, num_pred_layer=self.num_decoder_layers + 1,
            as_two_stage=True,
            loss_cls=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2., loss_weight=1.),
            loss_bbox=dict(type="L1Loss", loss_weight=5.),
            loss_iou=dict(type="GIoULoss", loss_weight=2.),
            train_cfg=dict(assigner=dict(type="HungarianAssigner", match_costs=[
                dict(type="FocalLossCost", weight=2.),
                dict(type="BBoxL1Cost", weight=5., box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.)])),
            test_cfg=dict(max_per_img=self.num_queries))
        self.bbox_head.init_weights()
        self.dn_query_generator.box_noise_scale = 1.0
        self.aux_pos_trans = nn.ModuleList([nn.Linear(2*self.embed_dims, self.embed_dims) for _ in range(2)])
        self.aux_pos_trans_norm = nn.ModuleList([nn.LayerNorm(self.embed_dims) for _ in range(2)])
        for layer in self.aux_pos_trans:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def _flatten_features(self, features, masks):
        # Targets, CDN and auxiliary coordinates use the entire resized canvas.
        # Geometric background may occur on ANY edge or inside that canvas;
        # it is an attention mask, not a smaller top-left image extent.
        values = list(super()._flatten_features(
            features, [torch.zeros_like(mask) for mask in masks]))
        values[2] = torch.cat([mask.flatten(1) for mask in masks], dim=1)
        if values[2].all(dim=1).any():
            raise ValueError("Official-SFP requires at least one valid image region")
        return tuple(values)

    def _encoder_proposals(self, memory, memory_mask, spatial_shapes):
        proposals = []
        for level, (height, width) in enumerate(spatial_shapes):
            y, x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=memory.device),
                torch.arange(width, dtype=torch.float32, device=memory.device),
                indexing="ij")
            grid = (torch.stack((x, y), -1) + .5) / memory.new_tensor([width, height])
            sizes = torch.ones_like(grid) * .05 * (2.0 ** level)
            proposals.append(torch.cat((grid, sizes), -1).reshape(1, -1, 4))
        proposals = torch.cat(proposals, 1).expand(memory.shape[0], -1, -1)
        valid = ((proposals > .01) & (proposals < .99)).all(-1, keepdim=True)
        excluded = memory_mask.unsqueeze(-1) | ~valid
        logits = torch.log(proposals / (1.0 - proposals)).masked_fill(excluded, float("inf"))
        output = self.memory_trans_norm(self.memory_trans_fc(memory.masked_fill(excluded, 0.)))
        return output, logits

    def cooperative_decode(self, raw, targets, head_index):
        anchors = targets.coordinates
        query = self.aux_pos_trans_norm[head_index](
            self.aux_pos_trans[head_index](proposal_position_embedding(anchors, self.embed_dims)))
        hidden, refs = self.decoder(query=query, value=raw["_memory"],
            key_padding_mask=raw["_memory_mask"], self_attn_mask=targets.attention_mask(),
            reference_points=anchors, spatial_shapes=raw["_spatial_shapes"],
            level_start_index=raw["_level_start_index"], valid_ratios=raw["_valid_ratios"],
            reg_branches=self.bbox_head.reg_branches)
        return self.bbox_head(hidden, refs)

    def cooperative_losses(self, raw, targets, samples, head_index):
        logits, boxes = self.cooperative_decode(raw, targets, head_index)
        logits, boxes = logits.float(), boxes.float()
        positive, valid = targets.positive, targets.valid
        npos = reduce_mean(boxes.new_tensor(float(positive.sum()))).clamp_min(1).item()
        nneg = (valid & ~positive).sum()
        navg = reduce_mean(boxes.new_tensor(float(positive.sum() +
            nneg * self.bbox_head.bg_cls_weight))).clamp_min(1).item()
        target_boxes = targets.boxes.float()
        weights = positive[..., None].expand_as(target_boxes).float()
        factor = boxes.new_tensor([[s.metainfo["img_shape"][1], s.metainfo["img_shape"][0]] * 2
                                  for s in samples])[:, None]
        losses = {}
        for index, (scores, predicted) in enumerate(zip(logits, boxes, strict=True)):
            quality = scores.new_zeros(targets.labels.shape)
            quality[positive] = bbox_overlaps(
                bbox_cxcywh_to_xyxy(predicted[positive].detach()),
                bbox_cxcywh_to_xyxy(target_boxes[positive]), is_aligned=True)
            prefix = f"co{head_index}.d{index}"
            losses[prefix + ".loss_cls"] = self.bbox_head.loss_cls(
                scores.flatten(0, 1), (targets.labels.flatten(), quality.flatten()),
                valid.flatten().float(), avg_factor=navg)
            losses[prefix + ".loss_bbox"] = self.bbox_head.loss_bbox(
                predicted, target_boxes, weights, avg_factor=npos)
            losses[prefix + ".loss_giou"] = self.bbox_head.loss_iou(
                (bbox_cxcywh_to_xyxy(predicted)*factor).reshape(-1, 4),
                (bbox_cxcywh_to_xyxy(target_boxes)*factor).reshape(-1, 4),
                weights.reshape(-1, 4), avg_factor=npos)
        return losses
