"""Pinned R50-recipe RPN/RoI and ATSS auxiliary branches (train only)."""
from copy import deepcopy

import torch
from torch import nn
from mmengine.config import ConfigDict
from mmdet.models.dense_heads import RPNHead
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from mmengine.structures import InstanceData

from src.models.co_dino.o2m_head import CoATSSHead


class OfficialRPN(RPNHead):
    def predict_by_feat(self, cls_scores, bbox_preds, score_factors=None, **kwargs):
        # MMDet 2.x forced proposal inputs to FP32. With native AMP in 3.x,
        # decoded boxes can be float while scores remain half; MMCV's split
        # batched-NMS then fails its indexed score assignment. Keep the RPN
        # forward/loss AMP path, but decode and suppress proposals in FP32.
        with torch.autocast(device_type=cls_scores[0].device.type, enabled=False):
            return super().predict_by_feat(
                [score.float() for score in cls_scores],
                [box.float() for box in bbox_preds],
                None if score_factors is None else [s.float() for s in score_factors],
                **kwargs,
            )


class OfficialATSS(CoATSSHead):
    def loss_and_coordinates(self, features, samples):
        outputs = self(tuple(features))
        metas = [s.metainfo for s in samples]
        gt = [s.gt_instances for s in samples]
        losses = self.loss_by_feat(*outputs, gt, metas)
        anchors, flags = self.get_anchors([f.shape[-2:] for f in features], metas,
                                         device=features[0].device)
        level_counts = [len(a) for a in anchors[0]]
        rows = []
        with torch.no_grad():
            for a, f, g, meta in zip(anchors, flags, gt, metas, strict=True):
                targets = self._get_targets_single(torch.cat(a), torch.cat(f),
                    level_counts, g, meta, None, unmap_outputs=True)
                coords, labels, label_weights, boxes = targets[:4]
                # Ignored locations are not background samples.
                labels = labels.clone()
                labels[label_weights == 0] = -1
                rows.append((coords.detach(), labels, boxes.detach()))
        return losses, rows


class OfficialAuxiliaryHeads(nn.Module):
    def __init__(self, channels=256, num_classes=1):
        super().__init__()
        self.atss = OfficialATSS(num_classes=num_classes, in_channels=channels,
            feat_channels=channels, stacked_convs=1, strides=(4,8,16,32,64,128),
            topk=9, max_positive_queries=300, loss_weight=12.)
        self.proposal_cfg = ConfigDict(nms_pre=4000, max_per_img=1000,
            nms=dict(type="nms", iou_threshold=.7), min_bbox_size=0)
        self.rpn = OfficialRPN(in_channels=channels, feat_channels=channels,
            anchor_generator=dict(type="AnchorGenerator", octave_base_scale=4,
                scales_per_octave=3, ratios=[.5,1.,2.], strides=[4,8,16,32,64,128]),
            bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.]*4, target_stds=[1.]*4),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=12.),
            loss_bbox=dict(type="L1Loss", loss_weight=12.),
            train_cfg=ConfigDict(assigner=dict(type="MaxIoUAssigner", pos_iou_thr=.7,
                neg_iou_thr=.3, min_pos_iou=.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=.5,
                    neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=-1, pos_weight=-1, debug=False),
            test_cfg=self.proposal_cfg)
        self.roi = StandardRoIHead(
            bbox_roi_extractor=dict(type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=channels, featmap_strides=[4,8,16,32,64], finest_scale=56),
            bbox_head=dict(type="Shared2FCBBoxHead", in_channels=channels,
                fc_out_channels=1024, roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.]*4,
                    target_stds=[.1,.1,.2,.2]), reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=12.),
                loss_bbox=dict(type="GIoULoss", loss_weight=120.)),
            train_cfg=ConfigDict(assigner=dict(type="MaxIoUAssigner", pos_iou_thr=.5,
                neg_iou_thr=.5, min_pos_iou=.5, match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True), pos_weight=-1, debug=False),
            test_cfg=ConfigDict(score_thr=0., nms=dict(type="nms", iou_threshold=.5), max_per_img=100))
        self.rpn.init_weights()
        self.roi.init_weights()

    def roi_loss_and_coordinates(self, features, proposals, samples):
        sampled = []
        for i, (proposal, sample) in enumerate(zip(proposals, samples, strict=True)):
            instances = InstanceData(priors=proposal.bboxes.detach())
            assignment = self.roi.bbox_assigner.assign(instances, sample.gt_instances, None)
            sampled.append(self.roi.bbox_sampler.sample(assignment, instances,
                sample.gt_instances, feats=[f[i:i+1] for f in features]))
        rois = bbox2roi([s.priors for s in sampled])
        prediction = self.roi._bbox_forward(features, rois)
        targets = self.roi.bbox_head.get_targets(sampled, self.roi.train_cfg)
        losses = self.roi.bbox_head.loss(prediction["cls_score"], prediction["bbox_pred"],
                                        rois, *targets)
        # Same truncation as CoStandardRoIHead before cooperative target selection.
        minimum = min(2000, *(len(s.priors) for s in sampled))
        rows, cursor = [], 0
        for result in sampled:
            n = len(result.priors)
            rows.append((result.priors[:minimum].detach(),
                targets[0][cursor:cursor+minimum].detach(),
                targets[2][cursor:cursor+minimum].detach()))
            cursor += n
        return losses, rows

    def loss_and_coordinates(self, features, samples):
        if len(features) != 6:
            raise ValueError("Auxiliary heads require encoder P2..P7, strides 4..128")
        rpn_samples = deepcopy(samples)
        for sample in rpn_samples:
            sample.gt_instances.labels = torch.zeros_like(sample.gt_instances.labels)
        rpn_losses, proposals = self.rpn.loss_and_predict(features, rpn_samples,
                                                         proposal_cfg=self.proposal_cfg)
        roi_losses, roi_rows = self.roi_loss_and_coordinates(features[:5], proposals, samples)
        atss_losses, atss_rows = self.atss.loss_and_coordinates(features, samples)
        losses = {}
        for branch, values in (("rpn", rpn_losses), ("roi", roi_losses), ("atss", atss_losses)):
            for name, value in values.items():
                if "loss" in name:  # RoI accuracy is a metric, never a loss.
                    losses[f"{branch}.{name}"] = value
        return losses, (roi_rows, atss_rows)
