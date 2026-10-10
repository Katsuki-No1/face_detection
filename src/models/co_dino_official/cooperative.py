"""Co-DETR negative-filled cooperative targets, separate from invalid padding."""
from dataclasses import dataclass

import torch
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


@dataclass
class CooperativeTargets:
    coordinates: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    valid: torch.Tensor
    positive: torch.Tensor

    def attention_mask(self, heads=8):
        # Real negative samples intentionally participate in self-attention.
        if self.valid.all():
            return None
        b, n = self.valid.shape
        mask = (~self.valid[:, None, :]).expand(b, n, n).clone()
        # Invalid rows attend only themselves, preventing all-masked softmax NaN.
        mask |= (~self.valid[:, :, None]).expand(b, n, n)
        diag = torch.arange(n, device=mask.device)
        mask[:, diag, diag] = False
        return mask[:, None].expand(b, heads, n, n).reshape(b * heads, n, n)


def build_cooperative_targets(rows, samples, num_classes, maximum=300):
    """Rows are detached pixel-xyxy (coordinates, labels, regression targets).

    Matches upstream get_aux_targets with use_zero_padding=False. Randomly
    subsample positives; fill to max(9, min(max positives, 300)) with genuine
    negatives. Only the absent-negative edge case adds masked invalid slots.
    """
    counts = [int(((labels >= 0) & (labels < num_classes)).sum())
              for _, labels, _ in rows]
    count = max(9, min(max(counts, default=0), maximum))
    coordinates = rows[0][0].new_zeros((len(rows), count, 4))
    boxes = torch.zeros_like(coordinates)
    labels_out = rows[0][1].new_full((len(rows), count), num_classes)
    valid = torch.zeros_like(labels_out, dtype=torch.bool)
    positive = torch.zeros_like(valid)
    for i, ((coords, labels, targets), sample) in enumerate(zip(rows, samples, strict=True)):
        pos = torch.where((labels >= 0) & (labels < num_classes))[0]
        neg = torch.where(labels == num_classes)[0]
        if len(pos) > count:
            pos = pos[torch.randperm(len(pos), device=pos.device)[:count]]
        remaining = count - len(pos)
        neg = neg[torch.randperm(len(neg), device=neg.device)[:remaining]]
        indices = torch.cat((pos, neg))
        h, w = sample.metainfo["img_shape"][:2]
        factor = coords.new_tensor([w, h, w, h])
        n = len(indices)
        coordinates[i, :n] = bbox_xyxy_to_cxcywh(coords[indices] / factor)
        boxes[i, :n] = bbox_xyxy_to_cxcywh(targets[indices] / factor)
        labels_out[i, :n] = labels[indices]
        valid[i, :n] = True
        positive[i, :len(pos)] = True
    return CooperativeTargets(coordinates.detach(), labels_out, boxes.detach(), valid, positive)
