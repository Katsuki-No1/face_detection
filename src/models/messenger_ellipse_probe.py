"""Frozen baseline FACE/ellipse decoder with a replaceable final DINO map."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.dinov3_deimv2_ellipse.official import ellipse_contour_points


def ellipse_parts(predicted, target, weights, circle_ratio=1.05, points=36):
    """Per-face version of the baseline final-query ellipse geometry losses."""
    center = F.smooth_l1_loss(predicted[:, :2], target[:, :2], reduction='none').sum(-1)
    axes = F.smooth_l1_loss(predicted[:, 2:4], target[:, 2:4], reduction='none').sum(-1)
    eligible = target[:, 2] / target[:, 3].clamp_min(1e-6) >= circle_ratio
    angle = (1 - torch.cos(2 * (predicted[:, 4] - target[:, 4]))) * eligible
    distances = torch.cdist(ellipse_contour_points(predicted, points=points),
                            ellipse_contour_points(target, points=points), compute_mode='donot_use_mm_for_euclid_dist')
    contour = .5 * (distances.min(2).values.mean(1) + distances.min(1).values.mean(1))
    parts = torch.stack((center, axes, angle, contour), -1)
    weight = parts.new_tensor([weights[k] for k in ('center', 'axes', 'angle', 'chamfer')])
    return parts @ weight, parts, eligible


class FrozenEllipseProbe(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector.eval().requires_grad_(False)
        # Dynamic feature-map positions and anchors; no learned weight changes.
        detector.encoder.eval_spatial_size = None
        detector.decoder.eval_spatial_size = None
        self.face_label = detector.category_to_label[2]

    def encode(self, rgb):
        backbone = self.detector.backbone
        normalized = backbone._normalize(rgb)
        maps = backbone.dino.forward_intermediate_feature_maps(normalized, layer_indexes=backbone.interaction_indexes)
        details = backbone.sta(normalized) if backbone.use_sta else [None] * 3
        levels = []
        for i in range(2):
            feature = F.interpolate(maps[i], scale_factor=2 ** (1-i), mode='bilinear', align_corners=False)
            if details[i] is not None:
                feature = torch.cat((feature, details[i]), 1)
            levels.append(backbone.norms[i](backbone.convs[i](feature)))
        detail = details[2] if details[2] is not None else maps[-1][:, :0, ::2, ::2]
        return torch.stack(maps, 1), levels, detail

    def forward(self, final_map, levels, detail):
        backbone = self.detector.backbone
        feature = F.interpolate(final_map, scale_factor=.5, mode='bilinear', align_corners=False)
        feature = torch.cat((feature, detail), 1)
        last = backbone.norms[2](backbone.convs[2](feature))
        return self.detector.decoder(self.detector.encoder([*levels, last]))

    def select(self, output):
        # No GT, artificial mask, or clean feature participates in query selection.
        score = output['pred_logits'][..., self.face_label].sigmoid()
        index = score.detach().argmax(1)
        batch = torch.arange(len(score), device=score.device)
        return output['pred_ellipses'][batch, index], score[batch, index], index
