from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KeypointMetrics:
    thresholds: tuple[float, ...] = (0.05, 0.10)
    errors: list[torch.Tensor] = field(default_factory=list)
    visible: list[torch.Tensor] = field(default_factory=list)
    predicted_visible: list[torch.Tensor] = field(default_factory=list)
    visibility_targets: list[torch.Tensor] = field(default_factory=list)
    visibility_masks: list[torch.Tensor] = field(default_factory=list)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor,
        *,
        visibility_probability: torch.Tensor | None = None,
        visibility_target: torch.Tensor | None = None,
        normalization: torch.Tensor | None = None,
    ) -> None:
        prediction = prediction.detach().cpu().float()
        target = target.detach().cpu().float()
        visible = visible.detach().cpu().bool()
        if normalization is None:
            # Inter-ocular normalization when both eyes are labelled; fallback to crop width.
            interocular = (target[:, 0] - target[:, 1]).norm(dim=-1)
            eye_visible = visible[:, 0] & visible[:, 1]
            normalization = torch.where(eye_visible, interocular, torch.ones_like(interocular))
        normalization = normalization.detach().cpu().float().clamp_min(1e-6)
        error = (prediction - target).norm(dim=-1) / normalization.unsqueeze(-1)
        self.errors.append(error)
        self.visible.append(visible)
        if visibility_probability is not None:
            self.predicted_visible.append(visibility_probability.detach().cpu().float() >= 0.5)
            self.visibility_targets.append(
                (visible if visibility_target is None else visibility_target).detach().cpu().bool()
            )
            self.visibility_masks.append(visible)

    def compute(self) -> dict[str, float]:
        if not self.errors:
            return {"nme": float("nan"), **{f"pck_{value:.2f}": float("nan") for value in self.thresholds}}
        error = torch.cat(self.errors)
        visible = torch.cat(self.visible)
        values = error[visible]
        result = {
            "nme": float(values.mean()) if values.numel() else float("nan"),
            "visible_points": float(visible.sum()),
            "instances": float(visible.shape[0]),
        }
        for threshold in self.thresholds:
            result[f"pck_{threshold:.2f}"] = float((values <= threshold).float().mean()) if values.numel() else float("nan")
        if self.predicted_visible:
            predicted = torch.cat(self.predicted_visible)
            visibility_target = torch.cat(self.visibility_targets)
            visibility_mask = torch.cat(self.visibility_masks)
            predicted = predicted[visibility_mask]
            visibility_target = visibility_target[visibility_mask]
            true_positive = (predicted & visibility_target).sum().float()
            false_positive = (predicted & ~visibility_target).sum().float()
            false_negative = (~predicted & visibility_target).sum().float()
            precision = true_positive / (true_positive + false_positive).clamp_min(1)
            recall = true_positive / (true_positive + false_negative).clamp_min(1)
            result["visibility_precision"] = float(precision)
            result["visibility_recall"] = float(recall)
            result["visibility_f1"] = float(2 * precision * recall / (precision + recall).clamp_min(1e-8))
        return result
