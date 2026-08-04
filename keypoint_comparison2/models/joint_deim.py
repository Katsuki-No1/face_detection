from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models import build_detector_from_config
from tools.dinov3_deimv2_common import load_checkpoint, load_config


def _gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    for _ in range(tensor.ndim - 2):
        view = view.unsqueeze(-1)
    return torch.gather(tensor, 1, view.expand(-1, -1, *tensor.shape[2:]))


def _candidate_indices(
    logits: torch.Tensor,
    *,
    maximum: int,
    matched: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    scores = logits.sigmoid().amax(-1)
    count = min(int(maximum), scores.shape[1])
    top = torch.topk(scores, count, dim=1).indices
    if matched is None:
        return top
    rows: list[torch.Tensor] = []
    for batch_index, (prediction_indices, _target_indices) in enumerate(matched):
        required = prediction_indices.to(device=top.device)
        merged = torch.unique(torch.cat((required, top[batch_index])), sorted=False)
        if merged.numel() < count:
            merged = torch.cat((merged, top[batch_index, : count - merged.numel()]))
        rows.append(merged[:count])
    return torch.stack(rows)


def _block_diagonal_landmark_mask(faces: int, landmarks: int, device: torch.device) -> torch.Tensor:
    groups = torch.arange(faces, device=device).repeat_interleave(landmarks)
    return groups[:, None] != groups[None, :]


class LandmarkQueryDecoder(nn.Module):
    """Generate five semantic queries from every DEIMv2 face object query."""

    def __init__(
        self,
        dimension: int,
        *,
        layers: int = 4,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.landmark_queries = nn.Parameter(torch.empty(5, dimension))
        nn.init.trunc_normal_(self.landmark_queries, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=dimension,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dimension)
        self.coordinate_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 2),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, 1),
        )

    @staticmethod
    def _inside_boxes(local: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        center = boxes[..., :2].unsqueeze(2)
        size = boxes[..., 2:].clamp_min(1e-5).unsqueeze(2)
        top_left = center - 0.5 * size
        return (top_left + local * size).clamp(0.0, 1.0)

    def forward(
        self,
        face_queries: torch.Tensor,
        face_boxes: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        batch, faces, dimension = face_queries.shape
        query = face_queries.unsqueeze(2) + self.landmark_queries.view(1, 1, 5, dimension)
        query = query.reshape(batch, faces * 5, dimension)
        mask = _block_diagonal_landmark_mask(faces, 5, query.device)
        auxiliary: list[torch.Tensor] = []
        visibility = query.new_zeros((batch, faces, 5))
        for layer in self.layers:
            query = layer(query, memory, tgt_mask=mask)
            normalized = self.norm(query).reshape(batch, faces, 5, dimension)
            local = self.coordinate_head(normalized).sigmoid()
            auxiliary.append(self._inside_boxes(local, face_boxes))
            visibility = self.visibility_head(normalized).squeeze(-1)
        return auxiliary[-1], visibility, auxiliary[:-1]


class DEIMv2JointFaceLandmarks(nn.Module):
    """DEIMv2 face queries with five child landmark queries per selected face."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        detector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(config)
        detector_cfg = dict(cfg.get("detector", {}))
        if detector is None:
            config_path = Path(str(detector_cfg["config"]))
            checkpoint_path = Path(str(detector_cfg["checkpoint"]))
            if not config_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "The trained DEIMv2-L face detector config/checkpoint is required for the joint model"
                )
            detector = build_detector_from_config(load_config(config_path))
            missing, unexpected = load_checkpoint(detector, checkpoint_path, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"DEIMv2-L detector checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
        self.face_detector = detector
        dimension = int(cfg.get("dimension", getattr(self.face_detector.decoder, "hidden_dim", 224)))
        self.maximum_face_queries = int(cfg.get("maximum_face_queries", 100))
        self.landmark_decoder = LandmarkQueryDecoder(
            dimension,
            layers=int(cfg.get("landmark_decoder_layers", 4)),
            heads=int(cfg.get("heads", 8)),
            feedforward=int(cfg.get("feedforward", dimension * 4)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
        self.keypoint_loss_weight = float(cfg.get("keypoint_loss_weight", 10.0))
        self.visibility_loss_weight = float(cfg.get("visibility_loss_weight", 1.0))
        self.auxiliary_loss_weight = float(cfg.get("auxiliary_loss_weight", 0.5))
        self._latest_face_queries: torch.Tensor | None = None
        decoder_layers = self.face_detector.decoder.decoder.layers
        decoder_layers[-1].register_forward_hook(self._capture_face_queries)

    def _capture_face_queries(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        self._latest_face_queries = output

    @staticmethod
    def _memory(encoded: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([feature.flatten(2).transpose(1, 2) for feature in encoded], dim=1)

    def _landmark_losses(
        self,
        predictions: torch.Tensor,
        visibility_logits: torch.Tensor,
        auxiliary: list[torch.Tensor],
        candidates: torch.Tensor,
        matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        coordinate_terms: list[torch.Tensor] = []
        visibility_terms: list[torch.Tensor] = []
        auxiliary_terms: list[torch.Tensor] = []
        for batch_index, ((prediction_indices, target_indices), target) in enumerate(zip(matches, targets)):
            if prediction_indices.numel() == 0:
                continue
            candidate_row = candidates[batch_index]
            positions = []
            valid_targets = []
            for prediction_index, target_index in zip(prediction_indices.tolist(), target_indices.tolist()):
                found = torch.nonzero(candidate_row == int(prediction_index), as_tuple=False)
                if found.numel():
                    positions.append(int(found[0, 0]))
                    valid_targets.append(int(target_index))
            if not positions:
                continue
            position_tensor = torch.tensor(positions, device=predictions.device, dtype=torch.long)
            target_tensor = torch.tensor(valid_targets, device=predictions.device, dtype=torch.long)
            target_coordinates = target["keypoints"].to(predictions.device)[target_tensor]
            target_visible = target["visible"].to(predictions.device)[target_tensor].to(predictions.dtype)
            target_labeled = target.get("labeled", target["visible"]).to(predictions.device)[target_tensor].to(
                predictions.dtype
            )
            prediction = predictions[batch_index, position_tensor]
            weight = target_labeled
            denominator = weight.sum().clamp_min(1.0)
            coordinate_terms.append(
                (F.smooth_l1_loss(prediction, target_coordinates, reduction="none").sum(-1) * weight).sum()
                / denominator
            )
            visibility_terms.append(
                (
                    F.binary_cross_entropy_with_logits(
                        visibility_logits[batch_index, position_tensor],
                        target_visible,
                        reduction="none",
                    )
                    * weight
                )
                .sum()
                / denominator
            )
            if auxiliary:
                auxiliary_terms.append(
                    sum(
                        (
                            F.smooth_l1_loss(
                                value[batch_index, position_tensor],
                                target_coordinates,
                                reduction="none",
                            ).sum(-1)
                            * weight
                        ).sum()
                        / denominator
                        for value in auxiliary
                    )
                    / len(auxiliary)
                )
        zero = predictions.sum() * 0.0
        loss_keypoint = torch.stack(coordinate_terms).mean() if coordinate_terms else zero
        loss_visibility = torch.stack(visibility_terms).mean() if visibility_terms else zero
        loss_auxiliary = torch.stack(auxiliary_terms).mean() if auxiliary_terms else zero
        return {
            "loss_keypoint": loss_keypoint,
            "loss_visibility": loss_visibility,
            "loss_landmark_aux": loss_auxiliary,
        }

    def forward(
        self,
        images: torch.Tensor,
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        official_targets = None
        if targets is not None:
            official_targets = self.face_detector._targets_to_official(
                list(targets),
                (int(images.shape[-2]), int(images.shape[-1])),
                device=images.device,
                dtype=images.dtype,
            )
        features = self.face_detector.backbone(images)
        encoded = self.face_detector.encoder(list(features))
        self._latest_face_queries = None
        raw = self.face_detector.decoder(encoded, official_targets)
        if self._latest_face_queries is None:
            raise RuntimeError("Failed to capture DEIMv2 decoder face object queries")
        number_of_queries = raw["pred_boxes"].shape[1]
        face_queries = self._latest_face_queries[:, -number_of_queries:, :]

        matches = None
        if official_targets is not None:
            matches = self.face_detector.criterion.matcher(
                {"pred_logits": raw["pred_logits"], "pred_boxes": raw["pred_boxes"]},
                official_targets,
                epoch=int(getattr(self.face_detector, "current_epoch", 0)),
            )["indices"]
        candidates = _candidate_indices(
            raw["pred_logits"],
            maximum=self.maximum_face_queries,
            matched=matches,
        )
        selected_queries = _gather(face_queries, candidates)
        selected_boxes = _gather(raw["pred_boxes"], candidates)
        selected_logits = _gather(raw["pred_logits"], candidates)
        keypoints, visibility_logits, auxiliary = self.landmark_decoder(
            selected_queries,
            selected_boxes,
            self._memory(encoded),
        )
        output: dict[str, Any] = {
            "pred_logits": selected_logits,
            "pred_boxes": selected_boxes,
            "pred_keypoints": keypoints,
            "visibility_logits": visibility_logits,
            "candidate_indices": candidates,
            "auxiliary_keypoints": auxiliary,
            "raw_detector_output": raw,
        }
        if "pred_ellipses" in raw:
            output["pred_ellipses"] = _gather(raw["pred_ellipses"], candidates)
        if targets is None or official_targets is None or matches is None:
            output["scores"] = selected_logits.sigmoid().amax(-1)
            output["visibility"] = visibility_logits.sigmoid()
            return output
        face_losses = self.face_detector.criterion(
            raw,
            official_targets,
            epoch=int(getattr(self.face_detector, "current_epoch", 0)),
        )
        formatted_face = self.face_detector._format_losses(face_losses)
        landmark_losses = self._landmark_losses(
            keypoints,
            visibility_logits,
            auxiliary,
            candidates,
            matches,
            targets,
        )
        output.update({f"face_{key}": value for key, value in formatted_face.items()})
        output.update(landmark_losses)
        output["loss_total"] = (
            formatted_face["loss_total"]
            + self.keypoint_loss_weight * landmark_losses["loss_keypoint"]
            + self.visibility_loss_weight * landmark_losses["loss_visibility"]
            + self.auxiliary_loss_weight * landmark_losses["loss_landmark_aux"]
        )
        return output
