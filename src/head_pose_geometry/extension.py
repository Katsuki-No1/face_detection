from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .crops import HeadCropBuilder
from .fusion import FusionEvidence, UncertaintyGatedGeometryFusion
from .geometry import CanonicalHeadGeometryAdapter
from .heads import HeadDetection
from .losses import HeadPoseGeometryLoss
from .pose_encoder import PoseEncoderAdapter


class HeadPoseGeometryExtension(nn.Module):
    """Detector-independent crop, pose, geometry, loss, and query bridge."""

    def __init__(
        self,
        *,
        crop_builder: HeadCropBuilder,
        pose_encoder: PoseEncoderAdapter,
        geometry_adapter: CanonicalHeadGeometryAdapter | None,
        fusion: UncertaintyGatedGeometryFusion | None,
        criterion: HeadPoseGeometryLoss | None,
        pose_query_dimension: int | None = None,
        query_injection_temperature: float = 0.20,
        use_valid_pixel_and_border_inputs: bool = True,
        augmentation_consistency_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.crop_builder = crop_builder
        self.pose_encoder = pose_encoder
        self.geometry_adapter = geometry_adapter
        self.fusion = fusion
        self.criterion = criterion
        self.query_injection_temperature = float(query_injection_temperature)
        self.use_valid_pixel_and_border_inputs = bool(use_valid_pixel_and_border_inputs)
        self.augmentation_consistency_enabled = bool(augmentation_consistency_enabled)
        if pose_query_dimension is None:
            self.pose_query_projection = None
            self.geometry_query_projection = None
            self.query_injection_gate = None
        else:
            self.pose_query_projection = nn.Linear(
                pose_encoder.embedding_dimension,
                int(pose_query_dimension),
            )
            self.geometry_query_projection = (
                nn.Linear(5 + 10 + 3, int(pose_query_dimension))
                if geometry_adapter is not None
                else None
            )
            self.query_injection_gate = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            gate_output = self.query_injection_gate[-1]
            assert isinstance(gate_output, nn.Linear)
            nn.init.zeros_(gate_output.weight)
            nn.init.constant_(gate_output.bias, -1.5)

    @property
    def geometry_enabled(self) -> bool:
        return self.geometry_adapter is not None

    @property
    def fusion_enabled(self) -> bool:
        return self.geometry_adapter is not None and self.fusion is not None

    def _one_image(
        self,
        image: torch.Tensor,
        detections: list[HeadDetection],
        *,
        training_crop: bool,
        generator: torch.Generator | None,
    ) -> dict[str, Any]:
        crop = self.crop_builder.build(
            image,
            detections,
            training=training_crop,
            generator=generator,
        )
        pose = self.pose_encoder(crop.rgb)
        output: dict[str, Any] = {**pose, "crop": crop}
        height, width = int(image.shape[-2]), int(image.shape[-1])
        normalizer = crop.head_boxes_xyxy.new_tensor(
            (width, height, width, height)
        ).clamp_min(1.0)
        normalized_heads = crop.head_boxes_xyxy / normalizer
        valid_fraction = crop.valid_mask.mean(dim=(1, 2, 3))
        output["normalized_head_boxes_xyxy"] = normalized_heads
        output["valid_fraction"] = valid_fraction
        if self.geometry_adapter is None:
            return output
        geometry_border = crop.border_distances
        geometry_valid_fraction = valid_fraction
        if not self.use_valid_pixel_and_border_inputs:
            geometry_border = torch.zeros_like(geometry_border)
            geometry_valid_fraction = torch.ones_like(geometry_valid_fraction)
        geometry = self.geometry_adapter(
            rotation_matrix=pose["rotation_matrix"],
            pose_embedding=pose["pose_embedding"],
            head_boxes_xyxy=normalized_heads,
            border_distances=geometry_border,
            head_confidence=crop.head_confidence,
            valid_fraction=geometry_valid_fraction,
        )
        output.update(geometry)
        output["geometry_embedding"] = torch.cat(
            (
                geometry["amodal_ellipse"],
                geometry["projected_keypoints"].flatten(1),
                geometry["face_observability"].unsqueeze(-1),
                geometry["geometry_uncertainty"].unsqueeze(-1),
                valid_fraction.unsqueeze(-1),
            ),
            dim=-1,
        )
        return output

    def forward(
        self,
        images: torch.Tensor,
        detections: Sequence[Sequence[HeadDetection]],
        *,
        training_crop: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, Any]]:
        if images.ndim != 4 or images.shape[0] != len(detections):
            raise ValueError("images and detections batch sizes must match")
        outputs = [
            self._one_image(
                image,
                list(rows),
                training_crop=training_crop,
                generator=generator,
            )
            for image, rows in zip(images, detections, strict=True)
        ]
        if (
            training_crop
            and self.geometry_adapter is not None
            and self.augmentation_consistency_enabled
        ):
            references = [
                self._one_image(
                    image,
                    list(rows),
                    training_crop=False,
                    generator=None,
                )
                for image, rows in zip(images, detections, strict=True)
            ]
            for output, reference in zip(outputs, references, strict=True):
                output["consistency_rotation_matrix"] = reference["rotation_matrix"]
                output["consistency_amodal_ellipse"] = reference["amodal_ellipse"]
                output["consistency_projected_keypoints"] = reference[
                    "projected_keypoints"
                ]
        return outputs

    def inject_pose_queries(
        self,
        independent_pose: Mapping[str, Any],
        extension_outputs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Condition current pose queries without replacing their direct path."""

        if self.pose_query_projection is None or self.query_injection_gate is None:
            return dict(independent_pose)
        result = dict(independent_pose)
        original = independent_pose["pose_embeddings"]
        updated = original.clone()
        diagnostics: list[dict[str, torch.Tensor]] = []
        for batch_index, output in enumerate(extension_outputs):
            count = int(output["rotation_matrix"].shape[0])
            if count == 0:
                diagnostics.append({})
                continue
            head_boxes = output["normalized_head_boxes_xyxy"].to(original)
            head_centers = 0.5 * (head_boxes[:, :2] + head_boxes[:, 2:])
            head_sizes = (head_boxes[:, 2:] - head_boxes[:, :2]).clamp_min(1.0e-4)
            query_centers = independent_pose["pred_pose_boxes"][batch_index, :, :2].to(
                original
            )
            distance = torch.cdist(query_centers, head_centers, p=1)
            scale = head_sizes.mean(dim=-1).clamp_min(1.0e-4)
            assignment_logits = -distance / (
                self.query_injection_temperature * scale.unsqueeze(0)
            ).clamp_min(1.0e-4)
            assignment = torch.softmax(assignment_logits, dim=-1)
            query_proximity = torch.exp(assignment_logits.amax(dim=-1)).clamp(0.0, 1.0)
            pose_token = self.pose_query_projection(output["pose_embedding"])
            if (
                self.geometry_query_projection is not None
                and "amodal_ellipse" in output
            ):
                pose_token = pose_token + self.geometry_query_projection(
                    output["geometry_embedding"]
                )
            input_valid_fraction = (
                output["valid_fraction"]
                if self.use_valid_pixel_and_border_inputs
                else torch.ones_like(output["valid_fraction"])
            )
            reliability = (
                output["pose_confidence"]
                * output["crop"].head_confidence
                * input_valid_fraction
            ).to(original)
            gate_features = torch.stack(
                (
                    output["pose_uncertainty"],
                    output["crop"].head_confidence,
                    input_valid_fraction,
                ),
                dim=-1,
            ).to(original)
            head_gate = torch.sigmoid(self.query_injection_gate(gate_features)).squeeze(
                -1
            )
            weighted_token = pose_token * (reliability * head_gate).unsqueeze(-1)
            query_delta = (assignment @ weighted_token) * query_proximity.unsqueeze(-1)
            updated[batch_index] = original[batch_index] + query_delta
            diagnostics.append(
                {
                    "query_to_head_assignment": assignment,
                    "head_gate": head_gate,
                    "query_proximity": query_proximity,
                    "query_delta_norm": torch.linalg.vector_norm(
                        query_delta,
                        dim=-1,
                    ),
                }
            )
        result["pose_embeddings"] = updated
        result["head_pose_query_injection"] = diagnostics
        return result

    @staticmethod
    def detections_from_targets(
        targets: Sequence[Mapping[str, Any]],
        *,
        image_size: tuple[int, int],
    ) -> list[list[HeadDetection]]:
        height, width = image_size
        batches = []
        for target in targets:
            raw_boxes = target.get("head_boxes")
            if raw_boxes is None:
                raise ValueError(
                    "Head-pose geometry training requires target['head_boxes']"
                )
            boxes = torch.as_tensor(raw_boxes).detach().cpu().reshape(-1, 4)
            confidence = (
                torch.as_tensor(
                    target.get("head_confidence", torch.ones(boxes.shape[0]))
                )
                .detach()
                .cpu()
                .reshape(-1)
            )
            sources = target.get("head_sources")
            if sources is None:
                sources = ["ground_truth"] * int(boxes.shape[0])
            if len(sources) != boxes.shape[0]:
                raise ValueError("head_sources length must match head_boxes")
            batches.append(
                [
                    HeadDetection(
                        box_xyxy=(
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ),
                        confidence=float(score),
                        instance_id=index,
                        valid_image_region_xyxy=(
                            0.0,
                            0.0,
                            float(width),
                            float(height),
                        ),
                        source=str(source),
                    )
                    for index, (box, score, source) in enumerate(
                        zip(boxes, confidence, sources, strict=True)
                    )
                ]
            )
        return batches

    def losses(
        self,
        outputs: Sequence[Mapping[str, Any]],
        targets: Sequence[Mapping[str, Any]],
        *,
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if not outputs:
            raise ValueError("outputs must not be empty")
        if self.geometry_adapter is None or self.criterion is None:
            zero = outputs[0]["rotation_matrix"].sum() * 0.0
            return {"loss_head_pose_geometry": zero}
        height, width = image_size
        totals: dict[str, torch.Tensor] = {}
        contributing = 0
        for output, target in zip(outputs, targets, strict=True):
            count = int(output["rotation_matrix"].shape[0])
            if count == 0:
                continue
            all_ellipses = torch.as_tensor(
                target["ellipses"],
                device=output["rotation_matrix"].device,
                dtype=output["rotation_matrix"].dtype,
            ).reshape(-1, 5)
            all_keypoints = torch.as_tensor(
                target["keypoints"],
                device=all_ellipses.device,
                dtype=all_ellipses.dtype,
            ).reshape(-1, 5, 2)
            face_count = int(all_ellipses.shape[0])
            mapping = torch.as_tensor(
                target.get(
                    "geometry_target_indices",
                    torch.arange(count),
                ),
                device=all_ellipses.device,
                dtype=torch.long,
            ).reshape(-1)
            if mapping.shape[0] != count:
                raise ValueError(
                    "geometry_target_indices must align with HEAD detections: "
                    f"{mapping.shape[0]} != {count}"
                )
            valid_mapping = (mapping >= 0) & (mapping < face_count)
            head_indices = torch.nonzero(
                valid_mapping,
                as_tuple=False,
            ).squeeze(-1)
            if head_indices.numel() == 0:
                continue
            face_indices = mapping[head_indices]
            selected_output: dict[str, Any] = {}
            for key, value in output.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == count
                ):
                    selected_output[key] = value[head_indices]
                else:
                    selected_output[key] = value
            ellipses = all_ellipses[face_indices]
            keypoints = all_keypoints[face_indices]
            selected_count = int(head_indices.shape[0])
            ellipse_normalizer = ellipses.new_tensor(
                (width, height, max(width, height), max(width, height), 1.0)
            )
            if keypoints.numel() and float(keypoints.detach().abs().amax()) > 1.5:
                keypoints = keypoints / keypoints.new_tensor((width, height))
            head_boxes = selected_output["normalized_head_boxes_xyxy"]
            all_labeled = torch.as_tensor(
                target.get("labeled", target.get("keypoint_mask")),
                device=ellipses.device,
                dtype=torch.bool,
            ).reshape(face_count, 5)
            all_visible = torch.as_tensor(
                target.get(
                    "visible",
                    target.get("labeled", target.get("keypoint_mask")),
                ),
                device=ellipses.device,
                dtype=torch.bool,
            ).reshape(face_count, 5)
            loss_target: dict[str, torch.Tensor] = {
                "ellipses": ellipses / ellipse_normalizer,
                "ellipse_mask": torch.as_tensor(
                    target.get(
                        "ellipse_mask",
                        torch.ones(face_count, dtype=torch.bool),
                    ),
                    device=ellipses.device,
                    dtype=torch.bool,
                ).reshape(face_count)[face_indices],
                "keypoints": keypoints,
                "keypoint_labeled": all_labeled[face_indices],
                "keypoint_visible": all_visible[face_indices],
                "head_boxes_xyxy": head_boxes,
            }
            if "rotation_matrix" in target:
                loss_target["rotation_matrix"] = torch.as_tensor(
                    target["rotation_matrix"],
                    device=ellipses.device,
                    dtype=ellipses.dtype,
                ).reshape(face_count, 3, 3)[face_indices]
            for key in (
                "template_vertices",
                "template_supervision_mask",
                "weak_perspective_scale",
                "translation",
                "shape_code",
                "projection_parameter_mask",
            ):
                if key in target:
                    value = torch.as_tensor(
                        target[key],
                        device=ellipses.device,
                    )
                    if value.is_floating_point():
                        value = value.to(dtype=ellipses.dtype)
                    if value.ndim > 0 and value.shape[0] == face_count:
                        value = value[face_indices]
                    loss_target[key] = value
            loss_target["pose_supervision_mask"] = torch.as_tensor(
                target.get(
                    "pose_supervision_mask",
                    torch.zeros(face_count, dtype=torch.bool),
                ),
                device=ellipses.device,
                dtype=torch.bool,
            ).reshape(face_count)[face_indices]
            loss_target["pose_weight"] = torch.as_tensor(
                target.get("pose_weight", torch.ones(face_count)),
                device=ellipses.device,
                dtype=ellipses.dtype,
            ).reshape(face_count)[face_indices]
            if selected_output["rotation_matrix"].shape[0] != selected_count:
                raise RuntimeError("Internal geometry selection failed")
            per_image = self.criterion(selected_output, loss_target)
            for key, value in per_image.items():
                totals[key] = totals.get(key, value * 0.0) + value
            contributing += 1
        if contributing == 0:
            return self.criterion.zero_losses(outputs[0]["rotation_matrix"])
        return {key: value / contributing for key, value in totals.items()}

    def fuse_training_queries(
        self,
        refined: Mapping[str, Any],
        extension_outputs: Sequence[Mapping[str, Any]],
        *,
        face_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        pose_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        geometry_target_indices: Sequence[torch.Tensor],
    ) -> dict[str, Any]:
        if not self.fusion_enabled:
            return dict(refined)
        fusion = self.fusion
        assert fusion is not None
        fused = dict(refined)
        fused_ellipses = refined["pred_face_ellipses"].clone()
        fused_keypoints = refined["pred_keypoints"].clone()
        fusion_rows: list[dict[str, torch.Tensor]] = []
        for batch_index, output in enumerate(extension_outputs):
            face_source, face_target = face_matches[batch_index]
            pose_source, pose_target = pose_matches[batch_index]
            face_for_target = {
                int(target): int(source)
                for source, target in zip(
                    face_source.tolist(),
                    face_target.tolist(),
                    strict=True,
                )
            }
            pose_for_target = {
                int(target): int(source)
                for source, target in zip(
                    pose_source.tolist(),
                    pose_target.tolist(),
                    strict=True,
                )
            }
            mapping = geometry_target_indices[batch_index].to(
                device=fused_ellipses.device,
                dtype=torch.long,
            )
            if mapping.shape[0] != output["rotation_matrix"].shape[0]:
                raise ValueError(
                    "geometry_target_indices must align with extension output"
                )
            geometry_for_target = {
                int(target): geometry_index
                for geometry_index, target in enumerate(mapping.tolist())
                if int(target) >= 0
            }
            common = sorted(
                set(face_for_target) & set(pose_for_target) & set(geometry_for_target)
            )
            if not common:
                fusion_rows.append({})
                continue
            target_index = torch.tensor(
                common,
                device=fused_ellipses.device,
                dtype=torch.long,
            )
            face_index = torch.tensor(
                [face_for_target[value] for value in common],
                device=fused_ellipses.device,
                dtype=torch.long,
            )
            pose_index = torch.tensor(
                [pose_for_target[value] for value in common],
                device=fused_ellipses.device,
                dtype=torch.long,
            )
            geometry_index = torch.tensor(
                [geometry_for_target[value] for value in common],
                device=fused_ellipses.device,
                dtype=torch.long,
            )
            direct_score = (
                refined["pred_face_logits"][
                    batch_index,
                    face_index,
                ]
                .sigmoid()
                .amax(dim=-1)
            )
            keypoint_visibility = refined["pred_visibility_logits"][
                batch_index,
                pose_index,
            ].sigmoid()
            border = output["crop"].border_distances[geometry_index]
            valid_fraction = output["valid_fraction"][geometry_index]
            if not self.use_valid_pixel_and_border_inputs:
                border = torch.zeros_like(border)
                valid_fraction = torch.ones_like(valid_fraction)
            evidence = FusionEvidence(
                pose_uncertainty=output["pose_uncertainty"][geometry_index],
                geometry_uncertainty=output["geometry_uncertainty"][geometry_index],
                direct_ellipse_uncertainty=1.0 - direct_score,
                mean_keypoint_visibility=keypoint_visibility.mean(dim=-1),
                visible_keypoint_fraction=(keypoint_visibility >= 0.5)
                .to(keypoint_visibility)
                .mean(dim=-1),
                face_observability=output["face_observability"][geometry_index],
                truncation_ratio=1.0 - valid_fraction,
                minimum_border_distance=border.amin(dim=-1),
                head_confidence=output["crop"].head_confidence[geometry_index],
            )
            result = fusion(
                pose_embedding=output["pose_embedding"][geometry_index],
                direct_ellipse=refined["pred_face_ellipses"][
                    batch_index,
                    face_index,
                ],
                direct_keypoints=refined["pred_keypoints"][
                    batch_index,
                    pose_index,
                ],
                geometry_ellipse=output["amodal_ellipse"][geometry_index],
                geometry_keypoints=output["projected_keypoints"][geometry_index],
                evidence=evidence,
            )
            fused_ellipses[batch_index, face_index] = result["fused_ellipse"]
            fused_keypoints[batch_index, pose_index] = result["fused_keypoints"]
            fusion_rows.append(
                {
                    **result,
                    "target_indices": target_index,
                    "geometry_indices": geometry_index,
                    "face_query_indices": face_index,
                    "pose_query_indices": pose_index,
                }
            )
        fused["pred_face_ellipses"] = fused_ellipses
        fused["pred_keypoints"] = fused_keypoints
        fused["head_pose_geometry_fusion"] = fusion_rows
        return fused

    def fuse_inference_queries(
        self,
        refined: Mapping[str, Any],
        extension_outputs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not self.fusion_enabled:
            return dict(refined)
        fusion = self.fusion
        assert fusion is not None
        fused = dict(refined)
        fused_ellipses = refined["pred_face_ellipses"].clone()
        fused_keypoints = refined["pred_keypoints"].clone()
        rows: list[dict[str, torch.Tensor]] = []
        for batch_index, output in enumerate(extension_outputs):
            head_boxes = output["normalized_head_boxes_xyxy"]
            if head_boxes.shape[0] == 0:
                rows.append({})
                continue
            pose_boxes = refined["pred_pose_boxes"][batch_index]
            pose_centers = pose_boxes[:, :2]
            head_size = (head_boxes[:, 2:] - head_boxes[:, :2]).clamp_min(1.0e-6)
            head_center = 0.5 * (head_boxes[:, :2] + head_boxes[:, 2:])
            distance = (
                (head_center[:, None, :] - pose_centers[None, :, :]).abs()
                / head_size[:, None, :]
            ).sum(dim=-1)
            # Greedy uniqueness is deterministic and only links external HEADs
            # to existing pose queries. FACE linkage reuses learned association.
            pairs = []
            used: set[int] = set()
            used_faces: set[int] = set()
            for head_index in range(int(head_boxes.shape[0])):
                ordering = torch.argsort(distance[head_index], stable=True)
                pose_index = next(
                    (
                        int(value)
                        for value in ordering.tolist()
                        if int(value) not in used
                    ),
                    int(ordering[0]),
                )
                used.add(pose_index)
                face_ordering = torch.argsort(
                    refined["association_logits"][batch_index, :, pose_index],
                    descending=True,
                    stable=True,
                )
                face_index = next(
                    (
                        int(value)
                        for value in face_ordering.tolist()
                        if int(value) not in used_faces
                    ),
                    int(face_ordering[0]),
                )
                used_faces.add(face_index)
                pairs.append((head_index, face_index, pose_index))
            head_index = torch.tensor(
                [item[0] for item in pairs],
                device=head_boxes.device,
                dtype=torch.long,
            )
            face_index = torch.tensor(
                [item[1] for item in pairs],
                device=head_boxes.device,
                dtype=torch.long,
            )
            pose_index = torch.tensor(
                [item[2] for item in pairs],
                device=head_boxes.device,
                dtype=torch.long,
            )
            direct_score = (
                refined["pred_face_logits"][
                    batch_index,
                    face_index,
                ]
                .sigmoid()
                .amax(dim=-1)
            )
            keypoint_visibility = refined["pred_visibility_logits"][
                batch_index,
                pose_index,
            ].sigmoid()
            border = output["crop"].border_distances[head_index]
            valid_fraction = output["valid_fraction"][head_index]
            if not self.use_valid_pixel_and_border_inputs:
                border = torch.zeros_like(border)
                valid_fraction = torch.ones_like(valid_fraction)
            evidence = FusionEvidence(
                pose_uncertainty=output["pose_uncertainty"][head_index],
                geometry_uncertainty=output["geometry_uncertainty"][head_index],
                direct_ellipse_uncertainty=1.0 - direct_score,
                mean_keypoint_visibility=keypoint_visibility.mean(dim=-1),
                visible_keypoint_fraction=(keypoint_visibility >= 0.5)
                .to(keypoint_visibility)
                .mean(dim=-1),
                face_observability=output["face_observability"][head_index],
                truncation_ratio=1.0 - valid_fraction,
                minimum_border_distance=border.amin(dim=-1),
                head_confidence=output["crop"].head_confidence[head_index],
            )
            result = fusion(
                pose_embedding=output["pose_embedding"][head_index],
                direct_ellipse=refined["pred_face_ellipses"][
                    batch_index,
                    face_index,
                ],
                direct_keypoints=refined["pred_keypoints"][
                    batch_index,
                    pose_index,
                ],
                geometry_ellipse=output["amodal_ellipse"][head_index],
                geometry_keypoints=output["projected_keypoints"][head_index],
                evidence=evidence,
            )
            fused_ellipses[batch_index, face_index] = result["fused_ellipse"]
            fused_keypoints[batch_index, pose_index] = result["fused_keypoints"]
            rows.append(
                {
                    **result,
                    "head_indices": head_index,
                    "face_query_indices": face_index,
                    "pose_query_indices": pose_index,
                }
            )
        fused["pred_face_ellipses"] = fused_ellipses
        fused["pred_keypoints"] = fused_keypoints
        fused["head_pose_geometry_fusion"] = rows
        return fused


__all__ = ["HeadPoseGeometryExtension"]
