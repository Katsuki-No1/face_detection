"""New model identity: non-cascade full-image FACE bbox[/ellipse]."""
import torch
from torch import nn

from src.models.co_dino.model import DINOv3CoDINODetector, _sum_loss_value
from src.models.dino_backbone import build_dino_backbone
from .auxiliary import OfficialAuxiliaryHeads
from .cooperative import build_cooperative_targets
from .query_head import OfficialQueryHead
from .sfp import OfficialSFP
from .provenance import ARCHITECTURE_VERSION


class SingleMapBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        if int(config.get("patch_size", 16)) != 16:
            raise ValueError("SFP contract requires stride-16 patch tokens")
        self.dino = build_dino_backbone(config)
        # Dataset tensors are RGB [0,1], as in the existing STA path. Preserve
        # the DINO ImageNet normalization when replacing STA with SFP.
        self.register_buffer("image_mean",torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer("image_std",torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)

    def forward(self, images):
        images = (images-self.image_mean.to(images))/self.image_std.to(images)
        return (self.dino(images),)


class OfficialCoDINODetector(DINOv3CoDINODetector):
    architecture_version = ARCHITECTURE_VERSION

    def forward_raw(self, images, targets=None, *, padding_mask=None):
        mask = self._resolve_padding_mask(images, targets, padding_mask)
        features = self.extract_features(images)
        masks = self._feature_masks(mask, features)
        samples = self._build_data_samples(images, targets) if targets is not None else None
        raw = self.query_head(features, masks, batch_data_samples=samples)
        memory = raw["_memory"]
        encoded, cursor = [], 0
        for feature in features:
            b, c, h, w = feature.shape
            encoded.append(memory[:, cursor:cursor+h*w].transpose(1,2).reshape(b,c,h,w))
            cursor += h*w
        encoded.append(raw["encoded_p6"])
        raw.update(_batch_data_samples=samples, _features=features,
                   _encoded_features=tuple(encoded), _feature_masks=masks)
        raw["_o2m_losses"], raw["_cooperative_targets"] = {}, []
        if self.training and samples is not None and self.o2m_head is not None:
            losses, coordinate_rows = self.o2m_head.loss_and_coordinates(tuple(encoded), samples)
            raw["_o2m_losses"] = losses
            raw["_cooperative_targets"] = [build_cooperative_targets(
                rows, samples, self.query_head.num_classes) for rows in coordinate_rows]
        return raw

    def loss(self, images, targets, *, padding_mask=None):
        raw = self.forward_raw(images, targets, padding_mask=padding_mask)
        samples = raw["_batch_data_samples"]
        losses = self.query_head.loss(raw, samples)
        losses.pop("loss_total", None)
        losses.update({f"o2m.{k}": _sum_loss_value(v) for k,v in raw["_o2m_losses"].items()})
        for index, targets in enumerate(raw["_cooperative_targets"]):
            losses.update(self.query_head.cooperative_losses(raw, targets, samples, index))
        losses["loss_total"] = sum(losses.values())
        # Trainer reporting aliases must not be counted a second time in total.
        losses["loss_bbox_l1"] = losses["loss_bbox"]
        ellipses = [v for k,v in losses.items() if k.startswith("loss_ellipse_")]
        if ellipses:
            losses["loss_ellipse"] = sum(ellipses)
        return losses


def build_official_codino(config, *, backbone_module=None):
    model = config["model"]
    query = model["query_head"]
    dims = int(query.get("embed_dims", 256))
    if int(query.get("num_feature_levels", 5)) != 5:
        raise ValueError("Official SFP Co-DINO requires five transformer scales")
    backbone = backbone_module if backbone_module is not None else SingleMapBackbone(model["backbone"])
    neck = OfficialSFP(int(model["backbone"].get("embed_dim",384)), dims,
                       bool(model.get("neck",{}).get("use_checkpoint",False)))
    head = OfficialQueryHead(num_classes=int(model.get("num_classes",1)), embed_dims=dims,
        num_feature_levels=5, num_queries=int(query.get("num_queries",300)),
        num_encoder_layers=int(query.get("num_encoder_layers",6)),
        num_decoder_layers=int(query.get("num_decoder_layers",6)),
        feedforward_channels=int(query.get("feedforward_channels",2048)),
        num_denoising_queries=int(query.get("num_denoising_queries",100)),
        ellipse=model.get("ellipse_head",{}))
    auxiliary = OfficialAuxiliaryHeads(dims, head.num_classes) if model.get("o2m_head",{}).get("enabled",True) else None
    inference = config.get("inference",{})
    detector = OfficialCoDINODetector(backbone=backbone, neck=neck, query_head=head,
        o2m_head=auxiliary, category_ids=config.get("data",{}).get("category_ids",[1]),
        score_threshold=float(inference.get("score_thr",.001)),
        detections_per_image=int(inference.get("detections_per_image",100)))
    if model.get("init_checkpoint"):
        from .transfer import load_wider_checkpoint
        detector.checkpoint_transfer_report = load_wider_checkpoint(detector, model["init_checkpoint"])
    return detector
