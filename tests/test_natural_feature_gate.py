import numpy as np
import torch

from keypoint_comparison1.landmark_detection.geometry import build_crop_transform
from src.data.natural_gate_teacher import patch_teacher
from src.models.natural_feature_gate import NaturalFeatureGate, supervised_losses


def test_visible_nose_opens_valley_on_occluded_eye_mouth_edge():
    # Long axis vertical, with nose laterally displaced from the two occluded points.
    ellipse = [127.5,127.5,120.,110.,np.pi/2]
    crop = build_crop_transform([0,0,255,255],output_size=256,scale_factor=1)
    xy = np.array([[80.,70.],[168.,70.],[105.,128.],[80.,190.],[168.,190.]])
    target, weight = patch_teacher(ellipse,xy,np.array([2,1,2,2,1]),crop)
    assert target[4,10] > .9 and target[11,10] > .9
    assert target[8,10] < .15 and weight[8,10] > .8
    unknown, confidence = patch_teacher(ellipse,xy,np.zeros(5),crop)
    assert np.all(confidence == 0) and np.isfinite(unknown).all()


def test_coordinate_and_soft_label_gradients_reach_same_logit():
    torch.manual_seed(42)
    x=torch.randn(2,4,3,3)
    r=torch.randn_like(x)
    model=NaturalFeatureGate(4)
    logits,fused=model(x,r,torch.full((1,4,1,1),.5))
    head=torch.nn.Linear(4,10).requires_grad_(False)
    coordinates=head(fused.mean((2,3))).reshape(2,5,2).sigmoid()
    target=torch.full((2,3,3),.38)
    truth=torch.zeros_like(coordinates)
    ce,position=supervised_losses(logits,coordinates,target,torch.ones_like(target),truth,torch.ones(2,5))
    for loss in [ce,position]:
        grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert all(p.grad is None for p in head.parameters())
