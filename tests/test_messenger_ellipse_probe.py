import math
import torch
from src.models.messenger_ellipse_probe import ellipse_parts, FrozenEllipseProbe
from src.models.dinov3_deimv2_ellipse.official import EllipseDEIMCriterion

WEIGHTS = dict(center=5., axes=5., angle=1., chamfer=2.)


def test_matches_existing_baseline_loss_for_noncircular_targets():
    p = torch.tensor([[.4,.51,.25,.13,.3],[.5,.4,.3,.1,-.4]], requires_grad=True)
    t = torch.tensor([[.5,.5,.3,.15,.4],[.5,.5,.25,.12,-.2]])
    criterion = object.__new__(EllipseDEIMCriterion)
    criterion.ellipse_weights = WEIGHTS
    criterion.circle_ratio_threshold = 1.05
    criterion.contour_points = 36
    expected = criterion._ellipse_losses({'pred_ellipses':p[:,None]},
        [{'ellipses':x[None]} for x in t],[(torch.tensor([0]),torch.tensor([0]))]*2)
    loss, _, _ = ellipse_parts(p,t,WEIGHTS)
    torch.testing.assert_close(loss.mean(),sum(expected.values()),atol=2e-6,rtol=2e-5)
    loss.mean().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_geometry_penalizes_translation_axes_and_axial_angle():
    t = torch.tensor([[.5,.5,.3,.1,.2]])
    same = ellipse_parts(t,t,WEIGHTS)[0]
    assert same.item()<1e-6
    for slot,change in [(0,.1),(2,.08),(4,.3)]:
        p=t.clone();p[:,slot]+=change
        assert ellipse_parts(p,t,WEIGHTS)[0].item()>same.item()+.001
    p=t.clone();p[:,4]+=math.pi
    assert ellipse_parts(p,t,WEIGHTS)[0].item()<1e-5


def test_circular_angle_is_not_supervised():
    t=torch.tensor([[.5,.5,.2,.2,0.]])
    p=t.clone();p[:,4]=.8
    _,parts,eligible=ellipse_parts(p,t,WEIGHTS)
    assert not eligible.item() and parts[0,2]==0


def test_encoding_uses_detector_configured_layers_and_query_selection_has_no_gt():
    class Dino(torch.nn.Module):
        def forward_intermediate_feature_maps(self, x, *, layer_indexes):
            assert layer_indexes == (5,8,11)
            return tuple(x.new_full((len(x),4,16,16),i) for i in layer_indexes)
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.dino=Dino(); self.interaction_indexes=(5,8,11); self.use_sta=False
            self.convs=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
            self.norms=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        def _normalize(self,x):return x
    detector=torch.nn.Module();detector.backbone=Backbone();detector.encoder=torch.nn.Identity();detector.decoder=torch.nn.Identity()
    detector.category_to_label={2:0}
    probe=FrozenEllipseProbe(detector)
    maps,levels,detail=probe.encode(torch.zeros(1,3,256,256))
    assert maps[:,0].mean()==5 and maps[:,1].mean()==8 and maps[:,2].mean()==11
    assert [x.shape[-1] for x in levels]==[32,16] and detail.shape==(1,0,8,8)
    predictions=torch.arange(15.).reshape(1,3,5)
    ellipse,score,index=probe.select(dict(pred_logits=torch.tensor([[[-1.],[2.],[0.]]]),pred_ellipses=predictions))
    assert index.item()==1
    torch.testing.assert_close(ellipse,predictions[:,1])
