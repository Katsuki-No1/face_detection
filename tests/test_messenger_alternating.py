import numpy as np
import torch
from src.data.messenger_diverse_occlusion import make_occlusion, BANDS

def fixture():
    y,x=np.mgrid[:256,:256]
    return np.full((256,256,3),127,np.uint8), ((x-128)/98)**2+((y-128)/78)**2<=1, np.array([[.4,.4],[.6,.4],[.5,.5],[.4,.6],[.6,.6]])

def test_mask_truth_coverage_locations_and_determinism():
    rgb,face,xy=fixture(); locations=set()
    for v in range(9):
        image,mask,patch,out,point,meta=make_occlusion(rgb,face,xy,101,v)
        again=make_occlusion(rgb,face,xy,101,v)
        assert np.array_equal(image,again[0])
        assert np.array_equal(image[~mask],rgb[~mask])
        assert not np.any(mask & ~face)
        lo,hi=BANDS[v%3];assert lo-.0001<=meta['fraction']<=hi+.0001
        assert np.array_equal(patch,mask.reshape(16,16,16,16).any((1,3)))
        ix,iy=np.rint(xy*255).astype(int).T
        assert np.array_equal(point,mask[iy,ix])
        assert not (patch&out).any()
        locations.add(meta['location'])
    assert locations==set(range(9))

def test_mask_does_not_depend_on_target_coordinates():
    rgb,face,xy=fixture()
    a=make_occlusion(rgb,face,xy,31,4)
    b=make_occlusion(rgb,face,np.ones_like(xy)*.5,31,4)
    assert np.array_equal(a[1],b[1])

def test_alternating_freezing_and_mask_gradient():
    r=torch.nn.Conv2d(3,3,1);h=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,2))
    x=torch.randn(2,3,2,2);mask=torch.tensor([[[True,False],[False,True]]]*2)
    h.requires_grad_(False)
    prediction=h(torch.where(mask[:,None],r(x),x));prediction.square().mean().backward()
    assert r.weight.grad.abs().sum()>0 and all(p.grad is None for p in h.parameters())
    r.zero_grad(set_to_none=True);r.requires_grad_(False);h.requires_grad_(True)
    h(torch.where(mask[:,None],r(x),x).detach()).square().mean().backward()
    assert all(p.grad is None for p in r.parameters())
    assert h[1].weight.grad.abs().sum()>0
