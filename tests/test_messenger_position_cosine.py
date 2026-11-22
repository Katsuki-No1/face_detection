import inspect
import json
import numpy as np
import torch
from torch import nn
import scripts.run_messenger_position_primary as previous
import scripts.run_messenger_position_cosine as current
from src.models.messenger_feature_diffusion import FeatureDiffusion


class Head(nn.Module):
    def forward(self,x,**kwargs):
        return {'decoded_coordinates':x[:,:10].mean((2,3)).reshape(len(x),5,2).sigmoid()}


def test_only_training_plan_mse_differs_and_gate_is_unchanged():
    a=json.loads(json.dumps(previous.PLAN));b=json.loads(json.dumps(current.PLAN))
    assert {k for k in a if a[k]!=b[k]}=={'mse_weight'}
    assert a['mse_weight']==.01 and b['mse_weight']==0
    for name in ('generate','balanced_mean','train_gate','gate_pass','candidates'):
        assert inspect.getsource(getattr(previous,name))==inspect.getsource(getattr(current,name))


def test_loss_and_gradient_remove_exactly_mse():
    torch.manual_seed(4)
    class Model:pass
    model=Model();model.head=Head();x=torch.randn(1,384,4,4);z=torch.randn_like(x);r=torch.randn_like(x,requires_grad=True)
    mask=torch.zeros(1,4,4,dtype=torch.bool);mask[:,:,:2]=True
    point=torch.tensor([[True,False,True,False,True]]);valid=torch.ones_like(point);xy=torch.zeros(1,5,2)
    args=(model,r,x,z,mask,~mask,xy,point,valid)
    old,parts,_=previous.endpoint_loss(*args);new,now,_=current.endpoint_loss(*args)
    torch.testing.assert_close(new,now['position']+.01*now['cosine'])
    torch.testing.assert_close(old-new,.01*parts['mse'])
    go=torch.autograd.grad(old,r,retain_graph=True)[0];gn=torch.autograd.grad(new,r,retain_graph=True)[0]
    gm=torch.autograd.grad(.01*parts['mse'],r)[0]
    torch.testing.assert_close(go-gn,gm,atol=1e-7,rtol=1e-4)


def test_full_sixteen_step_position_gradient_still_reaches_first_step():
    torch.set_num_threads(2);torch.manual_seed(42)
    net=FeatureDiffusion(torch.zeros(384),torch.ones(384));outputs=[];original=net.denoise
    def observed(*args):
        value=original(*args);value[0].retain_grad();outputs.append(value[0]);return value
    net.denoise=observed
    result=current.generate(net,torch.randn(1,3,384,4,4),np.array([7]))
    Head()(result)['decoded_coordinates'].norm(dim=-1).mean().backward()
    assert len(outputs)==16
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in outputs)
