import numpy as np
import torch
from torch import nn
from scripts.run_messenger_position_primary import generate, endpoint_loss, balanced_mean, PLAN
from src.models.messenger_feature_diffusion import FeatureDiffusion


class Head(nn.Module):
    def forward(self,x,**kwargs):
        xy=x[:,:10].mean((2,3)).reshape(len(x),5,2).sigmoid()
        return {'decoded_coordinates':xy}


def test_full_sampler_every_step_receives_position_gradient():
    torch.set_num_threads(2);torch.manual_seed(42)
    net=FeatureDiffusion(torch.zeros(384),torch.ones(384))
    maps=torch.randn(1,3,384,4,4);ids=np.array([7]);outputs=[]
    original=net.denoise
    def observed(*args):
        value=original(*args);value[0].retain_grad();outputs.append(value[0]);return value
    net.denoise=observed
    result=generate(net,maps,ids)
    assert len(outputs)==16
    reference=net.sample(maps,ids,42,16)
    torch.testing.assert_close(result,reference)
    target=torch.zeros(1,5,2)
    loss=(Head()(result)['decoded_coordinates']-target).norm(dim=-1).mean()
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in outputs[:16])


def test_endpoint_never_sends_clean_target_to_head():
    class Model: pass
    model=Model();model.head=Head();seen=[]
    hook=model.head.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    x=torch.zeros(1,384,4,4);r=torch.ones_like(x,requires_grad=True);z=torch.full_like(x,7.)
    mask=torch.zeros(1,4,4,dtype=torch.bool);mask[:,:,:2]=True
    point=torch.tensor([[True,False,True,False,True]])
    loss,parts,err=endpoint_loss(model,r,x,z,mask,~mask,torch.zeros(1,5,2),point,torch.ones_like(point))
    assert len(seen)==1
    torch.testing.assert_close(seen[0],torch.where(mask[:,None],r.detach(),x))
    torch.testing.assert_close(loss,parts['position']+.01*parts['mse']+.01*parts['cosine'])
    loss.backward();assert r.grad is not None and r.grad.norm()>0
    hook.remove()


def test_position_balances_valid_strata_and_handles_empty():
    error=torch.tensor([1.,3.,10.]);a=torch.tensor([True,True,False]);b=~a
    assert balanced_mean(error,a,b)==6
    assert balanced_mean(error,a,torch.zeros_like(b))==2
    assert PLAN['clean_head'] is False
