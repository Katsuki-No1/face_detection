import torch
from src.models.messenger_feature_diffusion import FeatureDiffusion,coefficients,noise_for


def test_v_parameterization_roundtrip():
    torch.manual_seed(7);z=torch.randn(4,384,4,4);noise=torch.randn_like(z)
    t=torch.tensor([0.,.25,.8,1.]);a,s=coefficients(t)
    xt=a*z+s*noise;v=a*noise-s*z
    torch.testing.assert_close(a*xt-s*v,z,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(s*xt+a*v,noise,atol=1e-6,rtol=1e-5)


def test_sampling_noise_identity_and_seed():
    together=noise_for([3,8],(384,4,4),'cpu',42)
    torch.testing.assert_close(together[1],noise_for([8],(384,4,4),'cpu',42)[0],atol=0,rtol=0)
    assert not torch.equal(together,noise_for([3,8],(384,4,4),'cpu',43))


def test_oracle_v_sampler_recovers_target():
    net=FeatureDiffusion(torch.zeros(384),torch.ones(384));target=torch.randn(2,384,4,4)
    def oracle(maps,xt,t):
        a,s=coefficients(t);v=(a*xt-target)/s
        return v,None,None
    net.denoise=oracle
    out=net.sample(torch.zeros(2,3,384,4,4),[1,2],steps=4)
    torch.testing.assert_close(out,target,atol=2e-6,rtol=1e-5)


def test_training_has_noisy_condition_and_memory_gradients():
    torch.set_num_threads(2);torch.manual_seed(42)
    net=FeatureDiffusion(torch.zeros(384),torch.ones(384))
    maps=torch.randn(2,3,384,4,4);clean=torch.randn(2,384,4,4)
    (v,branches,_),target,r=net.training_pair(maps,clean,torch.tensor([.3,.8]),torch.randn_like(clean))
    assert v.shape==clean.shape and branches.shape==(2,3,384,4,4)
    ((v-target).square().mean()+r.square().mean()).backward()
    for parameter in [net.noisy.weight,net.time[0].weight,*[b.patch.weight for b in net.branches]]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0


def test_sampler_reproducible_batching():
    torch.set_num_threads(2);torch.manual_seed(42)
    net=FeatureDiffusion(torch.zeros(384),torch.ones(384)).eval();maps=torch.randn(2,3,384,4,4)
    with torch.no_grad():
        together=net.sample(maps,[3,8],steps=2)
        separate=net.sample(maps[1:],[8],steps=2)
    torch.testing.assert_close(together[1:],separate,atol=2e-5,rtol=2e-5)
