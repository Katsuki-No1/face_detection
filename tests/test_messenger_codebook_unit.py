import pytest
import torch
from torch.nn import functional as F

from src.models.messenger_codebook_unit import ARMS, UnitVectorCodebook, make_recovery
from scripts import report_messenger_codebook_unit as report


def test_unit_lookup_decay_loss_and_position_gradient():
    codebook = UnitVectorCodebook(torch.tensor([[20.,0.],[0.,.4]]), 'gradient')
    latent = torch.tensor([[[[2.]],[[4.]]]], requires_grad=True)
    value, details = codebook(latent)
    torch.testing.assert_close(value.flatten(), torch.tensor([0.,1.]))
    torch.testing.assert_close(codebook(latent*30)[0], value)
    error = (F.normalize(latent,dim=1)-value.detach()).square().mean(1)
    torch.testing.assert_close(details['commitment'],error)
    torch.testing.assert_close(details['code_loss'],error)
    value[:,0].sum().backward()
    assert latent.grad.norm()>0 and codebook.embedding.grad[1].norm()>0
    assert codebook.embedding.grad[0].norm()==0
    optimizer=torch.optim.AdamW(codebook.parameters(),lr=.1)
    optimizer.step(); codebook.end_step()
    torch.testing.assert_close(codebook.embedding.norm(dim=1),torch.ones(2))
    assert codebook.embedding[1,0]!=0


def test_ema_uses_directions_and_is_frozen_during_evaluation():
    centers=torch.tensor([[5.,0.],[-2.,0.]])
    codebook=UnitVectorCodebook(centers,'ema',decay=.5)
    latent=torch.tensor([[[[4.,3.]],[[3.,4.]]]],requires_grad=True)
    first=codebook.embedding.detach().clone()
    output,details=codebook(latent)
    output[:,1].sum().backward()
    assert latent.grad.norm()>0 and codebook.embedding.grad is None
    torch.testing.assert_close(codebook.embedding,first)
    codebook.end_step()
    target=F.normalize(F.normalize(latent.detach(),dim=1).mean((0,2,3)),dim=0)
    torch.testing.assert_close(codebook.embedding[0],F.normalize((first[0]+target)*.5,dim=0))
    torch.testing.assert_close(codebook.embedding[1],first[1])
    assert int(codebook.ema_steps)==1
    before=codebook.embedding.detach().clone()
    codebook.eval(); codebook(latent*7);codebook.end_step()
    torch.testing.assert_close(codebook.embedding,before)
    assert int(codebook.ema_steps)==1


@pytest.mark.parametrize('width',[16,32])
def test_E_initial_output_equal_to_D_and_cleanup_scope(width,tmp_path):
    torch.set_num_threads(2)
    basis=torch.eye(384)[:,:width];mean=torch.zeros(384);codes=torch.randn(4096,width)
    maps=torch.randn(1,3,384,4,4)
    a=make_recovery(f'D_d{width}_k4096',basis,mean,codes).eval()
    b=make_recovery(f'E_d{width}_k4096',basis,mean,codes).eval()
    with torch.no_grad():torch.testing.assert_close(a(maps)[0],b(maps)[0],rtol=0,atol=0)
    assert len(ARMS)==12
    assert len(report.weight_paths(tmp_path))==28
