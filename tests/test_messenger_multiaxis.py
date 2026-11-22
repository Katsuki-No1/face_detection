import torch
import pytest
from src.models.messenger_multiaxis import MultiAxisRecovery, reconstruction_loss, branch_specs, region_mean


@pytest.mark.parametrize('family,n', [('scale',3),('level',3),('level_scale',9)])
def test_branches_gradients_shapes_and_fusion(family,n):
    torch.set_num_threads(2)
    model=MultiAxisRecovery(family,channels=12,width=24)
    x=torch.randn(2,3,12,4,4,requires_grad=True)
    y,r,w=model(x)
    assert r.shape==(2,n,12,4,4)
    torch.testing.assert_close(w.sum(1),torch.ones_like(w[:,0]))
    torch.testing.assert_close(y,(r*w).sum(1))
    loss,_=reconstruction_loss((y,r,w),torch.randn_like(y),torch.ones(2,4,4),torch.zeros(2,4,4),'C')
    loss.backward()
    for branch in model.branches:
        assert branch.patch.weight.grad.abs().sum()>0
    assert torch.isfinite(x.grad).all()


def test_same_initialization_and_loss_contract():
    torch.set_num_threads(2)
    a=MultiAxisRecovery('scale',12,24)
    b=MultiAxisRecovery('level_scale',12,24)
    for x,y in zip(a.branches[0].parameters(),b.branches[6].parameters()):
        torch.testing.assert_close(x,y)
    o=a(torch.randn(2,3,12,4,4));z=torch.randn_like(o[0]);cov=torch.ones(2,4,4);ext=torch.zeros_like(cov)
    la,d=reconstruction_loss(o,z,cov,ext,'A')
    lb,_=reconstruction_loss(o,z,cov,ext,'B')
    lc,_=reconstruction_loss(o,z,cov,ext,'C')
    torch.testing.assert_close(lb-la,d['branch'])
    torch.testing.assert_close(lc-lb,.1*d['consistency'])
    pairs=[(o[1][:,i]-o[1][:,j]).square().mean(1) for i in range(3) for j in range(i+1,3)]
    torch.testing.assert_close(d['consistency'],region_mean(torch.stack(pairs).mean(0),cov,ext))


def test_target_cell_is_available():
    torch.set_num_threads(2)
    model=MultiAxisRecovery('scale',12,24)
    x=torch.randn(1,3,12,4,4,requires_grad=True)
    # Factor-four branch has one memory token containing the target cell.
    y,_=model.branches[-1](x[:,2])
    y[0,:,0,0].square().sum().backward()
    assert x.grad[0,2,:,0,0].abs().sum()>0


def test_retention_only_after_evaluation(tmp_path,monkeypatch):
    import scripts.run_messenger_multiaxis as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    folder=tmp_path/'arm';folder.mkdir()
    p=folder/'best_recovery.pt';p.write_bytes(b'test checkpoint')
    with pytest.raises(RuntimeError):runner.prune_checkpoints(folder)
    assert p.exists()
    (folder/'evaluation.json').write_text('{}')
    (folder/'natural_points.npz').write_bytes(b'finished')
    other=folder/'parent.pt';other.write_bytes(b'keep')
    runner.prune_checkpoints(folder)
    assert not p.exists() and other.exists()
    assert 'completed' in (folder/'checkpoint_deletion.json').read_text()
