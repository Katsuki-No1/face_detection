import json
import numpy as np
import pytest
import torch
from scripts.run_messenger_multiaxis_convergence import Plateau,save,load_checkpoint,checkpoint,PLAN


def test_plateau_requires_minimum_lr_and_cooldown():
    c=Plateau(lr=1e-4,floor=2.5e-5,minimum=20,reduce_patience=3,stop_patience=5)
    done=[]
    for epoch in range(21):done.append(c.step(epoch,[1.,2.]))
    assert not any(done[:20]) and done[20]
    assert c.lr==2.5e-5 and c.last_reduction==6


def test_either_objective_improving_prevents_convergence():
    c=Plateau(lr=1e-6,floor=1e-6,minimum=2,reduce_patience=2,stop_patience=3)
    for epoch in range(30):assert not c.step(epoch,[1.,2.-.01*epoch])
    assert c.last_improvement==29


def test_bounded_noise_is_not_meaningful_progress():
    c=Plateau(lr=1e-6,floor=1e-6,minimum=2,stop_patience=5)
    for epoch in range(5):assert not c.step(epoch,[1.-epoch*.00001])
    assert c.step(5,[1.-5*.00001])


def test_nonfinite_and_changed_objective_rejected():
    c=Plateau(1e-4,1e-6,20)
    c.step(0,[1.,2.])
    with pytest.raises(ValueError):c.step(1,[float('nan'),2.])
    with pytest.raises(ValueError):c.step(1,[1.])


def test_resume_reproduces_optimizer_rng_and_controller(tmp_path):
    torch.manual_seed(42);net=torch.nn.Linear(3,2);opt=torch.optim.AdamW(net.parameters(),lr=.001)
    rng=np.random.default_rng(42);c=Plateau(.001,.0001,10)
    x=torch.randn(5,3)
    def step(model,optimizer,generator):
        order=generator.permutation(5);loss=model(x[order]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    step(net,opt,rng);c.step(0,[1.]);save(tmp_path/'last.pt',checkpoint(net,opt,0,c,rng,'identity'))
    step(net,opt,rng)
    ck=load_checkpoint(tmp_path/'last.pt','identity');other=torch.nn.Linear(3,2);other.load_state_dict(ck['state'])
    oo=torch.optim.AdamW(other.parameters(),lr=.001);oo.load_state_dict(ck['optimizer'])
    rr=np.random.default_rng();rr.bit_generator.state=ck['rng'];step(other,oo,rr)
    for a,b in zip(net.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert Plateau(**ck['controller']).last_improvement==0
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'last.pt','wrong')


def test_no_per_arm_checkpoint_deletion_in_successor():
    from pathlib import Path
    import scripts.run_messenger_multiaxis_convergence as runner
    source=Path(runner.__file__).read_text()
    assert 'prune_checkpoints(' not in source and '.unlink(' not in source
    assert PLAN['epochs']>PLAN['R_min_epochs'] and PLAN['gate_epochs']>PLAN['g_min_epochs']
