import json
import sys
import numpy as np
import pytest
import torch
from scripts import gate_messenger_codebook as gates
from scripts import run_messenger_codebook as run


def test_quality_gate_requires_finite_improvement_in_correct_objective():
    baseline=dict(position=.1,objective=.2)
    assert not gates.improve([baseline,dict(position=.1,objective=.1)],'R')
    assert gates.improve([baseline,dict(position=.09,objective=.19)],'R',True)
    assert not gates.improve([baseline,dict(position=.09,objective=.21)],'R',True)
    assert not gates.improve([baseline,dict(position=float('nan'),objective=.1)],'R')


def proof(tmp_path,monkeypatch):
    monkeypatch.setattr(gates,'OUT',tmp_path)
    run.base.write(tmp_path/'plan.json',dict(gate_plan=gates.GATE_PLAN,training_plan=run.PLAN,hashes={},
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__)))
    run.base.write(tmp_path/'initialization.json',dict(code_sha256={'4096':'a','8192':'b'}))
    for arm in gates.ARMS:
        for phase in gates.PHASES:
            folder=tmp_path/arm/phase
            run.base.write(folder/'report.json',dict(passed=True))
            run.base.write(folder/'success.json',dict(passed=True,context='same',report_sha256=run.base.sha(folder/'report.json')))
    hashes={str(p.relative_to(tmp_path)):run.base.sha(p) for p in tmp_path.rglob('*.json')}
    run.base.write(tmp_path/'completion.json',dict(passed=True,context='same',artifacts=hashes))


def test_authentication_requires_all_24_markers(tmp_path,monkeypatch):
    proof(tmp_path,monkeypatch)
    assert gates.authenticate()['passed']
    (tmp_path/gates.ARMS[-1]/'small_scale/success.json').unlink()
    with pytest.raises(FileNotFoundError):
        gates.authenticate()


def test_authentication_rejects_modified_evidence(tmp_path,monkeypatch):
    proof(tmp_path,monkeypatch)
    run.base.write(tmp_path/gates.ARMS[0]/'overfit/report.json',dict(passed=False))
    with pytest.raises(ValueError,match='evidence changed'):
        gates.authenticate()


def test_authentication_requires_identical_code_initialization(tmp_path,monkeypatch):
    proof(tmp_path,monkeypatch)
    assert gates.authenticate(code_hashes={'4096':'a','8192':'b'})['passed']
    with pytest.raises(ValueError,match='initial dictionaries'):
        gates.authenticate(code_hashes={'4096':'wrong','8192':'b'})


def test_subset_stays_inside_role():
    cache=dict(parent=np.repeat(np.arange(9),3),syn_fit=np.arange(12),syn_dev=np.arange(12,21))
    np.testing.assert_array_equal(gates.subset(cache,'fit',2),np.arange(6))
    np.testing.assert_array_equal(gates.subset(cache,'dev',2),np.arange(12,18))


def test_shared_epoch_accumulates_position_over_effective_batch(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(run.base,'gpu',lambda value:value)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.ones(1))
        def forward(self,maps):
            return maps[:,2]+self.bias,{}
    def head(_,features):
        return {'decoded_coordinates':features[:,:10].mean((2,3)).reshape(-1,5,2)}
    monkeypatch.setattr(run.base,'run_head',head)
    torch.manual_seed(42)
    n=16
    cache=dict(syn=torch.randn(n,3,384,4,4),clean=torch.randn(n,3,384,4,4),parent=np.arange(n),
       cov=torch.ones(n,4,4,dtype=torch.bool),ext=torch.zeros(n,4,4,dtype=torch.bool),
       point=torch.rand(n,5)>.6,state=torch.full((n,5),2),xy=torch.zeros(n,5,2))
    basis=torch.eye(384)[:,:16]
    a=Net();b=Net();opt_a=torch.optim.SGD(a.parameters(),lr=.01);opt_b=torch.optim.SGD(b.parameters(),lr=.01)
    run.train_epoch(a,SimpleNamespace(head=None),cache,basis,'continuous16',opt_a,np.arange(n))
    r,details=b(cache['syn'])
    parts,_=run.loss_parts(SimpleNamespace(head=None),r,cache['syn'][:,2],cache['clean'][:,2],
        cache['cov'],cache['ext'],cache['xy'],cache['point'],cache['state']>0,basis,'continuous16',details)
    parts['objective'].backward();torch.nn.utils.clip_grad_norm_(b.parameters(),5);opt_b.step()
    torch.testing.assert_close(a.bias,b.bias,rtol=1e-6,atol=1e-6)
