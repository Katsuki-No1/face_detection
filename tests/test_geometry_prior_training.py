import copy

import pytest
import torch

from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case
from keypoint_comparison1.landmark_detection.geometry_prior_recovery import ConditionalFeatureDiscriminator
from keypoint_comparison1.landmark_detection.geometry_prior_training import PriorUpdateRuntime, OptimizerCommitError


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_case(adversarial=True,anomalous=True):
    model,images,geometry,support,truth,criterion=model_case(anomalous)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    d=ConditionalFeatureDiscriminator(24,hidden=8) if adversarial else None
    dopt=torch.optim.AdamW(d.parameters(),lr=.001) if d is not None else None
    runtime=PriorUpdateRuntime(model,criterion,opt,contract={'test':'paired-v1'},
        discriminator=d,discriminator_optimizer=dopt,adversarial_weight=.01 if adversarial else 0.,amp=False)
    return runtime,(images,images,geometry,support,truth)


def same_state(left,right):
    assert set(left)==set(right)
    for key in left:
        if torch.is_tensor(left[key]):torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        else:assert left[key]==right[key]


@pytest.mark.parametrize('adversarial',[False,True])
def test_repeated_updates_and_exact_checkpoint_resume(adversarial):
    runtime,batch=make_case(adversarial)
    generator=torch.Generator().manual_seed(123)
    scheduler=torch.optim.lr_scheduler.StepLR(runtime.optimizer,step_size=1,gamma=.9)
    observer=copy.deepcopy(runtime.model.observer.state_dict())
    assert runtime.update_window(*batch,microbatch=1)['status']=='updated'
    scheduler.step()
    state=copy.deepcopy(runtime.checkpoint(epoch=1,schedulers={'g':scheduler},loader_generator=generator))
    continued=runtime.update_window(*batch,microbatch=1)
    scheduler.step()
    other,_=make_case(adversarial)
    other_scheduler=torch.optim.lr_scheduler.StepLR(other.optimizer,step_size=1,gamma=.9)
    other_generator=torch.Generator().manual_seed(456)
    assert other.restore(state,schedulers={'g':other_scheduler},loader_generator=other_generator)==1
    resumed=other.update_window(*batch,microbatch=1)
    other_scheduler.step()
    assert continued==resumed
    same_state(runtime.model.state_dict(),other.model.state_dict())
    same_state(runtime.ema.module.state_dict(),other.ema.module.state_dict())
    same_state(runtime.model.observer.state_dict(),observer)
    same_state(runtime.ema.module.observer.state_dict(),observer)
    same_state(runtime.ema.module.bank.state_dict(),runtime.model.bank.state_dict())
    if adversarial:same_state(runtime.discriminator.state_dict(),other.discriminator.state_dict())
    assert torch.equal(generator.get_state(),other_generator.get_state())
    assert other.ema.updates==2 and not other.model.observer.training


def test_empty_anomaly_window_never_updates_discriminator_or_weight_decay():
    runtime,batch=make_case(anomalous=False)
    before=copy.deepcopy(runtime.discriminator.state_dict())
    result=runtime.update_window(*batch,microbatch=1)
    assert result['status']=='updated' and not result['discriminator_updated']
    same_state(before,runtime.discriminator.state_dict())
    assert not runtime.d_optimizer.state


def test_nonfinite_d_gradient_updates_neither_optimizer():
    runtime,batch=make_case()
    g=copy.deepcopy(runtime.model.state_dict());d=copy.deepcopy(runtime.discriminator.state_dict())
    handle=next(runtime.discriminator.parameters()).register_hook(lambda x:torch.full_like(x,float('nan')))
    with pytest.raises(FloatingPointError,match='Non-finite gradients'):
        runtime.update_window(*batch,microbatch=1)
    handle.remove()
    same_state(g,runtime.model.state_dict());same_state(d,runtime.discriminator.state_dict())
    assert runtime.step==0 and runtime.ema.updates==0
    assert all(p.grad is None for p in runtime.model.parameters())


def test_oom_before_commit_replays_smaller_microbatch(monkeypatch):
    runtime,batch=make_case()
    original=runtime.model.forward
    def limited(images,**kwargs):
        if len(images)>1:raise torch.OutOfMemoryError('injected activation OOM')
        return original(images,**kwargs)
    monkeypatch.setattr(runtime.model,'forward',limited)
    result=runtime.update_window(*batch,microbatch=2)
    assert result['status']=='updated' and result['microbatch']==1 and result['effective_batch']==2
    assert result['attempts']==[dict(microbatch=2,error='oom_before_optimizer_commit')]


def test_commit_error_requires_genuine_checkpoint_restore(monkeypatch):
    runtime,batch=make_case()
    state=copy.deepcopy(runtime.checkpoint(epoch=0))
    original=runtime.d_optimizer.step
    def fail(*a,**kw):raise torch.OutOfMemoryError('injected commit OOM')
    monkeypatch.setattr(runtime.d_optimizer,'step',fail)
    with pytest.raises(OptimizerCommitError,match='commit'):
        runtime.update_window(*batch,microbatch=1)
    with pytest.raises(RuntimeError,match='incomplete'):
        runtime.checkpoint(epoch=0)
    with pytest.raises(OptimizerCommitError,match='Restore'):
        runtime.update_window(*batch,microbatch=1)
    monkeypatch.setattr(runtime.d_optimizer,'step',original)
    runtime.restore(state)
    assert runtime.update_window(*batch,microbatch=1)['status']=='updated'


def test_resume_contract_and_optimizer_ownership_fail_closed():
    runtime,_=make_case()
    state=copy.deepcopy(runtime.checkpoint(epoch=0));state['contract']['experiment']['test']='different'
    with pytest.raises(ValueError,match='contract'):runtime.restore(state)
    with pytest.raises(ValueError,match='G optimizer'):
        PriorUpdateRuntime(runtime.model,runtime.criterion,torch.optim.Adam(runtime.model.base.head.parameters()),contract={},amp=False)


def test_observer_remains_float32_inside_student_autocast():
    runtime,batch=make_case(False)
    observed=[]
    handle=runtime.model.observer.register_forward_hook(lambda m,a,out:observed.append(out.dtype))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=runtime.model(batch[0],geometry=batch[2],support=batch[3])
    handle.remove()
    assert observed==[torch.float32]
    assert output['prior_diagnostics']['observation'].dtype==torch.float32
