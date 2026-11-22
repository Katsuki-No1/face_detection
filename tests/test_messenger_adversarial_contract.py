import copy
import torch
from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case
from scripts import messenger_geometry_adversarial_contract as contract


def test_separate_discriminator_rng_and_two_optimizer_resume(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(contract.D_CONTRACT,'channels',24)
    monkeypatch.setitem(contract.D_CONTRACT,'hidden',8)
    def make():
        model,images,geometry,support,truth,criterion=model_case(True)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        before=torch.get_rng_state().clone()
        runtime=contract.AdversarialRuntime(model,criterion,optimizer,contract={'test':'selected'},amp=False)
        assert torch.equal(before,torch.get_rng_state())
        schedulers={name:torch.optim.lr_scheduler.StepLR(opt,1,.9) for name,opt in [('g',optimizer),('d',runtime.d_optimizer)]}
        return runtime,schedulers,(images,images,geometry,support,truth)
    runtime,schedulers,batch=make()
    assert runtime.update_window(*batch,microbatch=1)['discriminator_updated']
    for scheduler in schedulers.values():scheduler.step()
    state=copy.deepcopy(runtime.checkpoint(epoch=1,schedulers=schedulers))
    expected=runtime.update_window(*batch,microbatch=1)
    other,other_schedulers,_=make()
    assert other.restore(state,schedulers=other_schedulers)==1
    assert other.update_window(*batch,microbatch=1)==expected
    for key,value in runtime.discriminator.state_dict().items():
        torch.testing.assert_close(value,other.discriminator.state_dict()[key],rtol=0,atol=0)
    assert other.d_optimizer.state and other.optimizer.state
