import copy

import pytest
import torch
from torch.utils.data import DataLoader

from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case
from keypoint_comparison1.landmark_detection.dataset import collate_landmark_batch
from keypoint_comparison1.landmark_detection.geometry_prior_training import PriorUpdateRuntime
from keypoint_comparison1.landmark_detection.geometry_prior_epochs import PriorEpochDriver,save_epoch,finalize_checkpoints,METRIC
from keypoint_comparison1.landmark_detection.training import _scheduler


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(max_epochs=4,patience=2):
    torch.manual_seed(42)
    model,images,geometry,support,truth,criterion=model_case(True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    runtime=PriorUpdateRuntime(model,criterion,optimizer,contract={'test':'epoch'},amp=False)
    samples=[]
    for i in [0,1,0]:
        target=copy.deepcopy(truth[i]);target.update(clean_image=images[i],geometry_prior=dict(geometry=geometry[i],support=support[i]))
        samples.append((images[i],target))
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(samples,batch_size=2,shuffle=True,generator=generator,collate_fn=collate_landmark_batch)
    scheduler=_scheduler(optimizer,total_steps=max_epochs*2,warmup_steps=2,minimum_ratio=.01)
    driver=PriorEpochDriver(runtime,schedulers={'g':scheduler},loader_generator=generator,effective_batch=2,
        microbatch=2,freeze_backbone_epochs=1,mixing_warmup_epochs=1,minimum_epoch=2,patience=patience,maximum_epochs=max_epochs)
    return driver,loader


def validate(_):return {METRIC:.5,'eligible_face_crops':2}


def test_staged_freeze_partial_window_and_exact_epoch_resume():
    driver,loader=case()
    before=copy.deepcopy(driver.runtime.model.base.backbone.state_dict())
    first=driver.run_epoch(loader,validate)
    assert first['backbone_frozen'] and first['train_seen']==3 and driver.microbatch==2
    assert not driver.runtime.model.base.backbone.training
    for k,v in before.items():torch.testing.assert_close(v,driver.runtime.model.base.backbone.state_dict()[k],rtol=0,atol=0)
    state=copy.deepcopy(driver.checkpoint())
    second=driver.run_epoch(loader,validate)
    expected=copy.deepcopy(driver.runtime.model.state_dict())
    restored,other_loader=case();restored.restore(state)
    actual=restored.run_epoch(other_loader,validate)
    assert not actual['backbone_frozen'] and actual['step']==second['step']
    for k,v in expected.items():
        if torch.is_tensor(v):torch.testing.assert_close(v,restored.runtime.model.state_dict()[k],rtol=0,atol=0)
        else:assert v==restored.runtime.model.state_dict()[k]
    assert driver.schedulers['g'].get_last_lr()==restored.schedulers['g'].get_last_lr()
    restored.run_epoch(other_loader,validate)
    assert restored.should_stop and restored.epoch==3
    with pytest.raises(RuntimeError,match='stopping boundary'):restored.run_epoch(other_loader,validate)


def test_retention_and_resume_contract(tmp_path):
    driver,loader=case(max_epochs=10,patience=30)
    for epoch in range(1,11):
        driver.run_epoch(loader,validate);save_epoch(driver,tmp_path)
        snapshots=list(tmp_path.glob('epoch_*.pt'))
        assert len(snapshots)<=1
        if epoch==6:assert snapshots[0].name=='epoch_005.pt'
    assert list(tmp_path.glob('epoch_*.pt'))[0].name=='epoch_010.pt'
    finalize_checkpoints(driver,tmp_path)
    assert {p.name for p in tmp_path.iterdir()}=={'last.pt','best.pt'}
    payload=torch.load(tmp_path/'last.pt',weights_only=False)
    other,_=case(max_epochs=9)
    with pytest.raises(ValueError,match='contract'):other.restore(payload)


def test_metric_failure_does_not_complete_epoch_or_allow_save():
    driver,loader=case()
    with pytest.raises(ValueError,match='metric'):
        driver.run_epoch(loader,lambda _: {METRIC:float('nan'),'eligible_face_crops':2})
    assert driver.epoch==0
    with pytest.raises(RuntimeError,match='Restore a complete checkpoint'):
        driver.run_epoch(loader,validate)
    with pytest.raises(RuntimeError,match='validated epoch'):driver.checkpoint()


def test_warmup_immediately_decays_and_missing_logger_rejected():
    from keypoint_comparison1.landmark_detection.geometry_prior_epochs import wandb_epoch_logger
    driver,_=case()
    rates=[]
    for _ in range(4):
        driver.runtime.optimizer.step();driver.schedulers['g'].step();rates.append(driver.schedulers['g'].get_last_lr()[0])
    assert rates[1]>rates[0] and rates[2]<rates[1] and rates[3]<rates[2]
    with pytest.raises(ValueError,match='W&B'):wandb_epoch_logger(None)
