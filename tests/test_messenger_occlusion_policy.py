import copy
import random
from collections import Counter

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data.messenger_policy_occlusion import (
    POLICY_BINS, balanced_bins, in_bin, propose, compose, PolicyOcclusionDataset, audit_actions,
)
from src.models.messenger_occlusion_policy import (
    CompositionPolicy, FullCropDiscriminator, position_error, reinforce_loss, policy_losses, PolicyUpdate,
)
from src.models.messenger_occlusion_training import F1Checkpoint,T_METRIC


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(previous)


def fixture_rows():
    torch.manual_seed(6)
    rows=[]
    for i in range(4):
        target=dict(keypoints=torch.tensor([[.4,.4],[.6,.6]]),keypoint_visibility=torch.tensor([2,2]),
            labeled=torch.tensor([True,True]),visible=torch.tensor([True,True]),
            geometry_prior=dict(geometry=torch.zeros(5),support=torch.ones(4,4,dtype=torch.bool),
                                predicted_ellipse_crop=torch.tensor([32.,32.,24.,20.,0.])))
        candidates=[]
        for j in range(3):
            donor=torch.full((3,64,64),.2+.3*j);mask=torch.zeros(1,64,64,dtype=torch.bool)
            mask[:,8+j:52+j,8:52]=True
            candidates.append(dict(donor=donor,mask=mask,preview=F.adaptive_avg_pool2d(donor,16),
                features=torch.tensor([.1,.1,.8,.8,.7,float(j==0),0.,0.,1.,0.]),
                metadata=dict(overlap=sum(POLICY_BINS[i])/2,material_id=j,bbox=[8,8+j,52,52+j])))
        rows.append(dict(clean=torch.rand(3,64,64),real=torch.rand(3,64,64),target=target,candidates=candidates,
            metadata=dict(index=i,requested=True,bin=i,reason=None)))
    return rows


class TinyTask(nn.Module):
    def __init__(self):
        super().__init__();self.xy=nn.Linear(3,4);self.presence=nn.Parameter(torch.tensor(-4.))
        self.pipeline=nn.Module();self.pipeline.bank=nn.Linear(1,1).requires_grad_(False)
        self.pipeline.observer=nn.Linear(1,1).requires_grad_(False)
    def position(self,images,batch):
        coords=self.xy(images.mean((-1,-2))).reshape(-1,2,2).sigmoid()
        return position_error(coords,batch['target'],batch['covered'])
    def supervised(self,images,batch):
        return self.position(images,batch).mean()+.01*self.presence.square()


def setup():
    g=CompositionPolicy(hidden=4);d=FullCropDiscriminator(hidden=4);t=TinyTask()
    opts={k:torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
          for k,m in [('g',g),('d',d),('t',t)]}
    return g,d,t,opts,PolicyUpdate(g,d,t,opts)


@pytest.mark.parametrize('phase',['g','d','t'])
def test_gradient_ownership_and_source_detach(phase):
    rows=fixture_rows()
    for row in rows:
        row['clean'].requires_grad_();row['real'].requires_grad_()
        for c in row['candidates']:
            c['donor'].requires_grad_();c['preview'].requires_grad_();c['features'].requires_grad_()
    g,d,t,_,_=setup()
    losses=policy_losses(g,d,t,rows);losses[phase].backward()
    for name,module in [('g',g),('d',d),('t',t)]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())==(name==phase)
    for row in rows:
        assert row['clean'].grad is None and row['real'].grad is None
        for c in row['candidates']:
            assert c['donor'].grad is None and c['preview'].grad is None and c['features'].grad is None
    assert all(p.requires_grad for p in d.parameters()) and t.training


@pytest.mark.parametrize('kind',['position','natural'])
def test_policy_sign_exact_expected_gradient_and_update(kind):
    # Enumerate actions, weight by detached sampling probabilities: exact
    # expectation of REINFORCE, compared with differentiated expected cost.
    logits=torch.tensor([.2,-.1],requires_grad=True);prob=logits.softmax(0)
    natural=torch.tensor([.1,.8],requires_grad=True)
    position=torch.tensor([.1,.8],requires_grad=True)
    nw,pw=(0.,1.) if kind=='position' else (1.,0.)
    cost=nw*natural-pw*position
    exact=torch.autograd.grad((prob*cost.detach()).sum(),logits,retain_graph=True)[0]
    terms=[]
    for j in range(2):
        loss,_=reinforce_loss(logits.log_softmax(0)[j:j+1],natural[j:j+1],position[j:j+1],
                              natural_weight=nw,position_weight=pw)
        terms.append(prob[j].detach()*loss)
    sum(terms).backward()
    torch.testing.assert_close(logits.grad,exact)
    assert natural.grad is None and position.grad is None
    updated=(logits.detach()-.1*logits.grad).softmax(0)
    assert (updated[1]>prob[1]) if kind=='position' else (updated[1]<prob[1])


def test_position_gradient_opposes_g_and_does_not_use_presence_or_missing():
    rows=fixture_rows();target=rows[0]['target']
    target['keypoints'][1]=float('nan');target['labeled'][1]=False
    xy=torch.full((1,2,2),.8,requires_grad=True)
    loss=position_error(xy,[target],torch.ones(1,2,dtype=torch.bool)).mean()
    positive=torch.autograd.grad(loss,xy,retain_graph=True)[0]
    negative=torch.autograd.grad(-loss,xy)[0]
    torch.testing.assert_close(positive,-negative)
    assert positive[0,1].abs().sum()==0 and positive[0,0].sum()>0
    g,d,t,_,_=setup();rows=fixture_rows()
    torch.manual_seed(12);one=policy_losses(g,d,t,rows)['position']
    with torch.no_grad():t.presence.fill_(100.)
    torch.manual_seed(12);two=policy_losses(g,d,t,rows)['position']
    torch.testing.assert_close(one,two,rtol=0,atol=0)


def test_non_saturating_and_full_crop_spatial_reach():
    rows=fixture_rows();g,d,t,_,_=setup()
    logits=torch.tensor([-5.,0.,5.],requires_grad=True)
    F.softplus(-logits).sum().backward()
    torch.testing.assert_close(logits.grad,logits.detach().sigmoid()-1)
    # The complete crop, including corners far from keypoints, reaches D.
    image=torch.rand(1,3,64,64,requires_grad=True);d(image).sum().backward()
    assert image.grad[:,:,:8,:8].abs().sum()>0
    assert image.grad[:,:,-8:,-8:].abs().sum()>0
    with pytest.raises(ValueError):d(torch.rand(1,8,64,64))


def test_exact_opaque_composition_and_full_occlusion_labels():
    row=fixture_rows()[0];clean=row['clean'];donor=torch.ones_like(clean)
    row['target']['keypoints'][1]=torch.tensor([1.1,.5])
    mask=torch.ones(1,64,64,dtype=torch.bool)
    image,target,covered=compose(clean,donor,mask,row['target'])
    assert torch.equal(image,donor) and target['keypoint_visibility'].tolist()==[1,0]
    assert covered.tolist()==[True,False]
    assert torch.equal(target['keypoints'],row['target']['keypoints'])
    mask[:,:,32:]=False
    image,_,_=compose(clean,donor,mask,row['target'])
    assert torch.equal(image[~mask.expand_as(image)],clean[~mask.expand_as(image)])


def test_bin_balance_actual_overlap_and_endpoint():
    schedule=balanced_bins(range(1000),seed=42,epoch=0,probability=.5)
    assert Counter(schedule.values())=={i:125 for i in range(4)}
    assert in_bin(1.,3) and not in_bin(.4,1) and not any(in_bin(.5,i) for i in range(4))
    face=np.zeros((64,64),bool);face[10:54,10:54]=True
    for i in range(4):
        result=propose(np.full((40,30,3),128,np.uint8),None,face,i,random.Random(i))
        assert result is not None
        _,mask,meta=result
        assert in_bin((mask&face).sum()/face.sum(),i) and meta['overlap']==(mask&face).sum()/face.sum()
    report=audit_actions([dict(requested=True,bin=i,applied=i!=3,overlap=sum(POLICY_BINS[i])/2,reason='failed') for i in range(4)])
    assert not report['equal_applied_distribution'] and report['skips']=={'failed':1}


def test_train_only_and_missing_real_never_hardness_only():
    with pytest.raises(ValueError):PolicyOcclusionDataset(None,None,split='val')
    rows=fixture_rows();g,d,t,opts,runtime=setup();runtime.update(rows,microbatch=2)
    before={k:copy.deepcopy(m.state_dict()) for k,m in [('g',g),('d',d)]}
    for row in rows:row['real']=None
    result=runtime.update(rows,microbatch=2)
    assert not result['g_updated'] and not result['d_updated']
    for k,m in [('g',g),('d',d)]:
        for n,v in before[k].items():torch.testing.assert_close(v,m.state_dict()[n],rtol=0,atol=0)


def test_joint_actions_use_material_scale_placement_and_no_recipient_rgb():
    g=CompositionPolicy(hidden=4)
    previews=torch.rand(3,3,16,16);features=torch.rand(3,10);context=torch.rand(5)
    before=g(previews,features,context)
    changed=features.clone();changed[:,0:4]+=1.
    assert not torch.equal(before,g(previews,changed,context))
    assert not torch.equal(before,g(previews*.1,features,context))
    a=fixture_rows();b=copy.deepcopy(a)
    for row in b:row['clean'].fill_(.123)
    d=FullCropDiscriminator(hidden=4);t=TinyTask()
    torch.manual_seed(1);first=policy_losses(g,d,t,a)['records']
    torch.manual_seed(1);second=policy_losses(g,d,t,b)['records']
    assert [r['action'] for r in first]==[r['action'] for r in second]
    assert [r['log_probability'] for r in first]==[r['log_probability'] for r in second]


def test_complete_resume_repeats_action_update_and_baseline():
    rows=fixture_rows();g,d,t,opts,run=setup()
    def policy(g,d,t,opts):
        return F1Checkpoint(contract={'scope':'test'},metric=T_METRIC,modules={'g':g,'d':d,'t':t},
            optimizers=opts,schedulers={},loader_generator=torch.Generator().manual_seed(42))
    p=policy(g,d,t,opts);run.update(rows,microbatch=2);p.record({T_METRIC:.4})
    saved=copy.deepcopy(p.state());first=run.update(rows,microbatch=2)
    g2,d2,t2,opts2,run2=setup();p2=policy(g2,d2,t2,opts2);p2.restore(saved)
    second=run2.update(rows,microbatch=2)
    assert [r['action'] for r in first['records']]==[r['action'] for r in second['records']]
    for key,m in p.modules.items():
        for name,value in m.state_dict().items():torch.testing.assert_close(value,p2.modules[key].state_dict()[name],rtol=0,atol=0)


def test_failed_bin_preserves_equal_success_counts():
    rows=fixture_rows();rows[3]['candidates']=[];rows[3]['metadata']['reason']='proposal_failed'
    g,d,t,opts,run=setup()
    result=run.update(rows,microbatch=2)
    assert not result['g_updated']
    report=audit_actions(result['records'])
    assert report['applied_per_bin']==[0,0,0,0]
    assert report['skips']=={'bin_balance_deferred':3,'proposal_failed':1}


def test_precommit_oom_reuses_actions_and_recovers(monkeypatch):
    import src.models.messenger_occlusion_policy as module
    rows=fixture_rows();g,d,t,opts,run=setup()
    original=module.policy_losses;calls=[]
    def fail_once(*args,**kwargs):
        calls.append([r['_sampled_action'] for r in args[3]])
        if len(calls)==1:raise torch.OutOfMemoryError('injected')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'policy_losses',fail_once)
    result=run.update(rows,microbatch=4)
    assert result['retries']==[4] and result['microbatch']==2
    assert calls[0]==calls[1]+calls[2]


def test_messenger_adapter_shared_position_and_real_head_gradient():
    from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case
    from src.models.messenger_occlusion_policy import PolicyMessengerTask
    parent,images,geo,support,targets,criterion=model_case()
    parent.observer.requires_grad_(False);parent.bank.requires_grad_(False)
    task=PolicyMessengerTask(parent,criterion,mean=[0,0,0],std=[1,1,1])
    pixels=images.sigmoid()
    batch=dict(target=targets,geometry=geo,support=support,clean=pixels,
               covered=torch.ones(2,5,dtype=torch.bool))
    task.eval()
    output=task.output(pixels,batch)
    expected=position_error(output['decoded_coordinates'],targets,batch['covered'])
    torch.testing.assert_close(task.position(pixels,batch),expected)
    task.supervised(pixels,batch).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parent.base.head.parameters())
    assert all(p.grad is None for p in parent.observer.parameters())
    assert all(p.grad is None for p in parent.bank.parameters())


def test_policy_config_rejects_old_bins_and_production_toggle():
    import json
    from pathlib import Path
    from scripts.train_messenger_natural_policy import validate_config
    cfg=json.loads(Path('configs/training_plans/messenger_natural_policy_v2.json').read_text())
    validate_config(cfg)
    bad=copy.deepcopy(cfg);bad['composition']['bins'][2]=[.4,.6]
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(cfg);bad['production']['enabled']=True
    with pytest.raises(ValueError):validate_config(bad)
