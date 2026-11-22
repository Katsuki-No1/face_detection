import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.models.messenger_occlusion_augmentation import (
    BoundedOccluderRGB,NaturalOcclusionDiscriminator,point_patches,separated_losses,
)
from src.models.messenger_occlusion_training import OcclusionCounters,SupervisionAudit,F1Checkpoint,O_METRIC,CenterUpdate,rng_state
from src.models.messenger_prototype_observer import attach_learnable_prototypes
from src.data.messenger_natural_occlusion import balanced_requests,place_material,CompositionAudit,select_material
from src.models.messenger_natural_task import NaturalUpdate
from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)


def batch():
    torch.manual_seed(62)
    clean=torch.rand(2,3,64,64);mask=torch.zeros(2,1,64,64,dtype=torch.bool);mask[:,:,16:48,16:48]=True
    return dict(clean=clean,donor=torch.rand_like(clean),mask=mask,real=torch.rand_like(clean),
        fake_xy=torch.tensor([[32.,32.]]*2),real_xy=torch.tensor([[32.,32.]]*2),slots=torch.tensor([0,4]),
        real_valid=torch.ones(2,dtype=torch.bool))


class TinyTask(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,1,3,padding=1)
        self.pipeline=nn.Module();self.pipeline.bank=nn.Linear(2,2).requires_grad_(False)
    def loss(self,image,batch,*,hardness=False):return (self.conv(image)-.7).square().mean()


def test_generator_preserves_outside_and_cannot_read_hidden_recipient():
    b=batch();g=BoundedOccluderRGB()
    with torch.no_grad():g.net[-1].bias.fill_(2)
    result=g(b['clean'],b['donor'],b['mask'])
    changed=b['clean'].clone();changed[b['mask'].expand_as(changed)]=1
    alternate=g(changed,b['donor'],b['mask'])
    assert torch.equal(result['image'],alternate['image'])
    outside=~b['mask'].expand_as(changed)
    assert torch.equal(result['image'][outside],b['clean'][outside])
    assert (result['image']-b['donor'])[~outside].abs().max()<=.120001
    assert result['image'].min()>=0 and result['image'].max()<=1


@pytest.mark.parametrize('phase',['g','d','t'])
def test_gradient_routes_are_disjoint(phase):
    b=batch();b['clean'].requires_grad_();b['donor'].requires_grad_()
    g=BoundedOccluderRGB();d=NaturalOcclusionDiscriminator();t=TinyTask()
    losses=separated_losses(g,d,t,b)
    losses[phase].backward()
    for name,m in [('g',g),('d',d),('t',t)]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())==(name==phase)
    assert b['clean'].grad is None and b['donor'].grad is None
    assert all(p.requires_grad for p in d.parameters()) and t.training


def test_hardness_has_gradient_through_frozen_task_input():
    b=batch();g=BoundedOccluderRGB();d=NaturalOcclusionDiscriminator();t=TinyTask()
    losses=separated_losses(g,d,t,b,realism_weight=0,preserve_weight=0,hardness_weight=1)
    losses['g'].backward()
    assert g.net[-1].bias.grad.abs().sum()>0
    assert all(p.grad is None for p in t.parameters())


def test_d_regions_reject_edge_shortcuts_and_empty_regions():
    images=torch.rand(3,3,64,64,requires_grad=True)
    patches,ids=point_patches(images,torch.tensor([[1.,1.],[32.,32.],[float('nan'),5.]]))
    assert ids.tolist()==[1] and patches.shape==(1,3,32,32)
    patches.sum().backward();assert images.grad[0].abs().sum()==0
    b=batch();b['real_valid'].zero_()
    losses=separated_losses(BoundedOccluderRGB(),NaturalOcclusionDiscriminator(),TinyTask(),b)
    assert losses['d'] is None and losses['matched_regions']==0


def test_request_balance_and_actual_support_severity():
    requests=balanced_requests(range(160),seed=42,epoch=1)
    assert len(requests)==80
    from collections import Counter
    assert sorted(Counter(requests.values()).values())==[10]*8
    assert requests==balanced_requests(range(160),seed=42,epoch=1)
    assert requests!=balanced_requests(range(160),seed=42,epoch=2)
    face=np.zeros((64,64),bool);face[12:52,12:52]=True
    for low,high in ((.05,.2),(.2,.4),(.4,.6),(.6,.8)):
        result=place_material(np.full((40,32,3),150,np.uint8),None,face,interval=(low,high),rng=random.Random(2))
        assert result is not None
        _,mask,meta=result
        assert low <= (mask&face).sum()/face.sum() < high
        assert meta['overlap']==(mask&face).sum()/face.sum()
    audit=CompositionAudit();audit.add([dict(requested=True,family='face',bin=0,applied=False)])
    assert not audit.compute()['passed']


def test_donor_selection_cannot_use_hidden_recipient_descriptor():
    class Bank:
        images={'a':{'file':'a','video':'v1'},'b':{'file':'b','video':'v2'}}
        faces=[dict(file='a',box=[0,0,16,16],ellipse=[8,8,6,4,0],id=1),
               dict(file='b',box=[0,0,16,16],ellipse=[8,8,6,4,0],id=2)]
        def read(self,row):return np.full((16,16,3),70 if row['file']=='a' else 190,np.uint8)
    one=select_material(Bank(),'face',{'video':'v1','descriptor':[0,0,0,0]},random.Random(42))
    two=select_material(Bank(),'face',{'video':'v1','descriptor':[255,255,255,255]},random.Random(42))
    assert np.array_equal(one[0],two[0]) and one[2]==two[2]
    assert one[2]['donor_video']=='v2'


def test_o_f1_counts_support_misses_and_audits_conflicting_patch():
    support=torch.ones(1,4,4,dtype=torch.bool);support[0,0,0]=False
    xy=torch.tensor([[[0.,0.],[.4,.4],[.4,.4],[.7,.7],[-1.,0.]]])
    states=torch.tensor([[1,1,2,0,2]])
    prior=dict(margin=torch.ones(1,4,4))
    counter=OcclusionCounters();counter.add(prior,xy,states,support,input_hw=(64,64));m=counter.compute()
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(1,1,1,0)
    assert m['occlusion_f1']==.5
    audit=SupervisionAudit(3,4,4);audit.add(torch.tensor([1]),xy,states,support,input_hw=(64,64))
    a=audit.compute();assert a['conflicting_sample_cells']==1
    assert a['supervised_cells_per_center']==[0,1,0]


def test_f1_selection_ties_small_improvements_resume_and_retention(tmp_path):
    model=nn.Linear(1,1);opt=torch.optim.Adam(model.parameters());gen=torch.Generator().manual_seed(4)
    def policy():return F1Checkpoint(contract={'source':'fixed'},metric=O_METRIC,modules={'o':model},
        optimizers={'o':opt},schedulers={},loader_generator=gen,minimum_epoch=2,patience=3)
    p=policy()
    for f1 in (.4,.5,.50001,.50001,.49):
        improved=p.record({O_METRIC:f1});p.save(tmp_path,improved=improved)
    best=torch.load(tmp_path/'best.pt',weights_only=False)
    assert best['epoch']==3 and best['best']==.50001
    restored=policy();restored.restore(p.state());assert restored.should_stop==p.should_stop
    with pytest.raises(ValueError):p.record({O_METRIC:.9},split='test')
    p.save(tmp_path,improved=False,complete=True)
    assert {x.name for x in tmp_path.iterdir()}=={'last.pt','best.pt'}


def test_center_update_microbatch_equivalence_and_empty_window():
    parent,images,geo,support,_,_=model_case()
    a=attach_learnable_prototypes(parent,enabled=True);b=copy.deepcopy(a)
    xy=torch.tensor([[[.4,.4],[.7,.7]]]*2);states=torch.tensor([[1,2],[2,2]])
    first=CenterUpdate(a,positive_weight=3);second=CenterUpdate(b,positive_weight=3)
    r1=first.update(images,geo,support,xy,states,microbatch=1)
    r2=second.update(images,geo,support,xy,states,microbatch=2)
    torch.testing.assert_close(a.bank.centers,b.bank.centers,rtol=1e-5,atol=1e-7)
    assert r1['loss']==pytest.approx(r2['loss'],abs=1e-6)
    before=a.bank.centers.detach().clone()
    r=first.update(images,geo,support,xy,states*0,microbatch=1)
    assert r['status']=='no_supervision_skipped' and torch.equal(before,a.bank.centers)


def test_gdt_committed_update_and_complete_resume_match():
    b=batch();b['covered']=torch.ones(2,5,dtype=torch.bool)
    def setup():
        g=BoundedOccluderRGB();d=NaturalOcclusionDiscriminator();t=TinyTask()
        opts={k:torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
              for k,m in [('g',g),('d',d),('t',t)]}
        runtime=NaturalUpdate(g,d,t,optimizers=opts)
        policy=F1Checkpoint(contract={'identity':'fixture'},metric=O_METRIC,modules={'g':g,'d':d,'t':t},
            optimizers=opts,schedulers={},loader_generator=torch.Generator().manual_seed(42))
        return runtime,policy
    runtime,policy=setup()
    fixed=copy.deepcopy(runtime.task.pipeline.bank.state_dict())
    result=runtime.update(b,microbatch=1)
    assert result['d_updated'] and result['step']==1
    assert all(torch.equal(v,runtime.task.pipeline.bank.state_dict()[k]) for k,v in fixed.items())
    policy.record({O_METRIC:.4});checkpoint=copy.deepcopy(policy.state())
    runtime.update(b,microbatch=1)
    restored,p2=setup();p2.restore(checkpoint);restored.update(b,microbatch=1)
    for key,m in policy.modules.items():
        for name,value in m.state_dict().items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,p2.modules[key].state_dict()[name],rtol=0,atol=0)


def test_g_and_d_skip_empty_supervision_even_with_optimizer_momentum():
    b=batch();b['covered']=torch.ones(2,5,dtype=torch.bool)
    g=BoundedOccluderRGB();d=NaturalOcclusionDiscriminator();t=TinyTask()
    opts={k:torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
          for k,m in [('g',g),('d',d),('t',t)]}
    runtime=NaturalUpdate(g,d,t,optimizers=opts)
    runtime.update(b,microbatch=2)
    before_g=[p.detach().clone() for p in g.parameters()];before_d=[p.detach().clone() for p in d.parameters()]
    b['mask'].zero_();b['real_valid'].zero_();b['covered'].zero_()
    result=runtime.update(b,microbatch=2)
    assert not result['g_updated'] and not result['d_updated']
    assert all(torch.equal(a,p) for a,p in zip(before_g,g.parameters()))
    assert all(torch.equal(a,p) for a,p in zip(before_d,d.parameters()))


def test_o_oom_replays_before_commit_at_smaller_microbatch(monkeypatch):
    parent,images,geo,support,_,_=model_case()
    m=attach_learnable_prototypes(parent,enabled=True);reference=copy.deepcopy(m)
    run=CenterUpdate(m,positive_weight=2);ref=CenterUpdate(reference,positive_weight=2)
    xy=torch.tensor([[[.4,.4],[.7,.7]]]*2);states=torch.tensor([[1,2],[2,2]])
    observe=m.observe;calls=[0]
    def failing(*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:raise torch.OutOfMemoryError('injected precommit OOM')
        return observe(*args,**kwargs)
    monkeypatch.setattr(m,'observe',failing)
    result=run.update(images,geo,support,xy,states,microbatch=2)
    ref.update(images,geo,support,xy,states,microbatch=1)
    assert result['retries']==[2] and result['microbatch']==1
    torch.testing.assert_close(m.bank.centers,reference.bank.centers,rtol=0,atol=0)


def test_cpu_rng_capture_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    def forbidden():raise AssertionError('CPU diagnostic must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',forbidden)
    assert rng_state()['cuda'] is None
