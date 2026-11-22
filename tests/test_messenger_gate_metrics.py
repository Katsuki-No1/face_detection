import copy

import pytest
import torch

from src.models.messenger_gate_metrics import GateCounters,gate_deltas,fixed_gate_probe


def case():
    gate=torch.tensor([[[[.001,.02],[.03,.04]]]],requires_grad=True)
    xy=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[1.1,.1],[float('nan'),0.]])
    targets=[dict(keypoints=xy,keypoint_visibility=torch.tensor([1,1,2,0,2,1]),
                  geometry_prior=dict(support=torch.tensor([[True,True],[False,True]])))]
    output=dict(decoded_coordinates=torch.nan_to_num(xy)[None],presence_logits=torch.full((1,6),-100.),
                prior_diagnostics=dict(gate=gate,anomalous=torch.tensor([[[True,False],[False,False]]])))
    return output,targets,torch.tensor([[True,False,False,False,False,False]])


def test_disjoint_groups_presence_independent_and_exclusions():
    out,targets,covered=case();c=GateCounters()
    c.add(out,targets,input_hw=(64,64),channel_adoption=torch.tensor([0.]),covered=covered,bins=[0])
    r=c.compute();g=r['groups']
    assert [g[k]['count'] for k in ('artificial_occluded','natural_occluded','visible')]==[1,1,1]
    assert g['artificial_occluded']['g_mean']==pytest.approx(.001)
    assert g['natural_occluded']['g_mean']==pytest.approx(.02)
    assert g['visible']['g_mean']==pytest.approx(.03)
    assert g['visible']['outside_support_count']==1 and g['visible']['effective_coefficient_mean']==0
    assert g['artificial_occluded']['effective_coefficient_mean']==pytest.approx(.0005)
    assert r['excluded']==dict(missing=1,outside_or_invalid=2)
    assert g['artificial_bin0']['count']==1
    assert out['prior_diagnostics']['gate'].grad is None


def test_counter_aggregation_empty_and_delta_are_honest():
    out,t,cov=case();initial=GateCounters();current=GateCounters()
    initial.add(out,t,input_hw=(64,64),channel_adoption=torch.tensor([0.]))
    out['prior_diagnostics']['gate']=out['prior_diagnostics']['gate'].detach()*2
    current.add(out,t,input_hw=(64,64),channel_adoption=torch.tensor([0.]))
    a,b=initial.compute(),current.compute();d=gate_deltas(b,a)
    assert a['groups']['artificial_occluded']['g_mean'] is None
    assert d['artificial_occluded']['g_mean_relative_change'] is None
    assert d['natural_occluded']['g_mean_relative_change']==pytest.approx(1.)
    current.add(out,t,input_hw=(64,64),channel_adoption=torch.tensor([0.]))
    with pytest.raises(ValueError):gate_deltas(current.compute(),a)


def test_fixed_probe_preserves_rng_modes_weights_and_input_population():
    from keypoint_comparison1.tests.test_geometry_prior_recovery import model_case
    from src.models.messenger_occlusion_policy import PolicyMessengerTask
    from test_messenger_occlusion_policy import fixture_rows
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        parent,images,geo,support,targets,criterion=model_case()
        task=PolicyMessengerTask(parent,criterion,mean=[0,0,0],std=[1,1,1]);task.train()
        rows=fixture_rows()[:2]
        for i,row in enumerate(rows):
            t=targets[i];t['keypoint_visibility']=torch.where(t['visible'],2,1)
            t['geometry_prior']=dict(geometry=geo[i],support=support[i],predicted_ellipse_crop=torch.tensor([32.,32.,24.,20.,0.]))
            row['target']=t
        before=copy.deepcopy(task.state_dict());modes=[m.training for m in task.modules()];rng=torch.get_rng_state()
        r=fixed_gate_probe(task,rows,batch_size=1)
        assert torch.equal(rng,torch.get_rng_state()) and modes==[m.training for m in task.modules()]
        for name,v in before.items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,task.state_dict()[name])
        assert r['synthetic']['groups']['artificial_occluded']['count']>0
        for key in r['synthetic']['groups']:
            assert r['synthetic']['groups'][key]['count']==r['paired_clean']['groups'][key]['count']
    finally:torch.set_num_threads(previous)


def test_validation_diagnostics_reuse_forward_and_preserve_f1():
    from keypoint_comparison1.landmark_detection.geometry_prior_validation import validate_prior
    from src.data.head_anchored_keypoint_cache import CONTRACT
    class Dataset:
        training=False;geometry_prior_grid_size=(2,2);input_size=64
        detector_crop_report=dict(matching_contract=CONTRACT,normalization_bbox_source='GT Head')
        def __len__(self):return 2
        def __getitem__(self,index):
            return torch.zeros(3,64,64),dict(keypoints=torch.zeros(5,2),
                keypoint_visibility=torch.tensor([1,2,2,0,1]),
                original_labeled=torch.tensor([True,True,True,False,True]),
                original_keypoints=torch.zeros(5,2),bbox_xyxy=torch.tensor([0.,0.,64.,64.]),
                inverse_affine=torch.eye(3)[:2],
                geometry_prior=dict(geometry=torch.ones(3),support=torch.ones(2,2,dtype=torch.bool)))
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,images,**kwargs):
            self.calls+=1;n=len(images)
            return dict(decoded_coordinates=torch.zeros(n,5,2),presence_logits=torch.ones(n,5),
                prior_diagnostics=dict(gate=torch.full((n,1,2,2),.01),anomalous=torch.ones(n,2,2,dtype=torch.bool)))
    model=Model();counters=GateCounters();dataset=Dataset()
    plain=validate_prior(model,dataset,device=torch.device('cpu'),batch_size=1)
    enriched=validate_prior(model,dataset,device=torch.device('cpu'),batch_size=1,
        diagnostics=lambda out,targets,hw:counters.add(out,targets,input_hw=hw,channel_adoption=torch.tensor([0.])))
    assert plain==enriched and model.calls==4 and counters.crops==2
    assert counters.compute()['groups']['natural_occluded']['count']==4
