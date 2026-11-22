import json
from pathlib import Path

import pytest

from src.models.messenger_policy_quality_override import assess_once


def arguments():
    path=Path('configs/training_plans/overrides/messenger_natural_policy_v2_small_scale_once_20261018.json')
    policy=json.loads(path.read_text())
    args=dict(gate_run_id=policy['target_gate_run_id'],config_path=policy['target_config'],stage='small_scale',split='val',
        initial_f1=.5,candidate_f1=.50000001,
        initial_validation_contract={'annotation_hash':'fixture','presence':.7,'nme':.1,'aggregation':'semantic_macro'},
        candidate_validation_contract={'annotation_hash':'fixture','presence':.7,'nme':.1,'aggregation':'semantic_macro'})
    return policy,args


@pytest.mark.parametrize('value,expected',[(.50000001,True),(.5,False),(.49,False)])
def test_strict_improvement_without_minimum_margin(value,expected):
    policy,args=arguments();args['candidate_f1']=value
    assert assess_once(policy,**args)['criterion_passed'] is expected


@pytest.mark.parametrize('field,value',[('gate_run_id','another_run'),('stage','overfit'),('split','test'),
    ('candidate_f1',float('nan')),('candidate_validation_contract',{'presence':.5}),
    ('config_path','configs/training_plans/messenger_natural_occlusion_v1.json')])
def test_exception_cannot_escape_its_scope(field,value):
    policy,args=arguments();args[field]=value
    with pytest.raises(ValueError):assess_once(policy,**args)
