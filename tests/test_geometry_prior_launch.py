import json
from pathlib import Path
import pytest
from keypoint_comparison1.landmark_detection.geometry_prior_launch import bind_run, verify_gates, write_json, STAGES


def test_resume_output_and_contract_are_bound(tmp_path):
    contract={'sources':{'a':'sha'},'mode':'production'}
    binding=bind_run(tmp_path/'run',contract,resume=False)
    assert binding==bind_run(tmp_path/'run',contract,resume=True)
    with pytest.raises(ValueError,match='identical'):
        bind_run(tmp_path/'run',{'sources':{'a':'changed'}},resume=True)
    with pytest.raises(ValueError,match='resume'):
        bind_run(tmp_path/'run',contract,resume=False)
    with pytest.raises(ValueError,match='unregistered'):
        bind_run(tmp_path/'missing',contract,resume=True)
    (tmp_path/'partial').mkdir();(tmp_path/'partial/log').touch()
    with pytest.raises(ValueError,match='nonempty'):
        bind_run(tmp_path/'partial',contract,resume=False)


def test_all_gates_and_safe_batch_are_mandatory(tmp_path):
    spec={'architecture':'k1'};identity={'config_sha256':'original','sources':{'a':'sha'}}
    with pytest.raises(FileNotFoundError):verify_gates(spec,identity,tmp_path)
    gate=tmp_path/'outputs/training_gates/k1'
    for stage in STAGES:
        write_json(gate/stage/'success.json',dict(status='passed',architecture='k1',gate_stage=stage,identity=identity))
    plan=dict(identity=identity,status='passed',candidates=[4,8,16],effective_batch=64,
        selected_microbatch=8,attempts=[dict(microbatch=8,status='safe')])
    write_json(gate/'gpu_step/batch_plan.json',plan)
    assert verify_gates(spec,identity,tmp_path)==8
    with pytest.raises(ValueError,match='stale'):
        verify_gates(spec,dict(identity,config_sha256='changed'),tmp_path)
    plan['selected_microbatch']=16;write_json(gate/'gpu_step/batch_plan.json',plan)
    with pytest.raises(ValueError,match='safe'):verify_gates(spec,identity,tmp_path)


def test_wandb_resume_requires_original_online_run(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from keypoint_comparison1.landmark_detection.geometry_prior_launch import start_wandb
    calls=[]
    run=SimpleNamespace(settings=SimpleNamespace(mode='online'),resumed=False,id='id',url='url',finish=lambda **kw:calls.append(kw))
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **kw:run))
    spec=dict(parent_config={'wandb':{'entity':'e','project':'p'}},architecture='k1',arm='k1')
    binding=dict(wandb_id='id',contract={})
    (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints/last.pt').touch()
    with pytest.raises(RuntimeError,match='original'):
        start_wandb(spec,binding,tmp_path,resume=True,diagnostic=True)
    assert calls==[{'exit_code':1}]
    run.resumed=True
    assert start_wandb(spec,binding,tmp_path,resume=True,diagnostic=True) is run
