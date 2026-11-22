import copy
import pytest
from scripts.run_messenger_geometry_prior_gate import assess,QualityFailure,certify_preflight,METRIC


def test_learning_rejects_flat_degrading_and_nonfinite():
    assert assess([.5,.501,.49],higher=True,minimum_delta=1e-4)['passed']
    assert not assess([.5,.5,.49],higher=True,minimum_delta=1e-4)['passed']
    assert assess([1,.9,.8],higher=False,minimum_delta=0)['passed']
    with pytest.raises(QualityFailure):assess([.5,float('nan')],higher=True,minimum_delta=1e-4)


def fixture_report():
    identity={'config_sha256':'same'};attempts=[]
    for batch,time in [(4,120),(8,102),(16,100)]:
        rows=[dict(step=s,backbone_frozen=False,amp_overflows=0,microbatch=batch,train_seconds=1,validation_seconds=1,train={'total':.5},validation={METRIC:.8,'eligible_face_crops':768}) for s in [2,4]]
        attempts.append(dict(status='safe',identity=identity,preflight_source_sha256='code',effective_batch=64,full_validation_instances=768,rows=rows,maximum_temperature=60,maximum_device_used_mib=4000,peak_reserved_mib=2000,projected_seconds=time,microbatch=batch))
    return dict(status='preflight_completed_not_formal_gate',identity=identity,preflight_source_sha256='code',candidates=[4,8,16],effective_batch=64,selected_microbatch=8,attempts=attempts)


def test_gpu_certification_requires_current_full_joint_evidence():
    report=fixture_report()
    assert certify_preflight(report,report['identity'],'code')['status']=='passed'
    for mutate in [lambda d:d.update(selected_microbatch=16),lambda d:d['attempts'][0]['rows'][1].update(backbone_frozen=True),lambda d:d['attempts'][0].update(maximum_temperature=83),lambda d:d.update(preflight_source_sha256='stale')]:
        bad=copy.deepcopy(report);mutate(bad)
        with pytest.raises(ValueError):certify_preflight(bad,report['identity'],'code')
