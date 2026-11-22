import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.messenger_natural_policy_runtime import ROOT,epoch_batches,subset_indices
from scripts.run_messenger_natural_policy_gate import learning_assessment,QualityFailure,checked_marker
from src.models.messenger_occlusion_training import T_METRIC


def test_balanced_windows_partition_and_replay():
    class Dataset:
        eligible=list(range(180))
        probability=.5
        def set_epoch(self,e):self.epoch=e
    data=Dataset();ids=list(range(256))
    batches=epoch_batches(data,ids,1)
    assert sorted(i for b in batches for i in b)==ids
    assert all(len(b)==64 for b in batches)
    for b in batches:
        counts=[sum(data.requests.get(i)==k for i in b) for k in range(4)]
        assert max(counts)-min(counts)<=1
    assert batches==epoch_batches(data,ids,1)
    assert batches!=epoch_batches(data,ids,2)
    assert len(set(subset_indices(6185,256)))==256


def context():
    c=json.loads((ROOT/'configs/training_plans/messenger_natural_policy_v2_execution_20261018.json').read_text())
    return (c,None,None,None,None,None,{})


def trajectory():
    return [dict(epoch=e,step=e*4,validation={T_METRIC:.8+1e-12},finite=True,full_validation_crops=768,
                 composition=dict(requested_per_bin=[10]*4,applied_per_bin=[10]*4)) for e in range(1,21)]


def test_real_assessment_calls_scoped_override_and_requires_full_budget():
    with patch('scripts.run_messenger_natural_policy_gate.validation_contract',return_value={'population':'fixed'}):
        result=learning_assessment(context(),trajectory(),{T_METRIC:.8})
        assert result['criterion_passed'] and 0<result['delta']<1e-4
        with pytest.raises(QualityFailure):learning_assessment(context(),trajectory()[:2],{T_METRIC:.8})
        rows=trajectory();rows[0]['validation'][T_METRIC]=float('nan')
        with pytest.raises(QualityFailure):learning_assessment(context(),rows,{T_METRIC:.8})
        rows=trajectory();rows[-1]['composition']['applied_per_bin'][0]=9
        with pytest.raises(QualityFailure):learning_assessment(context(),rows,{T_METRIC:.8})
        rows=trajectory();rows[0]['full_validation_crops']=767
        with pytest.raises(QualityFailure):learning_assessment(context(),rows,{T_METRIC:.8})


def test_ties_and_regressions_do_not_pass():
    with patch('scripts.run_messenger_natural_policy_gate.validation_contract',return_value={'population':'fixed'}):
        for f1 in (.8,.79):
            rows=trajectory()
            for r in rows:r['validation'][T_METRIC]=f1
            assert not learning_assessment(context(),rows,{T_METRIC:.8})['criterion_passed']


def test_gate_rejects_changed_evidence_and_contract(tmp_path):
    cfg={'gate_root':str(tmp_path)};contract={'fresh':True};ctx=(cfg,None,None,None,None,None,contract)
    path=tmp_path/'small_scale';path.mkdir();evidence=path/'metrics.json';evidence.write_text('{}')
    from scripts.messenger_natural_policy_runtime import file_hash
    marker=dict(status='passed',stage='small_scale',contract=contract,evidence={str(evidence):file_hash(evidence)})
    (path/'success.json').write_text(json.dumps(marker));checked_marker(ctx,'small_scale')
    evidence.write_text('{"changed":1}')
    with pytest.raises(ValueError):checked_marker(ctx,'small_scale')
    marker['contract']={'stale':True};(path/'success.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError):checked_marker(ctx,'small_scale')
