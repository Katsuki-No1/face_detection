"""Explicit, scoped numerical criterion only; never issues a training gate marker."""
import hashlib
import math
from pathlib import Path


def assess_once(override, *, gate_run_id, config_path, stage, split,
                initial_f1, candidate_f1, initial_validation_contract, candidate_validation_contract):
    if (override['schema'] != 'single-experiment-quality-override-v1'
        or override['override_id'] != 'messenger_natural_policy_v2_small_scale_once_20261018'
        or gate_run_id != override['target_gate_run_id']
        or stage != override['stage'] or stage != 'small_scale'
        or split != override['split'] or split != 'val'
        or override['metric'] != 'conditional_semantic_macro_f1_nme_0.10'
        or override['criterion'] != 'candidate > reference'
        or override['minimum_delta_exclusive'] != 0.
        or override['ties_pass'] or override['automatic_application']):
        raise ValueError('Override is not authorized for this gate/metric/run')
    if hashlib.sha256(Path(config_path).read_bytes()).hexdigest() != override['target_config_sha256']:
        raise ValueError('Override does not cover this config revision')
    if not initial_validation_contract or initial_validation_contract != candidate_validation_contract:
        raise ValueError('Validation population/checkpoint metric/threshold contract differs')
    a,b=float(initial_f1),float(candidate_f1)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a,b)):
        raise ValueError('Invalid F1')
    return dict(criterion_passed=b>a,delta=b-a,reference=a,candidate=b,
        override_id=override['override_id'],gate_run_id=gate_run_id,
        scope='numerical improvement criterion only; not a passed gate or production authorization')
