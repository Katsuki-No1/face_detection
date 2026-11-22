import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prior_selection',Path(__file__).resolve().parents[1]/'scripts/select_messenger_geometry_prior_k.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def candidates():
    return [dict(k=k,value=.8,architecture=f'k{k}',prior={'k':k}) for k in range(1,8)]


def test_validation_ranking_and_ties():
    rows=candidates();rows[2]['value']=.9;rows[4]['value']=.9
    assert module.choose(rows)['k']==3
    for invalid in [rows[:-1],rows+[rows[0]],[dict(r,value=math.nan) for r in rows]]:
        with pytest.raises(ValueError):module.choose(invalid)


def test_selection_freeze_is_immutable(tmp_path):
    rows=candidates();path=tmp_path/'selection.json'
    result=module.freeze(rows,path)
    assert result['selected_k']==1 and result['test_used'] is False
    assert module.freeze(rows,path)==result
    rows[1]['value']=.9
    with pytest.raises(ValueError,match='overwrite'):module.freeze(rows,path)
