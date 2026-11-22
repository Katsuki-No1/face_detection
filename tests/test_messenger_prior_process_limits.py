import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prior_process',Path(__file__).resolve().parents[1]/'scripts/run_messenger_geometry_prior_production.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_descriptor_headroom_preserves_hard_limit(monkeypatch):
    limits=[10240,1048576]
    monkeypatch.setattr(module.resource,'getrlimit',lambda _:tuple(limits))
    def set_limit(_,new):limits[:]=new
    monkeypatch.setattr(module.resource,'setrlimit',set_limit)
    assert module.ensure_descriptor_limit()==(65536,1048576)
    assert module.ensure_descriptor_limit()==(65536,1048576)
    limits[:]=[1024,4096]
    with pytest.raises(RuntimeError,match='hard limit'):
        module.ensure_descriptor_limit()
