import random
import numpy as np
import torch
import pytest
from scripts.real_face_for_dataset import isolated_crop_rng, SharedFORCropDataset


def draws():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_rng_isolation_and_exception_restore():
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    expected = draws()
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    with isolated_crop_rng(42):
        a = draws()
    assert draws() == expected
    with isolated_crop_rng(42):
        assert draws() == a
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    with pytest.raises(RuntimeError):
        with isolated_crop_rng(99):
            draws()
            raise RuntimeError('fixture')
    assert draws() == expected


def test_next_epoch_cannot_reuse_previous_composite_manifest():
    dataset = SharedFORCropDataset.__new__(SharedFORCropDataset)
    dataset.recipe_epoch = 1
    with pytest.raises(ValueError, match='matching composite manifest'):
        dataset.set_epoch(12)
