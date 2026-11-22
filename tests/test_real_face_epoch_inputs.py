from copy import deepcopy
import random
import numpy as np
import pytest
from scripts.real_face_epoch_inputs import SharedEpochInputs, recipe_seed
from test_real_face_full_image import Bank, inputs


def fixture():
    rgb, image, anns = inputs()
    image['metadata']['sha256'] = 'fixture-image'
    bank = Bank()
    bank.read = lambda row: rgb
    epoch = SharedEpochInputs.__new__(SharedEpochInputs)
    epoch.bank = bank
    epoch.annotations = {1: anns}
    return epoch, image


def test_replay_is_independent_of_global_rng_and_resume_order():
    epoch, image = fixture()
    original = deepcopy(epoch.annotations)
    for seed in range(40):
        rgb, anns, meta = epoch.render(image, seed=seed, epoch=12)
        if meta['applied']:
            break
    else:
        pytest.fail('Fixture never applied composition')
    random.seed(92)
    for _ in range(100):
        random.random()
    epoch.render(image, seed=seed+1, epoch=40)
    after = epoch.render(image, seed=seed, epoch=12)
    np.testing.assert_array_equal(rgb, after[0])
    assert anns == after[1] and meta == after[2]
    assert epoch.annotations == original
    states = np.array(anns[0]['keypoints']).reshape(5, 3)[:, 2]
    assert anns[0]['attributes']['keypoint_occluded'] == (states == 1).tolist()


def test_unrequested_and_failed_paste_keep_original_images():
    epoch, image = fixture()
    epoch.bank.full_face_annotations = {'donor.png': [dict(id=11, ellipse=[40,40,200,200,0])]}
    found = set()
    for seed in range(30):
        rgb, anns, meta = epoch.render(image, seed=seed, epoch=1)
        assert not meta['applied']
        np.testing.assert_array_equal(rgb, epoch.bank.read(None))
        assert anns[0]['keypoints'] == epoch.annotations[1][0]['keypoints']
        found.add(meta['requested'])
    assert found == {False, True}


def test_recipe_identity_and_invalid_epoch():
    assert recipe_seed(42,1,1) == recipe_seed(42,1,1)
    assert len({recipe_seed(42,1,1), recipe_seed(43,1,1), recipe_seed(42,2,1), recipe_seed(42,1,2)}) == 4
    with pytest.raises(ValueError):
        recipe_seed(42,0,1)
