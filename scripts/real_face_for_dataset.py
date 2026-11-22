"""Shared FOR crop augmentation with RNG independent of the trained model."""
from contextlib import contextmanager
import hashlib
import json
import random

import numpy as np
import torch
from keypoint_comparison1.landmark_detection.dataset import TopDownFaceLandmarkDataset


@contextmanager
def isolated_crop_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed % 2**63)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class SharedFORCropDataset(TopDownFaceLandmarkDataset):
    """One recipe epoch per manifest; both arms consume the identical manifest.

    This wrapper is intended for DataLoader workers, not concurrent threads in
    one process. It preserves the calling process/model random streams.
    """
    def __init__(self, *, recipe_epoch, **kwargs):
        if type(recipe_epoch) is not int or recipe_epoch < 1:
            raise ValueError('Positive recipe epoch required')
        if kwargs.get('face_overlap_query_dim') is None:
            raise ValueError('Both FOR arms must use the shared query-bearing cache')
        self.recipe_epoch = recipe_epoch
        super().__init__(**kwargs)
        self.set_epoch(recipe_epoch)

    def set_epoch(self, epoch):
        if epoch != self.recipe_epoch:
            raise ValueError('A new recipe epoch requires its matching composite manifest')
        super().set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.samples[index]
        identity = ["real-face-for-crop-v1", self.seed, self.recipe_epoch,
                    sample['image']['id'], sample['annotation']['id']]
        seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:16], 'big')
        with isolated_crop_rng(seed):
            return super().__getitem__(index)
