import hashlib
import numpy as np
import pytest
from scripts.real_face_composite_context import extract_composite_context


def test_context_uses_actual_composite_and_rejects_clean_cache_identity():
    composite = np.ones((20, 20, 3), np.uint8)
    meta = dict(applied=True, rgb_sha256=hashlib.sha256(composite.tobytes()).hexdigest(),
                mask_sha256='binary-support')
    class Detector:
        def predict(self, rgb):
            assert rgb is composite
            return [dict(face_query_index=0, face_query_features=[float(rgb.sum())]*224,
                         ellipse=[10, 10, 5, 4, 0], detector_bbox_xyxy=[5, 6, 15, 14])]
    identity = dict(checkpoint_sha256='checkpoint', config_sha256='config')
    result = extract_composite_context(Detector(), composite, meta, detector_identity=identity, query_dim=224)
    assert result['shared_arms'] == ['for_off', 'for_on']
    assert result['predictions'][0]['face_query_features'][0] == composite.sum()
    with pytest.raises(ValueError, match='RGB differs'):
        extract_composite_context(Detector(), np.zeros_like(composite), meta,
                                  detector_identity=identity, query_dim=224)
    with pytest.raises(ValueError, match='Invalid composite'):
        extract_composite_context(Detector(), composite, meta, detector_identity=identity, query_dim=256)


def test_duplicate_queries_are_rejected():
    rgb = np.ones((4, 4, 3), np.uint8)
    meta = dict(applied=True, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), mask_sha256='mask')
    class Detector:
        def predict(self, rgb):
            row = dict(face_query_index=1, face_query_features=[0.]*256,
                       ellipse=[2, 2, 1, 1, 0], detector_bbox_xyxy=[1, 1, 3, 3])
            return [row, dict(row)]
    with pytest.raises(ValueError, match='Invalid composite'):
        extract_composite_context(Detector(), rgb, meta,
            detector_identity=dict(checkpoint_sha256='checkpoint', config_sha256='config'), query_dim=256)
