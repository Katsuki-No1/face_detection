from copy import deepcopy
import hashlib
import numpy as np
import pytest
from scripts.prepare_real_face_for_epoch import fresh_queries, check_record, object_sha, sha


class Detector:
    def __init__(self):
        self.rows=[dict(face_query_index=1,face_query_features=[0.]*224,
            ellipse=[10,10,5,3,0],detector_bbox_xyxy=[5,5,15,15])]
        self.calls=0

    def predict(self, rgb):
        self.calls+=1
        self.rgb=rgb.copy()
        return self.rows


def test_fresh_actual_pixels_and_invalid_context_rejected():
    d=Detector(); rgb=np.ones((20,20,3),np.uint8)
    digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    with pytest.raises(ValueError,match='RGB'):
        fresh_queries(d,rgb,'clean-image-hash')
    assert d.calls==0
    fresh_queries(d,rgb,digest)
    np.testing.assert_array_equal(d.rgb,rgb)
    d.rows.append(deepcopy(d.rows[0]))
    with pytest.raises(ValueError,match='query'):
        fresh_queries(d,rgb,digest)


def test_resume_authenticates_recipe_record_and_image(tmp_path):
    image=tmp_path/'fixture.bin';image.write_bytes(b'pixels')
    recipe=dict(image_id=1,seed=42,epoch=1)
    payload=dict(contract_sha256='contract',recipe_sha256=object_sha(recipe),
        image=dict(id=1,file_name=str(image)),image_sha256=sha(image))
    record=dict(payload=payload,payload_sha256=object_sha(payload))
    assert check_record(record,recipe,'contract')==payload
    with pytest.raises(ValueError,match='provenance'):
        check_record(record,dict(recipe,epoch=12),'contract')
    changed=deepcopy(record);changed['payload']['image']['id']=2
    with pytest.raises(ValueError,match='provenance'):
        check_record(changed,recipe,'contract')
    image.write_bytes(b'modified')
    with pytest.raises(ValueError,match='changed'):
        check_record(record,recipe,'contract')
