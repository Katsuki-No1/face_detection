import json

import pytest

from scripts.extract_messenger_geometry_features import crop_predictions, finish, reusable


def test_predictions_preserve_empty_images_and_reject_foreign_or_missing_ellipses():
    row=dict(image_id=1,score=.33,bbox_xyxy=[1,2,9,8],ellipse=[5,5,4,3,0])
    grouped=crop_predictions([row,dict(row,score=.32)], [1,2],.33)
    assert len(grouped['1'])==1 and grouped['2']==[]
    with pytest.raises(ValueError,match='outside train'):
        crop_predictions([dict(row,image_id=3)], [1,2],.33)
    with pytest.raises(ValueError,match='ellipse'):
        crop_predictions([dict(row,ellipse=[])], [1,2],.33)


def test_completed_stage_reuse_authenticates_identity_and_artifacts(tmp_path):
    identity=dict(split='train',sha='fixed')
    assert reusable(tmp_path,'features',identity) is None
    artifact=tmp_path/'features.pt';artifact.write_bytes(b'feature-test')
    finish(tmp_path,'features',identity,[artifact],records=10)
    assert reusable(tmp_path,'features',identity)['records']==10
    with pytest.raises(ValueError,match='Stale'):
        reusable(tmp_path,'features',dict(identity,sha='changed'))
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Corrupt'):
        reusable(tmp_path,'features',identity)
    assert not (tmp_path/'success.json').exists()
