import copy
import numpy as np
import pytest

from keypoint_comparison1.landmark_detection.geometry_prior_validation import (
    ConditionalSemanticCounters, head_validation_annotations, assign_validation_crops,
)


def native_case():
    return dict(images=[dict(id=1,file_name='val.jpg'),dict(id=2,file_name='empty.jpg')],
        annotations=[dict(id=7,image_id=1,bbox=[0,0,100,50],ellipse=[50,25,20,10,0],
                         attribute_valid=True,keypoints=[[10,10]]*5,keypoint_states=[2,0,0,0,0])])


def test_full_inventory_head_normalization_and_exclusions():
    native=native_case()
    invalid=copy.deepcopy(native['annotations'][0]);invalid.update(id=8,attribute_valid=False)
    missing=copy.deepcopy(invalid);missing.update(id=9,attribute_valid=True,keypoint_states=[0]*5)
    native['annotations'] += [invalid,missing]
    adapted,report=head_validation_annotations(native)
    assert len(adapted['images'])==2 and len(adapted['annotations'])==1
    assert adapted['annotations'][0]['bbox']==[0,0,100,50]
    assert adapted['annotations'][0]['normalization_bbox_source']=='GT Head'
    assert report['excluded']==dict(invalid_or_ambiguous_attributes=1,all_missing=1)
    assert native['annotations'][0]['keypoints']==[[10,10]]*5


def test_matching_uses_real_ellipse_and_retains_empty_image():
    native=native_case();adapted,_=head_validation_annotations(native)
    detection=dict(ellipse=[50,25,20,10,0],bbox_xyxy=[30,15,70,35],score=.8)
    result=assign_validation_crops(native,adapted,{1:[detection],2:[]})
    assert result['1'][0]['assigned_annotation_id']==7 and result['2']==[]
    assert result['1'][0]['ellipse']==detection['ellipse']
    with pytest.raises(ValueError,match='inventory'):
        assign_validation_crops(native,adapted,{1:[detection]})


def test_semantic_macro_is_not_micro_and_uses_gt_head_side():
    c=ConditionalSemanticCounters()
    target=dict(original_labeled=np.array([1,0,0,0,0],bool),bbox_xyxy=[0,0,100,50],
                original_keypoints=np.zeros((5,2)))
    points=np.zeros((1,5,2));points[0,0,0]=12  # inclusive 12/(1.2*100)=.10
    c.add(points,np.full((1,5),np.float32(.70)),[target])
    r=c.compute()
    assert r['semantic_tp']==[1,0,0,0,0]
    assert r['semantic_fp']==[0,1,1,1,1]
    assert r['conditional_semantic_macro_f1_nme_0.10']==.2
    assert not hasattr(c,'rows')
    points[0,0,0]=13
    c.add(points,np.ones((1,5)),[target])
    assert c.compute()['semantic_fn'][0]==1 and c.compute()['semantic_fp'][0]==1


def test_empty_gt_excluded_and_nonfinite_prediction_rejected():
    c=ConditionalSemanticCounters()
    t=dict(original_labeled=np.zeros(5,bool),bbox_xyxy=[0,0,100,50],original_keypoints=np.zeros((5,2)))
    c.add(np.zeros((1,5,2)),np.ones((1,5)),[t]);assert c.instances==0
    t['original_labeled'][0]=True
    with pytest.raises(ValueError,match='Invalid validation'):
        c.add(np.full((1,5,2),np.nan),np.ones((1,5)),[t])
