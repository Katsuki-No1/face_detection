import json
import math
import numpy as np
import pytest
import torch
from PIL import Image
from keypoint_comparison1.landmark_detection.geometry_prior_inputs import predicted_geometry_inputs
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation, TopDownFaceLandmarkDataset


def inputs(ellipse, **kwargs):
    settings=dict(affine=np.eye(2,3),flipped=False,source_size=(64,64),crop_size=(64,64),grid_size=(8,8))
    settings.update(kwargs)
    return predicted_geometry_inputs(dict(ellipse=ellipse),**settings)


def test_predicted_geometry_scale_rotation_and_flip():
    ellipse=[23,30,18,10,.4]
    normal=inputs(ellipse)
    flipped=inputs(ellipse,flipped=True)
    assert torch.equal(flipped["support"],normal["support"].flip(1))
    assert float(normal["geometry"][0]) == pytest.approx(36/64)
    assert float(normal["geometry"][1]) == pytest.approx(1.8)
    assert float(flipped["geometry"][2]) == pytest.approx(math.pi-.4)
    angle=.2;c,s=math.cos(angle),math.sin(angle)
    rotated=inputs(ellipse,affine=[[c,-s,10],[s,c,-5]])
    assert float(rotated["geometry"][2]) == pytest.approx(.6)
    scaled=inputs(ellipse,affine=[[.5,0,0],[0,.5,0]])
    assert float(scaled["geometry"][0]) == float(normal["geometry"][0])


def test_missing_prediction_empty_support_and_anisotropy_rejected():
    with pytest.raises(ValueError,match="GT fallback"):
        predicted_geometry_inputs(None,affine=np.eye(2,3),flipped=False,
            source_size=(64,64),crop_size=(64,64),grid_size=(8,8))
    with pytest.raises(ValueError,match="no feature-cell"):
        inputs([1000,1000,10,5,0])
    with pytest.raises(ValueError,match="isotropic"):
        inputs([32,32,10,5,0],affine=[[1,0,0],[0,2,0]])


def test_actual_dataset_uses_prediction_not_gt_and_disabled_is_identical(tmp_path):
    Image.fromarray(np.full((64,64,3),128,np.uint8)).save(tmp_path/"image.png")
    annotations=dict(images=[dict(id=1,file_name="image.png",width=64,height=64)],annotations=[
        dict(id=1,image_id=1,category_id=2,bbox=[14,18,36,24],ellipse=[32,30,18,12,0],
            keypoints=[25,25,2,39,25,2,32,30,2,27,37,2,37,37,2])])
    ann=tmp_path/"annotations.json";ann.write_text(json.dumps(annotations))
    cache=tmp_path/"predictions.json";pred=[35,31,16,10,.3]
    cache.write_text(json.dumps(dict(metadata=dict(crop_geometry="ellipse"),predictions={"1":[
        dict(ellipse=pred,bbox_xyxy=[14,18,50,42],score=.9)]})))
    kwargs=dict(annotation_path=ann,image_root=tmp_path,input_size=64,category_ids=(2,),
        detector_crop_cache=cache,augmentation=CropAugmentation())
    plain=TopDownFaceLandmarkDataset(**kwargs)
    opted=TopDownFaceLandmarkDataset(**kwargs,geometry_prior_grid_size=(4,4))
    a,ta=plain[0];b,tb=opted[0]
    assert torch.equal(a,b) and "geometry_prior" not in ta
    assert torch.equal(tb["geometry_prior"]["predicted_ellipse_source"],torch.tensor(pred,dtype=torch.float32))
    assert not torch.equal(tb["gt_ellipse"],tb["geometry_prior"]["predicted_ellipse_source"])
    for key in ta:
        if torch.is_tensor(ta[key]): assert torch.equal(ta[key],tb[key])
        else: assert ta[key]==tb[key]
    kwargs.pop("detector_crop_cache")
    with pytest.raises(ValueError,match="GT fallback"):
        TopDownFaceLandmarkDataset(**kwargs,geometry_prior_grid_size=(4,4))
