import numpy as np
import pytest
from scripts.diagnose_messenger_occlusion_observer import binary_metrics, point_cell


def test_rank_ties_do_not_make_random_detector_perfect():
    result=binary_metrics([1,0,1,0],[.2]*4,[True]*4)
    assert result['average_precision']==.5
    assert result['auroc']==.5
    assert result['tp']==2 and result['fp']==2
    assert len(result['pr_curve'])==1


def test_sparse_detection_and_pr_keep_false_positive_points():
    result=binary_metrics([1,0,1,0],[.9,.8,.2,.1],[1,1,0,0])
    assert result['precision']==.5 and result['recall']==.5
    assert result['average_precision']==pytest.approx(5/6)
    assert result['auroc']==.75
    assert result['fn']==1 and result['tn']==1


def test_outside_point_never_clips_onto_boundary_patch():
    assert point_cell([-1,10]) is None
    assert point_cell([256,10]) is None
    assert point_cell([float('nan'),10]) is None
    assert point_cell([255,255])==(15,15)
    assert point_cell([16,15.9])==(0,1)


def test_binary_mask_patch_pooling_preserves_pixel_coordinates():
    mask=np.zeros((256,256));mask[16:32,32:48]=1
    ratio=mask.reshape(16,16,16,16).mean((1,3))
    assert ratio.sum()==1 and ratio[1,2]==1
