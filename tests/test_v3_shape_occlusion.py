import cv2
import numpy as np
import pytest

from src.data.v3_shape_occlusion import SHAPES, occlude, silhouette


@pytest.mark.parametrize("kind", SHAPES)
def test_paste_respects_ellipse_coverage_and_original_pixels(kind):
    image=np.random.default_rng(2).integers(0,256,(256,256,3),dtype=np.uint8)
    yy,xx=np.ogrid[:256,:256]
    face=((xx-127)/95)**2+((yy-127)/45)**2<=1
    points=np.array([[99,113],[149,113],[127,123],[105,142],[143,142]])
    for seed in range(12):
        changed,mask=occlude(image,face,points,kind,seed)
        assert .10 <= (mask&face).sum()/face.sum() <= .40
        assert mask[points[:,1],points[:,0]].any()
        assert np.array_equal(changed[~mask],image[~mask])
        assert not np.array_equal(changed[mask],image[mask])
        assert np.array_equal(mask,occlude(image,face,points,kind,seed)[1])


def test_hand_has_four_separated_upper_fingers_and_connected_palm():
    stamp=silhouette("hand",np.random.default_rng(42))
    assert cv2.connectedComponents(stamp)[0]==2
    assert cv2.connectedComponents(stamp[:44])[0]==5
