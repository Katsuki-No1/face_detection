from copy import deepcopy
import random

import numpy as np
import pytest

from keypoint_comparison1.landmark_detection.real_occlusion import is_clean_annotation
from scripts.real_face_full_image import compose_full_image, transported_donor_face_pixels


class Bank:
    manifest_sha256 = 'fixture'
    images = {'recipient.png': {'video': 'recipient-video'}}
    full_face_annotations = {'donor.png': []}

    def eligible_sample(self, sample):
        return is_clean_annotation(sample['annotation'])

    def select(self, sample, rng, profile):
        return np.full((80, 80, 3), 180, np.uint8), [40, 40, 32, 28, 0], dict(
            donor_file='donor.png', donor_video='different-video',
            donor_annotation_id=10, donor_source_box=[0, 0, 80, 80])


def inputs():
    image = dict(id=1, file_name='recipient.png', width=384, height=384,
                 metadata=dict(split='train', source_video_id='recipient-video'))
    ann = dict(id=1, image_id=1, ellipse=[192, 192, 70, 60, .2],
               bbox=[120, 120, 144, 144], attributes=dict(face_occluded=False),
               keypoints=[160, 170, 2, 220, 170, 2, 190, 195, 2,
                          170, 215, 2, 215, 215, 2])
    rgb = np.random.default_rng(2).integers(0, 110, (384, 384, 3), dtype=np.uint8)
    return rgb, image, [ann]


def test_transport_preserves_exterior_and_supervision_and_is_reproducible():
    rgb, image, anns = inputs()
    original = deepcopy(anns)
    for seed in range(30):
        out, updated, mask, meta = compose_full_image(rgb, image=image, annotations=anns,
            target_id=1, bank=Bank(), rng=random.Random(seed))
        if meta['applied']:
            break
    else:
        pytest.fail('No placement in fixed fixture seed budget')
    assert mask.dtype == np.bool_ and mask.any()
    assert np.array_equal(out[~mask], rgb[~mask])
    assert not np.array_equal(out[mask], rgb[mask])
    assert anns == original
    assert updated[0]['ellipse'] == anns[0]['ellipse']
    assert updated[0]['bbox'] == anns[0]['bbox']
    old = np.array(anns[0]['keypoints']).reshape(5, 3)
    new = np.array(updated[0]['keypoints']).reshape(5, 3)
    np.testing.assert_array_equal(old[:, :2], new[:, :2])
    xy = old[:, :2].astype(int)
    np.testing.assert_array_equal(new[:, 2], np.where(mask[xy[:, 1], xy[:, 0]], 1, 2))
    assert meta['donor']['support_pixels'] > 0
    assert meta['donor']['include_in_recipient_keypoint_supervision'] is False
    repeated = compose_full_image(rgb, image=image, annotations=anns,
        target_id=1, bank=Bank(), rng=random.Random(seed))
    np.testing.assert_array_equal(out, repeated[0])
    assert meta == repeated[3]


def test_ineligible_bystander_is_rejected_without_pixel_or_label_edits():
    rgb, image, anns = inputs()
    other = deepcopy(anns[0]); other['id'] = 2
    other['attributes']['face_occluded'] = True
    anns.append(other)
    for seed in range(30):
        out, updated, mask, meta = compose_full_image(rgb, image=image, annotations=anns,
            target_id=1, bank=Bank(), rng=random.Random(seed))
        assert not meta['applied'] and not mask.any()
        np.testing.assert_array_equal(out, rgb)
        assert updated == anns
        if meta['reason'] == 'paste_touches_ineligible_recipient':
            break
    else:
        pytest.fail('Fixture did not exercise bystander rejection')


def test_additional_donor_face_cannot_be_pasted_without_labels():
    rgb, image, anns = inputs()
    bank = Bank(); bank.full_face_annotations = {'donor.png': [dict(id=11,
        ellipse=[40, 40, 200, 200, 0], attributes=dict(face_occluded=True))]}
    out, updated, mask, meta = compose_full_image(rgb, image=image, annotations=anns,
        target_id=1, bank=bank, rng=random.Random(42))
    assert meta['reason'] == 'additional_face_in_pasted_support'
    assert not mask.any() and updated == anns
    np.testing.assert_array_equal(out, rgb)


def test_donor_faces_clipped_out_of_actual_paste_are_not_rejected():
    # Left half of a100x100 patch lands outside the crop. Left face is absent;
    # right face survives and must still block unlabeled additional faces.
    args = ([0,0,100,100], [-50,0,50,100], (100,100), (200,200))
    assert transported_donor_face_pixels([20,50,10,10,0], *args) == 0
    assert transported_donor_face_pixels([75,50,10,10,0], *args) > 0


def test_donor_face_transport_matches_rgb_contributions():
    import cv2
    from src.data.landmark_occlusion import ellipse_region_mask
    ellipse=[55,30,20,7,.6]
    src=ellipse_region_mask(70,90,ellipse).astype(np.float32)
    resized=cv2.resize(src,(117,91),interpolation=cv2.INTER_LINEAR)
    canvas=np.zeros((64,80),np.float32)
    canvas[:64,:80]=resized[17:81,29:109]
    final=cv2.resize(canvas,(41,39),interpolation=cv2.INTER_NEAREST)
    assert transported_donor_face_pixels(ellipse,[0,0,90,70],[-29,-17,88,74],(80,64),(41,39)) == int((final>0).sum())


def test_validation_and_cross_video_mismatch_are_errors():
    rgb, image, anns = inputs()
    image['metadata']['split'] = 'val'
    with pytest.raises(ValueError, match='train-only'):
        compose_full_image(rgb, image=image, annotations=anns, target_id=1,
            bank=Bank(), rng=random.Random(42))
    image['metadata']['split'] = 'train'
    image['metadata']['source_video_id'] = 'not-authenticated'
    with pytest.raises(ValueError, match='authenticated'):
        compose_full_image(rgb, image=image, annotations=anns, target_id=1,
            bank=Bank(), rng=random.Random(42))
