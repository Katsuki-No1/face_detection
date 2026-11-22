import torch
import pytest
import hashlib
import json
import numpy as np
from PIL import Image
from keypoint_comparison1.landmark_detection.heads import LandmarkQueryHead, LandmarkQueryRefinementHead
from keypoint_comparison1.landmark_detection.model import DinoFaceLandmarkModel
from scripts.real_face_for_pair import initialize_matched_arm, prepare_bundle, sha


def model(enabled):
    head = LandmarkQueryHead(8, dimension=16, decoder_layers=1, heads=4,
                             feedforward=32, face_query_dim=224 if enabled else None)
    refined = LandmarkQueryRefinementHead(head, stages=[dict(radius_in_feature_cells=1,
                                              maximum_delta=.04, samples_per_axis=3)])
    return DinoFaceLandmarkModel(torch.nn.Conv2d(3, 8, 1), refined, variant='query_refinement')


def test_full_refinement_pair_initialization_and_forward():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    off = model(False).eval()
    torch.manual_seed(99)
    on = model(True).eval()
    report = initialize_matched_arm(off, on)
    assert report['conditioning_tensor_count'] > 0
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        a = off(x)
        b = on(x, face_query_features=torch.randn(2, 224), face_geometry=torch.randn(2, 10))
    for key in ['decoded_coordinates', 'visibility_logits']:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    on.register_buffer('unrelated_change', torch.tensor(1))
    with pytest.raises(ValueError, match='beyond'):
        initialize_matched_arm(off, on)


def test_reject_pretrained_nonzero_conditioning():
    off, on = model(False), model(True)
    for name, module in on.named_modules():
        if name.endswith('face_overlap'):
            with torch.no_grad():
                module.query_projection[-1].weight.fill_(1)
    with pytest.raises(ValueError, match='zero'):
        initialize_matched_arm(off, on)


@pytest.mark.parametrize('fault', ['pixels', 'context', 'split', 'donor_video'])
def test_bundle_rejects_stale_or_cross_split_context(tmp_path, fault):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path/'composite.png')
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    row = dict(image=dict(id=1, width=8, height=8, file_name='original.png',
        metadata=dict(split='train', sha256='source')),
        composite_file='composite.png', png_sha256=sha(tmp_path/'composite.png'),
        composition=dict(applied=True, split='train', rgb_sha256=rgb_sha,
            mask_sha256='mask', recipient_video='a', donor_video='b'),
        context=dict(detector_identity={}, query_dim=224, composite_rgb_sha256=rgb_sha,
            composition_mask_sha256='mask', context_source='fresh_composite_full_image_inference', predictions=[]),
        recipient_annotations=[])
    if fault == 'pixels':
        Image.fromarray(rgb+1).save(tmp_path/'composite.png')
    elif fault == 'context':
        row['context']['composite_rgb_sha256'] = 'clean-image'
    elif fault == 'split':
        row['image']['metadata']['split'] = 'test'
    else:
        row['composition']['donor_video'] = 'a'
    source = tmp_path/'composites.json'
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='mismatch'):
        prepare_bundle(source, tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()
