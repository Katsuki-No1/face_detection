from copy import deepcopy
import pytest
import torch
from scripts.prepare_real_face_for_validation import verify_record, object_sha
from scripts.real_face_for_validation import validate_for, CONTRACT


def test_resume_rejects_changed_source_contract_and_payload():
    payload = dict(image_id=7, image_sha256='pixels', contract_sha256='contract', predictions=[])
    record = dict(payload=payload, payload_sha256=object_sha(payload))
    assert verify_record(record, {'id':7}, 'contract', 'pixels') == payload
    for image, contract, pixels in [({'id':8},'contract','pixels'),
                                  ({'id':7},'changed','pixels'), ({'id':7},'contract','changed')]:
        with pytest.raises(ValueError, match='provenance'):
            verify_record(record,image,contract,pixels)
    changed = deepcopy(record)
    changed['payload']['predictions'] = [{}]
    with pytest.raises(ValueError, match='provenance'):
        verify_record(changed, {'id':7}, 'contract', 'pixels')


class ValidationSet:
    training = False
    face_overlap_query_dim = 224
    input_size = 11
    detector_crop_report = {'matching_contract':CONTRACT,'normalization_bbox_source':'GT Head'}

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(3,11,11), dict(
            inverse_affine=torch.tensor([[1.,0.,0.],[0.,1.,0.]]),
            bbox_xyxy=torch.tensor([0.,0.,10.,10.]),
            original_keypoints=torch.full((5,2),5.),
            original_labeled=torch.tensor([True,True,True,True,index==0]),
            face_query_features=torch.zeros(224), face_geometry=torch.zeros(10),
            face_query_index=torch.tensor(index))


class Probe(torch.nn.Module):
    def __init__(self, enabled):
        super().__init__()
        self.face_overlap_enabled = enabled
        self.seen = []

    def forward(self, images, **context):
        self.seen.append(set(context))
        return dict(decoded_coordinates=torch.full((len(images),5,2),.5),
                    presence_logits=torch.full((len(images),5),10.))


def test_same_cohort_macro_f1_with_only_conditioning_toggled():
    off, on = Probe(False), Probe(True)
    args = dict(dataset=ValidationSet(),device=torch.device('cpu'),batch_size=2)
    a,b = validate_for(off,**args), validate_for(on,**args)
    assert a == b
    assert a['semantic_tp'] == [2,2,2,2,1]
    assert a['semantic_fp'] == [0,0,0,0,1]
    assert a['conditional_semantic_macro_f1_nme_0.10'] == pytest.approx((4+2/3)/5)
    assert off.seen == [set()]
    assert on.seen == [{'face_query_features','face_geometry'}]
    invalid = ValidationSet()
    invalid.training = True
    with pytest.raises(ValueError,match='unaugmented'):
        validate_for(on,invalid,device=torch.device('cpu'),batch_size=2)
