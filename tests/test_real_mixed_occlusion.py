import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from PIL import Image

from keypoint_comparison1.landmark_detection.real_occlusion import (
    RealOcclusionBank,apply_real_mixed,digest,identity,is_clean_annotation,patch_avoids_faces)
from keypoint_comparison1.landmark_detection.real_recovery_metrics import RealRecoveryMeter
from keypoint_comparison1.landmark_detection.dataset import CropAugmentation,TopDownFaceLandmarkDataset


@pytest.fixture
def fixture_bank(tmp_path):
    images=[];anns=[]
    for i in range(2):
        file=f'{i}.png';Image.fromarray(np.full((256,256,3),70+80*i,np.uint8)).save(tmp_path/file)
        images.append(dict(id=i,file_name=file,width=256,height=256,metadata=dict(source_video_id=f'v{i}',split='train')))
        anns.append(dict(id=i,image_id=i,category_id=2,bbox=[30,30,196,196],ellipse=[128,128,98,90,0],
            keypoints=[60,128,2,196,128,2,128,128,2,100,170,2,155,170,2],attributes=dict(face_occluded=False)))
    annotation=tmp_path/'train.json';annotation.write_text(json.dumps(dict(images=images,annotations=anns)))
    rows=[dict(file=im['file_name'],video=f'v{i}',width=256,height=256,sha256=digest(tmp_path/im['file_name']),
               descriptor=[30,0,0,10],face_exclusion_boxes=[[30,30,226,226]]) for i,im in enumerate(images)]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema='real-occlusion-bank-v1',split='train',
        canonical_path=str(annotation),canonical_sha256=digest(annotation),adapter_sha256=digest(annotation),image_root=str(tmp_path),
        images=rows,faces=[dict(id=i,file=f'{i}.png',box=[25,25,231,231],ellipse=a['ellipse'],descriptor=[30,0,0,10]) for i,a in enumerate(anns)],
        eligible_recipients=[identity(f'{i}.png',a['ellipse']) for i,a in enumerate(anns)])))
    bank=RealOcclusionBank(manifest,digest(manifest),annotation_path=annotation,image_root=tmp_path)
    return bank,annotation,manifest,images,anns


def test_strict_original_eligibility():
    a=dict(attributes=dict(face_occluded=False),keypoints=[10,10,2]*5)
    assert is_clean_annotation(a)
    for flag in [True,None]:
        assert not is_clean_annotation(dict(a,attributes=dict(face_occluded=flag)))
    for state in [0,1]:
        b=dict(a,keypoints=a['keypoints'][:]);b['keypoints'][2]=state
        assert not is_clean_annotation(b)


def test_non_face_support_rejects_even_partial_face_overlap():
    assert patch_avoids_faces([0,0,10,10],[[10,10,20,20]])
    assert not patch_avoids_faces([0,0,11,11],[[10,10,20,20]])


def test_bank_provenance_and_validation_split_rejection(fixture_bank,tmp_path):
    bank,annotation,manifest,_,_=fixture_bank
    with pytest.raises(ValueError,match='hash mismatch'):
        RealOcclusionBank(manifest,'wrong',annotation_path=annotation,image_root=tmp_path)
    val=tmp_path/'val.json';val.write_text('{}')
    with pytest.raises(ValueError,match='Recipient annotation'):
        RealOcclusionBank(manifest,digest(manifest),annotation_path=val,image_root=tmp_path)


def test_actual_dataset_pair_deterministic_clean_teacher_and_missing(fixture_bank,tmp_path):
    bank,annotation,manifest,images,anns=fixture_bank
    aug=CropAugmentation(scale_jitter=0,center_jitter=0,rotation_deg=0,brightness=0,contrast=0,saturation=0,
        blur_probability=0,horizontal_flip_probability=1,occlusion_probability=0,
        paired_occlusion=dict(probability=1,appearance='real_mixed',material_manifest=str(manifest),material_manifest_sha256=digest(manifest)))
    ds=TopDownFaceLandmarkDataset(annotation_path=annotation,image_root=tmp_path,training=True,augmentation=aug)
    random.seed(42);x,t=ds[0]
    random.seed(42);xx,tt=ds[0]
    assert torch.equal(x,xx) and t['paired_occlusion']==tt['paired_occlusion']
    mask=t['occlusion_mask'];assert torch.equal(x[:,~mask],t['clean_image'][:,~mask])
    assert not t['synthetic_truncated'].any()
    assert t['paired_occlusion']['donor_video']!='v0'
    assert t['paired_occlusion']['applied']
    # A source point marked missing disqualifies the whole recipient, not just that point.
    ds.samples[0]['annotation']['keypoints'][2]=0
    assert ds._paired_occlusion_rng(0) is None
    assert ds[0][1]['occlusion_mask'].sum()==0


def test_family_does_not_switch_when_placement_fails(fixture_bank):
    bank,_,_,images,anns=fixture_bank
    bank.donor=lambda *args:None
    sample=dict(image=images[0],annotation=anns[0],gt_ellipse=anns[0]['ellipse'])
    rgb=np.full((256,256,3),70,np.uint8);points=np.array(anns[0]['keypoints']).reshape(5,3)[:,:2]
    families=[]
    for seed in range(100):
        out,mask,states,m=apply_real_mixed(rgb,sample=sample,face_ellipse=anns[0]['ellipse'],keypoints=points,
                                        keypoint_visibility=[2,2,0,2,2],bank=bank,rng=random.Random(seed))
        assert not m['applied'] and not mask.any() and np.array_equal(out,rgb) and states[2]==0
        families.append(m['requested_family'])
    assert 35<families.count('face')<65


def test_feature_metrics_separate_final_from_candidate_and_do_not_fake_empty_support():
    clean=torch.tensor([[[[1.,1.]],[[0.,0.]]]])
    raw=torch.tensor([[[0.,1.],[1.,0.]]]);candidate=torch.tensor([[[1.,0.],[1.,0.]]])
    masks=torch.tensor([[[True,False]]]);targets=[dict(paired_occlusion=dict(applied=True,requested_family='face'))]
    meter=RealRecoveryMeter();meter.update(dict(original_tokens=raw,candidate_tokens=candidate,recovered_tokens=raw),clean,masks,targets,patch_size=1)
    r=meter.report()
    assert r['face/masked']['metrics']['final_minus_raw_cosine']==0
    assert r['face/masked']['metrics']['candidate_cosine']==1
    assert r['non_face/masked']['metrics'] is None
    better=RealRecoveryMeter();better.update(dict(original_tokens=raw,candidate_tokens=candidate,recovered_tokens=candidate),clean,masks,targets,patch_size=1)
    assert better.report()['face/masked']['metrics']['final_minus_raw_cosine']>0


def test_recovery_on_off_configs_share_augmentation():
    from keypoint_comparison1.landmark_detection.config import load_landmark_config
    root=Path(__file__).resolve().parents[1]/'keypoint_comparison1/configs'
    on=load_landmark_config(root/'continuous_recovery_real_mixed.yaml').raw
    off=load_landmark_config(root/'continuous_recovery_real_mixed_control.yaml').raw
    assert on['data']==off['data'] and on['train']==off['train']
    assert on['model']['continuous_recovery']['enabled'] and not off['model']['continuous_recovery']['enabled']
    assert on['outputs']['root']!=off['outputs']['root']


def test_forced_family_stays_fixed_and_cannot_bypass_clean_recipient(fixture_bank):
    bank,_,_,images,anns=fixture_bank
    requested=[]
    def no_donor(family,*args):
        requested.append(family)
        return None
    bank.donor=no_donor
    sample=dict(image=images[0],annotation=anns[0])
    kwargs=dict(sample=sample,face_ellipse=anns[0]['ellipse'],
        keypoints=np.array(anns[0]['keypoints']).reshape(5,3)[:,:2],
        keypoint_visibility=[2]*5,bank=bank,rng=random.Random(42),force_family='non_face')
    rgb=np.zeros((256,256,3),np.uint8)
    _,_,_,meta=apply_real_mixed(rgb,**kwargs)
    assert requested==['non_face']*8 and meta['requested_family']=='non_face'
    sample['annotation']['attributes']['face_occluded']=True
    requested.clear()
    _,_,_,meta=apply_real_mixed(rgb,**kwargs)
    assert not requested and meta['reason']=='recipient_not_originally_unoccluded'


def test_real_mixed_schema_requires_explicit_matching_arm_and_locked_policy():
    import copy
    from keypoint_comparison1.landmark_detection.config import load_landmark_config
    from scripts.run_continuous_recovery_gate import validate_real_mixed_contract
    root=Path(__file__).resolve().parents[1]/'keypoint_comparison1/configs'
    on=load_landmark_config(root/'continuous_recovery_real_mixed.yaml').raw
    off=load_landmark_config(root/'continuous_recovery_real_mixed_control.yaml').raw
    assert validate_real_mixed_contract(on) and validate_real_mixed_contract(off)
    bad=copy.deepcopy(on);bad['model']['continuous_recovery']['enabled']=False
    with pytest.raises(ValueError,match='disagree'):validate_real_mixed_contract(bad)
    bad=copy.deepcopy(on);bad['augmentation_contract']['family_probabilities']['face']=.9
    with pytest.raises(ValueError,match='contract'):validate_real_mixed_contract(bad)
