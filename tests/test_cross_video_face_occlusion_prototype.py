"""Safety checks for the isolated train-only crop-compositing prototype."""
import random
import numpy as np
import pytest

from scripts.prototype_cross_video_face_occlusion import (
    recipient_eligible, choose_donor, place,
)
from scripts.opaque_boundary_gradient import opaque_boundary_gradient


@pytest.mark.parametrize('flag,states,valid,expected', [
    (False,[2]*5,True,True), (True,[2]*5,True,False),
    (None,[2]*5,True,False), (False,[2,2,2,2,1],True,False),
    (False,[2,2,2,2,0],True,False), (False,[2]*5,False,False),
])
def test_strict_no_occlusion_recipient(flag,states,valid,expected):
    ann=dict(attribute_valid=valid,source_face_index=0,keypoint_states=states,
             keypoints=[[50,50]]*5)
    assert bool(recipient_eligible(ann,dict(attributes=dict(face_occluded=flag)))) is expected


def test_donor_is_different_video_and_photometrically_bounded():
    t=dict(video='A',descriptor=[50,0,0,10])
    pool=[dict(id=1,video='A',descriptor=[50,0,0,10],ratio=3),
          dict(id=2,video='B',descriptor=[80,0,0,10],ratio=3),
          dict(id=3,video='C',descriptor=[52,1,0,11],ratio=3)]
    assert choose_donor(t,pool,random.Random(42),True)[1]['id']==3
    assert choose_donor(t,pool[:2],random.Random(42),True) is None
    assert choose_donor(t,pool,random.Random(42),False) is None


@pytest.mark.parametrize('route,lo,hi',[('light',0,.1),('medium',.1,.2),('strong',.2,.4)])
def test_coverage_and_keypoint_labels_use_entire_pasted_rectangle(route,lo,hi):
    target=np.full((256,256,3),70,np.uint8)
    donor=np.full((256,256,3),180,np.uint8)
    points=np.array([[50,128],[206,128],[128,128],[90,175],[165,175]])
    target_data=(target,[128,128,100,80,0],points)
    donor_data=(donor,[128,128,128,128,0],points)
    successes=0
    for seed in range(12):
        out,mask,meta=place(target_data,donor_data,random.Random(seed),route)
        assert np.all(target==70)  # teacher unchanged
        if not meta['applied']:
            assert not mask.any() and np.array_equal(out,target)
            continue
        successes+=1
        assert lo < meta['overlap'] <= hi
        assert meta['donor_face_overlap'] >= .01
        assert np.array_equal(np.any(out!=target,axis=-1),mask)
        expected=np.where(mask[points[:,1],points[:,0]],1,2).tolist()
        assert meta['states']==expected
        if route=='light':assert 1 in expected
    assert successes>0


def test_crop_clips_margin_and_marks_lost_points_missing(tmp_path,monkeypatch):
    import scripts.prototype_cross_video_face_occlusion as module
    from PIL import Image
    Image.fromarray(np.full((100,100,3),90,np.uint8)).save(tmp_path/'test.png')
    monkeypatch.setattr(module,'IMAGE_ROOT',tmp_path)
    r=dict(file='test.png',ellipse=[15,50,20,15,0],bbox=[0,20,50,60],
           points=[[15,50],[30,50],[90,50],[10,45],[10,55]],states=[2]*5)
    rgb,ellipse,pts,states=module.crop_record(r,target=True)
    assert rgb.shape[0]==256 and rgb.shape[1]<256
    assert states.tolist()==[2,2,0,2,2]
    assert np.array_equal(pts[2],[0,0])
    assert module.crop_record(r,target=False) is not None


def test_missing_slot_never_becomes_occluded():
    target=(np.full((256,256,3),70,np.uint8),[128,128,100,80,0],
            np.array([[0,0],[180,100],[128,128],[90,175],[165,175]]),[0,2,2,2,2])
    donor=(np.full((256,256,3),180,np.uint8),[128,128,128,128,0],target[2])
    for seed in range(5):
        _,_,meta=place(target,donor,random.Random(seed),'strong')
        if meta['applied']:assert meta['states'][0]==0


def test_opaque_correction_cannot_reveal_underlying_target():
    target=np.full((96,96,3),100,np.uint8)
    mask=np.zeros((96,96),bool);mask[16:80,16:80]=True
    pasted=target.copy();pasted[mask]=160
    out=opaque_boundary_gradient(target,pasted,mask,width=8)
    changed=target.copy();changed[mask]=np.random.default_rng(42).integers(0,256,(mask.sum(),3),dtype=np.uint8)
    other=opaque_boundary_gradient(changed,pasted,mask,width=8)
    assert np.array_equal(out,other)  # No target texture inside the opaque support.
    assert np.array_equal(out[~mask],target[~mask])
    assert np.all(out[30:66,30:66]==160)  # Donor interior is intact.
    assert np.any(out[mask]!=pasted[mask])


def test_gradient_matches_actual_paste_boundary_without_editing_recipient():
    target=np.full((96,96,3),15,np.uint8)
    # Spatially varying background checks dependence on placement, not crop alone.
    target[:,:,1]=np.arange(96,dtype=np.uint8)[None,:]
    mask=np.zeros((96,96),bool);mask[16:80,16:80]=True
    pasted=target.copy();pasted[mask]=240
    before=target.copy(); pasted_before=pasted.copy()
    out=opaque_boundary_gradient(target,pasted,mask,width=8)
    assert np.array_equal(out[16,17:79],target[15,17:79])
    assert np.array_equal(out[79,17:79],target[80,17:79])
    assert np.array_equal(out[17:79,16],target[17:79,15])
    assert np.array_equal(out[17:79,79],target[17:79,80])
    assert np.array_equal(out[~mask],target[~mask])
    assert np.array_equal(target,before) and np.array_equal(pasted,pasted_before)
    assert np.all(out[30:66,30:66]==240)


def test_gradient_width_and_strength_are_independent_and_keep_boundary_fixed():
    target=np.full((96,96,3),40,np.uint8)
    mask=np.zeros((96,96),bool);mask[8:88,8:88]=True
    pasted=target.copy();pasted[mask]=180
    narrow=opaque_boundary_gradient(target,pasted,mask,width=8,strength=1)
    wide=opaque_boundary_gradient(target,pasted,mask,width=20,strength=1)
    strong=opaque_boundary_gradient(target,pasted,mask,width=20,strength=2)
    assert np.all(narrow[48,20]==180)  # Beyond narrow band.
    assert np.all(wide[48,20]<180)
    assert np.all(strong[48,20]<wide[48,20])
    hidden=target.copy();hidden[mask]=255
    assert np.array_equal(strong,opaque_boundary_gradient(hidden,pasted,mask,width=20,strength=2))
    for out in [narrow,wide,strong]:
        assert np.array_equal(out[~mask],target[~mask])
        assert np.all(out[48,8]==40)
        assert np.all(out[48,48]==180)


@pytest.mark.parametrize('strength',[0,-1,float('nan'),float('inf')])
def test_invalid_gradient_strength(strength):
    rgb=np.zeros((16,16,3),np.uint8);mask=np.ones((16,16),bool)
    with pytest.raises(ValueError):opaque_boundary_gradient(rgb,rgb,mask,strength=strength)
