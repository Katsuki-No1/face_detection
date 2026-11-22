import numpy as np
import pytest
import torch
from src.data.messenger_balanced_occlusion import schedule,render,SHAPES
from src.models.messenger_codebook_reconstruction import make_recovery,CoveredCodebook
from src.models.messenger_codebook_teacher import teacher_blend

@pytest.mark.parametrize('epoch',[0,1,7])
def test_balanced_targets_diverse_masks_and_patch_consistency(epoch):
    y,x=np.ogrid[:256,:256];face=((x-128)/85)**2+((y-128)/100)**2<=1
    image=np.full((256,256,3),90,np.uint8);xy=np.array([[90,95],[165,95],[128,128],[100,170],[155,170]])
    specs=list(schedule(list(range(54)),'fit',epoch))
    assert np.bincount([s['bin'] for s in specs]).tolist()==[54]*5
    hashes=[]
    for s in specs:
        rgb,c,e,p,m,mask=render(image,face,xy,s)
        assert int(m['fraction']*5)==s['bin']
        assert abs(m['fraction']-s['target'])<.001
        assert np.array_equal(rgb[~mask],image[~mask])
        ij=np.rint(xy).astype(int)//16
        assert c[ij[:,1],ij[:,0]][p].all()
        hashes.append(m['mask_sha256'])
    assert len(set(hashes))==len(hashes)
    assert set(s['shape'] for s in specs)==set(SHAPES)
    assert set(s['anchor'] for s in specs)==set(range(9))


def test_epoch_and_split_identity():
    ids=list(range(30))
    assert list(schedule(ids,'dev',1))==list(schedule(ids,'dev',99))
    assert list(schedule(ids,'fit',1))!=list(schedule(ids,'fit',2))
    assert list(schedule(ids,'dev',0))!=list(schedule(ids,'eval',0))

@pytest.mark.parametrize('arm',['continuous16','continuous32','regression384','B_d16_k4096','D_d32_k4096','E_d16_k4096'])
def test_no_original_projection_bypass_and_fusion(arm):
    torch.set_num_threads(2)
    d=32 if '32' in arm else 16
    net=make_recovery(arm,torch.eye(384)[:,:d],torch.zeros(384),torch.randn(4096,d)).eval()
    # Constant predicted F must make output independent of original X.
    net.predict_features=lambda maps: torch.ones(maps.shape[0],384,*maps.shape[-2:])
    a=torch.randn(1,3,384,4,4);b=a*5+2
    with torch.no_grad():
        ra=net(a)[0];rb=net(b)[0]
    torch.testing.assert_close(ra,rb,rtol=0,atol=0)
    torch.testing.assert_close(teacher_blend(a[:,2],ra,torch.zeros(1,4,4,dtype=torch.bool)),a[:,2],rtol=0,atol=0)
    torch.testing.assert_close(teacher_blend(a[:,2],ra,torch.ones(1,4,4,dtype=torch.bool)),ra,rtol=0,atol=0)


def test_ema_ignores_uncovered_tokens_and_eval():
    code=CoveredCodebook(torch.eye(2),'ema',decay=.5)
    z=torch.tensor([[[[1.,0.]],[[1.,9.]]]])
    code(z,torch.tensor([[[True,False]]]))
    assert int(code._counts.sum())==1
    old=code.embedding.detach().clone();code.end_step()
    assert not torch.equal(old[0],code.embedding[0])
    torch.testing.assert_close(old[1],code.embedding[1])
    code.eval();old=code.embedding.detach().clone();code(z);code.end_step()
    torch.testing.assert_close(old,code.embedding)


def test_distribution_audit_roundtrip_identity(tmp_path):
    from src.data.messenger_reconstruction_cache import audit_records
    y,x=np.ogrid[:256,:256];face=((x-128)/85)**2+((y-128)/100)**2<=1
    image=np.full((256,256,3),90,np.uint8);xy=np.array([[90,95],[165,95],[128,128],[100,170],[155,170]])
    rows=[];masks=[]
    for spec in schedule([1,2],'fit',0):
        *_,meta,mask=render(image,face,xy,spec)
        rows.append(meta);masks.append(mask)
    first=audit_records(rows,np.stack(masks),[1,2],tmp_path,'fit',0)
    second=audit_records(rows,np.stack(masks),[1,2],tmp_path,'fit',0)
    assert first==second
