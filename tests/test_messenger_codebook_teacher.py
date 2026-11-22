import numpy as np
import pytest
import torch
from types import SimpleNamespace
from src.models.messenger_codebook_teacher import ARMS, make_recovery, teacher_blend
from scripts import run_messenger_codebook_teacher as run
from scripts import gate_messenger_codebook_teacher as gates
from scripts import report_messenger_codebook_teacher as report

@pytest.fixture(autouse=True)
def threads(): torch.set_num_threads(2)


def test_fixed_teacher_g_endpoints_and_gradient():
    x=torch.randn(2,384,4,4)
    r=torch.randn_like(x,requires_grad=True)
    g=torch.full((2,4,4),.25,requires_grad=True)
    y=teacher_blend(x,r,g)
    torch.testing.assert_close(y,.75*x+.25*r)
    y.sum().backward()
    assert g.grad is None
    torch.testing.assert_close(r.grad,torch.full_like(r,.25))
    torch.testing.assert_close(teacher_blend(x,r,torch.zeros_like(g)),x)
    torch.testing.assert_close(teacher_blend(x,r,torch.ones_like(g)),r)
    mask=torch.rand(2,4,4)>.5
    torch.testing.assert_close(teacher_blend(x,r,mask),torch.where(mask[:,None],r,x),rtol=0,atol=0)
    with pytest.raises(ValueError):teacher_blend(x,r,torch.full_like(g,2))


def test_R_position_head_receives_teacher_blend_not_clean_features(monkeypatch):
    seen=[]
    def head(_,features):
        seen.append(features.detach().clone())
        return {'decoded_coordinates':features[:,:10].mean((2,3)).reshape(-1,5,2)}
    monkeypatch.setattr(run.base,'run_head',head)
    x=torch.ones(1,384,4,4);r=torch.full_like(x,2.,requires_grad=True);z=torch.full_like(x,99.)
    mask=torch.zeros(1,4,4,dtype=torch.bool);mask[:,:2]=True
    point=torch.tensor([[True,False,True,False,False]])
    parts,_=run.loss_parts(SimpleNamespace(head=None),r,x,z,mask,~mask,torch.zeros(1,5,2),point,
        torch.ones_like(point),torch.eye(384)[:,:32],'continuous32',{})
    torch.testing.assert_close(seen[0],torch.where(mask[:,None],r,x))
    parts['position'].backward()
    assert r.grad[:,:,:2].abs().sum()>0 and r.grad[:,:,2:].abs().sum()==0


def test_fresh_controls_and_no_g_training(tmp_path):
    assert len(ARMS)==23 and ARMS[:3]==('continuous32','regression384','continuous16')
    assert run.PLAN['learned_g'] is False
    assert not hasattr(run,'train_gate') and not hasattr(gates,'fit_g')
    assert all(p.name in ('best_R.pt','last_R.pt') or p.suffix=='.npy' for p in report.weight_paths(tmp_path))
    basis=torch.eye(384)[:,:32];mean=torch.zeros(384)
    for arm in ('continuous16','regression384'):
        net=make_recovery(arm,basis,mean)
        result,_=run.generate(net,torch.randn(1,3,384,4,4))
        assert result.shape==(1,384,4,4) and torch.isfinite(result).all()
    with pytest.raises(ValueError,match='Only R'):
        run.initial(torch.nn.Linear(1,1),tmp_path,'g','x')

def test_gate_authentication_all_92_markers(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(gates,'OUT',tmp_path)
    run.base.write(tmp_path/'plan.json',dict(gate_plan=gates.GATE_PLAN,training_plan=run.PLAN,hashes={},
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__)))
    run.base.write(tmp_path/'initialization.json',dict(code_sha256={'d16_k4096':'abc'}))
    for arm in ARMS:
        for phase in gates.PHASES:
            folder=tmp_path/arm/phase
            run.base.write(folder/'report.json',dict(passed=True))
            run.base.write(folder/'success.json',dict(passed=True,context='same',report_sha256=run.base.sha(folder/'report.json')))
    hashes={str(p.relative_to(tmp_path)):run.base.sha(p) for p in tmp_path.rglob('*.json')}
    run.base.write(tmp_path/'completion.json',dict(passed=True,context='same',artifacts=hashes))
    assert gates.authenticate(code_hashes={'d16_k4096':'abc'})['passed']
    (tmp_path/ARMS[-1]/'small_scale/success.json').unlink()
    with pytest.raises(FileNotFoundError): gates.authenticate()


def test_complete_report_and_cleanup(tmp_path,monkeypatch):
    out=tmp_path/'comparison';out.mkdir()
    baseline=tmp_path/'saved';(baseline/'det_position').mkdir(parents=True)
    monkeypatch.setattr(run,'BASELINE',baseline)
    manifest=dict(roles={},faces=[],image_hashes={})
    run.base.write(baseline/'manifest.json',manifest)
    run.base.write(out/'manifest.json',manifest)
    run.base.write(out/'plan.json',{'hashes':{}})
    run.base.write(out/'training_completion.json',dict(all_converged=True,source_verified=True,fixed_parent_retrained=False,controls_retrained=True,learned_g=False))
    for name in ('comparison','normalization','teacher_audit','gate_authentication'):
        run.base.write(out/f'{name}.json',{})
    arrays={}
    for domain in ('natural','artificial'):
        methods=3
        rows=np.zeros((2,methods,5,2),dtype=np.float32);rows[...,0]=.03;rows[...,1]=.8
        data=dict(rows=rows,ids=np.arange(2),videos=np.array(['a','b']),
                  norms=np.ones((2,3,4,4)),point_norms=np.ones((2,3,5)),gate=np.ones((2,5))*.2)
        point=np.array([[True,False,True,False,True],[False,True,False,True,False]])
        data['state' if domain=='natural' else 'point']=np.where(point,1,2) if domain=='natural' else point
        np.savez(baseline/'det_position'/f'{domain}_seed42.npz',**data)
        data['teacher_g']=np.ones((2,4,4),dtype=np.float32)
        data['covered']=data['teacher_g'].astype(bool)
        data['face_occluded']=np.array([False,True])
        arrays[domain]=data
    for arm in ARMS:
        folder=out/arm;folder.mkdir()
        published=dict(identity=arm,code_usage=None,peak_gpu_bytes=0,inference_seconds=1)
        for stage in ('R',):
            history=[dict(epoch=0,dev=dict(position=.1),converged=True)]
            convergence=dict(selected_epoch=0,stop_epoch=0,converged=True)
            published[stage]=convergence
            run.base.write(folder/f'{stage}_history.json',history)
            run.base.write(folder/f'{stage}_convergence.json',convergence)
            for name in ('best','last'):
                torch.save(dict(identity=arm,converged=True,epoch=0,state={}),folder/f'{name}_{stage}.pt')
        run.base.write(folder/'initialization.json',dict(model_sha256='shared'))
        for domain,data in arrays.items():
            np.savez(folder/f'{domain}_seed42.npz',**data)
            masks={'occluded':data['state']==1,'visible':data['state']==2} if domain=='natural' else {'covered':data['point'],'uncovered':~data['point']}
            names=('raw','full_R','teacher_g')
            published[domain]={name:{group:run.reference.metrics(data['rows'][:,i],mask,data['videos'])
                              for group,mask in masks.items()} for i,name in enumerate(names)}
        published['artifacts']={path.name:run.base.sha(path) for path in folder.iterdir()}
        run.base.write(folder/'evaluation.json',published)
    for dimension in (16,32):
        for count in (4096,8192):
            np.save(out/f'initial_codes_d{dimension}_k{count}.npy',np.zeros((2,dimension)))
            run.base.write(out/f'initial_codes_d{dimension}_k{count}.json',{})
    report.report(out)
    complete=run.read(out/'numeric_completion.json')
    assert complete['all_numeric_evaluations_complete']
    assert len(complete['weights'])==50
    assert len(run.read(out/'comparison_audit.json')['conditions'])==23
    report.cleanup(out)
    report.report(out)  # Idempotent numerical report remains verifiable without weights.
    assert (baseline/'det_position/natural_seed42.npz').exists()




def test_boolean_teacher_statistics_regression():
    values=np.array([False,True,False,True])
    stats=run.teacher_statistics(values)
    assert stats == dict(mean=.5,median=.5,p90=1.)
    assert run.teacher_statistics(np.array([],dtype=bool)) == dict(mean=None,median=None,p90=None)
