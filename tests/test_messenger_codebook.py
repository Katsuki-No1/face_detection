import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.models.messenger_codebook import ARMS, VectorCodebook, directional_mse, make_recovery
from scripts import run_messenger_codebook as run
from scripts import report_messenger_codebook as report


torch.set_num_threads(2)


@pytest.fixture
def basis():
    return torch.eye(384)[:, :16]


def test_direction_weight_normalizes_and_rotates():
    generator = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(384, 384, generator=generator))
    x = (q[:, 0] + q[:, 17]).reshape(1,384,1,1).requires_grad_()
    loss = directional_mse(x,q[:,:16],4).sum()
    loss.backward()
    gradient = q.T @ x.grad.flatten()
    torch.testing.assert_close(gradient[0]/gradient[17],torch.tensor(4.))
    torch.testing.assert_close(directional_mse(x,q[:,:16],1),x.square().mean(1))
    all_axes = q.T.reshape(384,384,1,1)
    assert directional_mse(all_axes,q[:,:16],4).sum().item() == pytest.approx(1.)


@pytest.mark.parametrize('trainable',[False,True])
def test_hard_code_forward_position_gradient(trainable):
    codes=torch.tensor([[0.,0.],[2.,1.],[5.,5.]])
    quantizer=VectorCodebook(codes,trainable,chunk_size=1)
    latent=torch.tensor([[[[1.8]],[[1.2]]]],requires_grad=True)
    quantized,info=quantizer(latent)
    torch.testing.assert_close(quantized.flatten(),codes[1])
    assert info['indices'].item()==1
    # Position-only toy head: no code or commitment loss in this backward.
    (quantized[:,0]+2*quantized[:,1]).square().mean().backward()
    assert latent.grad.abs().sum()>0
    if trainable:
        assert quantizer.embedding.grad[1].abs().sum()>0
        assert quantizer.embedding.grad[[0,2]].abs().sum()==0
    else:
        assert quantizer.embedding.grad is None


def test_continuous_initial_projection_and_backward(basis):
    mean=torch.randn(384)
    net=make_recovery('continuous16',basis,mean)
    maps=torch.randn(2,3,384,4,4)
    restored,_=net(maps)
    expected=maps[:,2].clone()
    expected[:,16:]=mean[None,16:,None,None]
    torch.testing.assert_close(restored,expected,atol=1e-6,rtol=1e-5)
    restored.square().mean().backward()
    assert net.projection.weight.grad.abs().sum()>0
    assert net.context.branches[0].decode[-1].weight.grad.abs().sum()>0
    assert net.decoder_linear.weight.grad.abs().sum()>0


@pytest.mark.parametrize('count',[4096,8192])
def test_code_pairs_identical_and_no_384_skip(basis,count):
    codes=torch.randn(count,16)
    fixed=make_recovery(f'vq{count}_fixed',basis,torch.zeros(384),codes)
    trained=make_recovery(f'vq{count}_trainable',basis,torch.zeros(384),codes)
    assert run.base.tensor_digest(fixed)==run.base.tensor_digest(trained)
    maps=torch.randn(1,3,384,4,4)
    a,_=fixed(maps);b,_=trained(maps)
    torch.testing.assert_close(a,b)
    assert a.shape==(1,384,4,4)
    a.sum().backward();b.sum().backward()
    assert fixed.codebook.embedding.grad is None
    assert trained.codebook.embedding.grad.abs().sum()>0
    captured=[]
    handle=trained.decoder_linear.register_forward_pre_hook(lambda module,args:captured.append(args[0].shape))
    trained(maps);handle.remove()
    assert captured[0][1]==16
    assert not any('baseline' in arm for arm in ARMS)


def test_head_sees_masked_recovery_not_teacher(monkeypatch,basis):
    seen=[]
    def head(_,features):
        seen.append(features.detach().clone())
        return {'decoded_coordinates':features[:,:10].mean((2,3)).reshape(-1,5,2)}
    monkeypatch.setattr(run.base,'run_head',head)
    x=torch.ones(1,384,4,4);r=torch.full_like(x,2.,requires_grad=True);z=torch.full_like(x,99.)
    covered=torch.zeros(1,4,4,dtype=torch.bool);covered[:,:2]=True
    point=torch.tensor([[True,False,True,False,False]])
    parts,_=run.loss_parts(SimpleNamespace(head=None),r,x,z,covered,~covered,torch.zeros(1,5,2),point,
                           torch.ones_like(point),basis,'continuous16',{})
    torch.testing.assert_close(seen[0],torch.where(covered[:,None],r,x))
    parts['position'].backward()
    assert r.grad.abs().sum()>0
    assert r.grad[:,:,2:].abs().sum()==0


def test_dictionary_initialization_uses_only_synthetic_fit(basis):
    clean=torch.randn(4,3,384,4,4)
    cache=dict(clean=clean,parent=np.array([0,0,1,2,3]),syn_fit=np.array([0,1,2]))
    centers,audit=run.initial_codes(cache,basis,torch.zeros(384),4)
    clean[2:]=float('nan')
    other,_=run.initial_codes(cache,basis,torch.zeros(384),4)
    np.testing.assert_array_equal(centers,other)
    assert audit['fit_faces']==2 and audit['fit_patches']==32


def test_fit_basis_and_manifest():
    gram=np.diag(np.arange(384,dtype=float))
    b=run.basis_from_fit(gram)
    torch.testing.assert_close(b.T@b,torch.eye(16))
    assert b[383,0]==1
    a={'roles':{},'faces':[], 'image_hashes':{}}
    run.manifest_contract(a,a)
    with pytest.raises(ValueError):
        run.manifest_contract(a,dict(a,faces=[1]))


def prepared_output(tmp_path):
    weights=[]
    for path in report.weight_paths(tmp_path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'test weight')
        weights.append(dict(path=str(path.relative_to(tmp_path)),sha256=run.base.sha(path),bytes=path.stat().st_size))
    metric=tmp_path/'metrics.json';metric.write_text('{}')
    run.base.write(tmp_path/'numeric_completion.json',dict(all_numeric_evaluations_complete=True,
                   artifacts={'metrics.json':run.base.sha(metric)},weights=weights))
    return weights


def test_cleanup_requires_complete_and_preserves_other_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.cleanup(tmp_path)
    prepared_output(tmp_path)
    unrelated=tmp_path/'baseline.pt';unrelated.write_bytes(b'keep')
    report.cleanup(tmp_path);report.cleanup(tmp_path)
    assert unrelated.read_bytes()==b'keep'
    assert not any(path.exists() for path in report.weight_paths(tmp_path))
    assert run.read(tmp_path/'checkpoint_deletion.json')['phase']=='completed'


@pytest.mark.parametrize('target',['metrics.json','vq4096_fixed/best_R.pt'])
def test_cleanup_rejects_tampering(tmp_path,target):
    prepared_output(tmp_path)
    (tmp_path/target).write_bytes(b'changed')
    with pytest.raises(ValueError):
        report.cleanup(tmp_path)
    assert all(path.exists() for path in report.weight_paths(tmp_path))


def test_cleanup_rejects_symlink(tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    out=tmp_path/'run';out.mkdir()
    (out/ARMS[0]).symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):
        report.weight_paths(out)


def test_cleanup_resumes_partial_deletion(tmp_path):
    weights=prepared_output(tmp_path)
    run.base.write(tmp_path/'checkpoint_deletion.json',dict(phase='prepared',weights=weights))
    (tmp_path/weights[0]['path']).unlink()
    report.cleanup(tmp_path)
    assert not any(path.exists() for path in report.weight_paths(tmp_path))


def test_optimizer_resume_and_identity(tmp_path):
    net=torch.nn.Linear(2,1)
    opt,ctrl,rng,history,best,start,done=run.initial(net,tmp_path,'R','identity')
    loss=net(torch.ones(1,2)).square().sum();loss.backward();opt.step()
    history=[dict(epoch=0,dev=dict(position=.1),converged=False)]
    run.legacy.persist(tmp_path,'R',net,opt,ctrl,rng,history,.1,'identity',False)
    expected=run.base.tensor_digest(net)
    other=torch.nn.Linear(2,1)
    resumed=run.initial(other,tmp_path,'R','identity')
    assert run.base.tensor_digest(other)==expected
    assert resumed[5]==1 and resumed[4]==.1
    assert resumed[0].state_dict()['state']
    with pytest.raises(ValueError,match='identity'):
        run.initial(other,tmp_path,'R','wrong')


def test_complete_report_and_cleanup(tmp_path,monkeypatch):
    out=tmp_path/'comparison';out.mkdir()
    baseline=tmp_path/'saved';(baseline/'det_position').mkdir(parents=True)
    monkeypatch.setattr(run,'BASELINE',baseline)
    manifest=dict(roles={},faces=[],image_hashes={})
    run.base.write(baseline/'manifest.json',manifest)
    run.base.write(out/'manifest.json',manifest)
    run.base.write(out/'plan.json',{'hashes':{}})
    run.base.write(out/'training_completion.json',dict(all_converged=True,source_verified=True,baseline_retrained=False))
    for name in ('comparison','normalization','teacher_audit','gate_authentication'):
        run.base.write(out/f'{name}.json',{})
    arrays={}
    for domain in ('natural','artificial'):
        methods=3 if domain=='natural' else 4
        rows=np.zeros((2,methods,5,2),dtype=np.float32);rows[...,0]=.03;rows[...,1]=.8
        data=dict(rows=rows,ids=np.arange(2),videos=np.array(['a','b']),
                  norms=np.ones((2,3,4,4)),point_norms=np.ones((2,3,5)),gate=np.ones((2,5))*.2)
        point=np.array([[True,False,True,False,True],[False,True,False,True,False]])
        data['state' if domain=='natural' else 'point']=np.where(point,1,2) if domain=='natural' else point
        np.savez(baseline/'det_position'/f'{domain}_seed42.npz',**data)
        arrays[domain]=data
    for arm in ARMS:
        folder=out/arm;folder.mkdir()
        published=dict(identity=arm,code_usage=None,peak_gpu_bytes=0,inference_seconds=1)
        for stage in ('R','g'):
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
            names=('raw','full_R','learned_g') if domain=='natural' else ('raw','full_R','learned_g','mask_R')
            published[domain]={name:{group:run.reference.metrics(data['rows'][:,i],mask,data['videos'])
                              for group,mask in masks.items()} for i,name in enumerate(names)}
        published['artifacts']={path.name:run.base.sha(path) for path in folder.iterdir()}
        run.base.write(folder/'evaluation.json',published)
    for count in (4096,8192):
        np.save(out/f'initial_codes_{count}.npy',np.zeros((2,16)))
        run.base.write(out/f'initial_codes_{count}.json',{})
    report.report(out)
    complete=run.read(out/'numeric_completion.json')
    assert complete['all_numeric_evaluations_complete']
    assert len(complete['weights'])==26
    assert len(run.read(out/'comparison_audit.json')['conditions'])==6
    report.cleanup(out)
    report.report(out)  # Idempotent numerical report remains verifiable without weights.
    assert (baseline/'det_position/natural_seed42.npz').exists()


def test_semantic_metrics_distinguish_emission_and_position():
    rows=np.zeros((1,5,2));rows[...,0]=.2;rows[...,1]=.9
    rows[0,0,0]=.01;rows[0,1,1]=.2
    mask=np.ones((1,5),dtype=bool)
    metrics=report.semantic_metrics(rows,mask)
    assert metrics['per_slot']['right_eye']['missed']==1
    assert metrics['per_slot']['nose']['emitted']==1
    assert metrics['per_slot']['nose']['thresholds']['0.05']['f1']==0
    assert metrics['macro_f1']['0.05']['ordinary']==pytest.approx(.2)
    assert metrics['macro_f1']['0.05']['eye_weighted']==pytest.approx(2/7)
    mask[:,0]=False
    missing=report.semantic_metrics(rows,mask)
    assert missing['macro_f1']['0.05']['ordinary'] is None
    assert missing['per_slot']['left_eye']['mean_error'] is None


def test_gpu_busy_rejected(monkeypatch):
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**kw:'999999\n')
    with pytest.raises(RuntimeError,match='GPU is in use'):
        run.gpu_idle(False)


def test_queue_registration_does_not_schedule_training():
    import yaml
    config=yaml.safe_load((run.ROOT/'configs/training_queues/pending_models_tests_then_training.yaml').read_text())
    jobs={job['name']:job for job in config['jobs']}
    for arm in ARMS:
        job=jobs[f'messenger_codebook_{arm}_diagnostic']
        assert job['enabled'] is False and job['phase']=='evaluation'
        assert 'train' in job['command']


def test_saved_float32_threshold_boundaries_preserved():
    rows=np.full((2,5,2),.05,dtype=np.float32);rows[...,1]=.7
    mask=np.ones((2,5),dtype=bool)
    result=report.summarize(rows,mask,np.array(['a','b']),rows)
    published=run.reference.metrics(rows,mask,np.array(['a','b']))
    assert result['thresholds']['0.05']['joint']==published['thresholds']['0.05']['joint']==10
    assert result['presence_pass']==10
