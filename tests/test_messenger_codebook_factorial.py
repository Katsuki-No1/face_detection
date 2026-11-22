import io
import numpy as np
import pytest
import torch
from src.models.messenger_codebook_factorial import VectorCodebook, ARMS, arm_spec, make_recovery
from src.models.messenger_codebook import make_recovery as historical_model
from scripts import run_messenger_codebook_factorial as run
from scripts import gate_messenger_codebook_factorial as gates
from scripts import report_messenger_codebook_factorial as report

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_lookup_normalization_preserves_selected_code_magnitude():
    centers=torch.tensor([[2.,0.],[0.,10.]])
    x=torch.tensor([[[[1.]],[[2.]]]],requires_grad=True)
    raw=VectorCodebook(centers,False)
    unit=VectorCodebook(centers,True)
    a,_=raw(x);b,detail=unit(x)
    torch.testing.assert_close(a.flatten(),centers[0])
    torch.testing.assert_close(b.flatten(),centers[1])
    scaled,_=unit(x*10)
    torch.testing.assert_close(b,scaled)
    b.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x))
    assert unit.embedding.grad[1].abs().sum()>0
    assert unit.embedding.grad[0].abs().sum()==0


@pytest.mark.parametrize('normalized',[False,True])
def test_ema_deferred_accumulation_eval_and_resume(normalized):
    centers=torch.tensor([[2.,0.],[0.,2.],[-2.,-2.]])
    x=torch.tensor([[[[3.,4.]],[[.1,.2]]],[[[.1,.2]],[[3.,4.]]]],requires_grad=True)
    a=VectorCodebook(centers,normalized,'ema',decay=.5)
    b=VectorCodebook(centers,normalized,'ema',decay=.5)
    a.begin_step();out,_=a(x)
    out.sum().backward()
    assert a.embedding.grad is None and x.grad.abs().sum()>0
    torch.testing.assert_close(a.embedding,centers)
    b.begin_step()
    for part in x.split(1): b(part)
    a.end_step();b.end_step()
    torch.testing.assert_close(a.embedding,b.embedding)
    torch.testing.assert_close(a.embedding[2],centers[2])
    torch.testing.assert_close(a.embedding[:2],torch.tensor([[2.75,.075],[.075,2.75]]))
    assert int(a.ema_steps)==1
    before={k:v.clone() for k,v in a.state_dict().items()}
    a.eval();a(x);a.end_step()
    for k,v in before.items(): torch.testing.assert_close(a.state_dict()[k],v)
    payload=io.BytesIO();torch.save(a.state_dict(),payload);payload.seek(0)
    c=VectorCodebook(centers,normalized,'ema',decay=.5)
    c.load_state_dict(torch.load(payload,weights_only=True))
    a.train();c.train()
    a(x);c(x);a.end_step();c.end_step()
    for k,v in a.state_dict().items(): torch.testing.assert_close(c.state_dict()[k],v)


@pytest.mark.parametrize('count',[4096,8192])
def test_A16_is_historical_model_forward_and_gradient(count):
    torch.manual_seed(3)
    basis=torch.eye(384)[:,:16];mean=torch.randn(384);codes=torch.randn(count,16)
    old=historical_model(f'vq{count}_trainable',basis,mean,codes)
    new=make_recovery(f'A_d16_k{count}',basis,mean,codes)
    maps=torch.randn(1,3,384,4,4)
    a,_=old(maps);b,_=new(maps)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().mean().backward();b.square().mean().backward()
    for (ka,pa),(kb,pb) in zip(old.named_parameters(),new.named_parameters()):
        assert ka==kb
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


@pytest.mark.parametrize('dimension',[16,32])
def test_E_initial_parity_and_nonlocal_quantized_decoder(dimension):
    basis=torch.eye(384)[:,:dimension];mean=torch.zeros(384);codes=torch.randn(4096,dimension)
    d=make_recovery(f'D_d{dimension}_k4096',basis,mean,codes).eval()
    e=make_recovery(f'E_d{dimension}_k4096',basis,mean,codes).eval()
    maps=torch.randn(1,3,384,4,4)
    torch.testing.assert_close(d(maps)[0],e(maps)[0],rtol=0,atol=0)
    z=torch.randn(1,dimension,4,4,requires_grad=True)
    # D's decoder cannot carry a change at another position to the selected cell.
    dcell=(d.decoder_linear(z)+d.decoder_residual(z))[0,:,0,0].sum()
    dg=torch.autograd.grad(dcell,z)[0]
    assert dg[:,:,1:,1:].abs().sum()==0
    # A learned E residual has a true nonlocal path after quantization.
    torch.nn.init.normal_(e.spatial.output.weight,std=.01)
    eg=torch.autograd.grad(e.spatial(z)[0,:,0,0].sum(),z)[0]
    assert eg[:,:,1:,1:].abs().sum()>0


def test_scope_basis_and_no_eval_direction_fit():
    assert len(ARMS)==21 and ARMS[0]=='continuous32'
    gram=np.diag(np.arange(384,dtype=float))
    b=run.basis_from_fit(gram)
    assert b.shape==(384,32)
    torch.testing.assert_close(b.T@b,torch.eye(32))
    assert b[383,0]==1
    assert sum(arm_spec(a)['spatial'] for a in ARMS)==4
    assert run.PLAN['pretraining'] is False


def test_cleanup_scope_and_tamper_guard(tmp_path):
    weights=[]
    for p in report.weight_paths(tmp_path):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'weight')
        weights.append(dict(path=str(p.relative_to(tmp_path)),sha256=run.base.sha(p),bytes=p.stat().st_size))
    assert len(weights)==88
    metric=tmp_path/'metric.json';metric.write_text('{}')
    run.base.write(tmp_path/'numeric_completion.json',dict(all_numeric_evaluations_complete=True,
        artifacts={'metric.json':run.base.sha(metric)},weights=weights))
    extra=tmp_path/'baseline.pt';extra.write_bytes(b'keep')
    metric.write_text('changed')
    with pytest.raises(ValueError): report.cleanup(tmp_path)
    assert all(p.exists() for p in report.weight_paths(tmp_path))
    metric.write_text('{}');report.cleanup(tmp_path);report.cleanup(tmp_path)
    assert extra.read_bytes()==b'keep'
    assert not any(p.exists() for p in report.weight_paths(tmp_path))


def test_gate_authentication_all_84_markers(tmp_path,monkeypatch):
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
    run.base.write(out/'training_completion.json',dict(all_converged=True,source_verified=True,baseline_retrained=False))
    for name in ('comparison','normalization','teacher_audit','gate_authentication'):
        run.base.write(out/f'{name}.json',{})
    historical=run.ROOT/'outputs/diagnostics/messenger_codebook_20261119'
    fake_root=tmp_path/'project'
    fake_historical=fake_root/'outputs/diagnostics/messenger_codebook_20261119'
    (fake_historical/'continuous16').mkdir(parents=True)
    monkeypatch.setattr(run,'ROOT',fake_root)
    run.base.write(fake_historical/'numeric_completion.json',dict(artifacts={}))
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
        np.savez(fake_historical/'continuous16'/f'{domain}_seed42.npz',**data)
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
    for dimension in (16,32):
        for count in (4096,8192):
            np.save(out/f'initial_codes_d{dimension}_k{count}.npy',np.zeros((2,dimension)))
            run.base.write(out/f'initial_codes_d{dimension}_k{count}.json',{})
    report.report(out)
    complete=run.read(out/'numeric_completion.json')
    assert complete['all_numeric_evaluations_complete']
    assert len(complete['weights'])==88
    assert len(run.read(out/'comparison_audit.json')['conditions'])==21
    report.cleanup(out)
    report.report(out)  # Idempotent numerical report remains verifiable without weights.
    assert (baseline/'det_position/natural_seed42.npz').exists()
