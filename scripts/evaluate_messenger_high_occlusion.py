"""Frozen R evaluation under 90 high-area opaque occlusions per held-out face; known mask g."""
import fcntl, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import cv2
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_messenger_multiaxis as base
from scripts.run_messenger_adversarial_comparison import new_recovery, ARMS
from scripts.train_v3_shape_recovery_gate import Crops,stable,ellipse_aligned_rectangle,build_crop_transform
from src.data.v3_shape_occlusion import silhouette,SHAPES
from scripts.evaluate_messenger_success_thresholds import paired_interval
SOURCE=ROOT/'outputs/diagnostics/messenger_adversarial_comparison_20261119'
OUT=ROOT/'outputs/diagnostics/messenger_high_occlusion_20261119'
NAMES=('raw',*ARMS)

def read(p):return json.loads(p.read_text())
def support_for(a):
    box,angle=ellipse_aligned_rectangle(a['ellipse'])
    crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
    cx,cy=crop.original_to_crop(np.array([a['ellipse'][:2]]))[0]
    rx,ry=np.array(a['ellipse'][2:4])*255/crop.side
    yy,xx=np.ogrid[:256,:256]
    face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
    return face,face.reshape(16,16,16,16).mean((1,3))>=.5

def high_occlude(image,face,coord,kind,seed,lo,hi):
    """Choose a requested face-overlap area geometrically, never from model outputs."""
    rng=np.random.default_rng(seed)
    stamp=silhouette(kind,rng)
    distance=cv2.distanceTransform(stamp,cv2.DIST_L2,3)
    sy,sx=np.nonzero(distance>2)
    if not len(sx):sy,sx=np.nonzero(stamp)
    xy=np.rint(coord).astype(int)
    candidates=np.flatnonzero(face[xy[:,1],xy[:,0]])
    assert len(candidates)
    slot=int(rng.choice(candidates));pivot=int(rng.integers(len(sx)))
    center=(float(sx[pivot]),float(sy[pivot]))
    angle=float(rng.uniform(-180,180));target=float(rng.uniform(lo+.005,hi-.005))
    def render(scale):
        matrix=cv2.getRotationMatrix2D(center,angle,scale)
        matrix[:,2]+=coord[slot]-np.array(center)
        mask=cv2.warpAffine(stamp,matrix,(256,256),flags=cv2.INTER_NEAREST).astype(bool)
        return mask,float((mask&face).sum()/face.sum())
    lower=.001;upper=1.
    for _ in range(24):
        mask,fraction=render(upper)
        if fraction>=target:break
        upper*=2
    else:raise RuntimeError('Cannot bracket target coverage')
    for _ in range(18):
        scale=(lower+upper)/2
        mask,fraction=render(scale)
        if fraction<target:lower=scale
        else:upper=scale
    assert lo<=fraction<hi,(kind,lo,hi,target,fraction)
    assert mask[xy[slot,1],xy[slot,0]]
    texture=cv2.resize(rng.normal(0,18,(5,5,3)).astype('float32'),(256,256),interpolation=cv2.INTER_CUBIC)
    color=rng.uniform(28,225,3)
    image=image.copy();image[mask]=np.clip(color+texture[mask],0,255).astype(np.uint8)
    return image,mask,target

def variants(row,a):
    face,support=support_for(a)
    coord=row['xy']*255
    ix,iy=np.rint(coord).astype(int).T
    result=[]
    for bin_id,(lo,hi) in enumerate([(.4,.6),(.6,.8),(.8,.9)]):
        for kind_id,kind in enumerate(SHAPES):
            for repeat in range(10):
                seed=stable(f'highmask20261119:{a["id"]}:{bin_id}:{kind}:{repeat}')
                img,mask,target=high_occlude(row['image'],face,coord,kind,seed,lo,hi)
                assert np.array_equal(img[~mask],row['image'][~mask])
                c=support & mask.reshape(16,16,16,16).any((1,3))
                point=mask[iy,ix]
                assert c.any() and point.any()
                result.append(dict(image=img,cov=c,point=point,kind=kind_id,repeat=repeat,seed=seed,
                    fraction=float((face&mask).sum()/face.sum()),attempt=0,bin_id=bin_id,target=target))
    return result


def summarize(rows,point,videos,select=None):
    if select is None:select=np.ones(len(rows),bool)
    result={}
    for label,mask in [('covered',point),('uncovered',~point),('all',np.ones_like(point))]:
        m=mask & select[:,None]
        if not m.any():continue
        v=np.broadcast_to(videos[:,None],point.shape)[m]
        result[label]={}
        for k,name in enumerate(NAMES):
            er=rows[:,k,:,0][m].astype(float);pr=rows[:,k,:,1][m]
            ref=rows[:,0,:,0][m].astype(float)
            result[label][name]=dict(points=len(er),mean=float(er.mean()),median=float(np.median(er)),
                p90=float(np.quantile(er,.9)),location5=int((er<=.05).sum()),
                joint5=int(((er<=.05)&(pr>=.7)).sum()),location10=int((er<=.1).sum()),
                joint10=int(((er<=.1)&(pr>=.7)).sum()),emitted=int((pr>=.7).sum()),
                delta=paired_interval(er-ref,v),improved=int((er<ref).sum()),worsened=int((er>ref).sum()))
    return result

@torch.no_grad()
def run():
    OUT.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32=True
    with (OUT/'singleton.lock').open('a') as local, (ROOT/'outputs/training_queues/pending_models_tests_then_training/queue.lock').open('a') as lease:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                base.status(OUT,'waiting_for_canonical_lease');time.sleep(15)
        try:
            source_plan=read(SOURCE/'plan.json')
            for p,h in source_plan['hashes'].items():assert base.sha(ROOT/p)==h,p
            assert read(SOURCE/'numeric_completion.json')['complete']
            manifest=read(SOURCE/'manifest.json')
            hashes={**source_plan['hashes'],str(Path(__file__).relative_to(ROOT)):base.sha(__file__),
                    str((SOURCE/'manifest.json').relative_to(ROOT)):base.sha(SOURCE/'manifest.json')}
            for arm in ARMS:
                p=SOURCE/arm/'best_R.pt';hashes[str(p.relative_to(ROOT))]=base.sha(p)
                assert base.sha(p)==read(SOURCE/arm/'evaluation.json')['weights']['best_R.pt']
            base.write(OUT/'plan.json',dict(faces=128,patterns_per_face=90,shapes=list(SHAPES),repeats_per_bin=10,
                coverage_bins=[[.4,.6],[.6,.8],[.8,.9]],g='known binary patch mask intersected with face support, exactly as R training',
                names=NAMES,training=False,clean_head=False,source_hashes=hashes,
                seed='stable SHA256 highmask20261119:face:bin:kind:repeat; geometric scale search to target uniform area within each bin, no prediction-based selection',
                metrics='Euclidean distance / GT ellipse major diameter; presence threshold .7',
                scope='same reused train-derived eval128 faces; no fit/dev, no global val/test; 90 correlated masks per face; high-area extrapolation beyond R training10-40%',
                inference='FP32, TF32 on; backbone maps rounded to FP16 then restored to FP32 as original cache'))
            data=yaml.safe_load(base.CONFIG.read_text())
            corpus=read(base.ANN);images={a['id']:a for a in corpus['images']}
            ids=set(manifest['roles']['eval']['synthetic'])
            records=sorted([a for a in corpus['annotations'] if a['id'] in ids],key=lambda a:a['id'])
            assert len(records)==128
            evalvids={images[a['image_id']]['metadata']['source_video_id'] for a in records}
            prior=read(base.PRIOR)
            assert evalvids <= set(prior['roles']['eval'])
            assert not evalvids & (set(prior['roles']['fit'])|set(prior['roles']['dev']))
            for a in records:
                im=images[a['image_id']]
                assert im['metadata']['split']=='train'
                assert manifest['image_hashes'][str(a['image_id'])]==im['metadata']['sha256']
            ds=Crops(records,images,Path(data['data']['root']))
            model,_=base.build_landmark_model(data,variant='query_refinement',initialize_from_detector=False)
            base.load_landmark_checkpoint(model,base.BASE,prefer_ema=True,strict=True)
            model.cuda().eval().requires_grad_(False);fixed=base.tensor_digest(model)
            nets=[]
            for arm in ARMS:
                ck=torch.load(SOURCE/arm/'best_R.pt',map_location='cpu',weights_only=True)
                net=new_recovery();net.load_state_dict(ck['state']);net.eval().requires_grad_(False);nets.append(net)
            mean=torch.tensor(data['data']['mean'],device='cuda')[None,:,None,None]
            std=torch.tensor(data['data']['std'],device='cuda')[None,:,None,None]
            def infer(row,items):
                outputs=[]
                for jj in base.batches(len(items),4):
                    batch=[items[i] for i in jj]
                    rgb=torch.from_numpy(np.stack([v['image'] for v in batch])).cuda().permute(0,3,1,2).float()/255
                    maps=torch.stack(model.backbone.forward_intermediate_feature_maps((rgb-mean)/std,layer_indexes=[3,7,11]),1).half().float()
                    x=maps[:,2];mask=torch.tensor(np.stack([v['cov'] for v in batch]),device='cuda')
                    xy=torch.tensor(row['xy'],device='cuda',dtype=torch.float32)[None]
                    outputs_batch=[]
                    for value in [x,*[torch.where(mask[:,None],net(maps)[0],x) for net in nets]]:
                        assert torch.equal(value.masked_select(~mask[:,None]),x.masked_select(~mask[:,None]))
                        pred=base.run_head(model.head,value)
                        outputs_batch.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1).cpu().numpy())
                    outputs.append(np.stack(outputs_batch,1))
                return np.concatenate(outputs)
            # Reproduce the original feature-cache batch boundaries and head microbatches.
            base.status(OUT,'anchor_parity')
            cache=base.make_cache(OUT/'anchor_cache',model,data['data'])
            syn_ids=cache['syn_eval'];anchors=[]
            for b in base.batches(len(syn_ids),4):
                j=syn_ids[b];maps,z,mask,e=base.syn_batch(cache,j)
                x=maps[:,2];xy=base.gpu(cache['xy'][cache['parent'][j]])
                batch_rows=[]
                for value in [x,*[torch.where(mask[:,None],net(maps)[0],x) for net in nets]]:
                    pred=base.run_head(model.head,value)
                    batch_rows.append(torch.stack(((pred['decoded_coordinates']-xy).norm(dim=-1)*1.25,pred['presence_logits'].sigmoid()),-1).cpu().numpy())
                anchors.append(np.stack(batch_rows,1))
            anchor=np.concatenate(anchors);anchor_points=cache['point'][syn_ids].numpy()
            # Expanded generator uses precisely the same crop, mask-support and keypoint convention.
            for i,a in enumerate(records):
                row=ds[i]
                for k,(_,im,c,e,p,f) in enumerate(row['synthetic']):
                    j=syn_ids[3*i+k]
                    np.testing.assert_array_equal(c,cache['cov'][j].numpy())
                    np.testing.assert_array_equal(p,cache['point'][j].numpy())
            del cache
            checks={}
            for k,arm in enumerate(ARMS,1):
                old=np.load(SOURCE/arm/'artificial_seed42.npz')
                np.testing.assert_array_equal(anchor_points,old['point'])
                # Batch-size/TF32 rounding is reported; it must not change success counts.
                for new_index,old_index in [(0,0),(k,3)]:
                    diff=float(np.max(np.abs(anchor[:,new_index]-old['rows'][:,old_index])))
                    assert diff<1e-6,(arm,diff)
                    for threshold in [.05,.1]:
                        new=(anchor[:,new_index,:,0]<=threshold)&(anchor[:,new_index,:,1]>=.7)&anchor_points
                        prev=(old['rows'][:,old_index,:,0]<=threshold)&(old['rows'][:,old_index,:,1]>=.7)&anchor_points
                        assert np.array_equal(new,prev),(arm,threshold)
                    checks[f'{arm}/{new_index}']=diff
            np.savez_compressed(OUT/'anchor_predictions.npz',rows=anchor,point=anchor_points)
            base.write(OUT/'anchor_parity.json',dict(max_abs_differences=checks,success5_and10_exact=True,patterns=384))
            rows=[];meta=[];points=[];masks=[];border_counts=0
            for i,a in enumerate(records):
                row=ds[i];items=variants(row,a);rows.append(infer(row,items))
                xy=np.rint(row['xy']*255).astype(int)
                for item in items:
                    points.append(item['point']);masks.append(item['cov'])
                    cell=(xy//16).clip(0,15)
                    border_counts+=int((item['point'] & ~item['cov'][cell[:,1],cell[:,0]]).sum())
                    meta.append(dict(face=a['id'],video=images[a['image_id']]['metadata']['source_video_id'],
                        **{k:item[k] for k in ('kind','repeat','seed','fraction','attempt','bin_id','target')}))
                if i%8==0:base.status(OUT,'evaluating',faces=i+1,total=128,patterns=len(meta))
            rows=np.concatenate(rows);point=np.stack(points);videos=np.array([m['video'] for m in meta])
            assert rows.shape==(11520,4,5,2) and np.isfinite(rows).all()
            np.savez_compressed(OUT/'predictions.npz',rows=rows,point=point,videos=videos,patch_masks=np.stack(masks),
                face=np.array([m['face'] for m in meta]),kind=np.array([m['kind'] for m in meta]),
                fraction=np.array([m['fraction'] for m in meta]))
            base.write(OUT/'patterns.json',meta)
            groups={'overall':np.ones(len(rows),bool)}
            for k,kind in enumerate(SHAPES):groups[kind]=np.array([m['kind']==k for m in meta])
            fractions=np.array([m['fraction'] for m in meta])
            for lo,hi in [(.4,.6),(.6,.8),(.8,.9)]:
                groups[f'area_{lo:.1f}_{hi:.1f}']=(fractions>=lo)&(fractions<hi)
            summary={key:dict(patterns=int(sel.sum()),metrics=summarize(rows,point,videos,sel)) for key,sel in groups.items()}
            for k,arm in enumerate(ARMS,1):
                m=point;v=np.broadcast_to(videos[:,None],point.shape)[m]
                summary['overall']['metrics']['covered'][arm]['delta_vs_baseline_R']=paired_interval(
                    rows[:,k,:,0][m].astype(float)-rows[:,1,:,0][m],v)
            base.write(OUT/'summary.json',summary)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            overall=summary['overall']['metrics']
            for j,label in enumerate(['covered','uncovered']):
                axes[0].bar(np.arange(4)+(j-.5)*.35,[100*overall[label][n]['mean'] for n in NAMES],.35,label=label)
            axes[0].set_xticks(range(4),['Raw','Base R','Adv R','Adv+FM R']);axes[0].set_ylabel('Position error (% ellipse major diameter)');axes[0].legend()
            for k,n in enumerate(NAMES[1:]):
                stats=[summary[s]['metrics']['covered'][n]['delta'] for s in SHAPES]
                y=np.array([v['difference'] for v in stats])*100
                bounds=np.array([v['video_bootstrap_95'] for v in stats])*100
                axes[1].errorbar(np.arange(3)+(k-1)*.12,y,yerr=np.stack([y-bounds[:,0],bounds[:,1]-y]),fmt='o',label=n,capsize=3)
            axes[1].axhline(0,color='gray',lw=1);axes[1].set_xticks(range(3),SHAPES)
            axes[1].set_ylabel('R minus raw error (percentage points)');axes[1].legend(fontsize=8)
            fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
            lines=['Known-mask feature restoration: 128 faces x 90 high-area masks = 11520 cases.',
                'Frozen original best R and head; g=known mask, no learned g, no training, no clean-feature head.',
                'NME denominator: GT ellipse major diameter. Reused train-derived eval; correlated masks; video bootstrap5000.',
                'Covered point occurrences: '+str(int(point.sum()))]
            for label in ['covered','uncovered']:
                lines.append(label)
                for n,v in overall[label].items():
                    lines.append(f'{n}: error {100*v["mean"]:.5f}%; delta {100*v["delta"]["difference"]:.5f}pp; CI {[100*x for x in v["delta"]["video_bootstrap_95"]]}; location5 {v["location5"]}/{v["points"]}; joint5 {v["joint5"]}/{v["points"]}')
            (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
            assert base.tensor_digest(model)==fixed
            for p,h in hashes.items():assert base.sha(ROOT/p)==h,p
            base.write(OUT/'completion.json',dict(complete=True,patterns=11520,faces=128,videos=len(set(videos)),
                frozen_parent_verified=True,source_weights_verified=True,weights_generated=0,clean_head=False,learned_g=False,
                historical_anchor_verified=True,covered_point_outside_patch_gate=border_counts,
                artifacts={p.name:base.sha(p) for p in OUT.iterdir() if p.suffix in ('.npz','.json','.png','.txt') and p.name!='status.json'}))
            base.status(OUT,'completed',patterns=11520,faces=128)
        except Exception as e:
            base.status(OUT,'failed',error=repr(e));raise

if __name__=='__main__':run()
