"""Fixed global-test comparison; never train or select checkpoints on test."""
def evaluate_heldout(out):
    import pathlib,sys,json,hashlib,io,datetime,html,time
    import numpy as np
    import torch,yaml
    from PIL import Image,ImageDraw,ImageFont
    ROOT=pathlib.Path('/home/katsuki/projects/face_detection');sys.path.insert(0,str(ROOT))
    from scripts import run_messenger_code_consistency as exp
    from scripts.train_natural_feature_gate import build_crop_transform,ellipse_aligned_rectangle,warp_crop
    from src.data.messenger_balanced_occlusion import schedule,render,SHAPES
    OUT=pathlib.Path(out);OUT.mkdir(parents=True,exist_ok=True);RUN=exp.OUT;ARMS=list(exp.ARMS);NAMES=['raw']+ARMS
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    # Parent owns canonical lease; this function only evaluates frozen models.
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    config=yaml.safe_load(exp.base.CONFIG.read_text());annpath=pathlib.Path(config['data']['annotations']['test']);corpus=read(annpath);images={x['id']:x for x in corpus['images']}
    train=read(pathlib.Path(config['data']['annotations']['train']));trainvid={x['metadata']['source_video_id'] for x in train['images']};trainsha={x['metadata']['sha256'] for x in train['images']}
    detcfg=yaml.safe_load((ROOT/'configs/dinov3_deimv2/mal_alpha2_ellipse_dataset_v3_r3.yaml').read_text());assert detcfg['data']['annotation_paths']['train']==config['data']['annotations']['train']
    train_success=read(pathlib.Path(exp.base.BASE).parent.parent/'training_success.json');assert sha(exp.base.BASE)==train_success['details']['best_checkpoint_sha256'];assert sha(annpath)==train_success['identity']['inputs'][str(annpath)]
    anns={a['id']:a for a in corpus['annotations']}
    source={};excluded=[];yy,xx=np.ogrid[:256,:256]
    for a in corpus['annotations']:
     im=images[a['image_id']]
     if not(a.get('category_id')==2 and a.get('ellipse') and a.get('keypoints',[])[2::3]==[2]*5 and not a.get('attributes',{}).get('face_occluded',False)):continue
     assert im['metadata']['source_video_id'] not in trainvid and im['metadata']['sha256'] not in trainsha
     box,angle=ellipse_aligned_rectangle(a['ellipse']);crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
     xy=crop.original_to_crop(np.array(a['keypoints']).reshape(5,3)[:,:2])
     if not(np.isfinite(xy).all() and ((xy>=0)&(xy<=255)).all()):excluded.append(a['id']);continue
     data=(pathlib.Path(config['data']['root'])/im['file_name']).read_bytes();assert hashlib.sha256(data).hexdigest()==im['metadata']['sha256']
     rgb=np.asarray(Image.open(io.BytesIO(data)).convert('RGB'));cropped=warp_crop(rgb,crop)
     cx,cy=crop.original_to_crop(np.array([a['ellipse'][:2]]))[0];rx,ry=np.array(a['ellipse'][2:4])*255/crop.side
     face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
     source[a['id']]=dict(image=cropped,face=face,xy=xy,video=im['metadata']['source_video_id'])
    ids=sorted(source);specs=list(schedule(ids,'global_test',0));records=[];allgt=[];points=[];videos=[]
    print('Eligible faces',len(ids),'excluded crop',len(excluded),'videos',len({s['video'] for s in source.values()}),flush=True)
    # Audit/render all masks before inspecting any model output; RGB batches are regenerated deterministically.
    for sp in specs:
     s=source[sp['face_id']];im,c,e,p,meta,mask=render(s['image'],s['face'],s['xy'],sp)
     records.append(meta);allgt.append((s['xy']/255).astype('float32'));points.append(p);videos.append(s['video'])
    selection=[];uf=set();uv=set()
    for bin_id in range(5):
     for offset in [.05,.10,.15]:
      candidates=[i for i,r in enumerate(records) if r['bin']==bin_id and r['covered_points'] and r['face_id'] not in uf and videos[i] not in uv]
      i=min(candidates,key=lambda i:(abs(records[i]['fraction']-(.2*bin_id+offset)),records[i]['face_id']))
      selection.append(i);uf.add(records[i]['face_id']);uv.add(videos[i])
    write(OUT/'selection.json',dict(rule='Before inference: closest to bin lower edge +5/10/15 pp; >=1 covered point; distinct faces/videos across all 15 examples.',indices=selection,records=[records[i] for i in selection]))
    write(OUT/'occlusion_records.json',records)
    manifest=dict(split='global_test',annotation=str(annpath),annotation_sha256=sha(annpath),faces=ids,face_count=len(ids),image_count=len({anns[i]['image_id'] for i in ids}),videos=sorted({s['video'] for s in source.values()}),pairs=len(specs),per_bin=[len(ids)]*5,excluded_outside_crop=excluded,baseline_train_video_overlap=0,baseline_train_image_sha_overlap=0,detector_uses_same_train_annotation=True,checkpoint_selection='Per-arm best dev-position checkpoint; no test-based selection',crop='GT ellipse, scale1.25; same as previous evaluation',g='Known binary synthetic occlusion patch mask; fixed',clean_head_input=False)
    write(OUT/'manifest.json',manifest)
    with torch.no_grad():
     model,_=exp.base.build_landmark_model(config,variant='query_refinement',initialize_from_detector=False);exp.base.load_landmark_checkpoint(model,exp.base.BASE,prefer_ema=True,strict=True);model.eval().requires_grad_(False).cuda()
     norms=read(RUN/'normalization.json');basis=torch.tensor(norms['basis']);mean=torch.tensor(norms['mean']);codes=torch.from_numpy(np.load(RUN/'initial_codes_d16_k4096.npy'));nets={};before={}
     for a in ARMS:
      path=RUN/a/'best_R.pt';assert sha(path)==read(RUN/a/'evaluation.json')['artifacts']['best_R.pt']
      net=exp.make_recovery(a,basis[:,:16],mean,codes).cuda();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);net.eval().requires_grad_(False);nets[a]=net;before[a]=exp.base.tensor_digest(net)
     baseline_before=exp.base.tensor_digest(model);coord=[];pres=[];tick=time.monotonic()
     for start in range(0,len(specs),40):
      batch=specs[start:start+40];rgb=[];g=[]
      for sp in batch:
       s=source[sp['face_id']];im,c,*_=render(s['image'],s['face'],s['xy'],sp);rgb.append(im);g.append(c)
      maps=exp.dynamic.encode(model,np.stack(rgb),config['data']);outxy=[];outpr=[]
      for k in range(0,len(batch),4):
       x=maps[k:k+4].cuda().float();gate=torch.from_numpy(np.stack(g[k:k+4])).cuda();values=[x[:,2]]
       for a in ARMS:r,_=nets[a](x);values.append(exp.teacher_blend(x[:,2],r,gate))
       pr=[exp.base.run_head(model.head,v) for v in values];outxy.append(torch.stack([p['decoded_coordinates'] for p in pr],1).cpu().numpy());outpr.append(torch.stack([p['presence_logits'].sigmoid() for p in pr],1).cpu().numpy())
      coord.append(np.concatenate(outxy));pres.append(np.concatenate(outpr))
      if start%400==0:print('Evaluated',min(start+40,len(specs)),'/',len(specs),'seconds',round(time.monotonic()-tick,1),flush=True)
     coordinates=np.concatenate(coord);prob=np.concatenate(pres)
     assert exp.base.tensor_digest(model)==baseline_before
     assert all(exp.base.tensor_digest(nets[a])==before[a] for a in ARMS)
     del model,nets;torch.cuda.empty_cache()
    gtall=np.array(allgt);point=np.array(points);vid=np.array(videos);severity=np.array([r['bin'] for r in records]);err=np.linalg.norm(coordinates-gtall[:,None],axis=-1)*1.25
    np.savez_compressed(OUT/'evaluation.npz',coordinates=coordinates,presence=prob,errors=err,gt=gtall,covered_point=point,videos=vid,severity=severity,face_ids=np.array([r['face_id'] for r in records]),fraction=np.array([r['fraction'] for r in records]),methods=np.array(NAMES))
    def interval(diff,v):
     u=np.unique(v);s=np.array([diff[v==x].sum() for x in u]);n=np.array([(v==x).sum() for x in u]);ix=np.random.default_rng(42).integers(0,len(u),(5000,len(u)));values=s[ix].sum(1)/n[ix].sum(1)
     return dict(difference_pp=float(diff.mean()),video_bootstrap_95_pp=np.quantile(values,[.025,.975]).tolist())
    summary=[]
    for bin_id in [-1,0,1,2,3,4]:
     face=np.ones(len(specs),bool) if bin_id==-1 else severity==bin_id
     entry=dict(bin=bin_id,pairs=int(face.sum()),covered_points=int((point&face[:,None]).sum()),methods={})
     for mi,name in enumerate(NAMES):
      methods={}
      for region,mask in [('covered',point&face[:,None]),('uncovered',~point&face[:,None])]:
       er=err[:,mi][mask].astype(float);pr=prob[:,mi][mask];v=np.broadcast_to(vid[:,None],mask.shape)[mask]
       methods[region]=dict(points=len(er),mean_percent=float(er.mean()*100),p90_percent=float(np.quantile(er,.9)*100),within5_count=int((er<=.05).sum()),within5_percent=float((er<=.05).mean()*100),joint5_count=int(((er<=.05)&(pr>=.7)).sum()),within10_count=int((er<=.1).sum()),over20_count=int((er>.2).sum()),versus_raw=interval((er-err[:,0][mask])*100,v))
      entry['methods'][name]=methods
     summary.append(entry)
    write(OUT/'severity_summary.json',summary)

    # Paired comparisons on exactly the same held-out points and videos.
    mask=point;vv=np.broadcast_to(vid[:,None],mask.shape)[mask]
    pairs={}
    for label,a,b in [('js_s124',2,1),('js_s1248',4,3),('scale8_js0',3,1),('scale8_js001',4,2)]:
        pairs[label]=interval((err[:,a].astype(float)-err[:,b])[mask]*100,vv)
    write(OUT/'comparison.json',dict(conditions=summary,contrasts=pairs,scope='baseline-train-video-disjoint global test; GT crop and fixed synthetic g; single seed'))
    write(OUT/'audit.json',dict(completed=True,models_unchanged_during_inference=True,no_training_or_optimizer=True,
        no_clean_head_input=True,no_test_based_selection=True,baseline_checkpoint_sha256=sha(exp.base.BASE),
        checkpoints={a:sha(RUN/a/'best_R.pt') for a in ARMS},video_overlap=0,image_sha_overlap=0,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pairs=len(specs),faces=len(ids)))
    print('Heldout evaluation complete',json.dumps(pairs),flush=True)
