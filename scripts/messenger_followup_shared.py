"""Durable reports and checkpoint retention for the authorized follow-up queue."""
from pathlib import Path
import hashlib, json, os, shutil, datetime, html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=Path('/mnt/c/Users/佐藤/.codex/visualizations/2026/11/22/messenger-followup')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def retain(source,target):
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not source.exists():return False
    if target.exists() and source.stat().st_ino==target.stat().st_ino:return False
    tmp=target.with_suffix('.linktmp');tmp.unlink(missing_ok=True);os.link(source,tmp);tmp.replace(target);return True

def export_report(folder,title,images,gt,states,pred,prob,bins,videos,face_ids,errors,labels,checkpoint,bin_labels=None):
    """images: callable(row)->RGB; predictions already evaluated with the validation-selected best."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    bin_labels=bin_labels or [f'{i*20}-{(i+1)*20}%' for i in range(5)]
    fpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fpath),16);small=ImageFont.truetype(str(fpath),13)
    groups=[];selection=[]
    for b in range(5):
        rows=np.flatnonzero(bins==b).tolist();eligible=[i for i in rows if (states[i]>0).any()];selection_rows=eligible if len(eligible)>=3 else rows;selection_rows.sort(key=lambda i:hashlib.sha256(f'visual42:{face_ids[i]}'.encode()).hexdigest())
        picked=[];seen=set()
        for i in selection_rows:
            if str(videos[i]) not in seen:picked.append(i);seen.add(str(videos[i]))
            if len(picked)==3:break
        if len(picked)<3:
            picked += [i for i in selection_rows if i not in picked][:3-len(picked)]
        if len(picked)!=3:raise ValueError(f'Need three examples for bin {b}')
        selection.extend(picked)
        canvas=Image.new('RGB',(340*len(labels),80+3*390),'#111923');dr=ImageDraw.Draw(canvas)
        dr.text((12,10),f'{title} | {bin_labels[b]}',font=font,fill='white')
        dr.text((12,35),'Green + GT; pink circle prediction; orange = presence < 0.7',font=small,fill='white')
        for k,label in enumerate(labels):dr.text((340*k+12,58),label,font=font,fill='white')
        for j,i in enumerate(picked):
            rgb=np.asarray(images(i),dtype=np.uint8)
            for k,label in enumerate(labels):
                tile=Image.fromarray(rgb).resize((328,328));td=ImageDraw.Draw(tile)
                for point in range(5):
                    if states[i,point]>0:
                        x,y=gt[i,point]*327;td.line((x-5,y,x+5,y),fill='#30ff74',width=2);td.line((x,y-5,x,y+5),fill='#30ff74',width=2)
                    x,y=pred[i,k,point]*327;colour='#ff40dc' if prob[i,k,point]>=.7 else '#ffb52e'
                    td.ellipse((x-4,y-4,x+4,y+4),outline=colour,width=2)
                canvas.paste(tile,(340*k+6,80+390*j));valid=states[i]>0
                val=float(errors[i,k][valid].mean()*100) if valid.any() else None
                dr.text((340*k+10,413+390*j),f'Face {face_ids[i]} | labeled {int(valid.sum())}/5',font=small,fill='white')
                dr.text((340*k+10,435+390*j),'NME '+(f'{val:.2f}%' if val is not None else 'N/A: all missing'),font=small,fill='white')
        canvas.save(folder/f'bin_{b+1}.png')
        row={'bin':b,'label':bin_labels[b],'faces':len(rows),'methods':{}}
        for k,label in enumerate(labels):
            by={}
            for region,m in [('annotated',states>0),('visible',states==2),('occluded',states==1)]:
                mask=m&(bins==b)[:,None];e=errors[:,k][mask]
                by[region]={'points':int(len(e)),'mean_percent':float(e.mean()*100) if len(e) else None,'within5_percent':float((e<.05).mean()*100) if len(e) else None,'over20_percent':float((e>.2).mean()*100) if len(e) else None}
            row['methods'][label]=by
        groups.append(row)
    write(folder/'selection.json',{'policy':'fixed hash of face ID; prefer annotated coordinates; distinct videos when possible; independent of prediction errors','rows':selection,'face_ids':[int(face_ids[i]) for i in selection]})
    write(folder/'summary.json',groups);write(folder/'checkpoint.json',checkpoint)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for label in labels:ax.plot(range(5),[x['methods'][label]['annotated']['mean_percent'] for x in groups],marker='o',label=label)
    ax.set(xticks=range(5),xticklabels=bin_labels,ylabel='Annotated-point NME (%)',title=title);ax.legend();fig.savefig(folder/'metrics.png',dpi=150);plt.close(fig)
    txt='<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px/1.8 sans-serif}img{max-width:100%}</style>'
    txt+=f'<h1>{html.escape(title)}</h1><p>Validation-selected best checkpoint. Green: GT; pink/orange: predictions. Missing coordinates are excluded. GT ellipse crop. Test previously observed.</p><img src="metrics.png">'
    txt+=''.join(f'<h2>{html.escape(bin_labels[b])}</h2><img src="bin_{b+1}.png">' for b in range(5))
    (folder/'index.html').write_text(txt)
    artifacts={}
    for f in folder.glob('*.png'):
        with Image.open(f) as im:im.verify()
        artifacts[f.name]=sha(f)
    assert len(artifacts)==6
    for f in ['index.html','summary.json','selection.json','checkpoint.json']:artifacts[f]=sha(folder/f)
    write(folder/'visualization_completion.json',{'completed':True,'best_inference':True,'examples_per_bin':[3]*5,'artifacts':artifacts,'utc':stamp()})
    dest=(PUBLIC/folder.parent.parent.name/folder.parent.name) if folder.parent.name in ('natural_test','artificial_test') else (PUBLIC/folder.parent.name/folder.name);dest.mkdir(parents=True,exist_ok=True)
    for f in folder.iterdir():
        if f.is_file() and f.suffix in ('.html','.png','.json'):shutil.copy2(f,dest/f.name)
    return {'report':str(folder/'index.html'),'public':str(dest/'index.html'),'marker':str(folder/'visualization_completion.json')}

def synthetic_report(run,arm,prep,checkpoint):
    from src.data.messenger_balanced_occlusion import schedule,render
    from src.data.messenger_full_e_cache import build_crop_transform,ellipse_aligned_rectangle
    run,prep=Path(run),Path(prep);folder=run/arm
    with np.load(folder/'test/evaluation.npz') as a:arr={k:a[k].copy() for k in a.files}
    manifest=read(prep/'manifest.json');records=manifest['records'];rgb=np.load(prep/'cache/rgb.npy',mmap_mode='r');lookup={a['uid']:i for i,a in enumerate(records)}
    specs=list(schedule(manifest['roles']['eval'],'eval',0)); specmap={(s['face_id'],s['bin']):s for s in specs}; yy,xx=np.ogrid[:256,:256]
    def image(i):
        uid=int(arr['face_uid'][i]);idx=lookup[uid];rec=records[idx];box,angle=ellipse_aligned_rectangle(rec['ellipse']);crop=build_crop_transform(box,output_size=256,scale_factor=1.25,rotation_deg=angle)
        cx,cy=crop.original_to_crop(np.array([rec['ellipse'][:2]]))[0];rx,ry=np.array(rec['ellipse'][2:4])*255/crop.side
        face=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
        out,c,e,point,meta,mask=render(rgb[idx],face,np.array(rec['xy']),specmap[(uid,int(arr['severity'][i]))])
        np.testing.assert_array_equal(c,arr['teacher_g'][i]);np.testing.assert_array_equal(point,arr['point'][i]);return out
    states=np.where(arr['point'],1,2)
    return export_report(folder/'visualization',arm,image,arr['gt_xy'],states,arr['coordinates'],arr['rows'][... ,1],arr['severity'],arr['videos'],arr['face_uid'],arr['rows'][...,0],['Raw DINO',arm],checkpoint)

def verify_visualization(folder):
    folder=Path(folder);m=read(folder/'visualization_completion.json');assert m['completed'] and m['best_inference']
    for name,h in m['artifacts'].items():assert sha(folder/name)==h
    return m
