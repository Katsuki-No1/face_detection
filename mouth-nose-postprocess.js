/* Adopted mouth/nose ellipse post-processing, 2026-11-26.
 * Standalone: CommonJS require() or browser global MouthNoseGeometry.
 * face=[cx,cy,majorRadius,minorRadius,majorAxisAngleRadians].
 * Output ellipse=[cx,cy,axis1Radius,axis2Radius,axis1AngleRadians].
 * See README.md for input states, adoption scope and failure statuses.
 */
(function(scope){
'use strict';
const BaseGeometry=(function(){
  const defaults = {confidence: 0.7, close: 0.18, wide: 0.25, noseAngle: 30, profileRatio: 3, profileTurn: 45};
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1]], sub=(a,b)=>[a[0]-b[0],a[1]-b[1]], mul=(a,k)=>[a[0]*k,a[1]*k];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1], norm=a=>Math.hypot(...a), unit=a=>mul(a,1/norm(a));
  const xy=p=>[p.x,p.y], mean=ps=>mul(ps.reduce((s,p)=>add(s,xy(p)),[0,0]),1/ps.length);
  function derive(row, options={}) {
    const cfg={...defaults,...options}, [cx,cy,A,B,theta]=row.face, C=[cx,cy];
    if (![cx,cy,A,B,theta].every(Number.isFinite)||A<=0||B<=0) throw Error('Invalid face ellipse');
    // Confidence is point existence probability, not probability of visibility.
    const valid=row.points.filter(p=>[p.x,p.y].every(Number.isFinite)&&(row.source==='gt'?p.state>0:p.confidence>=cfg.confidence&&Number.isFinite(p.confidence)));
    const seen=type=>valid.filter(p=>p.type===type && p.state===2);
    const present=type=>valid.filter(p=>p.type===type && p.state>0);
    const eyes=seen('eye'), mouths=seen('mouth'), noses=present('nose');
    const ranked=ps=>[...ps].sort((a,b)=>(b.state===2)-(a.state===2)||b.confidence-a.confidence||a.x-b.x||a.y-b.y);
    const nose=ranked(noses)[0], mouth=ranked(mouths)[0];
    let down=[Math.cos(theta),Math.sin(theta)], orientation='画像の下方向（上下不明）';
    const orient=(vec,label)=>{if(norm(vec)>1e-8 && Math.abs(dot(vec,down))>0.05*A){if(dot(vec,down)<0)down=mul(down,-1);orientation=label;return true;}return false;};
    let directed=false;
    if(eyes.length&&mouths.length) directed=orient(sub(mean(mouths),mean(eyes)),'可視目→可視口');
    if(!directed&&eyes.length) directed=orient(sub(C,mean(eyes)),'可視目の反対側');
    if(!directed&&mouths.length) directed=orient(sub(mean(mouths),C),'可視口側');
    if(!directed&&nose) directed=orient(sub(xy(nose),C),'鼻側（暫定）');
    if(!directed&&down[1]<0) down=mul(down,-1);
    const across=[down[1],-down[0]], W=2*B;
    const pairRatio=ps=>ps.length>=2?norm(sub(xy(ps[0]),xy(ps[1])))/W:null;
    const eyeDistance=pairRatio(eyes), mouthDistance=pairRatio(mouths);
    const narrow=(ps,d)=>ps.length===1||(ps.length===2&&d<=cfg.close);
    // Both types must offer some visible evidence; all missing is not a profile.
    const pointProfile=narrow(eyes,eyeDistance)&&narrow(mouths,mouthDistance);
    const shapeProfile=A/B>=cfg.profileRatio;
    const profileEvidence=pointProfile||shapeProfile;
    let angleError=null, noseAbove=false, mouthAxis=null;
    if(mouths.length===2&&nose&&nose.state===2){
      const v=sub(xy(mouths[1]),xy(mouths[0])), toNose=sub(xy(nose),mean(mouths));
      if(norm(v)>1e-8&&norm(toNose)>1e-8){
        mouthAxis=unit(v);
        angleError=Math.asin(Math.min(1,Math.abs(dot(mouthAxis,unit(toNose)))))*180/Math.PI;
        noseAbove=dot(toNose,down)<-0.02*(2*A);
      }
    }
    const frontalEvidence=mouthDistance!==null&&mouthDistance>=cfg.wide&&noseAbove&&angleError<=cfg.noseAngle;
    let kind, center, r1,r2,angle,reason,turn=null;
    if(profileEvidence&&nose&&mouth&&norm(sub(xy(nose),xy(mouth)))>1e-6){
      kind='profile'; center=xy(mouth); r1=0.6*A;r2=0.6*A;
      const v=sub(xy(nose),center);angle=Math.atan2(v[1],v[0]);
      reason=(pointProfile?'目・口条件':'顔軸比≥3')+'／可視口中心の円';
      if(pointProfile&&shapeProfile)reason='目・口条件＋顔軸比≥3／可視口中心の円';
      if(nose.state===1)reason+='／鼻は遮蔽';
    }else if(frontalEvidence){
      kind='front';center=mean(mouths);r1=1.2*B;r2=4*A/7;
      angle=Math.atan2(mouthAxis[1],mouthAxis[0]);reason='口幅・鼻の上方向・直交角条件を通過';
    }else{
      const profileShape=profileEvidence||A/B>=cfg.profileRatio;
      kind=profileShape?'fallback_profile':'fallback_front';
      center=add(C,mul(down,0.6*A));
      if(profileShape){
        r1=0.6*A;r2=0.6*A;
        let lateral=0;
        if(nose)lateral=dot(sub(xy(nose),C),across);
        if(Math.abs(lateral)<0.02*B&&mouth)lateral=dot(sub(xy(mouth),C),across);
        const sign=lateral<0?-1:1;turn=sign*cfg.profileTurn;
        const rad=cfg.profileTurn*Math.PI/180;
        const axis=add(mul(down,Math.cos(rad)),mul(across,sign*Math.sin(rad)));
        angle=Math.atan2(axis[1],axis[0]);
        reason='横顔OR条件・鼻または可視口点なし／顔下端20%位置の円';
      }else{r1=1.2*B;r2=4*A/7;angle=Math.atan2(across[1],across[0]);reason='点条件不成立／顔の短軸方向に口鼻楕円の横軸';}
    }
    return {kind,ellipse:[...center,r1,r2,angle],orientation,reason,eyeDistance,mouthDistance,angleError,noseAbove,pointProfile,shapeProfile,profileEvidence,frontalEvidence,faceRatio:A/B,turn,axisReversed:r2>r1,usedMouth:kind==='profile'?mouth.id:null};
  }

return {defaults,derive};
})();
 function growSecondAxis(ellipse,point,margin){
  if(!(margin>0&&Number.isFinite(margin)))throw Error('Positive nose margin is required');
  const [cx,cy,a,b,t]=ellipse,dx=point[0]-cx,dy=point[1]-cy;
  const x=dx*Math.cos(t)+dy*Math.sin(t),y=-dx*Math.sin(t)+dy*Math.cos(t);
  const gap=a-Math.abs(x);
  if(gap<margin || (gap===margin&&Math.abs(y)>1e-12))return {status:'width_limited',ellipse:[...ellipse],beforeSatisfied:false,afterSatisfied:false,noseLocal:[x,y],extraFullAxis:0};
  let required=0,n=256;
  if(gap===margin){required=Math.sqrt(a*margin);}
  else{
   // Near a tangent at the fixed horizontal end, refine rather than
   // incorrectly rejecting a feasible disk because of polygon conservatism.
   while(margin/Math.cos(Math.PI/n)>=gap&&n<65536)n*=2;
   const R=margin/Math.cos(Math.PI/n);
   if(R>=gap)return {status:'numerical_limit',ellipse:[...ellipse],beforeSatisfied:false,afterSatisfied:false,noseLocal:[x,y],extraFullAxis:0};
   for(let k=0;k<n;k++){
    const angle=2*Math.PI*k/n,px=x+R*Math.cos(angle),py=y+R*Math.sin(angle);
    const need=Math.abs(py)/Math.sqrt(1-(px/a)**2);required=Math.max(required,need);
   }
  }
  const afterB=Math.max(b,required*(1+1e-12));
  return {status:afterB>b+1e-8?'expanded':'already_satisfied',ellipse:[cx,cy,a,afterB,t],beforeSatisfied:b>=required,afterSatisfied:true,noseLocal:[x,y],extraFullAxis:2*(afterB-b),polygonVertices:n,radialConservatism:margin/Math.cos(Math.PI/n)-margin};
 }


function normalize(row){
 if(!row||!Array.isArray(row.face)||row.face.length!==5||!row.face.every(Number.isFinite)||row.face[2]<=0||row.face[3]<=0||row.face[2]<row.face[3])throw Error('face must be finite [cx,cy,A,B,theta], A>=B>0; theta in radians');
 if(!Array.isArray(row.points))throw Error('points must be an array');
 const ids=new Set(),counts={eye:0,nose:0,mouth:0};
 for(const p of row.points){
  if(!p||!Object.prototype.hasOwnProperty.call(counts,p.type)||![0,1,2].includes(p.state)||p.id===undefined||p.id===null||ids.has(p.id))throw Error('Points require unique ids, eye/nose/mouth types and state 0/1/2');
  ids.add(p.id);counts[p.type]++;
 }
 if(counts.eye>2||counts.nose>1||counts.mouth>2)throw Error('At most 2 eyes, 1 nose and 2 mouth points');
 const source=row.source??row.dataset??'model';
 if(!['gt','model'].includes(source)|| (row.dataset&&row.dataset!==source))throw Error('source and dataset must consistently identify model or gt');
 return {...row,source,dataset:source};
}
function derive(input,options={}){
 const row=normalize(input),result=BaseGeometry.derive(row,options);
 const confidence=options.confidence??BaseGeometry.defaults.confidence;
 const noses=row.points.filter(p=>p.type==='nose'&&p.state>0&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&(row.source==='gt'||Number.isFinite(p.confidence)&&p.confidence>=confidence));
 noses.sort((a,b)=>(b.state===2)-(a.state===2)||b.confidence-a.confidence||a.x-b.x||a.y-b.y);
 const nose=noses[0];
 if(result.kind==='profile'){
  const [fx,fy,A,B,faceAngle]=row.face,L=2*A,S=2*B;
  const mouth=row.points.find(p=>p.id===result.usedMouth);
  const t=result.ellipse[4],u=[-Math.sin(faceAngle),Math.cos(faceAngle)];
  const noseProjection=u[0]*Math.cos(t)+u[1]*Math.sin(t);
  const ambiguous=Math.abs(noseProjection)<1e-10;
  if(noseProjection>0){u[0]*=-1;u[1]*=-1;}
  const distance=ambiguous?0:Math.max(0,(A/B)/2.1-1)*S;
  const cx=mouth.x+distance*u[0],cy=mouth.y+distance*u[1],a=5*A/7,b=9*A/16;
  const dx=nose.x-cx,dy=nose.y-cy,x=dx*Math.cos(t)+dy*Math.sin(t),y=-dx*Math.sin(t)+dy*Math.cos(t);
  const radius=.06*L,n=256,R=radius/Math.cos(Math.PI/n);
  let required=0;
  for(let k=0;k<n;k++){const phi=2*Math.PI*k/n;required=Math.max(required,Math.hypot((x+R*Math.cos(phi))/a,(y+R*Math.sin(phi))/b));}
  const scale=Math.max(1,required*(1+1e-12)),ellipse=[cx,cy,a*scale,b*scale,t];
  return {...result,ellipse,axisReversed:false,policyApplied:true,adoptedPolicy:'profile_2.1_5of7_9of16_nose6pct',
   reason:'横顔採用処理：顔短軸に沿い鼻と反対へ移動、鼻6%円を包含',
   centerShift:{distance,direction:ambiguous?null:u,status:ambiguous?'ambiguous_nose_direction':distance>0?'shifted':'at_or_below_frontal_reference',frontalReference:2.1},
   noseMargin:{status:scale>1+1e-9?'expanded':'already_satisfied',point:[nose.x,nose.y],state:nose.state,radius,faceFullMajorDiameter:L,scale,afterSatisfied:true,polygonVertices:n}};
 }
 if(row.source==='model'&&result.kind==='front'){
  if(!nose)return {...result,policyApplied:false,adoptedPolicy:null,noseMargin:{status:'no_usable_nose',point:null,afterSatisfied:false}};
  const radius=.03*(2*row.face[2]),margin=growSecondAxis(result.ellipse,[nose.x,nose.y],radius);
  return {...result,ellipse:margin.ellipse,axisReversed:margin.ellipse[3]>margin.ellipse[2],policyApplied:true,adoptedPolicy:'model_front_nose3pct_second_axis',noseMargin:{...margin,point:[nose.x,nose.y],radius,faceFullMajorDiameter:2*row.face[2],state:nose.state}};
 }
 return {...result,policyApplied:false,adoptedPolicy:null,noseMargin:{status:'out_of_scope',point:null,afterSatisfied:null}};
}
const api={defaults:Object.freeze({...BaseGeometry.defaults}),derive,growSecondAxis};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
else scope.MouthNoseGeometry=api;
})(typeof globalThis!=='undefined'?globalThis:window);
