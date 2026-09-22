"use strict";
/* Faithful port of the repo's posture/ package. Constants come from
   reports/landmark_noise.json and reports/reference_distribution.json via
   scripts, so this page and the Python version agree by construction. */
const NOSE=0,LEYE=2,REYE=5,LEAR=7,REAR=8,LSH=11,RSH=12,LWR=15,RWR=16,
      LHIP=23,RHIP=24,LKNEE=25,RKNEE=26,LANK=27,RANK=28;
const NAMES=["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye",
"right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder",
"left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index",
"right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee",
"left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"];
const ZH={nose:"鼻",left_ear:"左耳",right_ear:"右耳",left_shoulder:"左肩",right_shoulder:"右肩",
left_hip:"左髋",right_hip:"右髋",left_ankle:"左踝",right_ankle:"右踝",left_knee:"左膝",right_knee:"右膝",
left_eye:"左眼",right_eye:"右眼"};
const EDGES=[[7,2],[2,0],[0,5],[5,8],[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],
[23,24],[23,25],[25,27],[27,31],[24,26],[26,28],[28,32]];
const CRIT=[LEAR,REAR,LSH,RSH,LHIP,RHIP,LANK,RANK];
const MLAB={head_over_hip:"头部前移",shoulder_protraction:"圆肩（肩前移）",trunk_sway:"躯干前后倾",
shoulder_tilt:"高低肩",pelvis_tilt:"骨盆侧倾",lateral_head_shift:"头部侧偏",forward_head:"耳肩角",
head_tilt:"头部侧倾",head_vs_shoulder_tilt:"头肩相对侧倾",knee_deviation:"膝关节角度"};
const ORDER=["head_over_hip","shoulder_protraction","trunk_sway","lateral_head_shift","shoulder_tilt",
"pelvis_tilt","forward_head","head_tilt","head_vs_shoulder_tilt","knee_deviation"];
const DIAG=new Set(C.diagnostic_only);

const D=Math.PI/180, deg=r=>r/D;
const px=(P,i)=>({x:P[i][0],y:P[i][1]}), vis=(P,i)=>P[i][2];
const dist=(a,b)=>Math.hypot(b.x-a.x,b.y-a.y);
const mid=(a,b)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2});
const angV=(lo,up,ant)=>deg(Math.atan2((up.x-lo.x)*ant,-(up.y-lo.y)));
function angH(l,r){let dx=r.x-l.x,dy=-(r.y-l.y);if(!dx&&!dy)return 0;
  let a=deg(Math.atan2(dy,dx));if(a>90)a-=180;else if(a<=-90)a+=180;return a;}
function angI(a,v,c){const v1={x:a.x-v.x,y:a.y-v.y},v2={x:c.x-v.x,y:c.y-v.y};
  const n1=Math.hypot(v1.x,v1.y),n2=Math.hypot(v2.x,v2.y);if(!n1||!n2)return NaN;
  return deg(Math.acos(Math.max(-1,Math.min(1,(v1.x*v2.x+v1.y*v2.y)/(n1*n2)))));}

function bodyScale(P,W,H){
  const sh=mid(px(P,LSH),px(P,RSH)), hip=mid(px(P,LHIP),px(P,RHIP));
  const inF=i=>{const p=px(P,i);return p.x>=0&&p.x<=W&&p.y>=0&&p.y<=H;};
  if(inF(LANK)&&inF(RANK)){const d=dist(sh,mid(px(P,LANK),px(P,RANK)));if(d>1)return d;}
  return dist(sh,hip)*2.6;  // shoulder-to-ankle is ~2.6x shoulder-to-hip
}
function sig(name,scale){const f=C.noise.per_landmark[name];return (f==null?C.noise.overall:f)*scale;}
const angSig=(sa,sb,span)=>span<=1e-6?NaN:deg(Math.hypot(sa,sb)/span);

function estimateView(P,W,H){
  const scale=Math.max(1,bodyScale(P,W,H));
  const lsh=px(P,LSH),rsh=px(P,RSH),lh=px(P,LHIP),rh=px(P,RHIP);
  const spread=Math.abs(lsh.x-rsh.x)/scale;
  const view=spread<=C.view.SIDE_MAX_SPREAD?"side":(spread>=C.view.FRONT_MIN_SPREAD?"front":"oblique");
  const yaw=deg(Math.asin(Math.min(1,spread/0.29)));
  const nose=px(P,NOSE),em=mid(px(P,LEAR),px(P,REAR));
  const href=Math.max(dist(px(P,LEAR),px(P,REAR)),scale*0.06);
  const dx=nose.x-em.x;
  return {view,spread,yaw,facing:dx>=0?1:-1,conf:href>0?Math.min(1,Math.abs(dx)/href):0,
          hipSpread:Math.abs(lh.x-rh.x)/scale,scale};
}
function nearSide(P,facing){
  const lv=vis(P,LEAR),rv=vis(P,REAR);let left;
  if(Math.abs(lv-rv)>0.15) left=lv>rv;
  else {const nx=px(P,NOSE).x;
        left=Math.abs(px(P,LEAR).x-nx)>Math.abs(px(P,REAR).x-nx);}
  return left?{ear:LEAR,sh:LSH,hip:LHIP,knee:LKNEE,ank:LANK}
             :{ear:REAR,sh:RSH,hip:RHIP,knee:RKNEE,ank:RANK};
}
function M(key,val,unc,plane,span,idx,note){return{key,val,unc,plane,span,idx,note};}

function computeSagittal(P,V){
  const s=V.scale,S=nearSide(P,V.facing),ant=V.facing,o={};
  const g=i=>sig(NAMES[i],s);
  const ear=px(P,S.ear),sh=px(P,S.sh),hip=px(P,S.hip),kn=px(P,S.knee),an=px(P,S.ank);
  let sp=dist(sh,ear);
  o.forward_head=M("forward_head",angV(sh,ear,ant),angSig(g(S.sh),g(S.ear),sp),"sagittal",sp,[S.sh,S.ear],
    "仅作辅助。测量跨度是全项目最短的（耳—肩），且用到定位最不稳的肩关键点，无法分辨自己的阈值。请看「头部前移」。");
  sp=dist(hip,sh);
  o.shoulder_protraction=M("shoulder_protraction",angV(hip,sh,ant),angSig(g(S.hip),g(S.sh),sp),"sagittal",sp,[S.hip,S.sh],
    "肩峰相对髋的前移角。与「耳肩角」共用肩关键点且符号相反，两者要一起看。");
  sp=dist(an,hip);
  o.trunk_sway=M("trunk_sway",angV(an,hip,ant),angSig(g(S.ank),g(S.hip),sp),"sagittal",sp,[S.ank,S.hip],
    "整个身体的前后倾。可用量程很窄：下限被噪声卡在约 6°，上限被中立站姿守卫卡在 12°。");
  sp=dist(hip,ear);
  o.head_over_hip=M("head_over_hip",angV(hip,ear,ant),angSig(g(S.hip),g(S.ear),sp),"sagittal",sp,[S.hip,S.ear],
    "本工具的主力头前引指标。不使用肩关键点，跨度是「耳肩角」的约四倍，因此精度高得多。");
  const inter=angI(hip,kn,an), legLen=dist(hip,an);
  if(legLen>1e-6&&!isNaN(inter)){
    const cross=(an.x-hip.x)*(kn.y-hip.y)-(an.y-hip.y)*(kn.x-hip.x);
    const sgn=(cross*ant)<0?1:-1, sp2=legLen/2;
    o.knee_deviation=M("knee_deviation",(180-inter)*sgn,angSig(g(S.knee),g(S.knee),sp2),"sagittal",sp2,
      [S.hip,S.knee,S.ank],"仅作守卫用，不作为体态判定。膝盖弯曲说明不是中立站姿，其余读数随之失效。");
  }
  return o;
}
function computeFrontal(P,V){
  const s=V.scale,o={},g=i=>sig(NAMES[i],s);
  const lsh=px(P,LSH),rsh=px(P,RSH),lh=px(P,LHIP),rh=px(P,RHIP),le=px(P,LEYE),re=px(P,REYE);
  let sp=dist(rsh,lsh);
  o.shoulder_tilt=M("shoulder_tilt",angH(rsh,lsh),angSig(g(RSH),g(LSH),sp),"frontal",sp,[RSH,LSH],
    "正值表示人物自己的左肩更高。相机翻滚角会 1:1 叠加到这一项，单张未标定照片无法与真实不对称区分。");
  sp=dist(rh,lh);
  o.pelvis_tilt=M("pelvis_tilt",angH(rh,lh),angSig(g(RHIP),g(LHIP),sp),"frontal",sp,[RHIP,LHIP],
    "只是骨盆的左右高低，不是骨盆前后倾——后者需要 ASIS/PSIS，本模型不提供。髋比肩窄，所以这是精度最差的正面读数。");
  sp=dist(re,le);
  o.head_tilt=M("head_tilt",angH(re,le),angSig(g(REYE),g(LEYE),sp),"frontal",sp,[REYE,LEYE],
    "仅作辅助。跨两眼测量，是全项目最短的跨度。");
  const shm=mid(lsh,rsh),hipm=mid(lh,rh);sp=dist(hipm,shm);
  if(sp>1e-6){const n=px(P,NOSE);
    o.lateral_head_shift=M("lateral_head_shift",deg(Math.atan2(n.x-shm.x,sp)),angSig(g(NOSE),g(LSH),sp),
      "frontal",sp,[NOSE,LSH,RSH],
      "头相对肩中线的水平偏移。实测是本工具精度最好的一项。注意：人稍微转身也会让鼻子偏离肩中线。");}
  if(o.shoulder_tilt&&o.head_tilt){
    o.head_vs_shoulder_tilt=M("head_vs_shoulder_tilt",o.head_tilt.val-o.shoulder_tilt.val,
      Math.hypot(o.head_tilt.unc,o.shoulder_tilt.unc),"frontal",Math.min(o.head_tilt.span,o.shoulder_tilt.span),
      o.head_tilt.idx.concat(o.shoulder_tilt.idx),
      "仅作辅助。相机翻滚在这个差值里会抵消，但它继承了「头部侧倾」的噪声。");}
  return o;
}
function computeAll(P,V){
  let o={};
  if(V.view==="side"||V.view==="oblique")Object.assign(o,computeSagittal(P,V));
  if(V.view==="front"||V.view==="oblique")Object.assign(o,computeFrontal(P,V));
  return o;
}
function outOfFrame(P,W,H){
  const m=C.guards.OUT_OF_FRAME_MARGIN*bodyScale(P,W,H),s=new Set();
  for(let i=0;i<33;i++){const p=px(P,i);
    if(p.x<-m||p.x>W+m||p.y<-m||p.y>H+m)s.add(i);}
  return s;
}
function guards(P,V,mets,W,H,S,others,origS){
  const out=[],oof=outOfFrame(P,W,H),g=C.guards;
  const named=[...oof].filter(i=>CRIT.includes(i)).map(i=>ZH[NAMES[i]]||NAMES[i]);
  if(named.length)out.push({sev:"warn",key:"landmarks_out_of_frame",
    msg:"以下关键点在画面之外，模型是推测出它们位置的："+named.join("、")+"。依赖这些点的指标已单独停用，其余读数不受影响。"});
  const xs=P.map(p=>p[0]),ys=P.map(p=>p[1]);
  const bb=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
  const dg=Math.hypot(bb[2]-bb[0],bb[3]-bb[1]);
  for(const o of (others||[])){const d=Math.hypot(o[2]-o[0],o[3]-o[1]);
    if(dg>0&&d/dg>=g.BYSTANDER_SIZE_RATIO){
      out.push({sev:"block",key:"multiple_people",msg:"画面中检测到不止一个人，无法确定测量的是哪一位。"});break;}}
  const shm=mid(px(P,LSH),px(P,RSH)),hipm=mid(px(P,LHIP),px(P,RHIP)),ankm=mid(px(P,LANK),px(P,RANK));
  const torso=dist(shm,hipm),leg=dist(hipm,ankm);
  if(leg>1e-6){const r=torso/leg;
    if(r<g.TORSO_LEG_RATIO_MIN||r>g.TORSO_LEG_RATIO_MAX)
      out.push({sev:"block",key:"implausible_proportions",
        msg:"检测到的躯干与腿部比例不符合站立人体，这通常意味着画面中并没有一个完整站立的人。"});}
  if(V.view==="oblique")out.push({sev:"block",key:"oblique_view",
    msg:"拍摄角度介于正面与侧面之间。这种角度下矢状面各项读数会被系统性压缩，且读数本身看不出这个偏差。请重新拍摄正对或完全侧对镜头的照片。"});
  if(V.view==="side"&&V.conf<0.25)out.push({sev:"warn",key:"facing_ambiguous",
    msg:"无法可靠判断人物朝向，矢状面读数的正负号可能相反。"});
  const kn=mets.knee_deviation,ts=mets.trunk_sway;
  if(kn&&!oof.has(LANK)&&!oof.has(RANK)&&kn.val>g.MAX_KNEE_FLEXION)
    out.push({sev:"block",key:"knee_flexed",msg:"膝关节明显弯曲，说明拍摄时不是中立站姿，此时的其他读数描述的是另一个姿势，不作判定。"});
  if(ts&&!oof.has(LANK)&&!oof.has(RANK)&&Math.abs(ts.val)>g.MAX_TRUNK_LEAN)
    out.push({sev:"block",key:"trunk_leaning",msg:"整个躯干明显前后倾斜，说明不是中立站姿，上方各项读数不作判定。"});
  if(V.view==="front"||V.view==="oblique"){
    const ankOK=!oof.has(LANK)&&!oof.has(RANK);
    if(ankOK){
      const hw=dist(px(P,LHIP),px(P,RHIP)),aw=dist(px(P,LANK),px(P,RANK));
      if(hw>1e-6&&aw/hw>g.MAX_STANCE_WIDTH_RATIO)
        out.push({sev:"block",key:"stance_too_wide",msg:"双脚分开过宽或正处于跨步中，不是中立站姿。"});
      const knees=[angI(px(P,LHIP),px(P,LKNEE),px(P,LANK)),angI(px(P,RHIP),px(P,RKNEE),px(P,RANK))].filter(a=>!isNaN(a));
      if(knees.length&&Math.max(...knees)<g.MIN_KNEE_EXTENSION_FRONT_DEG)
        out.push({sev:"block",key:"knees_not_extended",msg:"双膝都明显弯曲，说明人物是坐着或蹲着而不是站立。"});
      if(Math.abs(hipm.x-ankm.x)/S>g.MAX_WEIGHT_SHIFT_FRAC)
        out.push({sev:"warn",key:"weight_on_one_leg",
          msg:"重心明显偏向一侧腿。这会直接造成骨盆侧倾与高低肩的读数，但反映的是当时的站法，不一定是长期体态。"});
    }
    const raised=[];
    if((px(P,LWR).y-px(P,LHIP).y)/S<g.MIN_WRIST_BELOW_HIP_FRAC)raised.push("左");
    if((px(P,RWR).y-px(P,RHIP).y)/S<g.MIN_WRIST_BELOW_HIP_FRAC)raised.push("右");
    if(raised.length)out.push({sev:"warn",key:"arms_not_at_side",
      msg:"手臂没有自然垂放（抱臂、叉腰、插兜都会如此）。这会改变肩部关键点位置，肩相关读数请谨慎参考。"});
  }
  for(const k in mets){const m=mets[k];
    if(m.plane==="sagittal"&&Math.abs(m.val)>g.MAX_PLAUSIBLE_ANGLE)
      out.push({sev:"block",key:"implausible_reading",
        msg:"「"+(MLAB[k]||k)+"」读数超出人体可能范围，判定为测量失败而非体态问题。"});}
  // Judged on the ORIGINAL photo's pixels. The demo images are downscaled for
  // payload size, and every other quantity here is normalised so that does not
  // matter -- this one is absolute, so it would otherwise flip on resize alone.
  const sizePx=(origS||S);
  if(V.view!=="front"&&sizePx<g.MIN_SUBJECT_HEIGHT_PX)
    out.push({sev:"warn",key:"subject_too_small",
      msg:"人物在画面中偏小（肩到踝约 "+sizePx.toFixed(0)+" 像素，建议至少 "+g.MIN_SUBJECT_HEIGHT_PX+" 像素）。实测显示侧面各项读数的严重误差率在人物偏小时会上升约一半。"});
  return {findings:out,oof};
}
function classify(sp,v){const x=sp.direction==="positive_only"?v:Math.abs(v);
  return x>=sp.notable?"notable":(x>=sp.slight?"slight":"reference");}
function assess(S){
  const P=S.lm,W=S.w,H=S.h,V=estimateView(P,W,H);
  const mets=computeAll(P,V);
  const {findings,oof}=guards(P,V,mets,W,H,V.scale,S.others,S.oscale);
  const unavail={};
  for(const k in mets){const bad=mets[k].idx.filter(i=>oof.has(i)).map(i=>ZH[NAMES[i]]||NAMES[i]);
    if(bad.length)unavail[k]=bad;}
  let blocked=findings.some(f=>f.sev==="block");
  const judged=Object.keys(mets).filter(k=>!DIAG.has(k)&&C.thresholds[k]);
  if(judged.length&&judged.every(k=>unavail[k])){blocked=true;
    findings.push({sev:"block",key:"all_metrics_unavailable",
      msg:"这张照片里可用的关键点不足以支持任何一项判定，通常是人物被裁切得太多。"});}
  const verdicts={};
  if(!blocked)for(const k in mets){
    if(DIAG.has(k)||unavail[k])continue;
    const sp=C.thresholds[k];if(!sp)continue;
    const m=mets[k],band=classify(sp,m.val);
    const floored=!isNaN(m.unc)&&(sp.notable-sp.slight)>0&&m.unc>=(sp.notable-sp.slight);
    const lo=classify(sp,m.val-m.unc),hi=classify(sp,m.val+m.unc);
    const resolved=!isNaN(m.unc)&&lo===band&&hi===band&&!floored;
    const order={reference:0,slight:1,notable:2};
    const alts=[lo,hi].filter(b=>b!==band);
    const alt=alts.length?alts.reduce((a,b)=>order[a]>=order[b]?a:b):null;
    verdicts[k]={band,resolved,floored,alt,sp,
      advice:(floored||band==="reference")?[]:(C.advice[k]||[])};
  }
  return {V,mets,findings,unavail,blocked,verdicts};
}
