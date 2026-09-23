"use strict";
/* UI for the posture check: upload -> (model download) -> scan animation ->
   plain result. All judging is done by posture.js; this file only shows it. */
const $=s=>document.querySelector(s);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const nextFrame=()=>new Promise(r=>requestAnimationFrame(()=>r()));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const REDUCED=matchMedia("(prefers-reduced-motion: reduce)").matches;
const MAX_SIDE=2048;                 // cap on the analysed canvas; the model sees 256px crops anyway
const MB=Math.round(C.assets.bytes/1e6);
const LEVEL_RGB={notable:"#FF6A55",slight:"#FFB23F",borderline:"#B9C8D8"};
const VIEW_ZH={side:"侧面",front:"正面"};
const ICON={
  check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  alert:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M12 6.5v7M12 17.5v.01"/></svg>',
  move:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13" cy="4.5" r="2"/><path d="M8 21l3-6 3 2v4M6 11l3-3 4 1 3 3 3 1M11 15l1-5"/></svg>',
  camera:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5A2.5 2.5 0 0 1 6.5 6h1.6l1.4-2h5l1.4 2h1.6A2.5 2.5 0 0 1 20 8.5v9a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5z"/><circle cx="12" cy="13" r="3.6"/></svg>',
  x:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M7 7l10 10M17 7L7 17"/></svg>'
};

const state={results:{},last:null,busy:false};
const fileInput=$("#file");

/* ---------------- analysis engine (downloaded in the background) -------- */
let engineErr=null;const progressFns=new Set();
function startEngine(){
  return Engine.load({wasm:C.assets.wasm,modelParts:C.assets.model,totalBytes:C.assets.bytes,
    onProgress:f=>progressFns.forEach(fn=>fn(f))})
    .then(x=>{engineErr=null;return x;},e=>{engineErr=e;throw e;});
}
// Most people pick a photo within seconds of arriving; starting the download
// now means the wait is mostly over by then. Skipped when the browser asks to
// save data -- then it starts on the first upload instead.
if(!(navigator.connection&&navigator.connection.saveData))
  setTimeout(()=>startEngine().catch(()=>{}),900);

/* ---------------- screens ---------------- */
function show(name){
  for(const id of ["home","analyze","result"])$("#"+id).hidden=id!==name;
  $("#closeBtn").hidden=name==="home";
  window.scrollTo({top:0});
  hero.run(name==="home");
}
$("#closeBtn").addEventListener("click",()=>{if(!state.busy)show("home");});
$("#disclaimer").textContent=C.copy.disclaimer;

/* ---------------- photo handling ---------------- */
async function fromFile(file){
  const url=URL.createObjectURL(file);
  const img=new Image();img.src=url;
  await img.decode();   // <img> applies the EXIF orientation of phone photos
  const W0=img.naturalWidth,H0=img.naturalHeight,f=Math.min(1,MAX_SIDE/Math.max(W0,H0));
  const W=Math.round(W0*f),H=Math.round(H0*f);
  const cv=document.createElement("canvas");cv.width=W;cv.height=H;
  cv.getContext("2d").drawImage(img,0,0,W,H);
  return {src:url,W,H,f,canvas:cv};
}
// Same perturbation as guards.check_landmark_stability: 4% down and back up.
function perturbed(cv){
  const w=Math.max(1,Math.floor(cv.width*0.96)),h=Math.max(1,Math.floor(cv.height*0.96));
  const a=document.createElement("canvas");a.width=w;a.height=h;
  const ax=a.getContext("2d");ax.imageSmoothingQuality="high";ax.drawImage(cv,0,0,w,h);
  const b=document.createElement("canvas");b.width=cv.width;b.height=cv.height;
  const bx=b.getContext("2d");bx.imageSmoothingQuality="high";bx.drawImage(a,0,0,cv.width,cv.height);
  return b;
}

/* ---------------- analyze screen ---------------- */
const frame=$("#frame"),photo=$("#photo"),overlay=$("#overlay"),loader=$("#loader");
function fitFrame(fr,W,H,maxW,maxH){
  const k=Math.min(maxW/W,maxH/H);
  fr.style.width=Math.round(W*k)+"px";fr.style.height=Math.round(H*k)+"px";
}
function sizeStage(ph){
  const maxW=frame.parentElement.clientWidth-28;
  fitFrame(frame,ph.W,ph.H,maxW,Math.max(260,window.innerHeight*0.58));
}
function steps(i,viewText){
  const li=[...document.querySelectorAll("#steps li")];
  li.forEach((el,j)=>{el.classList.toggle("done",j<i);el.classList.toggle("on",j===i);});
  if(viewText)$("#stepView").textContent="判断拍摄角度："+viewText;
  else if(i<=2)$("#stepView").textContent="判断拍摄角度";
}
async function ensureEngine(){
  if(Engine.ready)return true;
  frame.classList.add("dim");loader.hidden=false;$("#loaderActions").hidden=true;
  $("#ring").hidden=false;
  $("#loaderTitle").textContent="正在准备分析模型";$("#loaderTitle").classList.remove("err");
  const setP=f=>{const p=Math.round(f*100);$("#ringPct").textContent=p+"%";
    $("#ringBar").style.strokeDashoffset=String(276.5*(1-f));
    $("#loaderSub").textContent="首次使用需要下载约 "+MB+" MB（已完成 "+Math.round(f*MB)+" MB），之后会快很多";};
  setP(0);progressFns.add(setP);
  try{await startEngine();return true;}
  catch(e){
    $("#ring").hidden=true;
    $("#loaderTitle").textContent="分析模型没能加载";$("#loaderTitle").classList.add("err");
    $("#loaderSub").textContent="可能是网络不稳定。可以重试，或先看看示例的分析效果。";
    $("#loaderActions").hidden=false;
    console.error(e);return false;
  }finally{progressFns.delete(setP);}
}
$("#retryBtn").addEventListener("click",()=>{if(state.pending)analyze(state.pending);});
$("#toSamplesBtn").addEventListener("click",()=>{state.busy=false;show("home");
  $("#samples").scrollIntoView({behavior:REDUCED?"auto":"smooth",block:"center"});});

/* job: {kind:"file", file} | {kind:"sample", sample} */
async function analyze(job){
  state.pending=job;state.busy=true;
  show("analyze");steps(0);
  const ctx=overlay.getContext("2d");ctx.clearRect(0,0,overlay.width,overlay.height);
  frame.classList.remove("scanning","dim");loader.hidden=true;
  let ph,S=null;
  if(job.kind==="file"){
    try{ph=await fromFile(job.file);}
    catch(e){state.busy=false;return finish({ph:null,S:null,r:null,rep:retakeOnly("unreadable")});}
  }else{
    const s=job.sample;ph={src:"data:image/jpeg;base64,"+s.jpg,W:s.w,H:s.h,f:1};
  }
  photo.src=ph.src;sizeStage(ph);sizeCanvas(overlay,frame);
  if(job.kind==="file"&&!(await ensureEngine())){state.busy=false;return;}
  frame.classList.remove("dim");loader.hidden=true;
  frame.classList.add("scanning");
  await nextFrame();await sleep(REDUCED?0:450);
  const t0=performance.now();
  if(job.kind==="file"){
    const det=Engine.detect(ph.canvas,ph.W,ph.H);
    if(det){
      const det2=Engine.detect(perturbed(ph.canvas),ph.W,ph.H);
      S={lm:det.lm,others:det.others,w:ph.W,h:ph.H,lm2:det2?det2.lm:null};
      S.oscale=estimateView(S.lm,S.w,S.h).scale/ph.f;
    }
  }else{
    const s=job.sample;S={lm:s.lm,lm2:s.lm2,others:s.others,w:s.w,h:s.h,oscale:s.oscale};
  }
  // Let the sweep read as a scan even when detection is instant.
  const left=(REDUCED?0:1100)-(performance.now()-t0);if(left>0)await sleep(left);
  frame.classList.remove("scanning");
  const r=S?assess(S):null;
  const rep=r?report(r):retakeOnly("no_pose");
  const entry={ph,S,r,rep};
  if(r)await scanReveal(entry);
  state.busy=false;
  finish(entry);
}
function retakeOnly(key){
  const c=key==="unreadable"?{title:"读不出这张图片",tip:"请换一张 JPG 或 PNG 格式的照片。"}:C.copy.retake.no_pose;
  return {status:"retake",retake:[Object.assign({key},c)],issues:[],borderline:[],good:[],not_measured:[],photo_tips:[],other_view:null};
}
async function scanReveal(entry){
  const {r,rep}=entry,marks=marksFor(entry,rep.issues.map((x,i)=>Object.assign({num:i+1},x)).concat(rep.borderline));
  const T={pts:0,edges:0,plumb:0,marks:0};
  const run=(key,ms)=>new Promise(res=>{
    if(REDUCED){T[key]=1;drawOverlay(overlay,entry,T,marks);return res();}
    const t0=performance.now();
    const tick=now=>{T[key]=Math.min(1,(now-t0)/ms);drawOverlay(overlay,entry,T,marks,now);
      T[key]<1?requestAnimationFrame(tick):res();};
    requestAnimationFrame(tick);
  });
  steps(1);await run("pts",750);
  steps(2,VIEW_ZH[r.V.view]||"斜侧");await run("edges",520);
  steps(3);await run("plumb",480);await run("marks",520);
  steps(4);await sleep(REDUCED?0:420);steps(5);await sleep(REDUCED?0:260);
}
function finish(entry){
  state.last=entry;
  if(entry.rep.status==="ok")state.results[entry.rep.view]=entry;
  entry.rep.status==="ok"?renderResult():renderRetake(entry);
  show("result");
  requestAnimationFrame(()=>drawResultPhotos());
}

/* ---------------- overlay drawing ---------------- */
function sizeCanvas(cv,box){
  const dpr=Math.min(2.5,window.devicePixelRatio||1),w=box.clientWidth,h=box.clientHeight;
  cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);cv._dpr=dpr;
}
const ease=t=>1-Math.pow(1-t,3);
const back=t=>{const c=1.9;return 1+(c+1)*Math.pow(t-1,3)+c*Math.pow(t-1,2);};
const JOINTS=[0,7,8,11,12,13,14,15,16,23,24,25,26,27,28,31,32];

/* Which landmark each finding points at, per view. */
function marksFor(entry,items){
  const {r}=entry,P=r.P,near=r.near;const out=[];
  for(const it of items){
    let a=null;
    if(it.key==="head_over_hip"&&near)a=near.ear;
    else if(it.key==="shoulder_protraction"&&near)a=near.sh;
    else if(it.key==="trunk_sway"&&near)a=near.ank;
    else if(it.key==="shoulder_tilt")a=P[11][1]<P[12][1]?11:12;
    else if(it.key==="lateral_head_shift")a=0;
    if(a!=null)out.push({a,num:it.num||0,level:it.level,key:it.key});
  }
  return out;
}
/* The vertical reference drawn on the photo is the one the readings are taken
   against: through the hip in a side view (head and shoulder are measured
   relative to it), through the shoulder midpoint in a front view (head shift
   is). The ankle's offset from the hip line is the trunk reading. */
function plumbX(r){
  const P=r.P;
  if(r.V.view==="side"&&r.near)return P[r.near.hip][0];
  return (P[11][0]+P[12][0])/2;
}
function drawOverlay(cv,entry,T,marks,now){
  const {r}=entry,P=r.P,ctx=cv.getContext("2d"),dpr=cv._dpr||1;
  const k=cv.width/dpr/r.W;
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,cv.width,cv.height);
  const X=i=>P[i][0]*k,Y=i=>P[i][1]*k;
  const lw=Math.max(1.6,Math.min(3.2,cv.width/dpr/150));
  // plumb / midline
  if(T.plumb>0){
    const x=plumbX(r)*k,top=Math.min(Y(0),Y(7),Y(8))-r.V.scale*k*0.08;
    const bottom=Math.min(cv.height/dpr,Math.max(Y(27),Y(28))+6);
    const y1=top+(bottom-top)*ease(T.plumb);
    ctx.save();ctx.setLineDash([6,6]);ctx.lineWidth=lw;
    ctx.strokeStyle="rgba(0,0,0,.45)";ctx.beginPath();ctx.moveTo(x+1,top+1);ctx.lineTo(x+1,y1+1);ctx.stroke();
    ctx.strokeStyle="#5FF0C8";ctx.shadowColor="rgba(95,240,200,.8)";ctx.shadowBlur=8;
    ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,y1);ctx.stroke();ctx.restore();
    ctx.fillStyle="#5FF0C8";ctx.beginPath();ctx.arc(x,top,lw*1.6,0,7);ctx.fill();
  }
  // skeleton
  if(T.edges>0){
    ctx.save();ctx.lineCap="round";
    for(const pass of [0,1]){
      ctx.lineWidth=pass?lw:lw+2.4;ctx.strokeStyle=pass?"rgba(255,255,255,.92)":"rgba(0,0,0,.35)";
      for(const [a,b] of EDGES){const e=ease(T.edges);
        ctx.beginPath();ctx.moveTo(X(a),Y(a));ctx.lineTo(X(a)+(X(b)-X(a))*e,Y(a)+(Y(b)-Y(a))*e);ctx.stroke();}
    }
    ctx.restore();
  }
  if(T.pts>0){
    const ys=JOINTS.map(Y),y0=Math.min(...ys),y1=Math.max(...ys);
    for(const i of JOINTS){
      const d=(Y(i)-y0)/Math.max(1,y1-y0),t=Math.max(0,Math.min(1,(T.pts*1.35-d*0.35)/1));
      if(t<=0)continue;const rr=Math.max(0,back(t))*lw*1.55;
      ctx.fillStyle="rgba(0,0,0,.35)";ctx.beginPath();ctx.arc(X(i),Y(i),rr+1.4,0,7);ctx.fill();
      ctx.fillStyle="#5FF0C8";ctx.beginPath();ctx.arc(X(i),Y(i),rr,0,7);ctx.fill();
    }
  }
  // findings
  if(T.marks>0&&marks.length){
    const px0=plumbX(r)*k;
    for(const m of marks){
      const col=LEVEL_RGB[m.level]||"#fff",x=X(m.a),y=Y(m.a),e=ease(T.marks);
      if(m.key==="shoulder_tilt"){
        ctx.save();ctx.lineWidth=lw+.6;ctx.strokeStyle=col;ctx.lineCap="round";
        ctx.beginPath();ctx.moveTo(X(11),Y(11));ctx.lineTo(X(11)+(X(12)-X(11))*e,Y(11)+(Y(12)-Y(11))*e);ctx.stroke();
        ctx.setLineDash([4,5]);ctx.lineWidth=lw*.8;ctx.strokeStyle="rgba(255,255,255,.75)";
        const yl=Math.min(Y(11),Y(12)),xa=Math.min(X(11),X(12))-12,xb=Math.max(X(11),X(12))+12;
        ctx.beginPath();ctx.moveTo(xa,yl);ctx.lineTo(xa+(xb-xa)*e,yl);ctx.stroke();ctx.restore();
      }else{
        ctx.save();ctx.lineWidth=lw+.8;ctx.strokeStyle=col;ctx.lineCap="round";
        ctx.beginPath();ctx.moveTo(px0,y);ctx.lineTo(px0+(x-px0)*e,y);ctx.stroke();ctx.restore();
      }
      const pulse=now&&T.marks<1?1+0.25*Math.sin(now/90):1;
      ctx.lineWidth=lw+.4;ctx.strokeStyle=col;ctx.fillStyle="rgba(0,0,0,.25)";
      ctx.beginPath();ctx.arc(x,y,(lw*4.2)*e*pulse,0,7);ctx.fill();ctx.stroke();
      if(m.num){
        const R=Math.max(9,lw*4),side=x>=px0?1:-1;
        let bx=x+side*(R*2.1),by=y-R*1.6;
        bx=Math.max(R+2,Math.min(cv.width/dpr-R-2,bx));by=Math.max(R+2,by);
        ctx.globalAlpha=e;ctx.fillStyle=col;ctx.beginPath();ctx.arc(bx,by,R,0,7);ctx.fill();
        ctx.fillStyle="#fff";ctx.font="700 "+Math.round(R*1.15)+"px Outfit, system-ui, sans-serif";
        ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(String(m.num),bx,by+.5);ctx.globalAlpha=1;
      }
    }
  }
}

/* ---------------- result screens ---------------- */
function merged(){
  const views=["side","front"].filter(v=>state.results[v]);
  const out={views,issues:[],borderline:[],good:[],nm:[],tips:[]};
  for(const v of views){const rep=state.results[v].rep;
    for(const x of rep.issues)out.issues.push(Object.assign({view:v},x));
    for(const x of rep.borderline)out.borderline.push(Object.assign({view:v},x));
    out.good.push(...rep.good);out.nm.push(...rep.not_measured);
    for(const t of rep.photo_tips)if(!out.tips.includes(t))out.tips.push(t);}
  out.issues.sort((a,b)=>(a.level==="notable"?0:1)-(b.level==="notable"?0:1));
  out.issues.forEach((x,i)=>x.num=i+1);
  return out;
}
function photoCards(views){
  return '<div class="photos'+(views.length>1?" two":"")+'">'+views.map(v=>
    '<div class="pcard"><span class="tag">'+VIEW_ZH[v]+'</span><div class="frame" data-view="'+v+'">'+
    '<img alt="'+VIEW_ZH[v]+'照片" src="'+state.results[v].ph.src+'"><canvas></canvas></div></div>').join("")+'</div>';
}
function drawResultPhotos(){
  const m=merged();
  document.querySelectorAll("#result .frame[data-view]").forEach(fr=>{
    const v=fr.dataset.view,entry=state.results[v],card=fr.parentElement;
    const two=m.views.length>1;
    fitFrame(fr,entry.ph.W,entry.ph.H,card.clientWidth-16,window.innerHeight*(two?0.42:0.52));
    const cv=fr.querySelector("canvas");sizeCanvas(cv,fr);
    const items=m.issues.filter(x=>x.view===v).concat(m.borderline.filter(x=>x.view===v));
    drawOverlay(cv,entry,{pts:1,edges:1,plumb:1,marks:1},marksFor(entry,items));
  });
}
function renderResult(){
  const m=merged(),n=m.issues.length,nb=m.borderline.length;
  const src=m.views.map(v=>VIEW_ZH[v]).join("和")+"照";
  let badge,h,p;
  if(n){badge=m.issues.some(x=>x.level==="notable")?"bad":"warn";
    h="发现 "+n+" 个体态问题";p="根据"+src+"分析 · 下面是具体情况和改善动作";}
  else if(nb){badge="ok";h="整体不错";p="有 "+nb+" 处在临界附近，可以留意";}
  else{badge="ok";h="体态不错";p="根据"+src+"分析，检查的几项都在正常范围内";}
  let html=photoCards(m.views);
  html+='<div class="summary rise"><div class="badge '+badge+'">'+(badge==="ok"?ICON.check:ICON.alert)+
        '</div><div><h2>'+esc(h)+'</h2><p>'+esc(p)+'</p></div></div>';
  m.issues.forEach((x,i)=>{
    html+='<article class="issue rise" style="animation-delay:'+(80+i*90)+'ms"><div class="issue-head">'+
      '<span class="num '+x.level+'">'+x.num+'</span><h3>'+esc(x.name)+'</h3><span class="lv '+x.level+'">'+esc(x.level_label)+'</span></div>'+
      '<p class="what">'+esc(x.summary)+'</p><p class="why">'+esc(x.why)+'</p>'+
      '<div class="tips"><h4>改善动作</h4>'+x.tips.map(t=>'<div class="tip"><i>'+ICON.move+'</i><b>'+esc(t.name)+'</b><span>'+esc(t.how)+'</span></div>').join("")+
      '</div></article>';
  });
  if(nb)html+='<div class="block rise"><h4>可以留意</h4>'+m.borderline.map(x=>
      '<div class="row"><div class="grow"><b>'+esc(x.name)+'</b><p>'+esc(x.summary)+'程度在正常与轻度之间。</p></div><span class="lv borderline">'+esc(x.level_label)+'</span></div>').join("")+'</div>';
  if(m.good.length)html+='<div class="block rise"><h4>这些方面不错</h4><div class="chips">'+
      m.good.map(g=>'<span class="chip">'+ICON.check+esc(g.name)+'</span>').join("")+'</div></div>';
  const notes=[];
  if(m.nm.length)notes.push("<b>这次没检查：</b>"+m.nm.map(x=>esc(x.name)+"（"+esc(x.reason)+"）").join("；"));
  if(m.tips.length)notes.push("<b>拍摄小建议：</b>"+m.tips.map(esc).join(" "));
  if(notes.length)html+=notes.map(t=>'<p class="note">'+t+'</p>').join("");
  const missing=["side","front"].find(v=>!state.results[v]);
  if(missing){const ov=C.copy.other_view[missing];
    html+='<button class="next rise" type="button" id="nextBtn">'+ICON.camera+'<div class="grow"><b>再拍一张'+esc(ov.label)+'照</b><span>还能检查'+esc(ov.covers)+'</span></div>'+
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></button>';}
  html+='<button class="again" type="button" id="againBtn">重新检测</button>';
  $("#result").innerHTML=html;
  const nb2=$("#nextBtn");if(nb2)nb2.addEventListener("click",()=>fileInput.click());
  $("#againBtn").addEventListener("click",()=>{state.results={};state.last=null;show("home");});
}
function renderRetake(entry){
  const rep=entry.rep,has=Object.keys(state.results).length>0;
  let html="";
  if(entry.ph)html+='<div class="photos"><div class="pcard"><div class="frame" data-retake="1"><img alt="上传的照片" src="'+entry.ph.src+'"><canvas></canvas></div></div></div>';
  html+='<div class="summary rise"><div class="badge warn">'+ICON.alert+'</div><div><h2>这张照片没法准确分析</h2><p>换一张照片就好，按下面的提示拍</p></div></div>';
  html+='<div class="retake rise">'+rep.retake.map(x=>'<div class="row"><span class="rx">'+ICON.x+'</span><div class="grow"><b>'+esc(x.title)+'</b><p>'+esc(x.tip)+'</p></div></div>').join("")+'</div>';
  html+='<button class="next rise" type="button" id="retakeBtn">'+ICON.camera+'<div class="grow"><b>换一张照片</b><span>全身入镜 · 正面或侧面 · 自然站直</span></div>'+
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></button>';
  if(has)html+='<button class="again" type="button" id="backBtn">返回上次结果</button>';
  $("#result").innerHTML=html;
  $("#retakeBtn").addEventListener("click",()=>fileInput.click());
  const b=$("#backBtn");if(b)b.addEventListener("click",()=>{renderResult();requestAnimationFrame(drawResultPhotos);window.scrollTo({top:0});});
  const fr=document.querySelector("#result .frame[data-retake]");
  if(fr){const img=fr.querySelector("img");
    const place=()=>{fitFrame(fr,entry.ph.W,entry.ph.H,fr.parentElement.clientWidth-16,window.innerHeight*0.42);
      if(entry.r){const cv=fr.querySelector("canvas");sizeCanvas(cv,fr);drawOverlay(cv,entry,{pts:1,edges:1,plumb:0,marks:0},[]);}};
    img.complete?place():img.addEventListener("load",place,{once:true});}
}
window.addEventListener("resize",()=>{
  if(!$("#result").hidden&&document.querySelector("#result .frame[data-view]"))drawResultPhotos();
});

/* ---------------- inputs ---------------- */
fileInput.addEventListener("change",()=>{
  const f=fileInput.files&&fileInput.files[0];fileInput.value="";
  if(f&&!state.busy)analyze({kind:"file",file:f});
});
$("#samples").innerHTML=SAMPLES.map((s,i)=>
  '<button class="sample" type="button" data-i="'+i+'"><img alt="'+esc(s.label)+'" src="data:image/jpeg;base64,'+s.thumb+'"><span>'+esc(s.label)+'</span></button>').join("");
$("#samples").addEventListener("click",e=>{const b=e.target.closest(".sample");
  if(b&&!state.busy)analyze({kind:"sample",sample:SAMPLES[+b.dataset.i]});});

/* ---------------- hero figure: a real detected skeleton, being scanned ---------------- */
const hero=(()=>{
  const cv=$("#heroCanvas"),s=SAMPLES[0],P=s.lm;let raf=0,on=false,col=null,frameN=0;
  const idx=JOINTS.filter(i=>P[i][1]<=s.h+20);
  function colors(){const cs=getComputedStyle(document.documentElement);
    return {ink:cs.getPropertyValue("--ink").trim()||"#0F221D",jade:cs.getPropertyValue("--jade").trim()||"#0F9E7B"};}
  function draw(now){
    const dpr=Math.min(2.5,window.devicePixelRatio||1),w=cv.clientWidth,h=cv.clientHeight;
    if(!w||!h)return;
    if(cv.width!==Math.round(w*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
    if(!col||frameN++%45===0)col=colors();
    const ctx=cv.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const xs=idx.map(i=>P[i][0]),ys=idx.map(i=>P[i][1]);
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const k=Math.min((w-24)/(x1-x0||1),(h-26)/(y1-y0));
    const ox=(w-(x1-x0)*k)/2-x0*k,oy=h-8-(y1-y0)*k-y0*k;
    const X=i=>P[i][0]*k+ox,Y=i=>P[i][1]*k+oy;
    const t=REDUCED?0.62:((now||0)%3400)/3400,beamY=-10+(h+20)*t;
    const near=P[7][2]>=P[8][2]?23:24,px0=X(near);
    ctx.setLineDash([4,5]);ctx.strokeStyle=col.jade;ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(px0,Math.min(Y(0),Y(7))-14);ctx.lineTo(px0,h-4);ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=col.ink;ctx.lineWidth=2.4;ctx.lineCap="round";ctx.globalAlpha=.85;
    for(const [a,b] of EDGES){ctx.beginPath();ctx.moveTo(X(a),Y(a));ctx.lineTo(X(b),Y(b));ctx.stroke();}
    ctx.globalAlpha=1;
    for(const i of idx){const near2=Math.max(0,1-Math.abs(Y(i)-beamY)/26);
      ctx.fillStyle=col.jade;ctx.beginPath();ctx.arc(X(i),Y(i),2.6+near2*2.4,0,7);ctx.fill();}
    const g=ctx.createLinearGradient(0,beamY-26,0,beamY);g.addColorStop(0,"rgba(15,158,123,0)");g.addColorStop(1,"rgba(15,158,123,.28)");
    ctx.fillStyle=g;ctx.fillRect(0,beamY-26,w,26);
    ctx.fillStyle=col.jade;ctx.fillRect(4,beamY,w-8,1.5);
    if(on&&!REDUCED)raf=requestAnimationFrame(draw);
  }
  return {run(v){on=v;cancelAnimationFrame(raf);if(v)raf=requestAnimationFrame(draw);}};
})();
hero.run(true);
