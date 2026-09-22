/* ---------- rendering ---------- */
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const VIEWZH={front:"正面",side:"侧面",oblique:"斜侧"};
let cur=0,imgs=[];

function drawShot(S,res){
  const cv=$("cv"),ctx=cv.getContext("2d");
  cv.width=S.w;cv.height=S.h;
  ctx.drawImage(imgs[cur],0,0,S.w,S.h);
  const P=S.lm,sc=S.h/760,lw=Math.max(1.6,2.4*sc),r=Math.max(2.2,3.4*sc);
  ctx.lineCap="round";
  ctx.strokeStyle="rgba(53,181,196,.85)";ctx.lineWidth=lw;
  for(const[a,b]of EDGES){ctx.beginPath();ctx.moveTo(P[a][0],P[a][1]);ctx.lineTo(P[b][0],P[b][1]);ctx.stroke();}
  for(let i=0;i<33;i++){
    const off=res.oofSet.has(i);
    ctx.beginPath();ctx.arc(P[i][0],P[i][1],r,0,7);
    if(off){ctx.strokeStyle="#e0796a";ctx.lineWidth=Math.max(1.4,2*sc);ctx.stroke();}
    else{ctx.fillStyle="rgba(255,255,255,.95)";ctx.fill();}
  }
}
function gauge(m,v){
  const sp=v?v.sp:null;if(!sp)return"";
  const sym=sp.direction!=="positive_only";
  const max=sp.notable*1.38, lo=sym?-max:-sp.notable*0.3, span=max-lo;
  const pc=x=>Math.max(0,Math.min(100,((x-lo)/span)*100));
  const bandCol=v.resolved?({reference:"var(--ok)",slight:"var(--warn)",notable:"var(--alert)"}[v.band]):"var(--muted)";
  let h='<div class="gauge"><div class="g-track"></div>';
  if(sym){
    h+=`<div class="g-zone z1" style="left:${pc(sp.slight)}%;width:${pc(sp.notable)-pc(sp.slight)}%"></div>`;
    h+=`<div class="g-zone z1" style="left:${pc(-sp.notable)}%;width:${pc(-sp.slight)-pc(-sp.notable)}%"></div>`;
    h+=`<div class="g-zone z2" style="left:${pc(sp.notable)}%;width:${100-pc(sp.notable)}%"></div>`;
    h+=`<div class="g-zone z2" style="left:0;width:${pc(-sp.notable)}%"></div>`;
    h+=`<div class="g-tick" style="left:${pc(-sp.slight)}%"></div><div class="g-tick" style="left:${pc(-sp.notable)}%"></div>`;
  }else{
    h+=`<div class="g-zone z1" style="left:${pc(sp.slight)}%;width:${pc(sp.notable)-pc(sp.slight)}%"></div>`;
    h+=`<div class="g-zone z2" style="left:${pc(sp.notable)}%;width:${100-pc(sp.notable)}%"></div>`;
  }
  h+=`<div class="g-tick" style="left:${pc(sp.slight)}%"></div><div class="g-tick" style="left:${pc(sp.notable)}%"></div>`;
  h+=`<div class="g-lab" style="left:${pc(sp.slight)}%">${sp.slight}</div>`;
  h+=`<div class="g-lab" style="left:${pc(sp.notable)}%">${sp.notable}</div>`;
  const a=m.val-m.unc,b=m.val+m.unc;
  if(b<lo||a>max){h+=`<div class="g-out" style="${m.val>max?"right:2px":"left:2px"}">超出量程</div>`;}
  else{h+=`<div class="g-bar" style="left:${pc(a)}%;width:${Math.max(0.8,pc(b)-pc(a))}%;background:${bandCol};opacity:.42"></div>`;
       h+=`<div class="g-dot" style="left:calc(${pc(m.val)}% - 1.25px);background:${bandCol}"></div>`;}
  return h+"</div>";
}
function bandText(v){
  const L={reference:"参考范围内",slight:"轻度倾向",notable:"明显倾向"};
  if(v.floored)return`<span class="band undet">测量精度不足以判定（±${v.unc.toFixed(1)}° 已超过「轻度倾向」整档的宽度 ${(v.sp.notable-v.sp.slight).toFixed(1)}°）</span>`;
  if(v.resolved)return`<span class="band ${v.band}">${L[v.band]}</span>`;
  return`<span class="band undet">${L[v.band]} / ${L[v.alt]||""}（读数落在阈值的不确定度范围内，无法区分）</span>`;
}
function render(i){
  cur=i;const S=SAMPLES[i],res=assess(S);
  res.oofSet=outOfFrame(S.lm,S.w,S.h);
  drawShot(S,res);
  $("shotcap").textContent=S.note+"。空心红点＝该关键点在画面之外，位置是推测的。";
  document.querySelectorAll(".samp").forEach((b,k)=>b.setAttribute("aria-pressed",k===i?"true":"false"));

  const V=res.V;
  $("chips").innerHTML=
    `<span class="chip on">视角 ${VIEWZH[V.view]||V.view}</span>`+
    `<span class="chip">躯干偏航 ${V.yaw.toFixed(0)}°</span>`+
    `<span class="chip">肩到踝 ${V.scale.toFixed(0)} px</span>`+
    `<span class="chip">${res.blocked?"不给出判定":"已给出判定"}</span>`;

  $("msgs").innerHTML=res.findings
    .sort((a,b)=>(a.sev==="block"?0:1)-(b.sev==="block"?0:1))
    .map(f=>`<div class="msg ${f.sev}">${esc(f.msg)}<br><code>${esc(f.key)}</code></div>`).join("");

  const un=Object.keys(res.unavail);
  $("unavail").innerHTML=un.length
    ? `<div class="unavail"><b>本次停用 ${un.length} 项</b>（依赖的关键点在画面外）：`+
      un.map(k=>`${MLAB[k]||k} — 缺 ${res.unavail[k].join("、")}`).join("；")+"。其余读数不受影响。</div>"
    : "";

  const keys=ORDER.filter(k=>res.mets[k]&&!res.unavail[k]);
  $("rows").innerHTML=keys.map(k=>{
    const m=res.mets[k],v=res.verdicts[k],d=DIAG.has(k);
    const vv=v?Object.assign({},v,{unc:m.unc}):null;
    let h=`<div class="row${d?" diag":""}"><div class="rhead">`+
      `<span class="rname">${esc(MLAB[k]||k)}${d?'<span class="sub">辅助项 · 不判定</span>':""}</span>`+
      `<span class="rval">${m.val>=0?"+":""}${m.val.toFixed(1)}°<span class="pm"> ±${m.unc.toFixed(1)}</span></span></div>`;
    if(vv){
      h+=gauge(m,vv);
      h+=`<div class="verdict">${bandText(vv)}`+
         `<span class="prov ${vv.sp.provenance==="population-percentile"?"measured":"guess"}">`+
         (vv.sp.provenance==="population-percentile"?`实测分位数 n=${vv.sp.n}`:"凭经验设定")+`</span></div>`;
      if(vv.advice.length)h+=`<ul class="advice">${vv.advice.map(a=>`<li>${esc(a)}</li>`).join("")}</ul>`;
    }
    h+=`<div class="advice" style="padding-left:0;font-size:12px;color:var(--faint);margin-top:6px">${esc(m.note)}</div>`;
    return h+"</div>";
  }).join("");
}
/* ---------- boot ---------- */
$("picker").innerHTML=SAMPLES.map((s,i)=>
  `<button class="samp" type="button" aria-pressed="false" data-i="${i}">`+
  `<img alt="${esc(s.title)}" src="data:image/jpeg;base64,${s.jpg}">`+
  `<span><b>${esc(s.title)}</b>${esc(s.note)}</span></button>`).join("");
let pending=SAMPLES.length;
SAMPLES.forEach((s,i)=>{const im=new Image();imgs[i]=im;
  im.onload=im.onerror=()=>{if(--pending===0)render(0);};
  im.src="data:image/jpeg;base64,"+s.jpg;});
$("picker").addEventListener("click",e=>{const b=e.target.closest(".samp");if(b)render(+b.dataset.i);});
</script>
