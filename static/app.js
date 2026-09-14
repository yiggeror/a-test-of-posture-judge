const $ = s => document.querySelector(s);
const drop = $('#drop'), fileInput = $('#file'), go = $('#go'), status = $('#status');
let chosen = null;

const BAND = {typical:'参考范围内', mild:'轻度倾向', pronounced:'明显倾向', unavailable:'未计算'};
const BASIS = {
  'guess':'无依据·纯猜测',
  'geometric-estimate':'几何推算（基于典型体尺假设）',
  'literature-adjacent':'文献相邻（构念相近但不可直接套用）'
};

function pick(f){
  if(!f) return;
  if(!f.type.startsWith('image/')){ setStatus('请选择图片文件。', true); return; }
  chosen = f; go.disabled = false;
  setStatus(`已选择：${f.name}（${(f.size/1024/1024).toFixed(2)} MB）`);
}
function setStatus(t, err){ status.textContent = t; status.classList.toggle('err', !!err); }

drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => pick(e.target.files[0]));
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('over');
}));
drop.addEventListener('drop', e => pick(e.dataTransfer.files[0]));

go.addEventListener('click', async () => {
  if(!chosen) return;
  go.disabled = true; setStatus('分析中…');
  const fd = new FormData();
  fd.append('image', chosen);
  fd.append('variant', $('#variant').value);
  if($('#view').value) fd.append('view', $('#view').value);
  try{
    const r = await fetch('/api/analyze', {method:'POST', body:fd});
    const d = await r.json();
    if(!d.ok){ setStatus(d.error || '分析失败。', true); $('#results').hidden = true; }
    else { setStatus(''); paint(d); }
  }catch(err){ setStatus('请求失败：' + err.message, true); }
  finally{ go.disabled = false; }
});

function fmt(v, unit){
  if(v === null || v === undefined) return '—';
  if(unit === 'deg') return v.toFixed(1) + '<span class="u">°</span>';
  return v.toFixed(3) + '<span class="u">×躯干长</span>';
}

function paint(d){
  $('#results').hidden = false;
  $('#overlay').src = d.overlay;

  const pl = d.plausibility, box = $('#plaus');
  if(pl && !pl.ok){
    box.hidden = false;
    box.innerHTML = `<strong>姿态合理性检查未通过 — 本次不输出任何数值。</strong>
      <ul>${pl.reasons.map(r=>`<li>${r}</li>`).join('')}</ul>
      <p class="fine">躯干倾角 ${pl.torso_tilt_deg}° · 头颈段/躯干 ${pl.head_torso_ratio}。
      MediaPipe 的 visibility 只估计遮挡，不判断「画面里是不是人」，
      因此它可能对非人体图像输出高置信度的完整骨架。此处用体态比例做二次校验。</p>`;
  } else {
    box.hidden = true;
  }
  $('#viewinfo').innerHTML =
    `视角判定：<strong>${d.view.label}</strong>` +
    (d.view.forced ? '（手动指定）' : `（自动；肩宽/躯干长 = ${d.view.ratio}）`) +
    ` · 图像 ${d.image.width}×${d.image.height}px`;

  $('#metrics').innerHTML = d.metrics.map(m => {
    const t = m.threshold;
    let thr = '';
    if(t){
      thr = `<p class="mthresh">阈值：轻度 ≥ ${t.mild}，明显 ≥ ${t.marked} ${t.unit}
             <span class="basis basis-${t.basis}">${BASIS[t.basis]}</span><br>
             <span>${t.note}</span></p>`;
    } else {
      thr = `<p class="mthresh"><span class="basis basis-guess">无阈值</span> 仅给出数值，不做判定。</p>`;
    }
    const cav = m.caveats.length
      ? `<ul class="mcav">${m.caveats.map(c=>`<li>${c}</li>`).join('')}</ul>` : '';
    return `<div class="metric">
      <div class="mhead">
        <span class="mname">${m.label}<span class="pill b-${m.band}">${BAND[m.band]}</span></span>
        <span class="mval">${fmt(m.value, m.unit)}</span>
      </div>
      ${m.detail ? `<p class="mdetail">${m.detail}</p>` : ''}
      ${thr}
      <p class="madvice">${m.advice}</p>
      ${cav}
    </div>`;
  }).join('');

  const tb = $('#lmtable').querySelector('tbody');
  tb.innerHTML = d.landmarks.map(l =>
    `<tr class="${l.visibility < 0.5 ? 'dim' : ''}">
       <td>${l.i}</td><td>${l.name}</td><td>${l.x}</td><td>${l.y}</td>
       <td>${l.visibility.toFixed(3)}</td></tr>`).join('');
}
