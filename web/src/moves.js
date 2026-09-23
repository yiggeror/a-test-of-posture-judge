"use strict";
/* Looping stick-figure demos for the exercises in posture/consumer_zh.json.
   Each move is two key poses on a 100x100 box (y down) plus props; the figure
   eases A -> B, holds, and eases back. Keyed by the exercise's name as shown
   to the user, so a renamed tip simply falls back to the plain icon.

   Side-view joints: H head, N neck, S shoulder, E elbow, W wrist, P hip,
   K knee, A ankle, T toe; a trailing 2 is the far limb, drawn lighter.
   Front-view joints: H, N, and l/r pairs (image left/right) for s e w p k a. */
const Moves = (() => {
  const SIDE = {H:[53,15],N:[50,25],S:[50,28],E:[51,44],W:[53,59],P:[49,57],K:[51,75],A:[49,92],T:[57,93],
                E2:[48,44],W2:[47,59],K2:[48,75],A2:[48,92],T2:[55,93]};
  const FRONT = {H:[50,13],N:[50,23],ls:[40,27],rs:[60,27],le:[37,43],re:[63,43],lw:[36,58],rw:[64,58],
                 lp:[44,55],rp:[56,55],lk:[44,74],rk:[56,74],la:[44,92],ra:[56,92]};
  const pose = (base, o) => Object.assign({}, base, o || {});
  const lift = (p, dy, keep) => { const q = {}; for (const k in p) q[k] = keep.includes(k) ? p[k] : [p[k][0], p[k][1] - dy]; return q; };

  const MOVES = {
    "收下巴": {view:"side", hi:["H"], props:[{floor:94}],
      A: pose(SIDE, {H:[62,18], N:[53,26]}), B: pose(SIDE, {H:[50,15], N:[50,25]})},
    "靠墙站": {view:"side", hi:["H","S"], props:[{wall:41},{floor:94}],
      A: pose(SIDE, {H:[56,17],N:[49,26],S:[47,29],E:[48,44],W:[50,58],P:[44,57],K:[46,75],A:[47,92],T:[55,93],E2:[46,44],W2:[45,58],K2:[45,75],A2:[46,92],T2:[54,93]}),
      B: pose(SIDE, {H:[47,14],N:[45,24],S:[44,28],E:[45,44],W:[47,58],P:[44,57],K:[46,75],A:[47,92],T:[55,93],E2:[43,44],W2:[44,58],K2:[45,75],A2:[46,92],T2:[54,93]})},
    "把屏幕抬高": {view:"side", hi:["H","E","W"], props:[{phone:"W"},{floor:94}],
      A: pose(SIDE, {H:[57,20],N:[51,27],S:[50,29],E:[55,43],W:[63,40]}),
      B: pose(SIDE, {H:[52,15],N:[50,25],S:[50,28],E:[58,31],W:[66,19]})},
    "门框拉伸": {view:"side", hi:["S","N","H"], props:[{post:58},{floor:94}],
      A: pose(SIDE, {H:[52,15],N:[49,25],S:[49,28],E:[58,28],W:[58,14],P:[48,57],K:[53,75],A:[56,92],T:[63,93],E2:[57,29],W2:[57,15],K2:[46,75],A2:[43,92],T2:[50,93]}),
      B: pose(SIDE, {H:[63,16],N:[59,25],S:[59,29],E:[58,29],W:[58,15],P:[52,57],K:[55,75],A:[56,92],T:[63,93],E2:[57,30],W2:[57,16],K2:[48,75],A2:[43,92],T2:[50,93]})},
    "夹背": {view:"side", hi:["S","E"], props:[{floor:94}],
      A: pose(SIDE, {H:[57,17],N:[53,26],S:[55,30],E:[57,46],W:[58,60]}),
      B: pose(SIDE, {H:[52,15],N:[49,25],S:[46,28],E:[45,44],W:[47,59],E2:[44,44],W2:[45,59]})},
    "靠墙天使": {view:"front", hi:["le","re","lw","rw"], props:[{panel:1},{floor:94}],
      A: pose(FRONT, {le:[31,35],re:[69,35],lw:[29,21],rw:[71,21]}),
      B: pose(FRONT, {le:[31,17],re:[69,17],lw:[27,5],rw:[73,5]})},
    "找准重心": {view:"side", hi:["P","H"], props:[{plumb:50},{floor:94}],
      A: pose(SIDE, {H:[50,16],N:[47,26],S:[46,29],E:[47,45],W:[49,59],P:[56,57],K:[54,75],E2:[45,45],W2:[46,59],K2:[52,75]}),
      B: pose(SIDE, {H:[51,13],N:[50,24],S:[50,27]})},
    "臀桥": {view:"side", hi:["N","P","K"], props:[{floor:93}],
      A: {H:[12,84],N:[20,87],S:[22,88],E:[33,90],W:[44,91],P:[50,88],K:[62,72],A:[71,91],T:[78,92]},
      B: {H:[12,84],N:[20,87],S:[22,88],E:[33,90],W:[44,91],P:[50,75],K:[64,68],A:[71,91],T:[78,92]}},
    "死虫式": {view:"side", hi:["S","E","W","P","K2","A2","T2"], props:[{floor:93}],
      A: {H:[12,82],N:[19,85],S:[22,86],E:[24,72],W:[25,58],P:[50,86],K:[52,68],A:[66,68],T:[70,64],
          E2:[25,72],W2:[26,58],K2:[53,68],A2:[67,68],T2:[71,64]},
      B: {H:[12,82],N:[19,85],S:[22,86],E:[12,77],W:[2,82],P:[50,86],K:[52,68],A:[66,68],T:[70,64],
          E2:[25,72],W2:[26,58],K2:[66,78],A2:[82,83],T2:[87,79]}},
    "提踵": {view:"side", hi:["P","K","A","T"], props:[{wall:72},{floor:94}],
      A: pose(SIDE, {E:[60,43],W:[70,49]}),
      B: Object.assign(lift(pose(SIDE, {E:[60,43],W:[70,49]}), 6, ["T","T2","W"]), {A:[50,87],A2:[49,87]})},
    "两侧轮换": {view:"front", hi:["ls","rs"], props:[{bag:1},{floor:94}],
      A: pose(FRONT, {ls:[40,25.5]}), B: pose(FRONT, {rs:[60,25.5]})},
    "侧颈拉伸": {view:"front", hi:["H","rw","re"], props:[{floor:94}],
      A: pose(FRONT), B: pose(FRONT, {H:[44,15],N:[48,23],rs:[60,28],re:[63,17],rw:[46,6]})},
    "侧平板": {view:"side", hi:["N","P","K","A"], props:[{floor:93}],
      A: {H:[16,70],N:[22,74],S:[24,77],E:[24,92],W:[37,92],P:[52,90],K:[70,91],A:[86,92],T:[91,90]},
      B: {H:[16,70],N:[22,74],S:[24,77],E:[24,92],W:[37,92],P:[52,84],K:[70,88],A:[86,92],T:[91,90]}},
    "把屏幕摆正": {view:"front", hi:["H"], props:[{screen:1},{floor:94}],
      A: pose(FRONT, {H:[55,14],N:[51,23]}), B: pose(FRONT)},
    "对镜自检": {view:"front", hi:["H"], props:[{mirror:1},{floor:94}],
      A: pose(FRONT, {H:[45,14],N:[48,23]}), B: pose(FRONT)},
  };
  // The advice for the back-shifted trunk reuses 靠墙站 and 找准重心 as they are.

  const SIDE_EDGES = [["N","S"],["S","E"],["E","W"],["N","P"],["P","K"],["K","A"],["A","T"]];
  const FAR_EDGES = [["S","E2"],["E2","W2"],["P","K2"],["K2","A2"],["A2","T2"]];
  const FRONT_EDGES = [["ls","rs"],["ls","le"],["le","lw"],["rs","re"],["re","rw"],["ls","lp"],["rs","rp"],
                       ["lp","rp"],["lp","lk"],["lk","la"],["rp","rk"],["rk","ra"]];
  const ease = t => t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3)/2;
  const CYCLE = [500, 750, 900, 750];   // hold A, A->B, hold B, B->A

  function mix(a, b, t) { const o = {}; for (const k in a) { const q = b[k] || a[k]; o[k] = [a[k][0] + (q[0]-a[k][0])*t, a[k][1] + (q[1]-a[k][1])*t]; } return o; }
  function phase(ms) {
    const total = CYCLE.reduce((s, x) => s + x, 0); let t = ms % total;
    if (t < CYCLE[0]) return 0; t -= CYCLE[0];
    if (t < CYCLE[1]) return ease(t / CYCLE[1]); t -= CYCLE[1];
    if (t < CYCLE[2]) return 1; t -= CYCLE[2];
    return 1 - ease(t / CYCLE[3]);
  }

  function draw(cv, name, ms, col) {
    const mv = MOVES[name]; if (!mv) return;
    const dpr = Math.min(2.5, window.devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w*dpr)) { cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr); }
    const ctx = cv.getContext("2d"), k = Math.min(w, h) / 100;
    ctx.setTransform(dpr*k, 0, 0, dpr*k, dpr*(w - 100*k)/2, dpr*(h - 100*k)/2);
    ctx.clearRect(-50, -50, 200, 200);
    const t = ms == null ? 1 : phase(ms), P = mix(mv.A, mv.B, t);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    // props
    for (const pr of mv.props) {
      ctx.strokeStyle = col.line; ctx.fillStyle = col.soft; ctx.lineWidth = 2;
      if (pr.floor) { ctx.beginPath(); ctx.moveTo(4, pr.floor); ctx.lineTo(96, pr.floor); ctx.stroke(); }
      if (pr.wall) { ctx.globalAlpha = .5; ctx.fillStyle = col.prop; ctx.fillRect(pr.wall - 4, 4, 4, 90); ctx.globalAlpha = 1; }
      if (pr.post) { ctx.globalAlpha = .5; ctx.fillStyle = col.prop; ctx.fillRect(pr.post, 4, 5, 90); ctx.globalAlpha = 1; }
      if (pr.panel) { ctx.fillRect(18, 2, 64, 92); }
      if (pr.plumb) { ctx.setLineDash([3, 3]); ctx.strokeStyle = col.hi; ctx.beginPath(); ctx.moveTo(pr.plumb, 3); ctx.lineTo(pr.plumb, 94); ctx.stroke(); ctx.setLineDash([]); }
      if (pr.mirror) { ctx.strokeStyle = col.prop; ctx.globalAlpha = .6; ctx.lineWidth = 3; ctx.strokeRect(22, 2, 56, 92); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(50, 4); ctx.lineTo(50, 92); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1; }
    }
    const L = (a, b, lw, c) => { ctx.strokeStyle = c; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke(); };
    const hot = new Set(mv.hi);
    if (mv.view === "side") {
      if (P.E2) for (const [a, b] of FAR_EDGES) if (P[b]) L(a, b, 4.2, hot.has(a) && hot.has(b) ? col.hi : col.far);
      for (const [a, b] of SIDE_EDGES) L(a, b, 4.6, hot.has(a) && hot.has(b) ? col.hi : col.ink);
    } else {
      for (const [a, b] of FRONT_EDGES) L(a, b, 4.2, hot.has(a) || hot.has(b) ? col.hi : col.ink);
      L("N", "ls", 4.2, col.ink); L("N", "rs", 4.2, col.ink);
    }
    // head and neck
    ctx.strokeStyle = hot.has("H") ? col.hi : col.ink; ctx.lineWidth = 4.2;
    ctx.beginPath(); ctx.moveTo(P.N[0], P.N[1]); ctx.lineTo(P.N[0] + (P.H[0]-P.N[0])*.4, P.N[1] + (P.H[1]-P.N[1])*.4); ctx.stroke();
    ctx.fillStyle = hot.has("H") ? col.hi : col.ink;
    ctx.beginPath(); ctx.arc(P.H[0], P.H[1], 7, 0, 7); ctx.fill();
    // props attached to the figure
    for (const pr of mv.props) {
      if (pr.phone) { const [x, y] = P[pr.phone]; ctx.fillStyle = col.hi; ctx.save(); ctx.translate(x, y); ctx.rotate(-.35); ctx.fillRect(-2, -8, 5, 10); ctx.restore(); }
      if (pr.bag) {
        const side = t < .5 ? "ls" : "rs", [sx, sy] = P[side], dir = side === "ls" ? -1 : 1;
        const a = Math.abs(t - .5) * 2, bx = sx + dir * 5, by = 50;
        ctx.globalAlpha = a; ctx.strokeStyle = col.hi; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(bx, by - 6); ctx.stroke();
        ctx.fillStyle = col.hi; ctx.fillRect(bx - 6, by - 6, 12, 11); ctx.globalAlpha = 1;
      }
      if (pr.screen) { const x = 50 + (1 - t) * 26; ctx.fillStyle = col.soft; ctx.strokeStyle = col.hi; ctx.lineWidth = 2.5; ctx.fillRect(x - 11, 28, 22, 15); ctx.strokeRect(x - 11, 28, 22, 15); ctx.beginPath(); ctx.moveTo(x, 43); ctx.lineTo(x, 48); ctx.stroke(); }
    }
  }

  // One animation loop for every demo on screen; off-screen ones sleep.
  let raf = 0, col = null, visible = new Set(), io = null;
  function colors() {
    const cs = getComputedStyle(document.documentElement), v = n => cs.getPropertyValue(n).trim();
    return {ink: v("--ink2") || "#33463F", far: v("--line") || "#D8E0DB", hi: v("--jade") || "#0F9E7B",
            line: v("--line") || "#D8E0DB", soft: v("--jade-soft") || "#DCF1E9", prop: v("--muted") || "#6A7C75"};
  }
  function loop(now) {
    for (const cv of visible) draw(cv, cv.dataset.move, now, col);
    raf = visible.size ? requestAnimationFrame(loop) : 0;
  }
  function mount(root) {
    col = colors();
    if (io) io.disconnect();
    visible = new Set();
    const cvs = [...root.querySelectorAll("canvas[data-move]")];
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window);
    for (const cv of cvs) draw(cv, cv.dataset.move, null, col);   // resting frame first
    if (still) return;
    io = new IntersectionObserver(es => {
      for (const e of es) e.isIntersecting ? visible.add(e.target) : visible.delete(e.target);
      if (visible.size && !raf) raf = requestAnimationFrame(loop);
    });
    cvs.forEach(cv => io.observe(cv));
  }
  function stop() { if (io) io.disconnect(); visible.clear(); cancelAnimationFrame(raf); raf = 0; }
  return {has: n => !!MOVES[n], mount, stop, names: () => Object.keys(MOVES)};
})();
