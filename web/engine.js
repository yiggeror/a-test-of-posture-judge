"use strict";
/* Loads MediaPipe PoseLandmarker in the browser, from files published next to
   the page.

   Why the unusual loading: the page may run under a CSP that allows scripts
   from a CDN but blocks fetch() to anything except its own origin. So the
   loader script comes from the CDN (a <script> tag), while the .wasm binary
   and the model are fetched from our own origin and handed over as bytes --
   which also gives us a real download-progress number for the loading screen.

   Settings mirror posture/landmarks.py exactly (IMAGE mode, up to 4 poses,
   0.5 detection/presence confidence, CPU). The multi-person guard needs
   numPoses > 1, and CPU keeps results identical to the Python reference. */
const Engine = (() => {
  const VISION_VERSION = "1.0.1";   // must match the Python mediapipe version
  const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + VISION_VERSION;
  let landmarker = null, loading = null;

  async function fetchBytes(url, onBytes) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("下载失败 " + url + " (" + r.status + ")");
    if (!r.body || !r.body.getReader) {
      const b = new Uint8Array(await r.arrayBuffer()); onBytes(b.length); return b;
    }
    const rd = r.body.getReader(), parts = []; let n = 0;
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      parts.push(value); n += value.length; onBytes(value.length);
    }
    const out = new Uint8Array(n); let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous";
      s.onload = res; s.onerror = () => rej(new Error("脚本加载失败 " + src));
      document.head.appendChild(s);
    });
  }

  /* opts: {base, loaderUrl, wasm, modelParts: [...paths], totalBytes, onProgress(frac)} */
  function load(opts) {
    if (loading) return loading;
    loading = (async () => {
      const base = opts.base || document.baseURI;
      let got = 0;
      const tick = n => { got += n; opts.onProgress && opts.onProgress(Math.min(0.99, got / opts.totalBytes)); };
      if (typeof Vision === "undefined") await loadScript(opts.bundleUrl || CDN + "/vision_bundle.js");
      const [wasm, ...parts] = await Promise.all(
        [opts.wasm, ...opts.modelParts].map(p => fetchBytes(new URL(p, base).href, tick)));
      let model;
      if (parts.length === 1) model = parts[0];
      else {
        model = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
        let o = 0; for (const p of parts) { model.set(p, o); o += p.length; }
      }
      // The emscripten loader picks up a pre-set Module and uses wasmBinary
      // instead of fetching the binary itself.
      self.Module = { wasmBinary: wasm.buffer };
      landmarker = await Vision.PoseLandmarker.createFromOptions(
        { wasmLoaderPath: opts.loaderUrl || CDN + "/wasm/vision_wasm_internal.js",
          wasmBinaryPath: new URL(opts.wasm, base).href },
        { baseOptions: { modelAssetBuffer: model, delegate: "CPU" },
          runningMode: "IMAGE", numPoses: 4,
          minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
          outputSegmentationMasks: false });
      opts.onProgress && opts.onProgress(1);
      return landmarker;
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  /* Returns {lm: [[x,y,vis]...33], others: [[x0,y0,x1,y1]...], n} in pixels of
     the source, or null when no person is found. First pose is the subject,
     exactly as in landmarks.detect(). */
  function detect(source, W, H) {
    const r = landmarker.detect(source);
    if (!r.landmarks || !r.landmarks.length) return null;
    const toPx = pose => pose.map(p => [p.x * W, p.y * H, p.visibility == null ? 0 : p.visibility]);
    const lm = toPx(r.landmarks[0]);
    const others = r.landmarks.slice(1).map(pose => {
      const xs = pose.map(p => p.x * W), ys = pose.map(p => p.y * H);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    });
    return { lm, others, n: r.landmarks.length };
  }

  return { load, detect, get ready() { return !!landmarker; } };
})();
