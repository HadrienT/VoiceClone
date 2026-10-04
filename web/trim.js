/** Éditeur de forme d'onde : sélectionner le passage à garder (ou à supprimer) avant l'import. */
import { encodeWav } from "./audio.js";
import { $, esc } from "./util.js";

async function decode(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < d.length; i++) mono[i] += d[i] / buf.numberOfChannels;
    }
    return { samples: mono, rate: buf.sampleRate };
  } finally { ctx.close(); }
}

const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

/**
 * Ouvre l'éditeur. Résout avec { blob, duration } (audio modifié) ou null si annulé.
 */
export async function openTrimmer(blob, name = "") {
  const { samples, rate } = await decode(blob);
  const total = samples.length / rate;
  const dlg = document.createElement("dialog");
  dlg.className = "trimmer";
  dlg.innerHTML = `<h3 style="margin:0 0 4px">✂ ${esc(name)}</h3>
    <p class="muted" style="margin:0 0 10px;font-size:13px">Glissez sur la forme d'onde pour sélectionner un passage
      (ex. retirer un rire, un bruit, une autre personne qui parle). Espace = écouter.</p>
    <div class="wave-wrap"><canvas></canvas><div class="sel hidden"></div><div class="cursor"></div></div>
    <div class="row" style="align-items:center;margin:10px 0 0">
      <span class="muted grow" data-info>Durée : ${fmt(total)}</span>
      <button class="btn small" data-play>▶ Écouter</button>
      <button class="btn small" data-all>Tout sélectionner</button>
    </div>
    <div class="actions">
      <button class="btn" data-cancel>Annuler</button>
      <button class="btn" data-cut disabled title="Supprime le passage sélectionné">✂ Supprimer la sélection</button>
      <button class="btn primary" data-keep disabled>Garder seulement la sélection</button>
    </div>`;
  document.body.append(dlg);
  const canvas = $("canvas", dlg);
  const wrap = $(".wave-wrap", dlg);
  const selEl = $(".sel", dlg);
  const cursor = $(".cursor", dlg);
  let sel = null; // [début, fin] en secondes
  let audio = null;
  let raf;

  function draw() {
    const w = (canvas.width = wrap.clientWidth * devicePixelRatio);
    const h = (canvas.height = 140 * devicePixelRatio);
    const g = canvas.getContext("2d");
    g.clearRect(0, 0, w, h);
    g.fillStyle = getComputedStyle(dlg).getPropertyValue("--accent") || "#7c5cff";
    const step = samples.length / w;
    for (let x = 0; x < w; x++) {
      let lo = 1, hi = -1;
      const a = Math.floor(x * step), b = Math.min(samples.length, Math.floor((x + 1) * step) + 1);
      for (let i = a; i < b; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi < lo) continue;
      g.fillRect(x, ((1 - hi) / 2) * h, 1, Math.max(1, ((hi - lo) / 2) * h));
    }
  }
  const toTime = (clientX) => {
    const r = wrap.getBoundingClientRect();
    return Math.max(0, Math.min(total, ((clientX - r.left) / r.width) * total));
  };
  function renderSel() {
    const valid = sel && sel[1] - sel[0] > 0.05;
    selEl.classList.toggle("hidden", !valid);
    $("[data-keep]", dlg).disabled = $("[data-cut]", dlg).disabled = !valid;
    if (!valid) { $("[data-info]", dlg).textContent = `Durée : ${fmt(total)}`; return; }
    selEl.style.left = `${(sel[0] / total) * 100}%`;
    selEl.style.width = `${((sel[1] - sel[0]) / total) * 100}%`;
    $("[data-info]", dlg).textContent = `Sélection : ${fmt(sel[0])} → ${fmt(sel[1])} (${(sel[1] - sel[0]).toFixed(1)} s) · total ${fmt(total)}`;
  }
  wrap.addEventListener("pointerdown", (e) => {
    wrap.setPointerCapture(e.pointerId);
    const t0 = toTime(e.clientX);
    sel = [t0, t0];
    const move = (ev) => { const t = toTime(ev.clientX); sel = [Math.min(t0, t), Math.max(t0, t)]; renderSel(); };
    const up = () => { wrap.removeEventListener("pointermove", move); wrap.removeEventListener("pointerup", up); };
    wrap.addEventListener("pointermove", move);
    wrap.addEventListener("pointerup", up);
    renderSel();
  });

  function stop() {
    audio?.pause();
    cancelAnimationFrame(raf);
    cursor.style.display = "none";
    $("[data-play]", dlg).textContent = "▶ Écouter";
  }
  function play() {
    if (audio && !audio.paused) return stop();
    const [a, b] = sel && sel[1] - sel[0] > 0.05 ? sel : [0, total];
    const clip = samples.subarray(Math.floor(a * rate), Math.floor(b * rate));
    audio = new Audio(URL.createObjectURL(encodeWav(clip, rate)));
    audio.play();
    $("[data-play]", dlg).textContent = "■ Stop";
    cursor.style.display = "block";
    const tick = () => {
      cursor.style.left = `${((a + audio.currentTime) / total) * 100}%`;
      if (!audio.paused && !audio.ended) raf = requestAnimationFrame(tick); else stop();
    };
    tick();
  }
  $("[data-play]", dlg).addEventListener("click", play);
  $("[data-all]", dlg).addEventListener("click", () => { sel = [0, total]; renderSel(); });
  dlg.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); play(); } });

  draw();
  dlg.showModal();
  new ResizeObserver(draw).observe(wrap);

  return new Promise((resolve) => {
    const finish = (result) => { stop(); dlg.close(); dlg.remove(); resolve(result); };
    $("[data-cancel]", dlg).addEventListener("click", () => finish(null));
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); finish(null); });
    $("[data-keep]", dlg).addEventListener("click", () => {
      const out = samples.slice(Math.floor(sel[0] * rate), Math.floor(sel[1] * rate));
      finish({ blob: encodeWav(out, rate), duration: out.length / rate });
    });
    $("[data-cut]", dlg).addEventListener("click", () => {
      const a = Math.floor(sel[0] * rate), b = Math.floor(sel[1] * rate);
      const out = new Float32Array(samples.length - (b - a));
      out.set(samples.subarray(0, a));
      out.set(samples.subarray(b), a);
      finish({ blob: encodeWav(out, rate), duration: out.length / rate });
    });
  });
}
