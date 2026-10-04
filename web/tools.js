/** Onglet Diagnostic : matériel et mémoire GPU, modèles chargés, réglages, versions, journal, filigrane. */
import { $, esc, api, toast, busy } from "./util.js";

const ago = (t) => {
  if (!t) return "–";
  const s = Date.now() / 1000 - t;
  return s < 60 ? "à l'instant" : s < 3600 ? `il y a ${Math.round(s / 60)} min` : `il y a ${Math.round(s / 3600)} h`;
};

let models = [];
let logAfter = 0;
let logTimer;

export async function showTools() {
  await Promise.all([loadDiag(), loadSettings(), loadLogs(true)]);
}

async function loadDiag() {
  const [d, ms] = await Promise.all([api("/api/diagnostics"), api("/api/models")]);
  models = ms;
  const sys = d.system;
  const gpus = sys.gpus || [];
  $("#diag-hw").innerHTML = (gpus.length ? gpus.map((g) => g.error ? `<div class="notice error">GPU ${g.index} : ${esc(g.error)}</div>` : `
      <div class="gpu ${g.selected ? "selected" : ""}">
        <div class="kv"><span>GPU ${g.index} — ${esc(g.name)} ${g.selected ? "<span class='badge ok'>utilisé</span>" : ""}</span>
          <b>${g.used_gb} / ${g.total_gb} Go</b></div>
        <div class="vram"><span class="other" style="width:${(100 * (g.used_gb - g.voiceclone_gb)) / g.total_gb}%"></span><span class="mine" style="width:${(100 * g.voiceclone_gb) / g.total_gb}%"></span></div>
        <small class="muted">VoiceClone : ${g.voiceclone_gb} Go · autres programmes : ${Math.max(0, g.used_gb - g.voiceclone_gb).toFixed(2)} Go · libre : ${g.free_gb} Go</small>
      </div>`).join("")
    : `<p>Calcul sur <b>${esc(sys.device.toUpperCase())}</b>${sys.torch ? ` · PyTorch ${esc(sys.torch)}` : " · PyTorch non installé"}.</p>`)
    + `<p class="muted" style="font-size:13px">Python ${esc(d.python)} · ${esc(d.platform)} · données : <code>${esc(d.data_dir)}</code> (${d.disk_free_gb ?? "?"} Go libres)</p>`;

  const loaded = d.loaded_models;
  const name = (id) => models.find((m) => m.id === id)?.name || id;
  $("#diag-loaded").innerHTML = loaded.length ? `<table class="report-table"><tr><th>Modèle</th><th>Mémoire GPU</th><th>Dernière utilisation</th><th></th></tr>
    ${loaded.map((m) => `<tr><td>${esc(name(m.id))} ${m.pinned ? "<span class='badge'>Live</span>" : ""}</td>
      <td>${m.memory_gb != null ? `${m.memory_gb} Go` : "–"}</td><td>${ago(m.last_used)}</td>
      <td><button class="btn small" data-diag-unload="${m.id}">Décharger</button></td></tr>`).join("")}</table>`
    : `<p class="muted">Aucun modèle en mémoire. Ils se chargent à la première utilisation.</p>`;

  $("#diag-checks").innerHTML = d.checks.map((c) => `<li>${c.ok ? "✅" : "⚠️"} ${esc(c.label)}${c.help ? ` — <code>${esc(c.help)}</code>` : ""}</li>`).join("");
  $("#diag-packages").innerHTML = `<table class="report-table"><tr><th>Bibliothèque</th><th>Version</th></tr>
    ${d.packages.map((p) => `<tr class="${p.version ? "" : "muted"}"><td>${esc(p.name)}</td><td>${esc(p.version || "—")}</td></tr>`).join("")}
    <tr><td>ffmpeg</td><td>${esc(d.ffmpeg || "—")}</td></tr></table>`;
  return d;
}

async function loadSettings() {
  const s = await api("/api/settings");
  $("#set-max").value = s.max_loaded_models;
  $("#set-auto").checked = s.auto_unload;
  $("#set-wm").checked = s.watermark;
  const py = s.engine_python || {};
  $("#set-python").innerHTML = models.filter((m) => !m.id.startsWith("fake")).map((m) => `<label class="field">${esc(m.name)}
      <input data-py="${m.id}" value="${esc(py[m.id] || "")}" placeholder="(Python du serveur)"></label>`).join("");
}

async function loadLogs(reset) {
  clearTimeout(logTimer);
  if (reset) { logAfter = 0; $("#diag-log").textContent = ""; }
  try {
    const r = await api(`/api/logs?after=${logAfter}&level=${$("#log-level").value}`);
    if (r.records.length) {
      const pre = $("#diag-log");
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
      pre.textContent += r.records.map((x) => `${new Date(x.t * 1000).toLocaleTimeString("fr-FR")} ${x.level.padEnd(7)} ${x.name}: ${x.msg}`).join("\n") + "\n";
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    }
    logAfter = r.last;
  } catch { /* serveur indisponible */ }
  if ($("#log-follow").checked && $("#tab-tools").classList.contains("active")) logTimer = setTimeout(loadLogs, 2000);
}

$("#log-level").addEventListener("change", () => loadLogs(true));
$("#log-follow").addEventListener("change", () => loadLogs());
$("#log-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#diag-log").textContent); toast("Journal copié.", "ok"); }
  catch { toast("Copie impossible : sélectionnez le texte à la main.", "error"); }
});
$("#diag-report").addEventListener("click", async () => {
  const [d, logs] = await Promise.all([api("/api/diagnostics"), api("/api/logs?level=INFO")]);
  const blob = new Blob([JSON.stringify({ ...d, logs: logs.records }, null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "voiceclone-diagnostic.json" });
  a.click();
});
$("#diag-refresh").addEventListener("click", (e) => busy(e.currentTarget, "…", showTools));
$("#diag-free").addEventListener("click", (e) => busy(e.currentTarget, "Libération…", async () => {
  const r = await api("/api/models/unload-idle", { method: "POST" });
  toast(r.unloaded.length ? `${r.unloaded.length} modèle(s) déchargé(s).` : "Rien à décharger.", "ok");
  await loadDiag();
}));
$("#diag-loaded").addEventListener("click", (e) => {
  const b = e.target.closest("[data-diag-unload]");
  if (b) busy(b, "…", async () => { await api(`/api/models/${b.dataset.diagUnload}/unload`, { method: "POST" }); await loadDiag(); });
});
$("#set-save").addEventListener("click", (e) => busy(e.currentTarget, "Enregistrement…", async () => {
  const engine_python = {};
  document.querySelectorAll("[data-py]").forEach((i) => { if (i.value.trim()) engine_python[i.dataset.py] = i.value.trim(); });
  try {
    await api("/api/settings", { method: "PATCH", json: {
      max_loaded_models: parseInt($("#set-max").value || "0", 10), auto_unload: $("#set-auto").checked,
      watermark: $("#set-wm").checked, engine_python } });
    toast("Réglages enregistrés (le Python par modèle s'applique au prochain chargement).", "ok");
  } catch (err) { toast(err.message, "error"); }
}));
$("#wm-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const out = $("#wm-result");
  out.innerHTML = '<span class="spinner"></span> Analyse…';
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await api("/api/watermark/detect", { method: "POST", body: fd });
    out.innerHTML = `<div class="notice ${r.detected ? "ok" : "info"}">${r.detected
      ? `✅ <b>Filigrane VoiceClone détecté</b> (score ${r.score}, seuil ${r.threshold}).`
      : `Aucun filigrane VoiceClone détecté (score ${r.score}, seuil ${r.threshold}).${r.duration < 5 ? " Extrait court : la détection est moins fiable sous ~5 s." : ""}`}
      ${r.perth != null ? `<br>Filigrane Resemble (Perth, Chatterbox) : ${r.perth > 0.5 ? "présent" : "absent"} (${r.perth.toFixed(2)}).` : ""}</div>`;
  } catch (err) { out.innerHTML = `<div class="notice error">${esc(err.message)}</div>`; }
  e.target.value = "";
});
