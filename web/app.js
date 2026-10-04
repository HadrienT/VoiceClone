import { Recorder, toWav, listMicrophones, fmtTime, SYSTEM_SOURCE, canCaptureSystemAudio } from "./audio.js";
import { BrowserLive, listBrowserDevices, canChooseOutput, micPermission, requestMic } from "./live-browser.js";

import { $, $$, esc, LANGS, langName, store, api, toast, fmtBytes, busy } from "./util.js";
import { watchJob, jobProgressHtml, refreshJobsPanel } from "./jobs.js";
import { renderSegments } from "./longtext.js";
import { showTools } from "./tools.js";

const READ_PROMPTS = [
  "Bonjour, je m'appelle comme vous voulez. Aujourd'hui, il fait beau et je vais enregistrer ma voix pour essayer ce logiciel. J'espère que le résultat sera bluffant !",
  "Le vif zéphyr jubile sur les kumquats du clown gracieux. Pourquoi ce chat noir boit-il du lait chaud dans un bol jaune ? Personne ne le sait vraiment.",
  "Hier soir, nous avons joué jusqu'à minuit. C'était intense : trois victoires, deux défaites, et un fou rire général quand Paul a raté son dernier saut.",
  "Attention, ceci est un message important. Merci de vérifier vos paramètres avant de continuer. Est-ce que tout le monde m'entend correctement ?",
  "J'adore les longues balades en forêt, surtout en automne, quand les feuilles craquent sous les pieds et que l'air sent la terre mouillée.",
  "Un, deux, trois, quatre, cinq. Les chaussettes de l'archiduchesse sont-elles sèches, archisèches ? Voilà une phrase bien difficile à prononcer !",
];

const state = { models: [], voices: [], tab: store.get("tab", "models"), filter: "all" };

// ---------------------------------------------------------------- navigation
function showTab(tab) {
  state.tab = tab;
  store.set("tab", tab);
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${tab}`));
  if (tab === "history") loadHistory();
  if (tab === "live") { loadDevices(); pollLive(); }
  if (tab === "tools") showTools().catch((err) => toast(err.message, "error"));
}
$$("#nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------------------------------------------------------- système
async function loadSystem() {
  try {
    const s = await api("/api/system");
    const g = (s.gpus || []).find((x) => x.selected);
    const dev = s.gpu ? `GPU <b>${esc(s.gpu)}</b>${g ? `<br>VRAM : <b>${g.used_gb}</b> / ${g.total_gb} Go` : ` (${s.vram_gb} Go)`}` : `<b>${esc(s.device.toUpperCase())}</b>`;
    $("#sys").innerHTML = `Calcul : ${dev}<br>PyTorch : <b>${s.torch ? esc(s.torch) : "non installé"}</b><br>v${esc(s.version)}`;
    $("#data-dir").textContent = s.data_dir + "/models";
  } catch {
    $("#sys").textContent = "Serveur injoignable";
  }
  clearTimeout(state.sysPoll);
  state.sysPoll = setTimeout(loadSystem, 10000); // mémoire GPU à jour
}

// ---------------------------------------------------------------- modèles
const isReady = (m) => m.downloaded && m.installed;

function modelLabel(m) {
  const flag = isReady(m) ? (m.loaded ? "● " : "✓ ") : !m.downloaded ? "⬇ " : "⚠ ";
  return flag + m.name;
}

function modelWarning(m) {
  if (!m) return "";
  if (!m.installed) return `Dépendances Python manquantes (${m.missing_packages.join(", ")}). Installez-les : <code>${esc(m.pip)}</code> puis redémarrez le serveur.`;
  if (!m.downloaded) return `Ce modèle n'est pas encore téléchargé (${esc(m.size_hint)}). <button class="link" data-goto-models>Aller aux modèles →</button>`;
  return "";
}

async function loadModels() {
  state.models = await api("/api/models");
  renderModels();
  refreshModelSelects();
  const readySig = state.models.filter(isReady).map((m) => m.id).join();
  if (readySig !== state.readySig) { state.readySig = readySig; if (state.voices.length) renderVoices(); }
  const active = state.models.some((m) => m.download?.status === "running" || m.download?.status === "pending" || m.loading);
  clearTimeout(state.modelPoll);
  if (active) state.modelPoll = setTimeout(loadModels, 1000);
}

function renderModels() {
  const list = state.models.filter((m) => state.filter === "all" || m.capabilities.includes(state.filter));
  const capName = { tts: "Texte → Voix", vc: "Voix → Voix", asr: "Transcription" };
  $("#model-list").innerHTML = list.map((m) => {
    const dl = m.download;
    const running = dl && (dl.status === "running" || dl.status === "pending");
    let status;
    if (running) {
      const pct = Math.round(dl.progress * 100);
      status = `<div class="progress"><div style="width:${pct}%"></div></div>
        <div class="progress-label"><span>${dl.total_bytes ? pct + " %" : "Préparation…"} · ${fmtBytes(dl.downloaded_bytes)}${dl.total_bytes ? " / " + fmtBytes(dl.total_bytes) : ""}</span>
        <span>${fmtBytes(dl.speed_bps)}/s</span></div>`;
    } else if (dl?.status === "error") {
      status = `<div class="notice error">Échec : ${esc(dl.error)}</div>`;
    } else status = "";
    const pip = !m.installed ? `<div class="notice warn">Paquets manquants : <b>${esc(m.missing_packages.join(", "))}</b>
        <div class="pip"><code>${esc(m.pip)}</code><button class="btn small" data-copy="${esc(m.pip)}">Copier</button></div></div>` : "";
    const loadErr = m.load_error ? `<div class="notice error">${esc(m.load_error)}</div>` : "";
    const buttons = [];
    if (!m.downloaded && !running) buttons.push(`<button class="btn primary small" data-dl="${m.id}">⬇ Télécharger</button>`);
    if (running) buttons.push(`<button class="btn small" data-cancel="${m.id}">Annuler</button>`);
    if (m.downloaded && m.installed && !m.loaded) buttons.push(`<button class="btn small" data-load="${m.id}" ${m.loading ? "disabled" : ""}>${m.loading ? '<span class="spinner"></span> Chargement…' : "Charger en mémoire"}</button>`);
    if (m.loaded) buttons.push(`<button class="btn small" data-unload="${m.id}">Décharger</button>`);
    if (m.downloaded) buttons.push(`<button class="btn small danger" data-del="${m.id}">Supprimer</button>`);
    return `<div class="card model">
      <div class="top"><h3>${esc(m.name)}</h3>
        <div class="badges">${m.loaded ? '<span class="badge ok">en mémoire</span>' : m.downloaded ? '<span class="badge ok">téléchargé</span>' : ""}</div></div>
      <div class="badges">${m.capabilities.map((c) => `<span class="badge ${c}">${capName[c]}</span>`).join("")}
        ${m.realtime ? '<span class="badge live">live</span>' : ""}</div>
      <p>${esc(m.description)}</p>
      ${m.notes ? `<p><i>${esc(m.notes)}</i></p>` : ""}
      <div class="meta">
        <span>Taille</span><b>${esc(m.size_hint)}</b>
        <span>Licence</span><b>${esc(m.license)}</b>
        <span>Langues</span><b>${m.languages.includes("*") ? "toutes" : m.languages.slice(0, 8).map(langName).join(", ") + (m.languages.length > 8 ? ` +${m.languages.length - 8}` : "")}</b>
        <span>Source</span><b>${m.repos.map((r) => `<a href="https://huggingface.co/${esc(r)}" target="_blank" rel="noopener">${esc(r)}</a>`).join("<br>")}</b>
      </div>
      ${pip}${status}${loadErr}
      <div class="actions">${buttons.join("")}</div>
    </div>`;
  }).join("") || `<div class="empty">Aucun modèle.</div>`;
}

$(".filters").addEventListener("click", (e) => {
  const b = e.target.closest("[data-filter]");
  if (!b) return;
  state.filter = b.dataset.filter;
  $$(".filters .chip").forEach((c) => c.classList.toggle("active", c === b));
  renderModels();
});

$("#model-list").addEventListener("click", async (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  try {
    if (t.dataset.copy) { await navigator.clipboard.writeText(t.dataset.copy); toast("Commande copiée", "ok"); return; }
    if (t.dataset.dl) { await api(`/api/models/${t.dataset.dl}/download`, { method: "POST" }); toast("Téléchargement lancé"); }
    if (t.dataset.cancel) { await api(`/api/models/${t.dataset.cancel}/cancel`, { method: "POST" }); toast("Annulation après le fichier en cours"); }
    if (t.dataset.load) await api(`/api/models/${t.dataset.load}/load`, { method: "POST" });
    if (t.dataset.unload) await api(`/api/models/${t.dataset.unload}/unload`, { method: "POST" });
    if (t.dataset.del) {
      if (!confirm("Supprimer les fichiers de ce modèle du disque ?")) return;
      await api(`/api/models/${t.dataset.del}`, { method: "DELETE" });
      toast("Modèle supprimé", "ok");
    }
  } catch (err) { toast(err.message, "error"); }
  loadModels();
});

document.addEventListener("click", (e) => {
  if (e.target.closest("[data-goto-models]")) showTab("models");
});

// --------------------------------------------------- sélecteurs partagés
function fillSelect(sel, items, { value, empty } = {}) {
  const prev = value ?? sel.value;
  sel.innerHTML = items.length
    ? items.map((i) => `<option value="${esc(i.value)}">${esc(i.label)}</option>`).join("")
    : `<option value="">${esc(empty || "—")}</option>`;
  if (items.some((i) => String(i.value) === String(prev))) sel.value = prev;
}

const modelsWith = (cap) => state.models.filter((m) => m.capabilities.includes(cap));
const getModel = (id) => state.models.find((m) => m.id === id);
const getVoice = (id) => state.voices.find((v) => v.id === id);

function refreshModelSelects() {
  const opts = (cap) => modelsWith(cap).map((m) => ({ value: m.id, label: modelLabel(m) }));
  fillSelect($("#tts-model"), opts("tts"), { value: $("#tts-model").value || store.get("tts.model") });
  const s2sCap = state.s2sMode === "asr_tts" ? "tts" : "vc";
  fillSelect($("#s2s-model"), opts(s2sCap), { value: $("#s2s-model").value || store.get(`s2s.model.${s2sCap}`) });
  fillSelect($("#s2s-asr"), opts("asr"), { value: store.get("asr.model") });
  const liveCap = state.liveMode === "asr_tts" ? "tts" : "vc";
  fillSelect($("#live-model"), opts(liveCap), { value: $("#live-model").value || store.get(`live.model.${liveCap}`) });
  fillSelect($("#live-asr"), opts("asr"), { value: store.get("asr.model") });
  fillSelect($("#live-say-model"), opts("tts"), { value: $("#live-say-model").value || store.get("live.say.model"), empty: "Aucun modèle de synthèse" });
  onTTSModelChange();
  onS2SModelChange();
  onLiveModelChange();
}

function refreshVoiceSelects() {
  const items = state.voices.map((v) => ({ value: v.id, label: `${v.name} (${v.duration}s)` }));
  for (const id of ["#tts-voice", "#s2s-voice", "#live-voice"]) {
    fillSelect($(id), items, { empty: "Aucune voix — créez-en une", value: $(id).value || store.get("voice") });
  }
  fillSelect($("#live-src"), [{ value: "", label: "Automatique" }, ...items], { value: $("#live-src").value });
}

function fillLangSelect(sel, model, preferred) {
  const langs = model && !model.languages.includes("*") ? model.languages : Object.keys(LANGS).filter((l) => l !== "auto" && l !== "zh-cn");
  fillSelect(sel, langs.map((l) => ({ value: l, label: langName(l) })), { value: preferred || sel.value || "fr" });
}

function renderParams(container, model, prefix) {
  const params = model?.params || [];
  const saved = store.get(`params.${model?.id}`, {});
  container.innerHTML = params.map((p) => {
    const v = saved[p.name] ?? p.default;
    return `<label class="param" title="${esc(p.help || "")}">${esc(p.label)} <span><b id="${prefix}-${p.name}-v">${v}</b></span>
      <input type="range" data-param="${p.name}" data-type="${p.type}" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}">
      ${p.help ? `<small class="muted">${esc(p.help)}</small>` : ""}</label>`;
  }).join("") || `<p class="muted">Aucun réglage pour ce modèle.</p>`;
  container.oninput = (e) => {
    const inp = e.target.closest("[data-param]");
    if (!inp) return;
    $(`#${prefix}-${inp.dataset.param}-v`).textContent = inp.value;
    store.set(`params.${model.id}`, readParams(container));
  };
}

function readParams(container) {
  const out = {};
  $$("[data-param]", container).forEach((i) => { out[i.dataset.param] = i.dataset.type === "int" ? parseInt(i.value, 10) : parseFloat(i.value); });
  return out;
}

function showWarning(el, model) {
  const w = modelWarning(model);
  el.innerHTML = w;
  el.classList.toggle("hidden", !w);
}

// ---------------------------------------------------------------- voix : création
const pending = []; // { blob, name, source, duration, url, transcript }
let micDevices = [];

function renderPending() {
  $("#pending-list").innerHTML = pending.map((p, i) => `<div class="pending-item">
      <span>${{ record: "🎤", system: "🖥️" }[p.source] || "📁"}</span>
      <span class="name" title="${esc(p.name)}">${esc(p.name)} · ${p.duration ? p.duration.toFixed(1) + " s" : ""}</span>
      <audio controls src="${p.url}"></audio>
      ${p.transcript ? `<label class="check" style="margin:0;font-size:12px" title="${esc(p.transcript)}"><input type="checkbox" data-read="${i}" ${p.useTranscript ? "checked" : ""}> j'ai lu le texte proposé</label>` : ""}
      <button class="btn small danger" data-rm="${i}">✕</button></div>`).join("");
  const total = pending.reduce((s, p) => s + (p.duration || 0), 0);
  if (pending.length) {
    $("#pending-list").insertAdjacentHTML("beforeend", `<div class="muted" style="font-size:13px">Total : ${total.toFixed(1)} s ${total < 10 ? "— ajoutez un peu plus d'audio pour un meilleur clonage (10-30 s)." : "✓"}</div>`);
  }
  updateCreateBtn();
}

function updateCreateBtn() {
  $("#create-voice").disabled = !(pending.length && $("#nv-name").value.trim() && $("#nv-consent").checked);
}

$("#pending-list").addEventListener("change", (e) => {
  const c = e.target.closest("[data-read]");
  if (c) pending[+c.dataset.read].useTranscript = c.checked;
});
$("#pending-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-rm]");
  if (!b) return;
  const [p] = pending.splice(+b.dataset.rm, 1);
  URL.revokeObjectURL(p.url);
  renderPending();
});
["#nv-name", "#nv-consent"].forEach((s) => $(s).addEventListener("input", updateCreateBtn));

async function addFiles(files) {
  for (const f of files) {
    try {
      const { blob, duration } = await toWav(f);
      pending.push({ blob, name: f.name, source: "upload", duration, url: URL.createObjectURL(blob) });
    } catch {
      // format non décodable par le navigateur : le serveur s'en chargera (ffmpeg)
      pending.push({ blob: f, name: f.name, source: "upload", duration: 0, url: URL.createObjectURL(f) });
    }
    if (!$("#nv-name").value) $("#nv-name").value = f.name.replace(/\.[^.]+$/, "").slice(0, 60);
  }
  renderPending();
}

function setupDropzone(zone, input, onFiles) {
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { onFiles([...input.files]); input.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("audio") || f.type.startsWith("video") || /\.(wav|mp3|m4a|ogg|flac|webm|opus|aac|mp4)$/i.test(f.name));
    if (!files.length) return toast("Déposez un fichier audio.", "error");
    onFiles(files);
  });
}
setupDropzone($("#dropzone"), $("#file-input"), addFiles);

function nextPrompt() {
  state.promptIdx = ((state.promptIdx ?? Math.floor(Math.random() * READ_PROMPTS.length)) + 1) % READ_PROMPTS.length;
  $("#read-prompt").textContent = READ_PROMPTS[state.promptIdx];
}
$("#next-prompt").addEventListener("click", nextPrompt);
nextPrompt();

const SYSTEM_HINT = `<b>🖥️ Son du PC</b> : enregistre ce que vous entendez dans votre casque (vidéo, Discord, jeu…).
  À l'étape suivante, le navigateur demande quoi partager : choisissez <b>« Écran entier »</b> puis cochez
  <b>« Partager aussi l'audio du système »</b> (Windows, Chrome ou Edge). Pour un seul onglet, choisissez l'onglet et
  cochez « Partager aussi l'audio de l'onglet ». L'image n'est pas enregistrée. Coupez la musique pour un clone propre.`;

async function loadMics() {
  try {
    micDevices = await listMicrophones();
  } catch { micDevices = []; /* permissions non accordées */ }
  const items = micDevices.map((d, i) => ({ value: d.deviceId, label: `🎤 ${d.label || `Micro ${i + 1}`}` }));
  if (!items.length) items.push({ value: "", label: "🎤 Micro par défaut" });
  if (canCaptureSystemAudio()) items.push({ value: SYSTEM_SOURCE, label: "🖥️ Son du PC" });
  $$(".source-select").forEach((sel) => {
    fillSelect(sel, items, { value: sel.value || store.get(`source.${sel.id}`) });
    updateSourceHint(sel);
  });
}

function updateSourceHint(sel) {
  const hint = $(`.system-hint[data-for="${sel.id}"]`);
  if (!hint) return;
  hint.innerHTML = SYSTEM_HINT;
  hint.classList.toggle("hidden", sel.value !== SYSTEM_SOURCE);
}

$$(".source-select").forEach((sel) => sel.addEventListener("change", () => {
  store.set(`source.${sel.id}`, sel.value);
  updateSourceHint(sel);
}));

/** Relie un bouton ● Enregistrer à un sélecteur de source, un vumètre et un chrono. */
function bindRecorder({ btn, select, meter, time, onResult }) {
  const finish = async () => {
    btn.classList.remove("recording");
    btn.textContent = "● Enregistrer";
    select.disabled = false;
    const res = await rec.stop();
    if (!res) return;
    if (res.duration < 1) return toast("Enregistrement trop court.", "error");
    onResult(res, select.value === SYSTEM_SOURCE ? "system" : "record");
  };
  const rec = new Recorder({
    onLevel: (p) => { meter.style.width = `${Math.min(100, p * 140)}%`; },
    onTick: (t) => { time.textContent = fmtTime(t); },
    onEnded: finish, // partage arrêté depuis la barre du navigateur
  });
  btn.addEventListener("click", async () => {
    if (rec.recording) return finish();
    try {
      await rec.start(select.value);
      btn.classList.add("recording");
      btn.textContent = "■ Arrêter";
      select.disabled = true;
      if (select.value !== SYSTEM_SOURCE && !micDevices.some((d) => d.label)) loadMics(); // libellés après autorisation
    } catch (err) {
      toast(`${select.value === SYSTEM_SOURCE ? "Capture du son du PC" : "Micro"} impossible : ${err.message}`, "error", 8000);
    }
  });
}

bindRecorder({
  btn: $("#rec-btn"), select: $("#rec-device"), meter: $("#rec-meter"), time: $("#rec-time"),
  onResult: (res, source) => {
    const n = pending.filter((p) => p.source === source).length + 1;
    const fromMic = source === "record";
    pending.push({
      ...res, source, url: URL.createObjectURL(res.blob),
      name: fromMic ? `Enregistrement ${n}` : `Son du PC ${n}`,
      // le texte proposé n'a de sens que si c'est vous qui l'avez lu au micro
      transcript: fromMic ? $("#read-prompt").textContent : "", useTranscript: fromMic,
    });
    renderPending();
    if (fromMic) nextPrompt();
  },
});

// ------------------------------------------------------- import intelligent
const smartPrefs = store.get("smart", {});
for (const [id, key] of [["#smart-on", "on"], ["#smart-denoise", "denoise"], ["#smart-transcribe", "transcribe"]]) {
  if (smartPrefs[key] !== undefined) $(id).checked = smartPrefs[key];
}
if (smartPrefs.target) $("#smart-target").value = smartPrefs.target;
function saveSmartPrefs() {
  store.set("smart", { on: $("#smart-on").checked, denoise: $("#smart-denoise").checked,
    transcribe: $("#smart-transcribe").checked, target: $("#smart-target").value });
  $(".smart-opts").classList.toggle("hidden", !$("#smart-on").checked);
}
["#smart-on", "#smart-denoise", "#smart-transcribe", "#smart-target"].forEach((id) => $(id).addEventListener("change", saveSmartPrefs));
saveSmartPrefs();

/** Envoie un enregistrement à l'import intelligent ; renvoie { voice, report }. */
async function smartImport(voiceId, blob, name, source, targetSeconds) {
  const fd = new FormData();
  fd.append("file", blob, name);
  fd.append("source", source);
  fd.append("enhance", $("#smart-denoise").checked);
  fd.append("target_seconds", targetSeconds ?? $("#smart-target").value);
  const asr = modelsWith("asr").find(isReady);
  if ($("#smart-transcribe").checked && asr) fd.append("transcribe_model_id", asr.id);
  return api(`/api/voices/${voiceId}/auto-import`, { method: "POST", body: fd });
}

function renderReport(voiceName, items) {
  const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  $("#prep-report").innerHTML = `<div class="top" style="display:flex;justify-content:space-between;align-items:center">
      <h2 style="margin:0">✨ Import intelligent — ${esc(voiceName)}</h2>
      <button class="btn small" data-close-report>Fermer</button></div>
    ${items.map(({ name, report: r }) => `
      <h3 style="margin-top:14px">${esc(name)} : ${r.duration} s analysées → <span style="color:var(--accent-2)">${r.kept_duration} s gardées</span></h3>
      <p class="muted" style="margin:0;font-size:13px">Bruit de fond ${r.noise_floor_db_before} dB${r.denoised ? ` → ${r.noise_floor_db_after} dB après débruitage` : " (propre, pas de débruitage)"}.
        Les passages gardés sont ajoutés à la voix, le meilleur en premier.</p>
      <table class="report-table"><tr><th></th><th>Passage</th><th>Score</th><th>Voix / bruit</th><th>Parole</th><th>Verdict</th></tr>
      ${r.segments.map((g) => `<tr class="${g.kept ? "kept" : "rejected"}">
        <td>${g.kept ? `✓ n°${r.kept_order.indexOf(g.index) + 1}` : "✗"}</td>
        <td>${fmt(g.start)} → ${fmt(g.end)} <span class="muted">(${g.duration} s)</span></td>
        <td><span class="scorebar"><i style="width:${g.score}%"></i></span>${Math.round(g.score)}</td>
        <td>${g.snr_db} dB</td><td>${Math.round(g.speech_ratio * 100)} %</td>
        <td>${g.kept ? "gardé" : esc(g.reason)}</td></tr>`).join("")}
      </table>`).join("")}`;
  $("#prep-report").classList.remove("hidden");
  $("#prep-report").scrollIntoView({ behavior: "smooth", block: "start" });
}
$("#prep-report").addEventListener("click", (e) => {
  if (e.target.closest("[data-close-report]")) $("#prep-report").classList.add("hidden");
});

$("#create-voice").addEventListener("click", (e) => busy(e.currentTarget, $("#smart-on").checked ? "Analyse et nettoyage…" : "Analyse…", async () => {
  try {
    const fd = new FormData();
    fd.append("name", $("#nv-name").value.trim());
    fd.append("language", $("#nv-lang").value);
    fd.append("consent", $("#nv-consent").checked);
    fd.append("consent_owner", $("#nv-consent-owner").value);
    const fileName = (p) => (p.source === "upload" ? p.name : `${p.name}.wav`);
    let voice;
    if ($("#smart-on").checked) {
      // voix vide, puis chaque enregistrement passe par l'import intelligent (durée cible partagée)
      voice = await api("/api/voices", { method: "POST", body: fd });
      const reports = [];
      let remaining = parseFloat($("#smart-target").value);
      for (const p of pending) {
        if (remaining < 3) break;
        const r = await smartImport(voice.id, p.blob, fileName(p), p.source, remaining);
        voice = r.voice;
        remaining -= r.report.kept_duration;
        reports.push({ name: p.name, report: r.report });
      }
      renderReport(voice.name, reports);
    } else {
      const [first, ...rest] = pending;
      fd.append("files", first.blob, fileName(first));
      fd.append("source", first.source);
      if (first.useTranscript) fd.append("transcript", first.transcript);
      voice = await api("/api/voices", { method: "POST", body: fd });
      for (const p of rest) {
        const f = new FormData();
        f.append("file", p.blob, fileName(p));
        f.append("source", p.source);
        if (p.useTranscript) f.append("transcript", p.transcript);
        voice = await api(`/api/voices/${voice.id}/samples`, { method: "POST", body: f });
      }
    }
    pending.splice(0).forEach((p) => URL.revokeObjectURL(p.url));
    renderPending();
    $("#nv-name").value = "";
    toast(`Voix « ${voice.name} » créée !`, "ok");
    store.set("voice", voice.id);
    await loadVoices();
  } catch (err) { toast(err.message, "error"); }
}));

// ---------------------------------------------------------------- voix : liste
async function loadVoices() {
  state.voices = await api("/api/voices");
  renderVoices();
  refreshVoiceSelects();
}

function renderVoices() {
  const asr = modelsWith("asr").filter(isReady);
  const prepModels = state.models.filter((m) => (m.capabilities.includes("tts") || m.capabilities.includes("vc")) && isReady(m));
  $("#voice-list").innerHTML = state.voices.map((v) => {
    const a = v.analysis || {};
    const t = Date.now();
    const prepared = Object.keys(v.prepared || {}).map((id) => `<span class="badge ok">✓ ${esc(getModel(id)?.name || id)}</span>`).join("");
    return `<div class="card voice" data-voice="${v.id}">
      <div class="top"><h3>${esc(v.name)}</h3>
        <div class="badges"><span class="badge">${esc(langName(v.language))}</span><span class="badge">${v.duration} s</span></div></div>
      ${v.samples.length ? `<audio controls preload="none" src="/api/voices/${v.id}/audio?t=${t}"></audio>` : '<p class="muted">Aucun échantillon.</p>'}
      ${a.warnings?.length ? `<ul class="warnings">${a.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : v.samples.length ? '<p class="muted" style="margin:0;font-size:13px">✓ Qualité audio correcte</p>' : ""}
      <div class="badges">${prepared}</div>
      <details data-details ${openDetails.has(v.id) ? "open" : ""}>
        <summary>Échantillons (${v.samples.length}) & transcription ${v.samples.length && v.samples.every((s) => s.transcript) ? "📝" : ""}</summary>
        <div class="samples">
          ${v.samples.map((s, i) => {
            const name = s.file.split("/").pop();
            return `<div class="sample-block">
              <div class="sample"><span class="name">${i === 0 && v.samples.length > 1 ? '<span class="badge ok" title="Référence principale : Chatterbox n\'écoute que ses 10 premières secondes">★ principal</span> ' : ""}${{ record: "🎤", system: "🖥️" }[s.source] || "📁"} ${esc(s.original_name || s.file)} · ${s.duration}s</span>
                <audio controls preload="none" src="/api/voices/${v.id}/audio?sample=${encodeURIComponent(name)}&t=${t}"></audio>
                ${v.samples.length > 1 ? `<span class="order-btns">
                  <button class="btn small" data-move="${esc(name)}" data-to="first" title="Mettre en premier (référence principale)" ${i === 0 ? "disabled" : ""}>★</button>
                  <button class="btn small" data-move="${esc(name)}" data-to="up" title="Monter" ${i === 0 ? "disabled" : ""}>↑</button>
                  <button class="btn small" data-move="${esc(name)}" data-to="down" title="Descendre" ${i === v.samples.length - 1 ? "disabled" : ""}>↓</button>
                </span>` : ""}
                <button class="btn small danger" data-rm-sample="${esc(name)}">✕</button></div>
              <textarea rows="2" data-sample-text="${esc(name)}" placeholder="Texte exact prononcé dans cet échantillon (vide = inconnu)">${esc(s.transcript)}</textarea>
            </div>`;
          }).join("")}
          <div class="row" style="margin:6px 0 0">
            <button class="btn small" data-smart-import title="Nettoie, découpe et ajoute seulement les meilleurs passages">✨ Import intelligent</button>
            <input type="file" accept="audio/*,video/*" hidden data-smart-input>
            <button class="btn small" data-add-file>＋ Fichier</button>
            <button class="btn small rec" data-add-rec>● Enregistrer</button>
            <input type="file" accept="audio/*,video/*" hidden data-file-input>
          </div>
          <p class="muted" style="margin:4px 0;font-size:12px">Le texte n'est utile qu'à F5-TTS : il utilise un échantillon de 12 s maximum
            (de préférence un échantillon dont le texte est rempli). Les autres modèles n'en ont pas besoin.</p>
          <div class="row" style="margin:0">
            ${v.samples.length ? '<button class="btn small" data-save-transcript>Enregistrer les textes</button>' : ""}
            ${asr.length && v.samples.length ? `<button class="btn small" data-auto-transcribe="${asr[0].id}">Transcrire avec ${esc(asr[0].name.split(" (")[0])}</button>` : ""}
          </div>
        </div>
      </details>
      <div class="prep">
        <select data-prep-model>${prepModels.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("") || '<option value="">Aucun modèle prêt</option>'}</select>
        <button class="btn small" data-prepare ${prepModels.length ? "" : "disabled"} title="Pré-calcule l'empreinte vocale pour ce modèle (sinon faite à la 1re utilisation)">🧬 Préparer</button>
        <button class="btn small danger" data-del-voice>Supprimer</button>
      </div>
    </div>`;
  }).join("") || `<div class="empty">Aucune voix pour l'instant. Importez ou enregistrez un échantillon ci-dessus.</div>`;
}

const cardRecorders = new Map();
const openDetails = new Set(); // voix dont le panneau « Échantillons » est ouvert

$("#voice-list").addEventListener("toggle", (e) => {
  const d = e.target.closest?.("[data-details]");
  if (!d) return;
  const id = d.closest("[data-voice]").dataset.voice;
  d.open ? openDetails.add(id) : openDetails.delete(id);
}, true);

$("#voice-list").addEventListener("click", async (e) => {
  const card = e.target.closest("[data-voice]");
  const btn = e.target.closest("button");
  if (!card || !btn) return;
  const id = card.dataset.voice;
  try {
    if (btn.hasAttribute("data-del-voice")) {
      if (!confirm("Supprimer définitivement cette voix ?")) return;
      await api(`/api/voices/${id}`, { method: "DELETE" });
      toast("Voix supprimée", "ok");
    } else if (btn.dataset.move) {
      const v = getVoice(id);
      const files = v.samples.map((x) => x.file.split("/").pop());
      const i = files.indexOf(btn.dataset.move);
      files.splice(i, 1);
      const j = { first: 0, up: Math.max(0, i - 1), down: Math.min(files.length, i + 1) }[btn.dataset.to];
      files.splice(j, 0, btn.dataset.move);
      await api(`/api/voices/${id}/order`, { method: "PUT", json: { files } });
    } else if (btn.dataset.rmSample) {
      await api(`/api/voices/${id}/samples/${encodeURIComponent(btn.dataset.rmSample)}`, { method: "DELETE" });
    } else if (btn.hasAttribute("data-smart-import")) {
      const input = $("[data-smart-input]", card);
      input.onchange = () => busy(btn, "Analyse…", async () => {
        try {
          const reports = [];
          let voice;
          for (const f of input.files) {
            let blob = f;
            try { blob = (await toWav(f)).blob; } catch { /* le serveur décodera */ }
            const r = await smartImport(id, blob, f.name.replace(/\.[^.]+$/, "") + ".wav", "upload");
            voice = r.voice;
            reports.push({ name: f.name, report: r.report });
          }
          input.value = "";
          if (voice) renderReport(voice.name, reports);
        } catch (err) { toast(err.message, "error"); }
        loadVoices();
      });
      input.click();
      return;
    } else if (btn.hasAttribute("data-add-file")) {
      const input = $("[data-file-input]", card);
      input.onchange = async () => {
        for (const f of input.files) await uploadSample(id, f, f.name, "upload");
        loadVoices();
      };
      input.click();
      return;
    } else if (btn.hasAttribute("data-add-rec")) {
      let rec = cardRecorders.get(id);
      if (!rec) {
        rec = new Recorder();
        cardRecorders.set(id, rec);
        await rec.start($("#rec-device").value);
        btn.classList.add("recording");
        btn.textContent = "■ Arrêter";
        return;
      }
      cardRecorders.delete(id);
      const res = await rec.stop();
      const system = $("#rec-device").value === SYSTEM_SOURCE;
      if (res && res.duration >= 1) await uploadSample(id, res.blob, system ? "son-du-pc.wav" : "record.wav", system ? "system" : "record");
    } else if (btn.hasAttribute("data-save-transcript")) {
      const texts = Object.fromEntries($$("[data-sample-text]", card).map((t) => [t.dataset.sampleText, t.value]));
      await api(`/api/voices/${id}/transcripts`, { method: "PUT", json: { texts } });
      toast("Textes enregistrés", "ok");
    } else if (btn.dataset.autoTranscribe) {
      const v = await busy(btn, "Transcription…", () => api(`/api/voices/${id}/transcribe`, { json: { model_id: btn.dataset.autoTranscribe } }));
      const done = v.samples.filter((s) => s.transcript).length;
      if (done) toast(`Transcription terminée : ${done}/${v.samples.length} échantillon(s) — textes visibles sous chaque échantillon, corrigez-les si besoin.`, "ok", 7000);
      else toast("Whisper n'a reconnu aucune parole dans ces échantillons. Vérifiez la langue de la voix ou tapez le texte à la main.", "error", 9000);
    } else if (btn.hasAttribute("data-prepare")) {
      const model = $("[data-prep-model]", card).value;
      const r = await busy(btn, "Préparation…", () => api(`/api/voices/${id}/prepare`, { json: { model_id: model } }));
      toast(`Empreinte vocale prête en ${r.seconds} s`, "ok");
      loadModels();
    } else return;
  } catch (err) { toast(err.message, "error"); }
  loadVoices();
});

async function uploadSample(id, blob, name, source) {
  let file = blob;
  try { file = (await toWav(blob)).blob; name = name.replace(/\.[^.]+$/, "") + ".wav"; } catch { /* le serveur décodera */ }
  const fd = new FormData();
  fd.append("file", file, name);
  fd.append("source", source);
  await api(`/api/voices/${id}/samples`, { method: "POST", body: fd });
  toast("Échantillon ajouté", "ok");
}

// ---------------------------------------------------------------- TTS
function onTTSModelChange() {
  const m = getModel($("#tts-model").value);
  if (!m) return;
  store.set("tts.model", m.id);
  const v = getVoice($("#tts-voice").value);
  fillLangSelect($("#tts-lang"), m, store.get("tts.lang") || v?.language);
  showWarning($("#tts-model-warning"), m);
  if (state.ttsParamsFor !== m.id) { renderParams($("#tts-params"), m, "tts"); state.ttsParamsFor = m.id; }
}
$("#tts-model").addEventListener("change", onTTSModelChange);
$("#tts-lang").addEventListener("change", () => store.set("tts.lang", $("#tts-lang").value));
/** Applique les réglages mémorisés pour cette voix (modèle, langue, paramètres). */
function applyVoiceSettings() {
  const v = getVoice($("#tts-voice").value);
  const st = v?.settings?.tts;
  if (!st) return;
  if (st.params && st.model_id) store.set(`params.${st.model_id}`, st.params);
  if (st.model_id && getModel(st.model_id)) {
    $("#tts-model").value = st.model_id;
    state.ttsParamsFor = null; // force le rendu des paramètres mémorisés
  }
  if (st.language) store.set("tts.lang", st.language);
  onTTSModelChange();
  if (st.language) $("#tts-lang").value = st.language;
}
$("#tts-voice").addEventListener("change", () => { store.set("voice", $("#tts-voice").value); applyVoiceSettings(); });
const LONG_TEXT = 600;
const isLongText = (t) => t.length > LONG_TEXT || /\[\s*pause/i.test(t) || /\n\s*\n/.test(t.trim());
function updateTTSCount() {
  const t = $("#tts-text").value;
  $("#tts-count").textContent = t.length;
  $("#tts-hint").textContent = isLongText(t.trim()) ? "mode texte long (phrase par phrase) ·" : "";
}
$("#tts-text").addEventListener("input", updateTTSCount);
$("#tts-text").value = store.get("tts.text", "");
updateTTSCount();

function showResult(el, blob, info) {
  if (el._url) URL.revokeObjectURL(el._url);
  el._url = URL.createObjectURL(blob);
  el.innerHTML = `<audio controls autoplay src="${el._url}"></audio>
    <div class="info"><span>${info}</span><a href="${el._url}" download="voiceclone-${Date.now()}.wav">⬇ Télécharger le WAV</a></div>`;
  el.classList.remove("hidden");
}

$("#tts-go").addEventListener("click", (e) => {
  const text = $("#tts-text").value.trim();
  if (!text) return toast("Écrivez un texte.", "error");
  if (!$("#tts-voice").value) return toast("Créez d'abord une voix.", "error");
  store.set("tts.text", $("#tts-text").value);
  const model = getModel($("#tts-model").value);
  if (isLongText(text)) return generateLong(e.currentTarget, text);
  return busy(e.currentTarget, model?.loaded ? "Génération…" : "Chargement du modèle…", async () => {
    try {
      const res = await api("/api/tts", {
        raw: true,
        json: { model_id: $("#tts-model").value, voice_id: $("#tts-voice").value, text, language: $("#tts-lang").value, params: readParams($("#tts-params")) },
      });
      const gen = res.headers.get("X-Generation-Seconds");
      const dur = res.headers.get("X-Audio-Seconds");
      showResult($("#tts-result"), await res.blob(), `${dur} s d'audio générées en ${gen} s`);
      // mémorise ces réglages comme préférés pour cette voix
      const settings = { tts: { model_id: $("#tts-model").value, language: $("#tts-lang").value, params: readParams($("#tts-params")) } };
      api(`/api/voices/${$("#tts-voice").value}`, { method: "PATCH", json: { settings } })
        .then((v) => { const i = state.voices.findIndex((x) => x.id === v.id); if (i >= 0) state.voices[i] = v; })
        .catch(() => {});
      loadModels();
    } catch (err) { toast(err.message, "error"); }
  });
});

function rememberTTSSettings() {
  const settings = { tts: { model_id: $("#tts-model").value, language: $("#tts-lang").value, params: readParams($("#tts-params")) } };
  api(`/api/voices/${$("#tts-voice").value}`, { method: "PATCH", json: { settings } })
    .then((v) => { const i = state.voices.findIndex((x) => x.id === v.id); if (i >= 0) state.voices[i] = v; })
    .catch(() => {});
}

/** Texte long : tâche de fond phrase par phrase, puis éditeur de phrases. */
function generateLong(btn, text) {
  const out = $("#tts-result");
  return busy(btn, "Génération longue…", async () => {
    try {
      const job = await api("/api/tts/long", {
        json: { model_id: $("#tts-model").value, voice_id: $("#tts-voice").value, text, language: $("#tts-lang").value, params: readParams($("#tts-params")) },
      });
      out.classList.remove("hidden");
      delete out.dataset.historyId;
      const done = await watchJob(job.id, (j) => { out.innerHTML = jobProgressHtml(j); });
      await renderSegments(out, done.result.history_id);
      toast(`Terminé : ${done.result.duration} s d'audio.`, "ok");
      rememberTTSSettings();
      loadModels();
    } catch (err) {
      out.innerHTML = `<div class="notice error">${esc(err.message)}</div>`;
      toast(err.message, "error");
    }
  });
}

$("#tts-compare").addEventListener("click", (e) => {
  const text = $("#tts-text").value.trim();
  if (!text) return toast("Écrivez un texte.", "error");
  if (!$("#tts-voice").value) return toast("Créez d'abord une voix.", "error");
  const ready = modelsWith("tts").filter(isReady);
  if (ready.length < 2) return toast("Il faut au moins deux modèles TTS installés et téléchargés pour comparer.", "error", 7000);
  const box = $("#tts-compare-box");
  box.classList.remove("hidden");
  box.innerHTML = `<h3 style="margin:0">⚖ Comparer — même texte, même voix</h3>
    <div class="badges">${ready.map((m) => `<label class="check" style="margin:0"><input type="checkbox" data-cmp="${m.id}" checked> ${esc(m.name)}</label>`).join("")}</div>
    <div class="actions" style="justify-content:flex-start"><button class="btn primary small" data-cmp-go>Lancer la comparaison</button></div>
    <div class="compare-grid"></div>`;
});
$("#tts-compare-box").addEventListener("click", (e) => {
  const go = e.target.closest("[data-cmp-go]");
  if (!go) return;
  const ids = $$("[data-cmp]:checked").map((c) => c.dataset.cmp);
  const grid = $(".compare-grid", e.currentTarget);
  grid.innerHTML = ids.map((id) => `<div class="card cmp" data-cmp-cell="${id}"><b>${esc(getModel(id).name)}</b><p class="muted"><span class="spinner"></span> en attente…</p></div>`).join("");
  return busy(go, "Comparaison…", async () => {
    for (const id of ids) {
      const cell = $(`[data-cmp-cell="${id}"]`, grid);
      const lang = getModel(id).languages.includes($("#tts-lang").value) ? $("#tts-lang").value : getModel(id).languages[0];
      try {
        const res = await api("/api/tts", { raw: true, json: { model_id: id, voice_id: $("#tts-voice").value, text: $("#tts-text").value.trim(), language: lang, params: store.get(`params.${id}`, {}) } });
        const url = URL.createObjectURL(await res.blob());
        cell.innerHTML = `<b>${esc(getModel(id).name)}</b><audio controls src="${url}"></audio>
          <p class="muted" style="margin:0;font-size:12px">${res.headers.get("X-Audio-Seconds")} s générées en ${res.headers.get("X-Generation-Seconds")} s · langue ${esc(langName(lang))}</p>`;
      } catch (err) {
        cell.innerHTML = `<b>${esc(getModel(id).name)}</b><p class="notice error" style="margin:4px 0 0">${esc(err.message)}</p>`;
      }
    }
    loadModels();
  });
});

// ---------------------------------------------------------------- S2S
state.s2sMode = store.get("s2s.mode", "vc");
let s2sSource = null;

function setSegMode(segEl, mode) {
  $$("button", segEl).forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
}

function onS2SModelChange() {
  const m = getModel($("#s2s-model").value);
  const cap = state.s2sMode === "asr_tts" ? "tts" : "vc";
  if (m) store.set(`s2s.model.${cap}`, m.id);
  fillLangSelect($("#s2s-lang"), m, getVoice($("#s2s-voice").value)?.language);
  const asr = getModel($("#s2s-asr").value);
  const warnModel = m && !isReady(m) ? m : state.s2sMode === "asr_tts" && asr && !isReady(asr) ? asr : m;
  showWarning($("#s2s-model-warning"), warnModel);
  if (!modelsWith(cap).length) $("#s2s-model-warning").classList.add("hidden");
  if (state.s2sParamsFor !== m?.id) { renderParams($("#s2s-params"), m, "s2s"); state.s2sParamsFor = m?.id; }
}

function applyS2SMode() {
  setSegMode($("#s2s-mode"), state.s2sMode);
  $$("#tab-s2s .asr-only").forEach((el) => el.classList.toggle("hidden", state.s2sMode !== "asr_tts"));
  refreshModelSelects();
}

$("#s2s-mode").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (!b) return;
  state.s2sMode = b.dataset.mode;
  store.set("s2s.mode", state.s2sMode);
  applyS2SMode();
});
$("#s2s-model").addEventListener("change", onS2SModelChange);
$("#s2s-asr").addEventListener("change", () => { store.set("asr.model", $("#s2s-asr").value); onS2SModelChange(); });

function setS2SSource(blob, name) {
  s2sSource = { blob, name };
  if (state.s2sUrl) URL.revokeObjectURL(state.s2sUrl);
  state.s2sUrl = URL.createObjectURL(blob);
  $("#s2s-source").innerHTML = `<span>Source : <b>${esc(name)}</b></span><audio controls src="${state.s2sUrl}"></audio>`;
  $("#s2s-source").classList.remove("hidden");
  $("#s2s-go").disabled = false;
}

setupDropzone($("#s2s-drop"), $("#s2s-file"), async ([f]) => {
  try { setS2SSource((await toWav(f)).blob, f.name); } catch { setS2SSource(f, f.name); }
});

bindRecorder({
  btn: $("#s2s-rec"), select: $("#s2s-device"), meter: $("#s2s-meter"), time: $("#s2s-time"),
  onResult: (res, source) => setS2SSource(res.blob, source === "system" ? "son du PC" : "enregistrement"),
});

$("#s2s-go").addEventListener("click", (e) => {
  if (!s2sSource) return;
  if (!$("#s2s-voice").value) return toast("Créez d'abord une voix.", "error");
  return busy(e.currentTarget, "Conversion…", async () => {
    try {
      const fd = new FormData();
      fd.append("file", s2sSource.blob, s2sSource.name.endsWith(".wav") ? s2sSource.name : "source.wav");
      fd.append("model_id", $("#s2s-model").value);
      fd.append("voice_id", $("#s2s-voice").value);
      fd.append("mode", state.s2sMode);
      fd.append("asr_model_id", $("#s2s-asr").value || "");
      fd.append("language", $("#s2s-lang").value || "fr");
      fd.append("params", JSON.stringify(readParams($("#s2s-params"))));
      const res = await api("/api/vc", { method: "POST", body: fd, raw: true });
      const transcript = res.headers.get("X-Transcript");
      const gen = res.headers.get("X-Generation-Seconds");
      showResult($("#s2s-result"), await res.blob(), `Converti en ${gen} s${transcript ? ` · « ${esc(decodeURIComponent(transcript))} »` : ""}`);
      loadModels();
    } catch (err) { toast(err.message, "error"); }
  });
});

// ---------------------------------------------------------------- LIVE
state.liveMode = store.get("live.mode", "vc");
const liveSliders = [
  ["#live-chunk", "#v-chunk", "chunk_ms"], ["#live-ctx", "#v-ctx", "context_ms"], ["#live-th", "#v-th", "silence_db"],
  ["#live-eos", "#v-eos", "end_silence_ms"], ["#live-ig", "#v-ig", "input_gain"], ["#live-og", "#v-og", "output_gain"],
];
const liveSaved = store.get("live.settings", {});
for (const [inp, out, key] of liveSliders) {
  if (liveSaved[key] !== undefined) $(inp).value = liveSaved[key];
  $(out).textContent = $(inp).value;
  $(inp).addEventListener("input", () => {
    $(out).textContent = $(inp).value;
    const s = store.get("live.settings", {});
    s[key] = parseFloat($(inp).value);
    store.set("live.settings", s);
  });
}

function onLiveModelChange() {
  const m = getModel($("#live-model").value);
  const cap = state.liveMode === "asr_tts" ? "tts" : "vc";
  if (m) store.set(`live.model.${cap}`, m.id);
  fillLangSelect($("#live-lang"), m, getVoice($("#live-voice").value)?.language);
  const asr = getModel($("#live-asr").value);
  const warnModel = state.liveMode === "passthrough" ? null : m && !isReady(m) ? m : state.liveMode === "asr_tts" && asr && !isReady(asr) ? asr : m;
  showWarning($("#live-model-warning") || $("#live-dev-error"), warnModel);
  if (state.liveParamsFor !== m?.id) { renderParams($("#live-params"), m, "live"); state.liveParamsFor = m?.id; }
}

function applyLiveMode() {
  setSegMode($("#live-mode"), state.liveMode);
  const asrMode = state.liveMode === "asr_tts";
  $$("#tab-live .asr-only").forEach((el) => el.classList.toggle("hidden", !asrMode));
  $$("#tab-live .vc-only").forEach((el) => el.classList.toggle("hidden", state.liveMode !== "vc"));
  $$("#tab-live .model-field").forEach((el) => el.classList.toggle("hidden", state.liveMode === "passthrough"));
  $("#live-params").classList.toggle("hidden", state.liveMode === "passthrough");
  refreshModelSelects();
}
// zone d'avertissement dédiée au modèle
$("#live-dev-error").insertAdjacentHTML("beforebegin", '<div id="live-model-warning" class="notice warn hidden"></div>');

$("#live-mode").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (!b || state.liveRunning) return;
  state.liveMode = b.dataset.mode;
  store.set("live.mode", state.liveMode);
  applyLiveMode();
});
$("#live-model").addEventListener("change", onLiveModelChange);
$("#live-asr").addEventListener("change", () => { store.set("asr.model", $("#live-asr").value); onLiveModelChange(); });

state.liveWhere = store.get("live.where", "browser");
const browserLive = new BrowserLive({
  onStatus: (st) => renderLiveStatus(st),
  onStopped: (message) => {
    if (message) toast(message, "error", 8000);
    renderLiveStatus({ state: message ? "error" : "idle", error: message });
  },
});
const deviceKey = () => `live.devices.${state.liveWhere}`;
const devLabel = (x) => `${x.virtual ? "★ " : ""}${x.name}${x.default ? " (défaut)" : ""}${x.hostapi ? ` — ${x.hostapi}` : ""}`;

function showDeviceNotice(html) {
  $("#live-dev-error").innerHTML = html || "";
  $("#live-dev-error").classList.toggle("hidden", !html);
}

function fillDeviceSelects(inputs, outputs) {
  const saved = store.get(deviceKey(), {});
  const def = (list) => list.find((x) => x.default)?.id ?? list[0]?.id ?? "";
  fillSelect($("#live-in"), inputs.map((x) => ({ value: x.id, label: devLabel(x) })), { value: saved.input ?? def(inputs) });
  const virt = outputs.find((x) => x.virtual)?.id;
  fillSelect($("#live-out"), outputs.map((x) => ({ value: x.id, label: devLabel(x) })), { value: saved.output ?? virt ?? def(outputs) });
  fillSelect($("#live-mon"), [{ value: "", label: "Aucun" }, ...outputs.map((x) => ({ value: x.id, label: devLabel(x) }))], { value: saved.monitor ?? "" });
}

async function loadDevices() {
  try {
    if (state.liveWhere === "browser") {
      const perm = await micPermission();
      const d = await listBrowserDevices();
      if (!d.labeled || perm === "denied") {
        fillDeviceSelects([], []);
        return showDeviceNotice(perm === "denied"
          ? `<b>L'accès au micro est bloqué pour cette page.</b> Cliquez sur l'icône à gauche de l'adresse
             (🔒 ou ⓘ) → <i>Micro</i> → <b>Autoriser</b>, puis rechargez la page (F5).`
          : `Pour lister votre micro, votre casque et VB-CABLE, Chrome doit autoriser l'accès au micro.
             <div class="actions" style="justify-content:flex-start"><button class="btn primary small" id="live-allow-mic">🎤 Autoriser l'accès au micro</button></div>`);
      }
      const outputs = d.outputs.map((x) => ({ ...x, name: x.label, default: x.id === "default" }));
      fillDeviceSelects(d.inputs.map((x) => ({ ...x, name: x.label, default: x.id === "default" })), outputs);
      if (!canChooseOutput()) {
        showDeviceNotice("Ce navigateur ne sait pas choisir la sortie audio : ouvrez VoiceClone dans <b>Chrome</b> ou <b>Edge</b> pour envoyer la voix vers VB-CABLE.");
      } else if (!outputs.some((x) => x.virtual)) {
        showDeviceNotice("Aucun câble audio virtuel détecté sur ce PC. Installez <a href=\"https://vb-audio.com/Cable/\" target=\"_blank\" rel=\"noopener\">VB-CABLE</a>, redémarrez le navigateur puis cliquez sur ↻.");
      } else showDeviceNotice("");
    } else {
      const d = await api("/api/realtime/devices");
      fillDeviceSelects(d.inputs, d.outputs);
      showDeviceNotice(d.outputs.some((x) => x.virtual) ? "" : "Aucun câble audio virtuel détecté sur le serveur.");
    }
  } catch (err) {
    fillDeviceSelects([], []);
    if (err.message === "NOT_SECURE") {
      return showDeviceNotice(`Le navigateur bloque le micro sur cette adresse (${esc(location.host)}). Ouvrez la page via
        <b>http://localhost:${esc(location.port || "80")}</b> (redirection de port VS Code / tunnel SSH) ou en HTTPS.`);
    }
    showDeviceNotice(state.liveWhere === "server"
      ? `${esc(err.message)}<br>Si VoiceClone tourne sur un serveur distant, choisissez « Audio de ce PC ».`
      : `Accès aux périphériques refusé : ${esc(err.message)}`);
  }
}
$("#live-refresh").addEventListener("click", loadDevices);
$("#live-dev-error").addEventListener("click", async (e) => {
  if (!e.target.closest("#live-allow-mic")) return;
  try {
    await requestMic();
  } catch (err) {
    toast(`Micro refusé : ${err.message}`, "error", 8000);
  }
  loadDevices();
});
["#live-in", "#live-out", "#live-mon"].forEach((s) => $(s).addEventListener("change", () => {
  store.set(deviceKey(), { input: $("#live-in").value, output: $("#live-out").value, monitor: $("#live-mon").value });
}));

$("#live-where").addEventListener("click", (e) => {
  const b = e.target.closest("[data-where]");
  if (!b || state.liveRunning) return;
  state.liveWhere = b.dataset.where;
  store.set("live.where", state.liveWhere);
  applyLiveWhere();
  loadDevices();
  pollLive(true);
});
function applyLiveWhere() {
  $$("#live-where button").forEach((x) => x.classList.toggle("active", x.dataset.where === state.liveWhere));
  $$("#tab-live .browser-only").forEach((el) => el.classList.toggle("hidden", state.liveWhere !== "browser"));
}
applyLiveWhere();

const intOrNull = (v) => (v === "" || v === undefined ? null : parseInt(v, 10));

function liveSettings() {
  const v = (s) => parseFloat($(s).value);
  return {
    mode: state.liveMode,
    model_id: $("#live-model").value || null,
    voice_id: $("#live-voice").value || null,
    asr_model_id: $("#live-asr").value || null,
    source_voice_id: $("#live-src").value || null,
    language: $("#live-lang").value || "fr",
    chunk_ms: v("#live-chunk"), context_ms: v("#live-ctx"), silence_db: v("#live-th"),
    end_silence_ms: v("#live-eos"), input_gain: v("#live-ig"), output_gain: v("#live-og"),
    params: readParams($("#live-params")),
    say_model_id: $("#live-say-model").value || null,
    warmup: $("#live-warmup").checked,
  };
}

/** Libellé et couleur du bouton selon l'état réel (busy() remet l'ancien libellé à la fin du clic). */
function syncLiveButton() {
  const btn = $("#live-toggle");
  btn.textContent = state.liveRunning ? "■ Arrêter" : "▶ Démarrer";
  btn.classList.toggle("running", state.liveRunning);
}

$("#live-toggle").addEventListener("click", (e) => busy(e.currentTarget, state.liveRunning ? "Arrêt…" : "Démarrage…", async () => {
  try {
    if (state.liveWhere === "browser") {
      if (state.liveRunning) {
        await browserLive.stop();
        renderLiveStatus({ state: "idle" });
      } else {
        renderLiveStatus({ state: "starting" });
        await browserLive.start(liveSettings(), { input: $("#live-in").value, output: $("#live-out").value, monitor: $("#live-mon").value },
          { opus: $("#live-opus").checked });
        toast("Live démarré — parlez !", "ok");
        loadModels();
      }
      return;
    }
    if (state.liveRunning) {
      await api("/api/realtime/stop", { method: "POST" });
    } else {
      await api("/api/realtime/start", {
        json: {
          ...liveSettings(),
          input_device: intOrNull($("#live-in").value),
          output_device: intOrNull($("#live-out").value),
          monitor_device: intOrNull($("#live-mon").value),
        },
      });
      toast("Live démarré — parlez !", "ok");
      loadModels();
    }
  } catch (err) {
    toast(err.message, "error", 8000);
    if (state.liveWhere === "browser") renderLiveStatus({ state: "error", error: err.message });
  }
  if (state.liveWhere === "server") pollLive(true);
}).then(syncLiveButton));

const dbToPct = (db) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));

function renderLiveStatus(s) {
  state.liveRunning = ["running", "starting", "reconnecting"].includes(s.state);
  if (!$("#live-toggle").disabled) syncLiveButton(); // pendant un clic, c'est la fin du clic qui le met à jour
  const names = { idle: "arrêté", starting: "démarrage…", running: "en direct 🔴", error: "erreur", reconnecting: "reconnexion…" };
  $("#live-state").textContent = names[s.state] || s.state;
  $("#live-in-meter").style.width = `${dbToPct(s.input_db ?? -120)}%`;
  $("#live-out-meter").style.width = `${dbToPct(s.output_db ?? -120)}%`;
  $("#live-proc").textContent = s.process_ms ? `${Math.round(s.process_ms)} ms` : "–";
  $("#live-lat").textContent = state.liveRunning && s.latency_ms ? `≈ ${Math.round(s.latency_ms)} ms` : "–";
  $("#live-rtt").textContent = state.liveRunning && state.liveWhere === "browser"
    ? `${s.rtt_ms != null ? `${Math.round(s.rtt_ms)} ms` : "…"} · ${s.codec === "opus" ? "Opus" : "PCM"}` : "–";
  if (s.notice) showLiveNotice(s.notice);
  renderMicState();
  $("#live-buf").textContent = state.liveRunning && s.buffer_ms !== undefined ? `${s.buffer_ms} ms` : "–";
  $("#live-chunks").textContent = state.liveRunning && s.chunks !== undefined ? `${s.chunks} / ${s.dropped}` : "–";
  $("#live-error").textContent = s.error || "";
  $("#live-error").classList.toggle("hidden", !s.error);
  const tr = s.transcripts || [];
  $("#live-transcripts").classList.toggle("hidden", !tr.length);
  $("#live-transcripts").innerHTML = tr.map((t) => `<div>${esc(t.text)}<small>${t.first_audio_ms ? `voix après ${t.first_audio_ms} ms` : ""}</small></div>`).join("");
  $$("#live-mode button, #live-where button").forEach((b) => { b.disabled = state.liveRunning; });
}

async function pollLive(force) {
  clearTimeout(state.livePoll);
  if (state.liveWhere !== "server") return; // en mode navigateur, le statut arrive par le WebSocket
  if (state.tab !== "live" && !force) return;
  try {
    renderLiveStatus(await api("/api/realtime/status"));
  } catch { /* serveur indisponible */ }
  state.livePoll = setTimeout(pollLive, state.liveRunning ? 150 : 1500);
}

// ---------------------------------------------------------------- historique
async function loadHistory() {
  try {
    state.history = await api("/api/history?limit=200");
    renderHistory();
  } catch (err) { toast(err.message, "error"); }
}

function renderHistory() {
  const q = $("#hist-q").value.trim().toLowerCase();
  const kind = $("#hist-kind").value;
  const favOnly = $("#hist-fav").checked;
  const items = (state.history || []).filter((h) => (!kind || h.kind === kind) && (!favOnly || h.favorite)
    && (!q || [h.text, h.voice_name, getModel(h.model_id)?.name || h.model_id].join(" ").toLowerCase().includes(q)));
  $("#history-list").innerHTML = items.map((h) => `<div class="card h-item">
      <div><div class="txt">${{ tts: "💬", long: "📜", book: "📚", s2s: "🔁", translate: "🌍" }[h.kind] || "🔊"} ${esc(h.title || h.text || (h.kind === "s2s" ? "Conversion de voix" : ""))}${h.watermark ? ' <span class="badge" title="Filigrane inaudible">wm</span>' : ""}</div>
        <div class="sub">${esc(h.voice_name || "")} · ${esc(getModel(h.model_id)?.name || h.model_id)} · ${h.duration} s · ${new Date(h.created_at * 1000).toLocaleString("fr-FR")}</div></div>
      <audio controls preload="none" src="/api/history/${h.id}/audio"></audio>
      <div class="badges">
        <button class="btn small" data-fav-hist="${h.id}" title="Favori">${h.favorite ? "★" : "☆"}</button>
        ${["tts", "long"].includes(h.kind) && h.text ? `<button class="btn small" data-reuse-hist="${h.id}" title="Reprendre ces réglages dans Texte → Voix">↻</button>` : ""}
        ${h.segments ? `<button class="btn small" data-seg-hist="${h.id}" title="Corriger phrase par phrase">✎</button>` : ""}
        <a class="btn small" href="/api/history/${h.id}/audio" download title="WAV">⬇</a>
        <a class="btn small" href="/api/history/${h.id}/audio?format=mp3" download title="MP3">MP3</a>
        <button class="btn small danger" data-del-hist="${h.id}">✕</button></div></div>`).join("")
    || `<div class="empty">${state.history?.length ? "Aucun résultat pour ces filtres." : "Rien pour l'instant. Vos générations apparaîtront ici."}</div>`;
}
["#hist-q", "#hist-kind", "#hist-fav"].forEach((id) => $(id).addEventListener("input", renderHistory));

$("#history-list").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del-hist]");
  const fav = e.target.closest("[data-fav-hist]");
  const reuse = e.target.closest("[data-reuse-hist]");
  const segBtn = e.target.closest("[data-seg-hist]");
  try {
    if (segBtn) {
      showTab("tts");
      await renderSegments($("#tts-result"), segBtn.dataset.segHist);
      return $("#tts-result").scrollIntoView({ behavior: "smooth" });
    }
    if (del) {
      await api(`/api/history/${del.dataset.delHist}`, { method: "DELETE" });
      state.history = state.history.filter((h) => h.id !== del.dataset.delHist);
    } else if (fav) {
      const h = state.history.find((x) => x.id === fav.dataset.favHist);
      Object.assign(h, await api(`/api/history/${h.id}`, { method: "PATCH", json: { favorite: !h.favorite } }));
    } else if (reuse) {
      const h = state.history.find((x) => x.id === reuse.dataset.reuseHist);
      if (getVoice(h.voice_id)) $("#tts-voice").value = h.voice_id;
      if (getModel(h.model_id)) { $("#tts-model").value = h.model_id; state.ttsParamsFor = null; }
      if (h.params) store.set(`params.${h.model_id}`, h.params);
      onTTSModelChange();
      if (h.language) $("#tts-lang").value = h.language;
      $("#tts-text").value = h.text;
      $("#tts-count").textContent = h.text.length;
      showTab("tts");
      return toast("Réglages repris : cliquez sur Générer.", "ok");
    } else return;
  } catch (err) { toast(err.message, "error"); }
  renderHistory();
});

// ---------------------------------------------------------------- démarrage
function initLangs() {
  fillSelect($("#nv-lang"), Object.keys(LANGS).filter((l) => l !== "auto" && l !== "zh-cn").map((l) => ({ value: l, label: langName(l) })), { value: "fr" });
}

(async function init() {
  initLangs();
  loadSystem();
  loadMics();
  try {
    await loadModels();
    await loadVoices();
  } catch (err) { toast(`Serveur injoignable : ${err.message}`, "error"); }
  applyS2SMode();
  applyLiveMode();
  showTab(state.tab);
  refreshJobsPanel();
})();

// ------------------------------------------------ Live : messages, micro, texte dit
let liveNoticeTimer;
function showLiveNotice(text) {
  $("#live-notice").textContent = text;
  $("#live-notice").classList.remove("hidden");
  clearTimeout(liveNoticeTimer);
  liveNoticeTimer = setTimeout(() => $("#live-notice").classList.add("hidden"), 6000);
}

const keyName = (code) => code.replace(/^Key/, "").replace(/^Digit/, "");
state.pttKey = store.get("live.ptt.key", "Space");

function renderMicState() {
  const mode = browserLive.micMode;
  $$("#tab-live .ptt-only").forEach((el) => el.classList.toggle("hidden", mode !== "ptt"));
  $("#live-ptt-key").textContent = keyName(state.pttKey);
  const open = browserLive.micOpen();
  const el = $("#live-mic-state");
  el.textContent = mode === "mute" ? "🔇 coupé" : mode === "ptt" ? (open ? "🎙 émission" : `maintenez ${keyName(state.pttKey)}`) : "🎙 ouvert";
  el.className = `badge ${open ? "on" : "off"}`;
}
function setMicMode(mode) {
  browserLive.micMode = mode;
  browserLive.pttDown = false;
  $("#live-mic-mode").value = mode;
  store.set("live.mic.mode", mode);
  renderMicState();
}
$("#live-mic-mode").addEventListener("change", (e) => setMicMode(e.target.value));
setMicMode(store.get("live.mic.mode", "open"));

$("#live-gate").addEventListener("input", (e) => {
  browserLive.gateDb = parseFloat(e.target.value);
  $("#v-gate").textContent = e.target.value;
  store.set("live.gate", e.target.value);
});
$("#live-gate").value = store.get("live.gate", "-90");
$("#live-gate").dispatchEvent(new Event("input"));
$("#live-opus").checked = store.get("live.opus", true);
$("#live-opus").addEventListener("change", (e) => store.set("live.opus", e.target.checked));
$("#live-warmup").checked = store.get("live.warmup", true);
$("#live-warmup").addEventListener("change", (e) => store.set("live.warmup", e.target.checked));

let capturingKey = false;
$("#live-ptt-key").addEventListener("click", () => {
  capturingKey = true;
  $("#live-ptt-key").textContent = "appuyez…";
});
const typing = (e) => e.target.closest("input, textarea, select, [contenteditable]");
document.addEventListener("keydown", (e) => {
  if (capturingKey) {
    e.preventDefault();
    capturingKey = false;
    state.pttKey = e.code;
    store.set("live.ptt.key", e.code);
    return renderMicState();
  }
  if (state.tab !== "live" || typing(e)) return;
  if (e.ctrlKey && e.code === "KeyM") {
    e.preventDefault();
    return setMicMode(browserLive.micMode === "mute" ? store.get("live.mic.unmute", "open") : (store.set("live.mic.unmute", browserLive.micMode), "mute"));
  }
  if (browserLive.micMode === "ptt" && e.code === state.pttKey) {
    e.preventDefault();
    if (!browserLive.pttDown) { browserLive.pttDown = true; renderMicState(); }
  }
});
document.addEventListener("keyup", (e) => {
  if (browserLive.micMode === "ptt" && e.code === state.pttKey) { browserLive.pttDown = false; renderMicState(); }
});
window.addEventListener("blur", () => { if (browserLive.pttDown) { browserLive.pttDown = false; renderMicState(); } });

// Texte → Live (Discord) + phrases favorites
const sayFavs = () => store.get("live.say.favs", []);
function renderSayFavs() {
  $("#live-say-favs").innerHTML = sayFavs().map((t, i) =>
    `<span class="fav-chip"><a href="#" data-say="${i}">${esc(t)}</a><button data-unfav="${i}" title="Retirer">×</button></span>`).join("")
    || '<small class="muted">Phrases favorites : tapez une phrase puis ☆ pour la garder sous la main.</small>';
}
async function liveSay(text) {
  text = (text || "").trim();
  if (!text) return;
  if (!state.liveRunning) return toast("Démarrez le Live d'abord.", "error");
  store.set("live.say.model", $("#live-say-model").value);
  try {
    if (state.liveWhere === "browser") browserLive.say(text);
    else await api("/api/realtime/say", { json: { text, model_id: $("#live-say-model").value || null } });
    showLiveNotice(`💬 « ${text.slice(0, 80)} » envoyé`);
  } catch (err) {
    toast(err.message, "error", 6000);
  }
}
$("#live-say-go").addEventListener("click", () => { liveSay($("#live-say-text").value); $("#live-say-text").value = ""; });
$("#live-say-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); $("#live-say-go").click(); }
});
$("#live-say-model").addEventListener("change", (e) => store.set("live.say.model", e.target.value));
$("#live-say-fav-add").addEventListener("click", () => {
  const t = $("#live-say-text").value.trim();
  if (!t) return toast("Tapez d'abord une phrase.", "error");
  store.set("live.say.favs", [...new Set([...sayFavs(), t])].slice(0, 30));
  renderSayFavs();
});
$("#live-say-favs").addEventListener("click", (e) => {
  const say = e.target.closest("[data-say]");
  const un = e.target.closest("[data-unfav]");
  if (say) { e.preventDefault(); liveSay(sayFavs()[+say.dataset.say]); }
  if (un) { const f = sayFavs(); f.splice(+un.dataset.unfav, 1); store.set("live.say.favs", f); renderSayFavs(); }
});
renderSayFavs();
