import { Recorder, toWav, listMicrophones, fmtTime, SYSTEM_SOURCE, canCaptureSystemAudio } from "./audio.js";

// ---------------------------------------------------------------- utilitaires
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const LANGS = {
  auto: "Détection auto", fr: "Français", en: "Anglais", es: "Espagnol", de: "Allemand", it: "Italien",
  pt: "Portugais", pl: "Polonais", tr: "Turc", ru: "Russe", nl: "Néerlandais", cs: "Tchèque", ar: "Arabe",
  "zh-cn": "Chinois", zh: "Chinois", hu: "Hongrois", ko: "Coréen", ja: "Japonais", hi: "Hindi", da: "Danois",
  el: "Grec", fi: "Finnois", he: "Hébreu", ms: "Malais", no: "Norvégien", sv: "Suédois", sw: "Swahili",
};
const langName = (c) => LANGS[c] || c;

const READ_PROMPTS = [
  "Bonjour, je m'appelle comme vous voulez. Aujourd'hui, il fait beau et je vais enregistrer ma voix pour essayer ce logiciel. J'espère que le résultat sera bluffant !",
  "Le vif zéphyr jubile sur les kumquats du clown gracieux. Pourquoi ce chat noir boit-il du lait chaud dans un bol jaune ? Personne ne le sait vraiment.",
  "Hier soir, nous avons joué jusqu'à minuit. C'était intense : trois victoires, deux défaites, et un fou rire général quand Paul a raté son dernier saut.",
  "Attention, ceci est un message important. Merci de vérifier vos paramètres avant de continuer. Est-ce que tout le monde m'entend correctement ?",
  "J'adore les longues balades en forêt, surtout en automne, quand les feuilles craquent sous les pieds et que l'air sent la terre mouillée.",
  "Un, deux, trois, quatre, cinq. Les chaussettes de l'archiduchesse sont-elles sèches, archisèches ? Voilà une phrase bien difficile à prononcer !",
];

const store = {
  get(k, d) { try { const v = localStorage.getItem("vc." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("vc." + k, JSON.stringify(v)); } catch { /* stockage indisponible */ } },
};

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* corps non JSON */ }
    throw new Error(msg);
  }
  if (opts.raw) return res;
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res;
}

function toast(msg, type = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms);
}

const fmtBytes = (b) => {
  if (!b) return "0 o";
  const u = ["o", "Ko", "Mo", "Go"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1024)));
  return `${(b / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
};

async function busy(btn, label, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${label}`;
  try { return await fn(); } finally { btn.disabled = false; btn.innerHTML = old; }
}

const state = { models: [], voices: [], tab: store.get("tab", "models"), filter: "all" };

// ---------------------------------------------------------------- navigation
function showTab(tab) {
  state.tab = tab;
  store.set("tab", tab);
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${tab}`));
  if (tab === "history") loadHistory();
  if (tab === "live") { loadDevices(); pollLive(); }
}
$$("#nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------------------------------------------------------- système
async function loadSystem() {
  try {
    const s = await api("/api/system");
    const dev = s.gpu ? `GPU <b>${esc(s.gpu)}</b> (${s.vram_gb} Go)` : `<b>${esc(s.device.toUpperCase())}</b>`;
    $("#sys").innerHTML = `Calcul : ${dev}<br>PyTorch : <b>${s.torch ? esc(s.torch) : "non installé"}</b><br>v${esc(s.version)}`;
    $("#data-dir").textContent = s.data_dir + "/models";
  } catch {
    $("#sys").textContent = "Serveur injoignable";
  }
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

$("#create-voice").addEventListener("click", (e) => busy(e.currentTarget, "Analyse…", async () => {
  try {
    const fd = new FormData();
    fd.append("name", $("#nv-name").value.trim());
    fd.append("language", $("#nv-lang").value);
    fd.append("consent", $("#nv-consent").checked);
    const [first, ...rest] = pending;
    fd.append("files", first.blob, first.source === "upload" ? first.name : `${first.name}.wav`);
    fd.append("source", first.source);
    if (first.useTranscript) fd.append("transcript", first.transcript);
    let voice = await api("/api/voices", { method: "POST", body: fd });
    for (const p of rest) {
      const f = new FormData();
      f.append("file", p.blob, p.source === "upload" ? p.name : `${p.name}.wav`);
      f.append("source", p.source);
      if (p.useTranscript) f.append("transcript", p.transcript);
      voice = await api(`/api/voices/${voice.id}/samples`, { method: "POST", body: f });
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
      <details>
        <summary>Échantillons (${v.samples.length}) & transcription</summary>
        <div class="samples">
          ${v.samples.map((s) => `<div class="sample"><span class="name">${{ record: "🎤", system: "🖥️" }[s.source] || "📁"} ${esc(s.original_name || s.file)} · ${s.duration}s</span>
            <audio controls preload="none" src="/api/voices/${v.id}/audio?sample=${encodeURIComponent(s.file.split("/").pop())}&t=${t}"></audio>
            <button class="btn small danger" data-rm-sample="${esc(s.file.split("/").pop())}">✕</button></div>`).join("")}
          <div class="row" style="margin:6px 0 0">
            <button class="btn small" data-add-file>＋ Fichier</button>
            <button class="btn small rec" data-add-rec>● Enregistrer</button>
            <input type="file" accept="audio/*,video/*" hidden data-file-input>
          </div>
          <label class="field">Transcription de la référence (utile pour F5-TTS)
            <textarea rows="3" data-transcript placeholder="Texte exact prononcé dans les échantillons…">${esc(v.transcript)}</textarea></label>
          <div class="row" style="margin:0">
            <button class="btn small" data-save-transcript>Enregistrer le texte</button>
            ${asr.length ? `<button class="btn small" data-auto-transcribe="${asr[0].id}">Transcrire avec ${esc(asr[0].name.split(" (")[0])}</button>` : ""}
          </div>
        </div>
      </details>
      <div class="prep">
        <select data-prep-model>${prepModels.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("") || '<option value="">Aucun modèle prêt</option>'}</select>
        <button class="btn small" data-prepare ${prepModels.length ? "" : "disabled"} title="Pré-calcule l'empreinte vocale pour ce modèle">🧬 Entraîner</button>
        <button class="btn small danger" data-del-voice>Supprimer</button>
      </div>
    </div>`;
  }).join("") || `<div class="empty">Aucune voix pour l'instant. Importez ou enregistrez un échantillon ci-dessus.</div>`;
}

const cardRecorders = new Map();

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
    } else if (btn.dataset.rmSample) {
      await api(`/api/voices/${id}/samples/${encodeURIComponent(btn.dataset.rmSample)}`, { method: "DELETE" });
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
      await api(`/api/voices/${id}`, { method: "PATCH", json: { transcript: $("[data-transcript]", card).value } });
      toast("Transcription enregistrée", "ok");
    } else if (btn.dataset.autoTranscribe) {
      await busy(btn, "Transcription…", () => api(`/api/voices/${id}/transcribe`, { json: { model_id: btn.dataset.autoTranscribe } }));
      toast("Transcription terminée", "ok");
    } else if (btn.hasAttribute("data-prepare")) {
      const model = $("[data-prep-model]", card).value;
      const r = await busy(btn, "Entraînement…", () => api(`/api/voices/${id}/prepare`, { json: { model_id: model } }));
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
$("#tts-voice").addEventListener("change", () => store.set("voice", $("#tts-voice").value));
$("#tts-text").addEventListener("input", () => { $("#tts-count").textContent = $("#tts-text").value.length; });
$("#tts-text").value = store.get("tts.text", "");
$("#tts-count").textContent = $("#tts-text").value.length;

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
  return busy(e.currentTarget, model?.loaded ? "Génération…" : "Chargement du modèle…", async () => {
    try {
      const res = await api("/api/tts", {
        raw: true,
        json: { model_id: $("#tts-model").value, voice_id: $("#tts-voice").value, text, language: $("#tts-lang").value, params: readParams($("#tts-params")) },
      });
      const gen = res.headers.get("X-Generation-Seconds");
      const dur = res.headers.get("X-Audio-Seconds");
      showResult($("#tts-result"), await res.blob(), `${dur} s d'audio générées en ${gen} s`);
      loadModels();
    } catch (err) { toast(err.message, "error"); }
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

async function loadDevices() {
  try {
    const d = await api("/api/realtime/devices");
    const lbl = (x) => `${x.virtual ? "★ " : ""}${x.name}${x.default ? " (défaut)" : ""} — ${x.hostapi}`;
    const saved = store.get("live.devices", {});
    const def = (list) => list.find((x) => x.default)?.id ?? "";
    fillSelect($("#live-in"), d.inputs.map((x) => ({ value: x.id, label: lbl(x) })), { value: saved.input ?? def(d.inputs) });
    const virt = d.outputs.find((x) => x.virtual)?.id;
    fillSelect($("#live-out"), d.outputs.map((x) => ({ value: x.id, label: lbl(x) })), { value: saved.output ?? virt ?? def(d.outputs) });
    fillSelect($("#live-mon"), [{ value: "", label: "Aucun" }, ...d.outputs.map((x) => ({ value: x.id, label: lbl(x) }))], { value: saved.monitor ?? "" });
    $("#live-dev-error").classList.add("hidden");
    if (!d.outputs.some((x) => x.virtual)) {
      $("#live-dev-error").innerHTML = "Aucun câble audio virtuel détecté. Installez-en un (voir le guide ci-dessous) pour envoyer la voix dans Discord.";
      $("#live-dev-error").classList.remove("hidden");
    }
  } catch (err) {
    $("#live-dev-error").textContent = err.message;
    $("#live-dev-error").classList.remove("hidden");
  }
}
$("#live-refresh").addEventListener("click", loadDevices);
["#live-in", "#live-out", "#live-mon"].forEach((s) => $(s).addEventListener("change", () => {
  store.set("live.devices", { input: $("#live-in").value, output: $("#live-out").value, monitor: $("#live-mon").value });
}));

const intOrNull = (v) => (v === "" || v === undefined ? null : parseInt(v, 10));

$("#live-toggle").addEventListener("click", (e) => busy(e.currentTarget, state.liveRunning ? "Arrêt…" : "Démarrage…", async () => {
  try {
    if (state.liveRunning) {
      await api("/api/realtime/stop", { method: "POST" });
    } else {
      const v = (s) => parseFloat($(s).value);
      await api("/api/realtime/start", {
        json: {
          mode: state.liveMode,
          model_id: $("#live-model").value || null,
          voice_id: $("#live-voice").value || null,
          asr_model_id: $("#live-asr").value || null,
          source_voice_id: $("#live-src").value || null,
          language: $("#live-lang").value || "fr",
          input_device: intOrNull($("#live-in").value),
          output_device: intOrNull($("#live-out").value),
          monitor_device: intOrNull($("#live-mon").value),
          chunk_ms: v("#live-chunk"), context_ms: v("#live-ctx"), silence_db: v("#live-th"),
          end_silence_ms: v("#live-eos"), input_gain: v("#live-ig"), output_gain: v("#live-og"),
          params: readParams($("#live-params")),
        },
      });
      toast("Live démarré — parlez !", "ok");
      loadModels();
    }
  } catch (err) { toast(err.message, "error"); }
  pollLive(true);
}));

const dbToPct = (db) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));

async function pollLive(force) {
  clearTimeout(state.livePoll);
  if (state.tab !== "live" && !force) return;
  try {
    const s = await api("/api/realtime/status");
    state.liveRunning = s.state === "running" || s.state === "starting";
    const btn = $("#live-toggle");
    if (!btn.disabled) {
      btn.textContent = state.liveRunning ? "■ Arrêter" : "▶ Démarrer";
      btn.classList.toggle("running", state.liveRunning);
    }
    const names = { idle: "arrêté", starting: "démarrage…", running: "en direct 🔴", error: "erreur" };
    $("#live-state").textContent = names[s.state] || s.state;
    $("#live-in-meter").style.width = `${dbToPct(s.input_db)}%`;
    $("#live-out-meter").style.width = `${dbToPct(s.output_db)}%`;
    $("#live-proc").textContent = s.process_ms ? `${Math.round(s.process_ms)} ms` : "–";
    $("#live-buf").textContent = state.liveRunning ? `${s.buffer_ms} ms` : "–";
    $("#live-chunks").textContent = state.liveRunning ? `${s.chunks} / ${s.dropped}` : "–";
    $("#live-error").textContent = s.error || "";
    $("#live-error").classList.toggle("hidden", !s.error);
    const tr = s.transcripts || [];
    $("#live-transcripts").classList.toggle("hidden", !tr.length);
    $("#live-transcripts").innerHTML = tr.map((t) => `<div>${esc(t.text)}<small>${t.first_audio_ms ? `voix après ${t.first_audio_ms} ms` : ""}</small></div>`).join("");
    $$("#live-mode button").forEach((b) => { b.disabled = state.liveRunning; });
  } catch { /* serveur indisponible */ }
  state.livePoll = setTimeout(pollLive, state.liveRunning ? 150 : 1500);
}

// ---------------------------------------------------------------- historique
async function loadHistory() {
  try {
    const items = await api("/api/history");
    $("#history-list").innerHTML = items.map((h) => `<div class="card h-item">
      <div><div class="txt">${h.kind === "tts" ? "💬" : "🔁"} ${esc(h.text || (h.kind === "s2s" ? "Conversion de voix" : ""))}</div>
        <div class="sub">${esc(h.voice_name || "")} · ${esc(getModel(h.model_id)?.name || h.model_id)} · ${h.duration} s · ${new Date(h.created_at * 1000).toLocaleString("fr-FR")}</div></div>
      <audio controls preload="none" src="/api/history/${h.id}/audio"></audio>
      <div class="badges"><a class="btn small" href="/api/history/${h.id}/audio" download>⬇</a>
        <button class="btn small danger" data-del-hist="${h.id}">✕</button></div></div>`).join("")
      || `<div class="empty">Rien pour l'instant. Vos générations apparaîtront ici.</div>`;
  } catch (err) { toast(err.message, "error"); }
}
$("#history-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-del-hist]");
  if (!b) return;
  await api(`/api/history/${b.dataset.delHist}`, { method: "DELETE" }).catch((err) => toast(err.message, "error"));
  loadHistory();
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
})();
