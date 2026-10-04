/** Onglet Entraînement : affiner XTTS sur une voix, modèles RVC (import ou entraînement via Applio). */
import { $, esc, api, toast, busy } from "./util.js";
import { watchJob, jobProgressHtml } from "./jobs.js";

let voices = [];
let models = [];

export async function showTraining() {
  [voices, models] = await Promise.all([api("/api/voices"), api("/api/models")]);
  const sel = $("#tr-voice");
  const prev = sel.value;
  sel.innerHTML = voices.map((v) => `<option value="${esc(v.id)}">${esc(v.name)} (${v.duration} s)</option>`).join("")
    || '<option value="">Aucune voix</option>';
  if (voices.some((v) => v.id === prev)) sel.value = prev;
  const asr = models.filter((m) => m.capabilities.includes("asr"));
  $("#tr-asr").innerHTML = `<option value="">Ne pas transcrire (extraits déjà transcrits seulement)</option>`
    + asr.map((m) => `<option value="${esc(m.id)}" ${m.downloaded && m.installed ? "" : "disabled"}>${esc(m.name)}</option>`).join("");
  const ready = asr.find((m) => m.downloaded && m.installed);
  if (ready) $("#tr-asr").value = ready.id;
  renderVoiceInfo();
  renderTrained();
}

function renderVoiceInfo() {
  const v = voices.find((x) => x.id === $("#tr-voice").value);
  const xtts = models.find((m) => m.id === "xtts-v2");
  if (!v) { $("#tr-info").innerHTML = ""; return; }
  const clips = [...v.samples, ...(v.training || [])];
  const done = clips.filter((s) => s.transcript);
  const secs = done.reduce((s, x) => s + x.duration, 0);
  const total = v.duration + (v.training_duration || 0);
  $("#tr-info").innerHTML = `<b>${esc(v.name)}</b> : ${fmtMin(total)} d'audio au total
    (référence de clonage ${v.duration} s + audio d'entraînement ${fmtMin(v.training_duration || 0)}) —
    ${done.length}/${clips.length} extraits transcrits (${fmtMin(secs)}).
    ${total < 120 ? "<br>⚠ Moins de 2 minutes : ajoutez de l'audio d'entraînement ci-dessous (idéal 5 à 30 min)." : ""}
    ${xtts && !(xtts.downloaded && xtts.installed) ? "<br>⚠ XTTS v2 doit être téléchargé et installé (onglet Modèles)." : ""}`;
  renderTrainingAudio(v);
  const rvc = v.settings?.rvc;
  $("#rvc-current").innerHTML = rvc?.pth
    ? `✅ Modèle RVC attaché (${esc(rvc.source || "")}${rvc.index ? ", avec index" : ", sans index"}).
       <button class="btn small danger" id="rvc-detach">Retirer</button>
       <br><small class="muted">Utilisez le modèle « RVC » dans Voix → Voix ou le Live avec cette voix.</small>`
    : `<span class="muted">Aucun modèle RVC pour cette voix.</span>`;
}

const fmtMin = (s) => (s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);

function renderTrainingAudio(v) {
  const list = v.training || [];
  const shown = list.slice(0, 200);
  $("#tr-audio").innerHTML = list.length ? `<div class="row" style="align-items:center;margin:0 0 6px">
      <b class="grow">${list.length} extraits · ${fmtMin(v.training_duration)}
        <small class="muted">(${list.filter((t) => t.transcript).length} transcrits)</small></b>
      <button class="btn small danger" id="tr-clear">Tout supprimer</button></div>
    <details><summary class="muted" style="cursor:pointer;font-size:13px">Écouter / retirer des extraits</summary>
      <ol class="chapter-list">${shown.map((t) => {
        const name = t.file.split("/")[1];
        return `<li><span class="grow" title="${esc(t.transcript || "")}">${esc(t.source || name)}
            <small class="muted">${t.duration} s${t.transcript ? ` · « ${esc(t.transcript.slice(0, 60))} »` : ""}</small></span>
          <audio controls preload="none" src="/api/voices/${v.id}/training-audio/${name}"></audio>
          <button class="btn small" data-tr-del="${name}" title="Retirer">✕</button></li>`;
      }).join("")}${list.length > shown.length ? `<li class="muted">… et ${list.length - shown.length} autres</li>` : ""}</ol>
    </details>` : '<p class="muted" style="margin:0">Aucun audio d\'entraînement pour cette voix.</p>';
}

async function uploadTraining(files) {
  const voiceId = $("#tr-voice").value;
  if (!voiceId) return toast("Créez d'abord une voix.", "error");
  const out = $("#tr-upload-progress");
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("enhance", $("#tr-enhance").checked);
  out.classList.remove("hidden");
  out.innerHTML = `<span class="spinner"></span> Envoi de ${files.length} fichier(s)…`;
  try {
    const job = await api(`/api/voices/${voiceId}/training-audio`, { method: "POST", body: fd });
    const done = await watchJob(job.id, (j) => { out.innerHTML = jobProgressHtml(j); });
    out.innerHTML = `<div class="notice ok">${done.result.reports.map((r) => `✅ ${esc(r.name)} : ${fmtMin(r.duration)} analysées →
      <b>${r.clips} extraits, ${fmtMin(r.kept_duration)} gardées</b>${r.rejected ? ` (${r.rejected} passages écartés)` : ""}${r.fallback ? `<br>⚠ ${esc(r.fallback)}` : ""}`).join("<br>")}</div>`;
    await showTraining();
  } catch (err) {
    out.innerHTML = `<div class="notice error">${esc(err.message)}</div>`;
  }
}

const drop = $("#tr-drop");
drop.addEventListener("click", () => $("#tr-file").click());
$("#tr-file").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; if (f.length) uploadTraining(f); });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const files = [...e.dataTransfer.files].filter((f) => /^(audio|video)/.test(f.type) || /\.(wav|mp3|m4a|ogg|flac|webm|opus|aac|mp4)$/i.test(f.name));
  if (files.length) uploadTraining(files); else toast("Déposez un fichier audio.", "error");
});

async function renderTrained() {
  const list = await api("/api/training/models");
  $("#tr-models").innerHTML = list.length ? list.map((m) => `<div class="kv"><span>🎓 <b>${esc(m.name)}</b>
      <small class="muted">${m.dataset ? `${m.dataset.seconds} s d'audio` : ""} · ${new Date(m.created_at * 1000).toLocaleString("fr-FR")}</small></span>
      <button class="btn small danger" data-del-model="${esc(m.id)}">Supprimer</button></div>`).join("")
    : '<p class="muted">Aucun modèle entraîné pour l\'instant.</p>';
}

async function runJob(btn, path, payload, out) {
  return busy(btn, "Lancement…", async () => {
    try {
      const job = await api(path, { json: payload });
      out.classList.remove("hidden");
      toast("Entraînement lancé : il continue en arrière-plan (suivi dans la barre latérale).", "ok", 7000);
      const done = await watchJob(job.id, (j) => { out.innerHTML = jobProgressHtml(j); });
      out.innerHTML = `<div class="notice ok">✅ Terminé.${done.result?.model_id ? ` Nouveau modèle : <b>${esc(done.result.model_id)}</b> (sélectionné pour cette voix dans Texte → Voix).` : ""}</div>`;
      await showTraining();
    } catch (err) {
      out.classList.remove("hidden");
      out.innerHTML = `<div class="notice error" style="white-space:pre-wrap">${esc(err.message)}</div>`;
    }
  });
}

$("#tr-voice").addEventListener("change", renderVoiceInfo);
$("#xtts-go").addEventListener("click", (e) => runJob(e.currentTarget, "/api/training/xtts", {
  voice_id: $("#tr-voice").value, epochs: +$("#xtts-epochs").value, batch_size: +$("#xtts-batch").value,
  grad_accum: +$("#xtts-accum").value, asr_model_id: $("#tr-asr").value || null,
  gpus: $("#xtts-gpus").value, precision: $("#xtts-precision").value,
}, $("#xtts-progress")));
$("#rvc-go").addEventListener("click", (e) => runJob(e.currentTarget, "/api/training/rvc", {
  voice_id: $("#tr-voice").value, epochs: +$("#rvc-epochs").value, batch_size: +$("#rvc-batch").value,
  sample_rate: +$("#rvc-sr").value,
}, $("#rvc-progress")));
$("#rvc-upload").addEventListener("click", (e) => busy(e.currentTarget, "Import…", async () => {
  const pth = $("#rvc-pth").files[0];
  if (!pth) return toast("Choisissez le fichier .pth du modèle.", "error");
  const fd = new FormData();
  fd.append("pth", pth);
  if ($("#rvc-index").files[0]) fd.append("index", $("#rvc-index").files[0]);
  try {
    await api(`/api/voices/${$("#tr-voice").value}/rvc`, { method: "POST", body: fd });
    toast("Modèle RVC attaché à la voix.", "ok");
    $("#rvc-pth").value = "";
    $("#rvc-index").value = "";
    await showTraining();
  } catch (err) { toast(err.message, "error"); }
}));
document.addEventListener("click", async (e) => {
  const trDel = e.target.closest("[data-tr-del]");
  if (trDel || e.target.closest("#tr-clear")) {
    if (!trDel && !confirm("Supprimer tout l'audio d'entraînement de cette voix ?")) return;
    try {
      await api(`/api/voices/${$("#tr-voice").value}/training-audio${trDel ? `/${trDel.dataset.trDel}` : ""}`, { method: "DELETE" });
      await showTraining();
    } catch (err) { toast(err.message, "error"); }
    return;
  }
  if (e.target.closest("#rvc-detach")) {
    await api(`/api/voices/${$("#tr-voice").value}/rvc`, { method: "DELETE" });
    return showTraining();
  }
  const del = e.target.closest("[data-del-model]");
  if (del && confirm("Supprimer ce modèle entraîné (fichiers compris) ?")) {
    try { await api(`/api/training/models/${del.dataset.delModel}`, { method: "DELETE" }); renderTrained(); }
    catch (err) { toast(err.message, "error"); }
  }
});
