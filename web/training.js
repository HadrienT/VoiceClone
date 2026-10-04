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
  const done = v.samples.filter((s) => s.transcript);
  const secs = done.reduce((s, x) => s + x.duration, 0);
  $("#tr-info").innerHTML = `<b>${esc(v.name)}</b> : ${v.samples.length} échantillon(s), ${v.duration} s —
    ${done.length} transcrit(s) (${secs.toFixed(1)} s).
    ${v.duration < 60 ? "<br>⚠ Moins d'une minute d'audio : l'affinage XTTS apporte surtout quelque chose à partir de 2-10 min de voix propre." : ""}
    ${xtts && !(xtts.downloaded && xtts.installed) ? "<br>⚠ XTTS v2 doit être téléchargé et installé (onglet Modèles)." : ""}`;
  const rvc = v.settings?.rvc;
  $("#rvc-current").innerHTML = rvc?.pth
    ? `✅ Modèle RVC attaché (${esc(rvc.source || "")}${rvc.index ? ", avec index" : ", sans index"}).
       <button class="btn small danger" id="rvc-detach">Retirer</button>
       <br><small class="muted">Utilisez le modèle « RVC » dans Voix → Voix ou le Live avec cette voix.</small>`
    : `<span class="muted">Aucun modèle RVC pour cette voix.</span>`;
}

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
