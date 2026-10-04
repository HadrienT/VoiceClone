/** Tâches de fond (textes longs, livres audio…) : suivi de progression et panneau latéral. */
import { $, esc, api, toast } from "./util.js";

const fmtEta = (s) => (s == null ? "" : s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

export function jobProgressHtml(j) {
  const pct = Math.round((j.progress || 0) * 100);
  return `<div class="job-progress">
    <div class="bar"><span style="width:${pct}%"></span></div>
    <div class="muted" style="font-size:12px">${esc(j.message || (j.state === "queued" ? "En file d'attente…" : ""))}
      · ${pct} %${j.eta_s != null ? ` · reste ≈ ${fmtEta(j.eta_s)}` : ""}
      <button class="link" data-cancel-job="${j.id}">Annuler</button></div></div>`;
}

/** Attend la fin d'une tâche en appelant onUpdate(job) à chaque étape. Renvoie la tâche terminée. */
export async function watchJob(id, onUpdate = () => {}) {
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    onUpdate(j);
    refreshJobsPanel();
    if (!["queued", "running"].includes(j.state)) {
      if (j.state === "error") throw new Error(j.error || "La tâche a échoué.");
      if (j.state === "cancelled") throw new Error("Tâche annulée.");
      return j;
    }
    await new Promise((r) => setTimeout(r, 700));
  }
}

let panelTimer;
export async function refreshJobsPanel() {
  clearTimeout(panelTimer);
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { return; }
  const active = jobs.filter((j) => ["queued", "running"].includes(j.state));
  const el = $("#jobs-mini");
  el.classList.toggle("hidden", !active.length);
  el.innerHTML = active.length ? `<b>⏳ ${active.length} tâche(s)</b>` + active.map((j) =>
    `<div class="job-mini"><span title="${esc(j.title)}">${esc(j.title)}</span>
      <div class="bar"><span style="width:${Math.round(j.progress * 100)}%"></span></div></div>`).join("") : "";
  if (active.length) panelTimer = setTimeout(refreshJobsPanel, 1500);
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-cancel-job]");
  if (!b) return;
  try { await api(`/api/jobs/${b.dataset.cancelJob}/cancel`, { method: "POST" }); toast("Annulation demandée.", "info"); }
  catch (err) { toast(err.message, "error"); }
});
