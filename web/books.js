/** Onglet Livre audio : texte / .txt / .md / .epub → chapitres → génération en tâche de fond → ZIP. */
import { $, esc, LANGS, langName, store, api, toast, busy } from "./util.js";
import { watchJob, jobProgressHtml } from "./jobs.js";

let chapters = [];
const fmtDur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

export async function showBooks() {
  const [models, voices] = await Promise.all([api("/api/models"), api("/api/voices")]);
  const tts = models.filter((m) => m.capabilities.includes("tts"));
  const keep = (sel, items, fallback) => {
    const prev = sel.value || fallback;
    sel.innerHTML = items.map((i) => `<option value="${esc(i.value)}">${esc(i.label)}</option>`).join("") || '<option value="">—</option>';
    if (items.some((i) => i.value === prev)) sel.value = prev;
  };
  keep($("#book-model"), tts.map((m) => ({ value: m.id, label: `${m.downloaded && m.installed ? "✓ " : "⚠ "}${m.name}` })), store.get("book.model") || store.get("tts.model"));
  keep($("#book-voice"), voices.map((v) => ({ value: v.id, label: v.name })), store.get("voice"));
  const m = tts.find((x) => x.id === $("#book-model").value);
  const langs = m && !m.languages.includes("*") ? m.languages : Object.keys(LANGS).filter((l) => l !== "auto" && l !== "zh-cn");
  keep($("#book-lang"), langs.map((l) => ({ value: l, label: langName(l) })), store.get("book.lang", "fr"));
  await renderLibrary();
}

function renderChapters() {
  const total = chapters.filter((c) => c.on).reduce((s, c) => s + c.text.length, 0);
  $("#book-chapters").innerHTML = chapters.length ? `<p class="muted" style="font-size:13px;margin:0 0 6px">${chapters.length} chapitre(s) ·
      ${total.toLocaleString("fr-FR")} caractères sélectionnés (≈ ${fmtDur(total / 15)} d'audio)</p>
    <ol class="chapter-list">${chapters.map((c, i) => `<li>
      <input type="checkbox" data-ch-on="${i}" ${c.on ? "checked" : ""}>
      <input data-ch-title="${i}" value="${esc(c.title)}">
      <span class="muted" title="${esc(c.text.slice(0, 400))}">${c.text.length.toLocaleString("fr-FR")} car. — ${esc(c.text.slice(0, 70))}…</span>
    </li>`).join("")}</ol>` : "";
  $("#book-go").disabled = !chapters.some((c) => c.on);
}

async function parseBlob(blob, name, pasted = false) {
  const fd = new FormData();
  fd.append("file", blob, name);
  const r = await api("/api/books/parse", { method: "POST", body: fd });
  chapters = r.chapters.map((c) => ({ ...c, on: true }));
  if (!$("#book-title").value || ($("#book-title").dataset.auto && !pasted)) {
    $("#book-title").value = pasted ? (chapters[0]?.title || "Livre audio") : r.title;
    $("#book-title").dataset.auto = "1";
  }
  renderChapters();
  if (!chapters.length) toast("Aucun texte trouvé dans ce fichier.", "error");
}

$("#book-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try { await parseBlob(f, f.name); } catch (err) { toast(err.message, "error"); }
  e.target.value = "";
});
$("#book-split").addEventListener("click", (e) => busy(e.currentTarget, "Découpage…", async () => {
  const text = $("#book-text").value.trim();
  if (!text) return toast("Collez un texte ou choisissez un fichier.", "error");
  try { await parseBlob(new Blob([text], { type: "text/plain" }), "texte.txt", true); } catch (err) { toast(err.message, "error"); }
}));
$("#book-title").addEventListener("input", (e) => delete e.target.dataset.auto);
$("#book-chapters").addEventListener("input", (e) => {
  const on = e.target.closest("[data-ch-on]");
  const title = e.target.closest("[data-ch-title]");
  if (on) chapters[+on.dataset.chOn].on = on.checked;
  if (title) chapters[+title.dataset.chTitle].title = title.value;
  if (on) renderChapters();
});
["#book-model", "#book-lang"].forEach((id) => $(id).addEventListener("change", () => {
  store.set(id === "#book-model" ? "book.model" : "book.lang", $(id).value);
  if (id === "#book-model") showBooks();
}));

$("#book-go").addEventListener("click", (e) => busy(e.currentTarget, "Génération…", async () => {
  const sel = chapters.filter((c) => c.on);
  if (!$("#book-voice").value) return toast("Créez d'abord une voix.", "error");
  const out = $("#book-progress");
  try {
    const r = await api("/api/books", { json: {
      title: $("#book-title").value.trim() || "Livre audio", chapters: sel.map((c) => ({ title: c.title, text: c.text })),
      model_id: $("#book-model").value, voice_id: $("#book-voice").value, language: $("#book-lang").value,
      params: store.get(`params.${$("#book-model").value}`, {}), format: $("#book-format").value,
      announce_titles: $("#book-announce").checked } });
    out.classList.remove("hidden");
    toast("Génération lancée : vous pouvez changer d'onglet, elle continue en arrière-plan.", "ok", 6000);
    renderLibrary();
    await watchJob(r.job.id, (j) => { out.innerHTML = jobProgressHtml(j); });
    out.classList.add("hidden");
    toast("Livre audio terminé !", "ok");
  } catch (err) {
    out.innerHTML = `<div class="notice error">${esc(err.message)}</div>`;
  }
  renderLibrary();
}));

async function renderLibrary() {
  const list = await api("/api/books");
  const label = { queued: "en attente", running: "en cours", done: "terminé", error: "erreur", cancelled: "annulé" };
  $("#book-library").innerHTML = list.map((b) => `<div class="card book">
      <div class="row" style="align-items:center;margin:0">
        <div class="grow"><b>📚 ${esc(b.title)}</b>
          <div class="muted" style="font-size:12px">${esc(b.voice_name)} · ${b.chapters.length} chapitre(s)
            ${b.duration ? ` · ${fmtDur(b.duration)}` : ""} · <span class="badge ${b.state === "done" ? "ok" : ""}">${label[b.state] || b.state}${b.state === "running" && b.progress != null ? ` ${Math.round(b.progress * 100)} %` : ""}</span>
            ${b.error ? `<span class="notice error" style="display:inline-block;margin:0;padding:2px 8px">${esc(b.error)}</span>` : ""}</div></div>
        ${b.zip ? `<a class="btn small primary" href="/api/books/${b.id}/zip">⬇ ZIP</a>` : ""}
        <button class="btn small danger" data-book-del="${b.id}">✕</button>
      </div>
      <details><summary class="muted" style="font-size:13px;cursor:pointer">Chapitres</summary>
        <ol class="chapter-list">${b.chapters.map((c, i) => `<li><span class="grow">${esc(c.title)}</span>
          ${c.file ? `<audio controls preload="none" src="/api/books/${b.id}/chapters/${i}"></audio><small class="muted">${fmtDur(c.duration)}</small>` : '<small class="muted">—</small>'}</li>`).join("")}</ol>
      </details></div>`).join("") || '<p class="muted">Aucun livre audio pour l\'instant.</p>';
}
$("#book-library").addEventListener("click", async (e) => {
  const d = e.target.closest("[data-book-del]");
  if (!d || !confirm("Supprimer ce livre audio ?")) return;
  try { await api(`/api/books/${d.dataset.bookDel}`, { method: "DELETE" }); renderLibrary(); } catch (err) { toast(err.message, "error"); }
});

