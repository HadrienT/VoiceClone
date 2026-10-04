/** Textes longs : éditeur phrase par phrase d'une génération (réécouter, corriger, régénérer une phrase). */
import { $, $$, esc, api, toast, busy } from "./util.js";

export async function renderSegments(container, historyId) {
  const item = await api(`/api/history/${historyId}`);
  const segs = item.segments || [];
  const stamp = Date.now();
  container.innerHTML = `<audio controls src="/api/history/${historyId}/audio?t=${stamp}"></audio>
    <div class="info"><span>${item.duration} s · ${segs.filter((s) => s.type === "text").length} phrases${item.watermark ? " · filigrane ✓" : ""}</span>
      <span><a href="/api/history/${historyId}/audio" download>⬇ WAV</a> · <a href="/api/history/${historyId}/audio?format=mp3" download>⬇ MP3</a></span></div>
    <details class="segments" open><summary>Phrase par phrase — corrigez une phrase et régénérez-la seule</summary>
    <ol class="seg-list">${segs.map((s, i) => s.type === "pause"
      ? `<li class="seg-pause">⏸ pause ${s.seconds} s</li>`
      : `<li class="sent" data-seg="${i}">
          <button class="btn small" data-seg-play="${i}" title="Écouter cette phrase">▶</button>
          <input value="${esc(s.text)}" data-seg-text="${i}">
          <button class="btn small" data-seg-regen="${i}" title="Régénérer cette phrase">↻</button>
          ${s.regenerated ? `<small class="muted">×${s.regenerated}</small>` : ""}
        </li>`).join("")}</ol></details>`;
  container.classList.remove("hidden");
  container.dataset.historyId = historyId;
}

let segAudio;
document.addEventListener("click", (e) => {
  const play = e.target.closest("[data-seg-play]");
  const regen = e.target.closest("[data-seg-regen]");
  const box = e.target.closest("[data-history-id]");
  if (!box || !(play || regen)) return;
  const hid = box.dataset.historyId;
  if (play) {
    segAudio?.pause();
    segAudio = new Audio(`/api/history/${hid}/segments/${play.dataset.segPlay}/audio?t=${Date.now()}`);
    segAudio.play();
    return;
  }
  const i = regen.dataset.segRegen;
  const text = $(`[data-seg-text="${i}"]`, box).value.trim();
  busy(regen, "", async () => {
    try {
      await api(`/api/history/${hid}/segments/${i}`, { json: { text } });
      await renderSegments(box, hid);
      $(`[data-seg="${i}"]`, box)?.classList.add("flash");
      toast("Phrase régénérée et réassemblée.", "ok");
    } catch (err) { toast(err.message, "error"); }
  });
});
document.addEventListener("keydown", (e) => {
  const input = e.target.closest?.("[data-seg-text]");
  if (input && e.key === "Enter") {
    e.preventDefault();
    $(`[data-seg-regen="${input.dataset.segText}"]`, input.closest("[data-history-id]")).click();
  }
});
export const segmentRows = (box) => $$(".sent", box).length;
