/** Utilitaires partagés par les modules de l'interface. */

export const $ = (sel, el = document) => el.querySelector(sel);
export const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const LANGS = {
  auto: "Détection auto", fr: "Français", en: "Anglais", es: "Espagnol", de: "Allemand", it: "Italien",
  pt: "Portugais", pl: "Polonais", tr: "Turc", ru: "Russe", nl: "Néerlandais", cs: "Tchèque", ar: "Arabe",
  "zh-cn": "Chinois", zh: "Chinois", hu: "Hongrois", ko: "Coréen", ja: "Japonais", hi: "Hindi", da: "Danois",
  el: "Grec", fi: "Finnois", he: "Hébreu", ms: "Malais", no: "Norvégien", sv: "Suédois", sw: "Swahili",
};
export const langName = (c) => LANGS[c] || c;

export const store = {
  get(k, d) { try { const v = localStorage.getItem("vc." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("vc." + k, JSON.stringify(v)); } catch { /* stockage indisponible */ } },
};

export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const res = await fetch(path, init);
  if (res.status === 401 && !path.startsWith("/api/login")) {
    location.href = `/login.html?next=${encodeURIComponent(location.pathname + location.hash)}`;
    throw new Error("Connexion requise");
  }
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

export function toast(msg, type = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms);
}

export const fmtBytes = (b) => {
  if (!b) return "0 o";
  const u = ["o", "Ko", "Mo", "Go"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1024)));
  return `${(b / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
};

export async function busy(btn, label, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${label}`;
  try { return await fn(); } finally { btn.disabled = false; btn.innerHTML = old; }
}

