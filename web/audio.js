// Outils audio côté navigateur : enregistrement micro, décodage, encodage WAV.

/** Encode un Float32Array mono en WAV PCM 16 bits. */
export function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

/**
 * Décode n'importe quel fichier audio/vidéo lisible par le navigateur et le convertit en WAV mono.
 * Renvoie { blob, duration } ou lève une erreur si le navigateur ne sait pas le décoder.
 */
export async function toWav(blob, targetRate = 24000) {
  const data = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(data);
  } finally {
    ctx.close();
  }
  const length = Math.ceil(decoded.duration * targetRate);
  const offline = new OfflineAudioContext(1, Math.max(1, length), targetRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination); // le mixage stéréo -> mono est fait par le contexte
  src.start();
  const rendered = await offline.startRendering();
  return { blob: encodeWav(rendered.getChannelData(0), targetRate), duration: decoded.duration };
}

export async function listMicrophones() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "audioinput");
}

/** Valeur spéciale de source : le son de l'ordinateur au lieu d'un micro. */
export const SYSTEM_SOURCE = "__system__";

/** Capture du son système possible dans ce navigateur ? (Chrome / Edge ; pas Firefox ni Safari) */
export const canCaptureSystemAudio = () => Boolean(navigator.mediaDevices?.getDisplayMedia) && !/firefox/i.test(navigator.userAgent);

async function openMicrophone(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Le micro n'est accessible que via localhost ou HTTPS (utilisez le tunnel SSH).");
  }
  return navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      // Audio brut : les traitements du navigateur dégradent le clonage
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });
}

/**
 * Capture le son qui sort de l'ordinateur (vidéo, Discord, jeu…) via le partage d'écran.
 * Sous Windows avec Chrome/Edge : choisir « Écran entier » puis cocher « Partager l'audio du système ».
 * L'image n'est jamais utilisée : la piste vidéo est coupée immédiatement.
 */
async function captureSystemAudio() {
  if (!canCaptureSystemAudio()) {
    throw new Error("Capture du son du PC impossible dans ce navigateur : utilisez Chrome ou Edge (via localhost / tunnel SSH).");
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: true, // obligatoire pour que le navigateur propose l'audio
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: false },
      systemAudio: "include",
      selfBrowserSurface: "exclude",
      surfaceSwitching: "exclude",
      monitorTypeSurfaces: "include",
    });
  } catch (err) {
    if (err.name === "NotAllowedError") throw new Error("Partage annulé.");
    throw err;
  }
  stream.getVideoTracks().forEach((t) => t.stop());
  if (!stream.getAudioTracks().length) {
    throw new Error("Aucun son partagé : choisissez « Écran entier » et cochez « Partager l'audio du système » (ou un onglet avec « Partager l'audio de l'onglet »).");
  }
  return new MediaStream(stream.getAudioTracks());
}

/** Enregistreur (micro ou son du PC) avec vumètre. */
export class Recorder {
  constructor({ onLevel, onTick, onEnded } = {}) {
    this.onLevel = onLevel || (() => {});
    this.onTick = onTick || (() => {});
    this.onEnded = onEnded || (() => {});
    this.recording = false;
  }

  /**
   * Démarre l'enregistrement.
   * - deviceId : micro à utiliser
   * - SYSTEM_SOURCE : son de l'ordinateur (ce qui sort dans le casque), capturé via le partage d'écran
   */
  async start(deviceId) {
    this.stream = deviceId === SYSTEM_SOURCE ? await captureSystemAudio() : await openMicrophone(deviceId);
    // L'utilisateur peut couper le partage depuis la barre du navigateur : on arrête proprement
    this.stream.getAudioTracks()[0].addEventListener("ended", () => this.recording && this.onEnded());
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    source.connect(this.analyser);

    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"]
      .find((m) => window.MediaRecorder?.isTypeSupported?.(m));
    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
    this.mediaRecorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.mediaRecorder.start(250);
    this.recording = true;
    this.startedAt = performance.now();
    this._loop();
  }

  _loop() {
    if (!this.recording) return;
    const buf = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buf);
    let peak = 0;
    for (const v of buf) peak = Math.max(peak, Math.abs(v));
    this.onLevel(peak);
    this.onTick((performance.now() - this.startedAt) / 1000);
    this._raf = requestAnimationFrame(() => this._loop());
  }

  /** Arrête et renvoie { blob (WAV), duration }. */
  async stop() {
    if (!this.recording) return null;
    this.recording = false;
    cancelAnimationFrame(this._raf);
    const done = new Promise((resolve) => (this.mediaRecorder.onstop = resolve));
    this.mediaRecorder.stop();
    await done;
    this.stream.getTracks().forEach((t) => t.stop());
    this.ctx.close();
    this.onLevel(0);
    const raw = new Blob(this.chunks, { type: this.mediaRecorder.mimeType || "audio/webm" });
    try {
      return await toWav(raw);
    } catch {
      return { blob: raw, duration: (performance.now() - this.startedAt) / 1000 };
    }
  }
}

export function fmtTime(sec) {
  const s = Math.floor(sec);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
