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

/** Enregistreur micro avec vumètre. */
export class Recorder {
  constructor({ onLevel, onTick } = {}) {
    this.onLevel = onLevel || (() => {});
    this.onTick = onTick || (() => {});
    this.recording = false;
  }

  async start(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Le micro n'est accessible que via localhost ou HTTPS.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        // Audio brut : les traitements du navigateur dégradent le clonage
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
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
