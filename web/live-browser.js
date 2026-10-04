// Live « via le navigateur » : le micro de CE PC est envoyé au serveur (WebSocket), la voix convertie
// revient et est jouée sur la sortie choisie (ex. CABLE Input de VB-CABLE, que Discord prend comme micro).
// Indispensable quand VoiceClone tourne sur un serveur distant sans carte son.

const CAPTURE_WORKLET = `
class VoiceCloneCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("voiceclone-capture", VoiceCloneCapture);
`;

const VIRTUAL_HINTS = ["cable", "vb-audio", "voicemeeter", "blackhole", "virtual", "loopback"];
export const isVirtualDevice = (label) => VIRTUAL_HINTS.some((h) => label.toLowerCase().includes(h));

/** Le navigateur sait-il choisir la sortie audio (setSinkId) ? Chrome/Edge oui, Firefox récent oui. */
export const canChooseOutput = () => typeof HTMLMediaElement !== "undefined" && "setSinkId" in HTMLMediaElement.prototype;

/** État de l'autorisation micro : "granted", "prompt", "denied" ou "unknown". */
export async function micPermission() {
  try {
    return (await navigator.permissions.query({ name: "microphone" })).state;
  } catch {
    return "unknown"; // navigateur sans l'API Permissions pour le micro
  }
}

/** Demande l'accès au micro (à appeler depuis un clic : Chrome affiche alors sa fenêtre à coup sûr). */
export async function requestMic() {
  const s = await navigator.mediaDevices.getUserMedia({ audio: true });
  s.getTracks().forEach((t) => t.stop());
}

/** Liste les entrées / sorties audio du PC (noms visibles seulement une fois le micro autorisé). */
export async function listBrowserDevices() {
  if (!window.isSecureContext || !navigator.mediaDevices?.enumerateDevices) {
    throw new Error("NOT_SECURE");
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const pick = (kind) => devices.filter((d) => d.kind === kind && d.deviceId && d.deviceId !== "communications")
    .map((d, i) => ({ id: d.deviceId, label: d.label || `${kind} ${i + 1}`, virtual: isVirtualDevice(d.label) }));
  return { inputs: pick("audioinput"), outputs: pick("audiooutput"), labeled: devices.some((d) => d.label) };
}

/** Opus via WebCodecs disponible pour cette fréquence ? (Chrome/Edge récents) */
async function opusSupported(sampleRate) {
  if (!("AudioEncoder" in window) || !("AudioDecoder" in window)) return false;
  if (![8000, 12000, 16000, 24000, 48000].includes(sampleRate)) return false;
  try {
    const enc = await AudioEncoder.isConfigSupported({ codec: "opus", sampleRate, numberOfChannels: 1, bitrate: 32000 });
    const dec = await AudioDecoder.isConfigSupported({ codec: "opus", sampleRate: 48000, numberOfChannels: 1 });
    return Boolean(enc.supported && dec.supported);
  } catch {
    return false;
  }
}

const dbOf = (x) => {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return 10 * Math.log10(sum / Math.max(1, x.length) + 1e-12);
};

export class BrowserLive {
  constructor({ onStatus, onStopped } = {}) {
    this.onStatus = onStatus || (() => {});
    this.onStopped = onStopped || (() => {});
    this.running = false;
    this.micMode = "open"; // open | ptt | mute
    this.pttDown = false;
    this.gateDb = -90; // porte de bruit navigateur (dB) : sous ce niveau, on n'envoie que du « silence »
    this.rttMs = 0;
    this.codec = "pcm";
  }

  /** Le micro est-il « ouvert » à cet instant (mode + touche) ? */
  micOpen() {
    return this.micMode === "open" || (this.micMode === "ptt" && this.pttDown);
  }

  /**
   * @param config   réglages envoyés au serveur (mode, model_id, voice_id…)
   * @param devices  { input, output, monitor } : deviceId du micro, de la sortie Discord, du retour casque
   * @param opts     { opus: bool } : compresser le flux si possible
   */
  async start(config, devices, opts = {}) {
    if (this.running) await this.stop();
    this._args = [config, devices, opts];
    this.stopping = false;
    this.lastStatus = { state: "starting" };
    await this._openAudio(devices, opts.opus !== false);
    this.codec = opts.opus !== false && (await opusSupported(this.capCtx.sampleRate)) ? "opus" : "pcm";
    await this._connect(config);
  }

  async _openAudio(devices, opus = false) {
    // 1. Micro (audio brut : les traitements du navigateur dégradent la conversion)
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: devices.input ? { exact: devices.input } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1,
      },
    });
    // Opus n'accepte que 8/12/16/24/48 kHz : on fixe 48 kHz (le navigateur rééchantillonne le micro)
    this.capCtx = new AudioContext(opus ? { latencyHint: "interactive", sampleRate: 48000 } : { latencyHint: "interactive" });
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await this.capCtx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const src = this.capCtx.createMediaStreamSource(this.stream);
    this.capNode = new AudioWorkletNode(this.capCtx, "voiceclone-capture");
    const mute = this.capCtx.createGain();
    mute.gain.value = 0; // le nœud doit être relié à la sortie pour tourner, mais sans bruit
    src.connect(this.capNode).connect(mute).connect(this.capCtx.destination);

    // 2. Lecture : contexte -> flux -> <audio> routé vers la sortie choisie (et le casque)
    this.playCtx = new AudioContext({ latencyHint: "interactive" });
    this.playDest = this.playCtx.createMediaStreamDestination();
    this.playTime = 0;
    this.outEl = await this._sinkElement(devices.output);
    this.monEl = devices.monitor && devices.monitor !== devices.output ? await this._sinkElement(devices.monitor) : null;
  }

  async _connect(config) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/realtime/ws`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    const started = new Promise((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify({ ...config, sample_rate: this.capCtx.sampleRate, codec: this.codec }));
      ws.onmessage = (e) => {
        if (this.stopping || ws !== this.ws) return; // messages en vol après l'arrêt / ancienne connexion
        if (typeof e.data !== "string") return this._receiveAudio(e.data);
        const msg = JSON.parse(e.data);
        if (msg.type === "started") {
          this.outRate = msg.out_sample_rate;
          this.codec = msg.codec || "pcm";
          this._setupCodecs();
          this.running = true;
          this.reconnects = 0;
          resolve();
        } else if (msg.type === "error") {
          if (this.running) this._fail(msg.detail); else reject(new Error(msg.detail));
        } else if (msg.type === "status") {
          this.lastStatus = msg;
          this.onStatus(this.status());
        } else if (msg.type === "pong") {
          this.rttMs = Math.round(performance.now() - msg.t);
        } else if (msg.type === "notice") {
          this.onStatus({ ...this.status(), notice: msg.detail });
        } else if (msg.type === "loading") {
          this.onStatus({ ...this.status(), state: "starting" });
        }
      };
      ws.onerror = () => reject(new Error("Connexion WebSocket impossible avec le serveur."));
      ws.onclose = () => {
        if (ws !== this.ws) return;
        if (this.running && !this.stopping) this._reconnect(config);
        else reject(new Error("Le serveur a fermé la connexion."));
      };
    });
    this._startCapture();
    clearInterval(this.pingTimer);
    this.pingTimer = setInterval(() => this._send(JSON.stringify({ type: "ping", t: performance.now() })), 2000);
    await started;
  }

  /** Coupure réseau : on retente avec un délai croissant (1, 2, 4, 8, 8 s) sans fermer l'audio. */
  async _reconnect(config) {
    this.running = false;
    for (let attempt = 1; attempt <= 5 && !this.stopping; attempt++) {
      this.onStatus({ ...this.status(), state: "reconnecting", notice: `Connexion perdue, nouvel essai (${attempt}/5)…` });
      await new Promise((r) => setTimeout(r, Math.min(8000, 1000 * 2 ** (attempt - 1))));
      if (this.stopping) return;
      try {
        await this._connect(config);
        this.onStatus({ ...this.status(), notice: "Reconnecté ✓" });
        return;
      } catch { /* on réessaie */ }
    }
    if (!this.stopping) this._fail("Connexion avec le serveur perdue (5 essais).");
  }

  _setupCodecs() {
    this.encoder?.close?.();
    this.decoder?.close?.();
    this.encoder = this.decoder = null;
    if (this.codec !== "opus") return;
    this.encoder = new AudioEncoder({
      output: (chunk) => {
        const buf = new Uint8Array(chunk.byteLength);
        chunk.copyTo(buf);
        this._send(buf.buffer);
      },
      error: (e) => this._fail(`Encodeur Opus : ${e.message}`),
    });
    this.encoder.configure({ codec: "opus", sampleRate: this.capCtx.sampleRate, numberOfChannels: 1, bitrate: 32000 });
    this.decoder = new AudioDecoder({
      output: (data) => {
        const f = new Float32Array(data.numberOfFrames);
        data.copyTo(f, { planeIndex: 0, format: "f32-planar" });
        this._play(f, data.sampleRate);
        data.close();
      },
      error: (e) => this._fail(`Décodeur Opus : ${e.message}`),
    });
    this.decoder.configure({ codec: "opus", sampleRate: 48000, numberOfChannels: 1 });
    this.decTs = 0;
  }

  _send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(data);
  }

  /** Envoi du micro par blocs de 20 ms : PCM 16 bits ou Opus ; micro coupé = message « silence ». */
  _startCapture() {
    const sr = this.capCtx.sampleRate;
    const block = Math.round(sr * 0.02);
    let pending = new Float32Array(0);
    let silent = 0; // échantillons de silence pas encore signalés
    let ts = 0;
    const flushSilence = () => {
      if (silent) this._send(JSON.stringify({ type: "silence", n: silent }));
      silent = 0;
    };
    this.capNode.port.onmessage = (e) => {
      if (!this.running || this.ws?.readyState !== WebSocket.OPEN) return;
      const merged = new Float32Array(pending.length + e.data.length);
      merged.set(pending);
      merged.set(e.data, pending.length);
      let off = 0;
      for (; off + block <= merged.length; off += block) {
        const frame = merged.subarray(off, off + block);
        this.inDb = dbOf(frame);
        if (!this.micOpen() || this.inDb < this.gateDb) {
          silent += block;
          if (silent >= sr / 10) flushSilence(); // regroupé par 100 ms : quasiment aucun débit
          continue;
        }
        flushSilence();
        if (this.codec === "opus" && this.encoder?.state === "configured") {
          this.encoder.encode(new AudioData({ format: "f32", sampleRate: sr, numberOfFrames: block, numberOfChannels: 1, timestamp: ts, data: frame.slice() }));
          ts += Math.round((block / sr) * 1e6);
        } else {
          const pcm = new Int16Array(block);
          for (let i = 0; i < block; i++) pcm[i] = Math.max(-1, Math.min(1, frame[i])) * 0x7fff;
          this._send(pcm.buffer);
        }
      }
      pending = merged.slice(off);
    };
  }

  /** Fait dire un texte tapé à la voix clonée (mêlé au flux envoyé vers Discord). */
  say(text) {
    if (!this.running) throw new Error("Démarrez d'abord le Live.");
    this._send(JSON.stringify({ type: "say", text }));
  }

  async _sinkElement(deviceId) {
    const el = new Audio();
    el.srcObject = this.playDest.stream;
    if (deviceId) {
      if (!canChooseOutput()) throw new Error("Ce navigateur ne permet pas de choisir la sortie audio : utilisez Chrome ou Edge.");
      await el.setSinkId(deviceId);
    }
    await el.play();
    return el;
  }

  _receiveAudio(buffer) {
    if (this.codec === "opus") {
      if (this.decoder?.state !== "configured") return;
      this.decoder.decode(new EncodedAudioChunk({ type: "key", timestamp: this.decTs, data: buffer }));
      this.decTs += 20000;
      return;
    }
    const pcm = new Int16Array(buffer);
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
    this._play(f, this.outRate || 24000);
  }

  _play(samples, rate) {
    if (!samples.length || !this.playCtx) return;
    const audioBuf = this.playCtx.createBuffer(1, samples.length, rate);
    audioBuf.copyToChannel(samples, 0);
    const node = this.playCtx.createBufferSource();
    node.buffer = audioBuf;
    node.connect(this.playDest);
    const now = this.playCtx.currentTime;
    // petit tampon anti-coupures ; si on a pris trop de retard, on se recale sur le direct
    if (this.playTime < now + 0.03 || this.playTime > now + 1.5) this.playTime = now + 0.08;
    node.start(this.playTime);
    this.playTime += audioBuf.duration;
  }

  status() {
    const buffer = this.playCtx ? Math.max(0, this.playTime - this.playCtx.currentTime) * 1000 : 0;
    const server = this.lastStatus?.latency_ms || 0;
    return {
      ...this.lastStatus,
      buffer_ms: Math.round(buffer),
      rtt_ms: this.rttMs,
      codec: this.codec,
      mic_open: this.micOpen(),
      // micro → serveur → retour : réseau + traitement serveur + tampon de lecture + ~40 ms de sortie audio
      latency_ms: server ? Math.round(server + this.rttMs + buffer + 40) : 0,
    };
  }

  _fail(message) {
    this.stop();
    this.onStopped(message);
  }

  async stop() {
    this.stopping = true;
    this.running = false;
    clearInterval(this.pingTimer);
    try { if (this.ws?.readyState === WebSocket.OPEN) this.ws.send("stop"); } catch { /* déjà fermé */ }
    try { this.ws?.close(); } catch { /* déjà fermé */ }
    try { this.encoder?.close(); } catch { /* déjà fermé */ }
    try { this.decoder?.close(); } catch { /* déjà fermé */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    for (const el of [this.outEl, this.monEl]) { if (el) { el.pause(); el.srcObject = null; } }
    await Promise.allSettled([this.capCtx?.close(), this.playCtx?.close()]);
    this.ws = this.stream = this.capCtx = this.playCtx = this.outEl = this.monEl = this.encoder = this.decoder = null;
  }
}
