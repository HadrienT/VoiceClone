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

export class BrowserLive {
  constructor({ onStatus, onStopped } = {}) {
    this.onStatus = onStatus || (() => {});
    this.onStopped = onStopped || (() => {});
    this.running = false;
  }

  /**
   * @param config   réglages envoyés au serveur (mode, model_id, voice_id…)
   * @param devices  { input, output, monitor } : deviceId du micro, de la sortie Discord, du retour casque
   */
  async start(config, devices) {
    if (this.running) await this.stop();
    this.stopping = false;
    this.lastStatus = { state: "starting" };

    // 1. Micro (audio brut : les traitements du navigateur dégradent la conversion)
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: devices.input ? { exact: devices.input } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1,
      },
    });
    this.capCtx = new AudioContext({ latencyHint: "interactive" });
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

    // 3. WebSocket vers le serveur
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/api/realtime/ws`);
    this.ws.binaryType = "arraybuffer";
    const started = new Promise((resolve, reject) => {
      this.ws.onopen = () => this.ws.send(JSON.stringify({ ...config, sample_rate: this.capCtx.sampleRate }));
      this.ws.onmessage = (e) => {
        if (typeof e.data !== "string") return this._play(e.data);
        const msg = JSON.parse(e.data);
        if (msg.type === "started") { this.outRate = msg.out_sample_rate; this.running = true; resolve(); }
        else if (msg.type === "error") { if (this.running) this._fail(msg.detail); else reject(new Error(msg.detail)); }
        else if (msg.type === "status") { this.lastStatus = msg; this.onStatus(this.status()); }
        else if (msg.type === "loading") { this.onStatus({ ...this.status(), state: "starting" }); }
      };
      this.ws.onerror = () => reject(new Error("Connexion WebSocket impossible avec le serveur."));
      this.ws.onclose = () => {
        if (this.running && !this.stopping) this._fail("Connexion avec le serveur perdue.");
        else reject(new Error("Le serveur a fermé la connexion."));
      };
    });

    // Envoi du micro par blocs de ~20 ms en PCM 16 bits
    const block = Math.round(this.capCtx.sampleRate * 0.02);
    let pending = new Float32Array(0);
    this.capNode.port.onmessage = (e) => {
      if (!this.running || this.ws.readyState !== WebSocket.OPEN) return;
      const merged = new Float32Array(pending.length + e.data.length);
      merged.set(pending);
      merged.set(e.data, pending.length);
      let off = 0;
      for (; off + block <= merged.length; off += block) {
        const pcm = new Int16Array(block);
        for (let i = 0; i < block; i++) pcm[i] = Math.max(-1, Math.min(1, merged[off + i])) * 0x7fff;
        this.ws.send(pcm.buffer);
      }
      pending = merged.slice(off);
    };

    try {
      await started;
    } catch (err) {
      await this.stop();
      throw err;
    }
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

  _play(buffer) {
    const pcm = new Int16Array(buffer);
    if (!pcm.length) return;
    const audioBuf = this.playCtx.createBuffer(1, pcm.length, this.outRate || 24000);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
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
    const buffer = this.playCtx ? Math.max(0, this.playTime - this.playCtx.currentTime) : 0;
    return { ...this.lastStatus, buffer_ms: Math.round(buffer * 1000) };
  }

  _fail(message) {
    this.stop();
    this.onStopped(message);
  }

  async stop() {
    this.stopping = true;
    this.running = false;
    try { if (this.ws?.readyState === WebSocket.OPEN) this.ws.send("stop"); } catch { /* déjà fermé */ }
    try { this.ws?.close(); } catch { /* déjà fermé */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    for (const el of [this.outEl, this.monEl]) { if (el) { el.pause(); el.srcObject = null; } }
    await Promise.allSettled([this.capCtx?.close(), this.playCtx?.close()]);
    this.ws = this.stream = this.capCtx = this.playCtx = this.outEl = this.monEl = null;
  }
}
