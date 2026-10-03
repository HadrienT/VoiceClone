import threading
import time

import numpy as np

from voiceclone.manager import EngineManager
from voiceclone.realtime import AudioFifo, RealtimeConfig, RealtimeSession
from voiceclone.voices import VoiceStore
from tests.conftest import make_voice_wav


def test_fifo_read_and_overflow():
    f = AudioFifo(1.0, 100)
    f.push(np.ones(60, dtype=np.float32))
    f.push(np.ones(60, dtype=np.float32) * 2)  # dépasse 100 : le premier bloc est jeté
    assert f.seconds == 0.6
    out = f.read(80)
    assert out[:60].tolist() == [2] * 60 and out[60:].tolist() == [0] * 20


def _session(fake_model, mode):
    store = VoiceStore()
    v = store.create("Live")
    store.add_sample(v.id, make_voice_wav(6.0))
    s = RealtimeSession(EngineManager(), store)
    s.cfg = RealtimeConfig(mode=mode, model_id="fake", asr_model_id="fake", voice_id=v.id,
                           chunk_ms=500, context_ms=200, crossfade_ms=20, silence_db=-50, end_silence_ms=300)
    s._prepare_models(s.cfg)
    s.in_sr, s.out_sr = 16000, 48000
    s._out = AudioFifo(30.0, s.out_sr)
    return s


def _feed(s, signal, block=320, realtime=False):
    for i in range(0, len(signal), block):
        s._in_q.put(signal[i:i + block].astype(np.float32))
        if realtime:
            time.sleep(block / s.in_sr)


def test_vc_loop_keeps_timing(fake_model):
    s = _session(fake_model, "vc")
    sr = s.in_sr
    t = np.arange(sr * 2) / sr
    speech = 0.3 * np.sin(2 * np.pi * 200 * t)
    th = threading.Thread(target=s._vc_loop)
    th.start()
    _feed(s, np.concatenate([speech, np.zeros(sr)]), realtime=True)  # 2 s de voix + 1 s de silence
    time.sleep(0.2)
    s._stop.set()
    th.join()
    out_seconds = s._out.seconds
    assert s.chunks >= 4 and s.dropped == 0
    # la sortie a ~ la même durée que la parole d'entrée (+ au plus un morceau de « traîne »)
    assert 1.9 < out_seconds < 2.6, out_seconds


def test_asr_loop_transcribes_utterance(fake_model):
    s = _session(fake_model, "asr_tts")
    sr = s.in_sr
    t = np.arange(sr) / sr
    _feed(s, np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 200 * t), np.zeros(sr)]))
    th = threading.Thread(target=s._asr_loop)
    th.start()
    time.sleep(1.0)
    s._stop.set()
    th.join()
    for t2 in s._threads:
        t2.join()
    assert [x["text"] for x in s.transcripts] == ["bonjour tout le monde"]
    assert s._out.seconds > 0
