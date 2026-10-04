import numpy as np

from voiceclone import enhance, prep


def _calls(monkeypatch, df=True, dm=True, fail=None):
    calls = []
    monkeypatch.setattr(enhance, "available", lambda: {"auto": True, "spectral": True, "deepfilter": df,
                                                        "demucs": dm, "demucs+deepfilter": df and dm})

    def fake(name):
        def f(x, sr):
            calls.append(name)
            if fail == name:
                raise RuntimeError("boum")
            return x * 0.5
        return f

    monkeypatch.setattr(enhance, "deepfilter", fake("deepfilter"))
    monkeypatch.setattr(enhance, "demucs_vocals", fake("demucs"))

    def spectral(x, sr, strength):
        calls.append("spectral")
        return x
    return calls, spectral


def test_auto_prefers_deepfilter(monkeypatch):
    calls, spectral = _calls(monkeypatch)
    _, info = enhance.run(np.ones(100, np.float32), 16000, "auto", spectral)
    assert calls == ["deepfilter"] and info["method"] == "deepfilter" and "fallback" not in info


def test_auto_without_deepfilter_uses_spectral(monkeypatch):
    calls, spectral = _calls(monkeypatch, df=False)
    _, info = enhance.run(np.ones(100, np.float32), 16000, "auto", spectral)
    assert calls == ["spectral"] and info["method"] == "spectral"


def test_chain_and_fallbacks(monkeypatch):
    calls, spectral = _calls(monkeypatch)
    enhance.run(np.ones(100, np.float32), 16000, "demucs+deepfilter", spectral)
    assert calls == ["demucs", "deepfilter"]
    calls, spectral = _calls(monkeypatch, dm=False)
    _, info = enhance.run(np.ones(100, np.float32), 16000, "demucs", spectral)
    assert calls == ["spectral"] and "pip install demucs" in info["fallback"]
    calls, spectral = _calls(monkeypatch, fail="deepfilter")
    _, info = enhance.run(np.ones(100, np.float32), 16000, "demucs+deepfilter", spectral)
    assert calls == ["demucs", "deepfilter", "spectral"] and "boum" in info["fallback"]


def test_clean_applies_explicit_method_even_on_clean_audio(monkeypatch):
    calls, _ = _calls(monkeypatch)
    sr = 16000
    t = np.arange(sr * 3) / sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)  # silences parfaits
    prep.clean(x, sr, True, "auto")
    assert calls == []  # rien à faire en automatique
    _, info = prep.clean(x, sr, True, "demucs")
    assert calls == ["demucs"] and info["method"] == "demucs"


def test_methods_endpoint(client):
    ms = client.get("/api/enhance/methods").json()
    assert {m["id"] for m in ms} >= {"auto", "spectral", "deepfilter", "demucs"}
    assert next(m for m in ms if m["id"] == "spectral")["available"] is True
