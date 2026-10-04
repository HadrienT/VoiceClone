import numpy as np

from voiceclone import prep

SR = 24000


def phrase(d, amp=0.25):
    t = np.arange(int(d * SR)) / SR
    env = (0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)) ** 2
    return amp * env * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6))


def recording(kinds, hiss=0.01, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for k in kinds:
        x = phrase(4.5)
        if k == "bruité":
            x = x + 0.08 * rng.standard_normal(len(x))
        if k == "saturé":
            x = np.clip(x * 6, -1, 1)
        parts += [x, np.zeros(SR // 2)]
    x = np.concatenate(parts + [np.zeros(3 * SR)])
    return (x + hiss * rng.standard_normal(len(x))).astype(np.float32)


def test_denoise_lowers_noise_floor_and_keeps_speech():
    x = recording(["propre"] * 3)
    y, info = prep.clean(x, SR)
    assert info["denoised"]
    assert info["noise_floor_db_after"] < info["noise_floor_db_before"] - 10
    assert abs(prep.speech_level_db(y, SR) - prep.speech_level_db(x, SR)) < 3  # la voix n'est pas écrasée


def test_clean_recording_is_not_denoised():
    _, info = prep.clean(recording(["propre"] * 3, hiss=0.0), SR)
    assert not info["denoised"]


def test_auto_prepare_rejects_bad_parts_and_orders_best_first():
    kinds = ["propre", "bruité", "propre", "saturé", "propre"]
    kept, report = prep.auto_prepare(recording(kinds), SR, target_s=60)
    verdicts = [s["reason"] or "gardé" for s in report["segments"]]
    assert verdicts == ["gardé", "trop de bruit", "gardé", "son saturé", "gardé"]
    assert len(kept) == 3 and len(report["kept_order"]) == 3
    scores = {s["index"]: s["score"] for s in report["segments"]}
    assert [scores[i] for i in report["kept_order"]] == sorted((scores[i] for i in report["kept_order"]), reverse=True)
    assert report["segments"][-1]["end"] < report["duration"] - 2  # silence final retiré


def test_target_duration_is_respected():
    kept, report = prep.auto_prepare(recording(["propre"] * 6), SR, target_s=10)
    assert 4 <= report["kept_duration"] <= 10 and len(kept) == 2


def test_split_still_works_for_scripts():
    parts = prep.split(recording(["propre"] * 6), SR, 4, 11)
    assert all(len(p) / SR <= 11.05 for p in parts) and len(parts) >= 3
