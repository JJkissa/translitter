"""Opt-in owned runtime and native worker integration. Synthetic speech only."""
import json
import math
import os
from pathlib import Path

from time import monotonic
from urllib.error import HTTPError
from urllib.request import Request

import numpy as np
import pytest
from test_audio import explicit_audio

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("TRANSLITTER_QUALITY_TESTS") != "1",
                              reason="Opt-in installed TranslateGemma runtime; no physical audio")


def test_owned_runtime_real_translation_browser_isolation_and_cleanup():
    from translitter.quality_runtime import QualityRuntime
    from translitter.quality import BASE_URL
    runtime = QualityRuntime(ROOT)
    directory = None
    try:
        client = runtime.start()
        directory = Path(runtime._socket_directory.name)
        assert directory.stat().st_mode & 0o777 == 0o700
        assert (directory / "api.sock").is_socket()
        assert "--port" not in runtime._process.args
        assert client.probe()
        greeting = client.translate("Hello, how are you today?", "en", "ru")
        assert "сегодня" in greeting.casefold(), greeting
        bread = client.translate("Tänään haluan ostaa leipä.", "fi", "en")
        assert "bread" in bread.casefold(), bread
        request = Request(BASE_URL + "/models", headers={"Origin": "https://untrusted.example"})
        try:
            with client._opener.open(request, timeout=3) as response:
                assert response.headers.get("Access-Control-Allow-Origin") not in ("*", "https://untrusted.example")
        except HTTPError as error:
            assert error.code in (400, 403)
        try:
            client._opener.open(BASE_URL.removesuffix("/v1") + "/slots", timeout=3)
        except HTTPError as error:
            assert error.code in (403, 404, 501)
        else:
            pytest.fail("Slot/prompt monitoring endpoint should not be exposed")
        print(json.dumps({"operation": "owned_runtime", "greeting": greeting, "partial_finnish": bread,
                          "browser_access": "blocked", "slot_endpoint": "disabled"}, ensure_ascii=False))
    finally:
        runtime.close()
    assert runtime._process is None and runtime._socket_directory is None
    assert directory is not None and not directory.exists()


def test_real_candidate_all_twelve_ordered_text_directions():
    from itertools import permutations
    from translitter.quality_runtime import QualityRuntime
    text = {"fi": "Haluan ostaa leipää tänään.", "en": "Today I want to buy bread.",
            "ru": "Сегодня я хочу купить хлеб.", "uk": "Сьогодні я хочу купити хліб."}
    anchors = {"fi": ("leip", "tänään"), "en": ("bread", "today"),
               "ru": ("хлеб", "сегодня"), "uk": ("хліб", "сьогодні")}
    runtime, rows = QualityRuntime(ROOT), []
    try:
        client = runtime.start()
        for source, target in permutations(text, 2):
            translated = client.translate(text[source], source, target)
            row = {"source": source, "target": target, "original": text[source],
                   "translation": translated, **client.last_metrics}
            rows.append(row)
            assert all(word in translated.casefold() for word in anchors[target]), row
    finally:
        runtime.close()
        (ROOT / ".local/quality-all-pairs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    assert len(rows) == 12 and runtime._process is None and runtime._socket_directory is None


@pytest.mark.parametrize("source,target,text,anchor,tail_anchor", [
    ("ru", "fi", "Сегодня я хочу купить хлеб. Завтра мы пойдем в магазин. Мой друг будет ждать возле вокзала.", "leip", "asem"),
    ("fi", "en", "Tänään haluan ostaa leipää. Huomenna menen kauppaan. Ystäväni odottaa asemalla.", "bread", "station"),
])
def test_native_worker_translates_complete_turn_only_after_release(explicit_audio, qtbot, monkeypatch, source, target, text, anchor, tail_anchor):
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly
    from translitter.languages import VOICE_FILES
    from translitter.worker import ModelWorker
    from translitter.engine import ModelEngine
    from translitter.ui import Window
    from translitter.quality_runtime import QualityRuntime
    managed = []
    runtime_start = QualityRuntime.start
    def own_start(runtime, *args, **kwargs):
        managed.append(runtime)
        return runtime_start(runtime, *args, **kwargs)
    monkeypatch.setattr(QualityRuntime, "start", own_start)
    recognition = []
    recognize = ModelEngine.recognize
    def observe(engine, samples, *args, **kwargs):
        assert not window._held[0], "Recognition itself must wait for PTT release"
        words = recognize(engine, samples, *args, **kwargs)
        recognition.append({"input_seconds": len(samples) / 16000, "words": words})
        return words
    monkeypatch.setattr(ModelEngine, "recognize", observe)
    voice = PiperVoice.load(ROOT / "models" / f"voice-{source}" / VOICE_FILES[source], use_cuda=False)
    chunks = list(voice.synthesize(text, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)))
    rate = chunks[0].sample_rate
    pcm = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
    divisor = math.gcd(rate, 16000)
    pcm = resample_poly(pcm, 16000 // divisor, rate // divisor).astype(np.float32)
    # Cross the former eight-second auto-submit boundary and hold through silence.
    pcm = np.pad(pcm, (1600, max(16000 * 2, 16000 * 9 - len(pcm))))
    hub, pairs, opened = explicit_audio
    worker = ModelWorker(ROOT)
    window = Window(ROOT, worker=worker, audio=hub)
    qtbot.addWidget(window)
    errors, results, early = [], [], []
    worker.error.connect(lambda generation, message: errors.append((generation, message)))
    started = monotonic()
    def collect(turn):
        results.append(turn)
        if window._held[0]:
            early.append({"seconds_from_press": monotonic() - started, "original": turn.original,
                          "translation": turn.translation, "listener": turn.listener})
    worker.result.connect(collect)
    try:
        qtbot.waitUntil(lambda: window.models_ready or bool(errors), timeout=45000)
        assert window.models_ready and not errors, errors
        for selector, pair in zip(window.selectors, pairs):
            selector.addItem(pair.label, pair)
            selector.setCurrentIndex(0)
        for selector, language in zip(window.language_selectors, (source, target)):
            selector.setCurrentIndex(selector.findData(language))
        window.translation_backend.setCurrentIndex(window.translation_backend.findData("translategemma"))
        assert not window.start_button.isEnabled() and not opened
        qtbot.waitUntil(lambda: window._translation_is_ready or bool(window._translation_problem), timeout=90000)
        assert window._translation_is_ready, window.status.text()
        assert not opened, "Backend selection must not open audio devices"
        window.start_session()
        assert window.active and window.mode.currentData() == "ptt"
        assert not hasattr(window, "live_ptt")
        window._press(0)
        started = monotonic()
        for index in range(0, len(pcm), 1600):
            hub.samples.emit(0, pcm[index:index + 1600])
            qtbot.wait(100)
            assert not recognition and not results and not errors
            assert not hub.played_bytes[1]
            assert not window.cards[0].original.toPlainText()
        assert window._held[0] and not early
        window._release(0)
        qtbot.waitUntil(lambda: bool(results) or bool(errors), timeout=30000)
        assert not errors, errors
        qtbot.waitUntil(lambda: bool(window.cards[0].translated.toPlainText()) and hub.played_bytes[1] > 0, timeout=5000)
        assert len(results) == 1 and len(recognition) == 1 and not early
        assert recognition[0]["input_seconds"] == len(pcm) / 16000
        assert [identifier for kind, identifier in opened if kind == "playback"] == [pairs[1].output_id]
        assert anchor in window.cards[0].translated.toPlainText().casefold()
        assert tail_anchor in window.cards[0].translated.toPlainText().casefold(), "Final released phrase was not translated"
        row = {"fixture": "realtime-fed deterministic Piper; real Window/worker/engine/AudioHub, fake physical Qt endpoints",
               "mode": "release-only PTT", "source": source, "target": target,
               "feed_seconds": len(pcm) / 16000, "early_results": early,
               "originals": [turn.original for turn in results], "translations": [turn.translation for turn in results],
               "recognition_snapshots": recognition}
        (ROOT / f".local/release-quality-{source}-{target}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False))
        window.stop_session()
        window.show()
        qtbot.wait(50)
        window.grab().save(str(ROOT / f".local/release-quality-{source}-{target}.png"))
    finally:
        window.close()
        qtbot.waitUntil(lambda: worker.isFinished(), timeout=45000)
        window.close()
    assert managed and all(runtime._process is None and runtime._socket_directory is None for runtime in managed)
