"""Opt-in native GUI/model/USB integration. Generated inputs, no saved mic audio."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import json
import math
import time
import numpy as np
from scipy.signal import resample_poly
from piper import PiperVoice, SynthesisConfig
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from translitter.ui import Window

(root / ".local").mkdir(exist_ok=True)
fixtures = []
for language, filename, phrase in [
    ("fi", "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx", "Hei, miten voit tänään?"),
    ("en", "en/en_US/lessac/medium/en_US-lessac-medium.onnx", "Hello, how are you today?"),
]:
    voice = PiperVoice.load(root / "models" / f"voice-{language}" / filename, use_cuda=False)
    chunks = list(voice.synthesize(phrase, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)))
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    rate = chunks[0].sample_rate
    divisor = math.gcd(rate, 16000)
    samples = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    fixtures.append(np.pad(samples, (3200, 6400)))
app = QApplication([])
app.setApplicationName("Translitter GUI check")
window = Window(root)
# Count real USB capture in AudioHub, but only generated fixtures may enter the
# captioned test conversation, including while a PTT key is held across inference.
window.audio.samples.disconnect(window._samples)
window.audio.samples.connect(
    lambda speaker, samples: window._samples(speaker, samples)
    if any(samples is fixture for fixture in fixtures) else None
)
window.show()
evidence = {"input_kind": "generated_speech_through_native_PTT_events", "turns": [], "errors": []}
state = {"step": 0, "deadline": time.monotonic() + 45, "done": False}


def abort(message):
    print("GUI_CHECK_FAILED", state["step"], message, flush=True)
    evidence["errors"].append(message)
    window.close()
    timer.stop()
    app.exit(1)


def key_event(speaker, pressed):
    # Send Qt events directly: XCB QTest native key injection is asynchronous
    # and did not reliably deliver simultaneous F-key holds in this harness.
    key = Qt.Key_F1 if speaker == 0 else Qt.Key_F2
    kind = QEvent.KeyPress if pressed else QEvent.KeyRelease
    QApplication.sendEvent(window.cards[speaker].original, QKeyEvent(kind, key, Qt.NoModifier))


def speak(speaker):
    key_event(speaker, True)
    assert window._held[speaker], f"Key press not handled for User {speaker + 1}"
    window.audio.samples.emit(speaker, fixtures[speaker])
    assert window._sizes[speaker] == len(fixtures[speaker]), f"Fixture not captured for User {speaker + 1}"
    key_event(speaker, False)
    assert not window._held[speaker] and window._busy[speaker] == 1, f"Key release did not submit User {speaker + 1}'s fixture"


def check():
    try:
        assert time.monotonic() < state["deadline"], "GUI verification timed out"
        if state["step"] == 0:
            if not window.models_ready or len(window.audio.headsets()) != 2:
                return
            window.refresh_devices()
            assert window.start_button.isEnabled()
            QTest.mouseClick(window.start_button, Qt.LeftButton)
            assert window.active and len(window.audio.sources) == 2
            window.audio.set_volume(0.30)
            evidence["headsets"] = [selector.currentText() for selector in window.selectors]
            # Hold User 2's generated English turn while User 1's Finnish
            # translation arrives: this is the independently reported overlap.
            key_event(1, True)
            window.audio.samples.emit(1, fixtures[1])
            assert window._held[1] and window._sizes[1] == len(fixtures[1]), f"Fixture capture: held={window._held[1]}, stored={window._sizes[1]}, expected={len(fixtures[1])}"
            speak(0)
            state["step"] = 1
        elif state["step"] == 1:
            if not window.cards[0].translated.toPlainText():
                return
            assert window._held[1]
            assert window._sizes[1] == len(fixtures[1]), "Incoming translation erased held speech"
            assert len(window._deferred[1]) == 1
            assert not window.audio.buffers and window.audio.played_bytes[1] == 0
            evidence["overlap"] = {"preserved_samples": window._sizes[1],
                "deferred_translations": len(window._deferred[1]), "playback_before_release": False}
            key_event(1, False)
            assert window._busy[1] == 1, "Release did not submit preserved English speech"
            state["step"] = 2
        elif state["step"] == 2:
            if not window.cards[1].translated.toPlainText() or window.audio.buffers:
                return
            for speaker in (0, 1):
                listener = 1 - speaker
                assert window.audio.played_bytes[listener] > 0
                assert not window._playback[listener] and not window._deferred[listener]
                evidence["turns"].append({"speaker": speaker, "listener": listener,
                    "original": window.cards[speaker].original.toPlainText(),
                    "translation": window.cards[speaker].translated.toPlainText(),
                    "pcm_bytes": window.audio.played_bytes[listener]})
            assert all(size > 0 for size in window.audio.capture_bytes)
            evidence["microphone_bytes_received"] = window.audio.capture_bytes.copy()
            assert window.grab().save(str(root / ".local/gui-live-check.png"))
            QTest.mouseClick(window.stop_button, Qt.LeftButton)
            assert not window.active and not window.audio.sources and not window.audio.sinks
            QTest.mouseClick(window.start_button, Qt.LeftButton)
            assert window.active
            # A second real translation waits behind User 1's held synthetic
            # input. Focus loss must cancel that input and drain the translation.
            key_event(0, True)
            window.audio.samples.emit(0, fixtures[0])
            assert window._held[0] and window._sizes[0] == len(fixtures[0])
            speak(1)
            state["step"] = 3
        elif state["step"] == 3:
            if not window._deferred[0]:
                return
            assert window._held[0] and window._sizes[0] == len(fixtures[0])
            QApplication.sendEvent(window, QEvent(QEvent.WindowDeactivate))
            assert not window._held[0] and not window._deferred[0]
            assert window._sizes[0] == 0 and window._busy[0] == 0
            assert "cancelled" in window.status.text().lower()
            evidence["focus_loss"] = {"cancelled_generated_samples": len(fixtures[0]),
                "partial_turn_submitted": False, "deferred_queue_drained": True,
                "status": window.status.text()}
            QApplication.sendEvent(window, QEvent(QEvent.WindowActivate))
            key_event(0, False)
            assert window._busy[0] == 0, "Late release submitted cancelled speech"
            state["step"] = 4
        elif state["step"] == 4:
            if window.audio.buffers:
                return
            assert window.audio.played_bytes[0] > 0
            evidence["focus_loss"]["completed_playback_bytes"] = window.audio.played_bytes[0]
            window.text_only.setChecked(True)
            assert window.submit_audio(0, np.zeros(16000, dtype=np.float32))
            state["step"] = 5
        elif state["step"] == 5:
            if window._busy[0]:
                return
            assert not window.audio.buffers
            assert "Busy" not in window.cards[0].state.text()
            QTest.mouseClick(window.stop_button, Qt.LeftButton)
            assert not window.active
            window.close()
            state["step"] = 6
        elif state["step"] == 6:
            if window.isVisible() or window.worker.isRunning():
                return
            state["done"] = True
            timer.stop()
            (root / ".local/gui-integration.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
            print("PASS_NATIVE_GUI_BOTH_PIPELINES_STOP_RESTART_AND_SILENCE", flush=True)
            app.quit()
    except Exception as exc:
        abort(f"{type(exc).__name__}: {exc}")

window.worker.error.connect(lambda generation, message: abort(message))
window.audio.failure.connect(lambda speaker, message: abort(message))
# Keep the event loop alive while testing asynchronous close.
app.setQuitOnLastWindowClosed(False)
timer = QTimer()
timer.timeout.connect(check)
timer.start(60)
app.exec()
window.audio.stop()
window.worker.shutdown()
assert window.worker.wait(15000), "Model thread failed to close"
assert state["done"] and not evidence["errors"], evidence
