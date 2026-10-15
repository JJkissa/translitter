"""Authorized real USB routing + offline pipeline check. No mic audio is saved."""
import json
import math
import os
import subprocess
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from scipy.signal import resample_poly
from piper import PiperVoice
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from translitter.audio import AudioHub
from translitter.core import make_routes
from translitter.worker import ModelWorker

(root / ".local").mkdir(exist_ok=True)
fixtures = []
for language, filename, text in [
    ("fi", "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx", "Hei, miten voit tänään?"),
    ("en", "en/en_US/lessac/medium/en_US-lessac-medium.onnx", "Hello, how are you today?"),
]:
    voice = PiperVoice.load(root / "models" / f"voice-{language}" / filename, use_cuda=False)
    chunks = list(voice.synthesize(text))
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    rate = chunks[0].sample_rate
    divisor = math.gcd(rate, 16000)
    samples = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    fixtures.append(np.pad(samples, (3200, 6400)))
app = QApplication([])
app.setApplicationName("Translitter pipeline check")
hub = AudioHub()
worker = ModelWorker(root)
evidence = {"input_kind": "generated_speech_not_human_microphone", "turns": [], "errors": [], "playback_states": []}
state = {"done": False, "playing": None, "failed": False}


def fail(message):
    evidence["errors"].append(message)
    state["failed"] = True
    hub.stop()
    worker.shutdown()
    app.quit()


def inspect_graph(listener):
    try:
        graph = json.loads(subprocess.check_output(["pw-dump"], text=True))
        props = {int(entry["id"]): entry.get("info", {}).get("props", {}) for entry in graph}
        ours = {node for node, prop in props.items() if str(prop.get("application.process.id")) == str(os.getpid())}
        ours |= {node for node, prop in props.items() if prop.get("client.id") in ours}
        connections = []
        for entry in graph:
            if entry.get("type") != "PipeWire:Interface:Link":
                continue
            info = entry.get("info", {})
            left = int(info.get("output-node-id", info.get("props", {}).get("link.output.node", -1)))
            right = int(info.get("input-node-id", info.get("props", {}).get("link.input.node", -1)))
            if left in ours or right in ours:
                connections.append({"from": props.get(left, {}).get("node.name"),
                                    "to": props.get(right, {}).get("node.name")})
        outgoing = {row["to"] for row in connections if str(row["to"]).startswith("alsa_output.")}
        incoming = {row["from"] for row in connections if str(row["from"]).startswith("alsa_input.")}
        assert outgoing == {hub.output_ids[listener]}, (outgoing, hub.output_ids[listener])
        assert incoming == {route.microphone_id for route in hub.routes}, incoming
        evidence["turns"][-1]["verified_graph"] = connections
        print("ROUTE_ISOLATION_PASS", listener, hub.output_ids[listener], flush=True)
    except Exception as exc:
        fail(f"Graph verification: {type(exc).__name__}: {exc}")


def receive(turn):
    try:
        assert turn.generation == 1
        assert turn.original and turn.translation and turn.listener == 1 - turn.speaker
        evidence["turns"].append({"speaker": turn.speaker, "listener": turn.listener,
            "original": turn.original, "translation": turn.translation,
            "source": turn.source_language, "target": turn.target_language,
            "elapsed_seconds": turn.elapsed, "rate": turn.rate, "audio_samples": len(turn.audio)})
        state["playing"] = turn.listener
        assert hub.play(turn.listener, turn.audio, turn.rate)
        QTimer.singleShot(200, lambda: inspect_graph(turn.listener))
    except Exception as exc:
        fail(f"Turn verification: {type(exc).__name__}: {exc}")


def on_playing(listener, active):
    evidence["playback_states"].append([listener, active])
    if not active and state["playing"] == listener:
        state["playing"] = None
        if len(evidence["turns"]) == 1:
            QTimer.singleShot(350, lambda: worker.submit(1, 1, fixtures[1]))
        elif len(evidence["turns"]) == 2:
            QTimer.singleShot(350, finish)


def begin():
    try:
        pairs = hub.headsets()
        assert len(pairs) == 2
        routes = make_routes(*pairs)
        evidence["routes"] = [vars(route) for route in routes]
        hub.set_volume(0.30)
        hub.start(routes)
        worker.set_session(1, routes)
        assert worker.submit(1, 0, fixtures[0])
    except Exception as exc:
        fail(f"Start verification: {type(exc).__name__}: {exc}")


def finish():
    try:
        assert len(evidence["turns"]) == 2
        assert all(turn.get("verified_graph") for turn in evidence["turns"])
        assert all(size > 0 for size in hub.capture_bytes), hub.capture_bytes
        assert all(size > 0 for size in hub.played_bytes), hub.played_bytes
        assert not hub.buffers, "Playback did not fully complete"
        assert not evidence["errors"]
        evidence["microphone_bytes_received"] = hub.capture_bytes.copy()
        evidence["pcm_bytes_submitted"] = hub.played_bytes.copy()
        evidence["complete_playback"] = True
        hub.stop()
        worker.stop_session(2)
        worker.shutdown()
        state["done"] = True
        (root / ".local/usb-pipeline-check.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
        print("PASS_BOTH_TRANSLATED_PIPELINES_AND_ISOLATED_USB_PLAYBACK", flush=True)
        app.quit()
    except Exception as exc:
        fail(f"Finish verification: {type(exc).__name__}: {exc}")

hub.failure.connect(lambda n, message: fail(message))
hub.playing.connect(on_playing)
worker.error.connect(lambda generation, message: fail(message))
worker.loading.connect(lambda text: print("LOAD", text, flush=True))
worker.ready.connect(lambda: QTimer.singleShot(800, begin))
worker.result.connect(receive)
QTimer.singleShot(45000, lambda: fail("Pipeline check timed out"))
worker.start()
app.exec()
worker.shutdown()
assert worker.wait(15000), "Model worker failed to join"
assert state["done"] and not state["failed"], evidence
