"""Engine unit contracts plus opt-in real offline speech; never device I/O."""
from contextlib import nullcontext
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from translitter.core import Route

ROUTE = Route(0, 1, "fi", "en", "mic0", "out1")


@pytest.fixture
def backends(monkeypatch, tmp_path):
    calls = []
    state = SimpleNamespace(original="Hei tänään", translation="Hello today", hook=lambda stage: None)
    def transcribe(samples, **kwargs):
        calls.append(("asr", kwargs))
        calls.append(("asr_samples", samples.copy()))
        def segments():
            state.hook("asr")
            yield SimpleNamespace(text=state.original, words=[
                SimpleNamespace(word=word, start=i * .2, end=(i + 1) * .2)
                for i, word in enumerate((state.original or "").split())])
        return segments(), SimpleNamespace(language=kwargs["language"])
    def generate(**kwargs):
        calls.append(("translate", kwargs))
        state.hook("translate")
        return [[1]]
    tokenizer = SimpleNamespace(
        decode=lambda *a, **kw: state.translation,
        __call__=None,
    )
    class Tokenizer:
        def __init__(self, edge=None):
            self.edge = edge
        def __call__(self, texts, **kwargs):
            calls.append(("edge", self.edge))
            calls.append(("tokenize", texts, kwargs))
            return {"input_ids": [[1]]}
        def decode(self, *args, **kwargs):
            return getattr(state, "translations", {}).get(self.edge, state.translation)
    def to_device(device):
        calls.append(("model_device", device))
        return model
    def evaluate():
        calls.append(("model_eval",))
        return model
    model = SimpleNamespace(eval=evaluate, to=to_device, generate=generate)
    def synthesize(text):
        calls.append(("tts", text))
        state.hook("tts")
        yield SimpleNamespace(audio_float_array=np.array([0.1, -0.2], dtype=np.float32), sample_rate=22050)
    voice = SimpleNamespace(synthesize=synthesize)
    def loader(name, obj):
        def load(path, **kwargs):
            calls.append((name, str(path), kwargs))
            state.hook(name)
            if name == "tokenizer_load":
                return Tokenizer(Path(path).name)
            if name == "voice_load":
                def speak(text):
                    calls.append(("voice_language", Path(path).name.split("_")[0]))
                    return synthesize(text)
                return SimpleNamespace(synthesize=speak)
            return obj
        return load
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=loader("whisper_load", SimpleNamespace(transcribe=transcribe))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        MarianTokenizer=SimpleNamespace(from_pretrained=loader("tokenizer_load", Tokenizer())),
        MarianMTModel=SimpleNamespace(from_pretrained=loader("marian_load", model))))
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=loader("voice_load", voice))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: calls.append(("threads", n)), inference_mode=nullcontext))
    for name in ("whisper-turbo", "whisper-small", "translate-fi-en", "translate-en-fi"):
        (tmp_path / "models" / name).mkdir(parents=True)
    for name in ("voice-fi/fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx",
                 "voice-en/en/en_US/lessac/medium/en_US-lessac-medium.onnx"):
        path = tmp_path / "models" / name
        path.parent.mkdir(parents=True)
        path.touch()
        Path(str(path) + ".json").touch()
    return tmp_path, calls, state


@pytest.mark.parametrize("source,target", [("fi", "ru"), ("ru", "uk"), ("uk", "en")])
def test_missing_optional_resources_fail_before_asr_but_baseline_still_works(backends, source, target):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    engine = ModelEngine(root)
    engine.load()
    route = Route(0, 1, source, target, "mic0", "out1")
    with pytest.raises(FileNotFoundError, match="Required local model asset missing:.*translate-"):
        engine.interpret(np.ones(160, dtype=np.float32), route, 1)
    assert not any(c[0] == "asr" for c in calls)
    assert len(engine._translations) == 2 and len(engine._voices) == 2
    assert engine.interpret(np.ones(160, dtype=np.float32), ROUTE, 2) is not None


@pytest.mark.parametrize("source,target", [("fi", "fi"), ("xx", "en"), ("en", "../uk")])
def test_unsupported_pair_fails_before_asr(backends, source, target):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    engine = ModelEngine(root)
    engine.load()
    with pytest.raises(ValueError):
        engine.interpret(np.ones(160, dtype=np.float32), Route(0, 1, source, target, "mic0", "out1"), 1)
    assert not any(c[0] == "asr" for c in calls)


def test_optional_voice_sidecar_is_required_before_asr(backends):
    from translitter.engine import ModelEngine
    from translitter.languages import VOICE_FILES
    root, calls, _ = backends
    install_multilingual_assets(root)
    (root / "models" / "voice-ru" / (VOICE_FILES["ru"] + ".json")).unlink()
    engine = ModelEngine(root)
    engine.load()
    with pytest.raises(FileNotFoundError, match="voice-ru"):
        engine.interpret(np.ones(160, dtype=np.float32), Route(0, 1, "fi", "ru", "mic0", "out1"), 1)
    assert not any(c[0] == "asr" for c in calls)


@pytest.mark.parametrize("boundary,load_count", [
    ("Loading ru → en translation", 0), ("tokenizer_load", 1), ("marian_load", 2),
    ("Loading uk voice", 4), ("voice_load", 5),
])
def test_cancellation_during_optional_loading_stops_before_next_backend(backends, boundary, load_count):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    install_multilingual_assets(root)
    engine = ModelEngine(root)
    engine.load()
    calls.clear()
    stopped = False
    def cancel_at(stage):
        nonlocal stopped
        if stage == boundary:
            stopped = True
    state.hook = cancel_at
    route = Route(0, 1, "ru", "uk", "mic0", "out1")
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1, lambda: stopped, cancel_at) is None
    assert len([c for c in calls if c[0].endswith("_load")]) == load_count
    assert not any(c[0] in ("asr", "translate", "tts") for c in calls)


def test_cancellation_between_pivot_legs_prevents_second_generation(backends):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    install_multilingual_assets(root)
    engine = ModelEngine(root)
    engine.load()
    stopped = False
    def cancel_at(stage):
        nonlocal stopped
        if stage == "translate":
            stopped = True
    state.hook = cancel_at
    route = Route(0, 1, "ru", "uk", "mic0", "out1")
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1, lambda: stopped) is None
    assert [c[1] for c in calls if c[0] == "edge"] == ["translate-ru-en"]
    assert not any(c[0] == "tts" for c in calls)


@pytest.mark.parametrize("invalid_edge,count", [("translate-ru-en", 1), ("translate-en-uk", 2)])
@pytest.mark.parametrize("invalid", ["", "<pad>", "\ufffd", None])
def test_invalid_pivot_or_final_output_never_synthesizes(backends, invalid_edge, count, invalid):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    install_multilingual_assets(root)
    state.translations = {invalid_edge: invalid}
    engine = ModelEngine(root)
    engine.load()
    route = Route(0, 1, "ru", "uk", "mic0", "out1")
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1) is None
    assert len([c for c in calls if c[0] == "translate"]) == count
    assert not any(c[0] == "tts" for c in calls)


def test_switching_every_pair_keeps_only_six_edges_and_four_voices(backends):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    install_multilingual_assets(root)
    engine = ModelEngine(root)
    engine.load()
    for source, target in permutations(("fi", "en", "ru", "uk"), 2):
        route = Route(0, 1, source, target, "mic0", "out1")
        assert engine.interpret(np.ones(160, dtype=np.float32), route, 1) is not None
        assert len(engine._translations) <= 6 and len(engine._voices) <= 4
    assert len(engine._translations) == 6 and len(engine._voices) == 4
    assert len([c for c in calls if c[0] == "marian_load"]) == 6
    assert len([c for c in calls if c[0] == "voice_load"]) == 4


def install_multilingual_assets(root):
    from translitter.languages import VOICE_FILES
    for language in ("ru", "uk"):
        for source, target in ((language, "en"), ("en", language)):
            directory = root / "models" / f"translate-{source}-{target}"
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("pytorch_model.bin", "config.json", "source.spm", "target.spm", "vocab.json"):
                (directory / name).touch()
        path = root / "models" / f"voice-{language}" / VOICE_FILES[language]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        Path(str(path) + ".json").touch()


@pytest.mark.parametrize("source,target", tuple(permutations(("fi", "en", "ru", "uk"), 2)))
def test_all_language_pairs_translate_in_order_with_source_asr_and_final_voice(backends, source, target):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    install_multilingual_assets(root)
    state.translations = {f"translate-{a}-{b}": f"translated {a} to {b}"
                          for a, b in permutations(("fi", "en", "ru", "uk"), 2)}
    engine = ModelEngine(root)
    engine.load()
    assert len(engine._translations) == 2 and len(engine._voices) == 2
    route = Route(0, 1, source, target, "mic0", "out1")
    for repeat in range(2):
        calls.clear()
        result = engine.interpret(np.ones(160, dtype=np.float32), route, 7)
        assert result is not None
        assert (result.source_language, result.target_language) == (source, target)
        edges = [(source, target)] if "en" in (source, target) else [(source, "en"), ("en", target)]
        assert [c[1] for c in calls if c[0] == "edge"] == [f"translate-{a}-{b}" for a, b in edges]
        inputs = [c[1][0] for c in calls if c[0] == "tokenize"]
        assert inputs == [state.original] + [state.translations[f"translate-{a}-{b}"] for a, b in edges[:-1]]
        assert result.translation == state.translations[f"translate-{edges[-1][0]}-{target}"]
        assert next(c[1]["language"] for c in calls if c[0] == "asr") == source
        assert [c[1] for c in calls if c[0] == "voice_language"] == [target]
        assert len(engine._translations) <= 6 and len(engine._voices) <= 4
        loads = len([c for c in calls if c[0] == "marian_load"])
        assert [c[1] for c in calls if c[0] == "model_device"] == ["cpu"] * loads
        assert len([c for c in calls if c[0] == "model_eval"]) == loads
        if repeat:
            assert not any(c[0].endswith("_load") for c in calls)
        assert all(c[2] == dict(local_files_only=True, trust_remote_code=False, weights_only=True)
                   for c in calls if c[0] == "marian_load")
        assert all(c[2] == dict(local_files_only=True) for c in calls if c[0] == "tokenizer_load")
        assert all(c[2] == dict(use_cuda=False) for c in calls if c[0] == "voice_load")


@pytest.mark.parametrize("device,precision", [("cuda", "int8_float16"), ("cpu", "int8")])
def test_load_is_lazy_idempotent_and_all_backends_are_offline(backends, device, precision):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    engine = ModelEngine(root, device=device)
    assert calls == []
    progress = []
    engine.load(progress.append)
    engine.load(progress.append)
    assert len([c for c in calls if c[0] == "whisper_load"]) == 1
    whisper = next(c for c in calls if c[0] == "whisper_load")
    assert whisper[1] == str(root / "models" / "whisper-turbo")
    assert whisper[2] == dict(device=device, compute_type=precision, cpu_threads=4, local_files_only=True)
    marian = [c for c in calls if c[0] == "marian_load"]
    assert len(marian) == 2
    assert all(c[2] == dict(local_files_only=True, trust_remote_code=False, weights_only=True) for c in marian)
    assert all(c[2] == dict(local_files_only=True) for c in calls if c[0] == "tokenizer_load")
    assert len([c for c in calls if c[0] == "voice_load"]) == 2
    assert all(c[2] == dict(use_cuda=False) for c in calls if c[0] == "voice_load")
    assert ("threads", 4) in calls
    assert progress and all(isinstance(item, str) for item in progress)


@pytest.mark.parametrize("method", ["recognize", "interpret"])
@pytest.mark.parametrize("length", [128001, 16000 * 30])
def test_long_released_audio_reaches_asr_whole_without_truncation(backends, method, length):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    engine = ModelEngine(root)
    engine.load()
    samples = np.linspace(-.5, .5, length, dtype=np.float32)
    if method == "recognize":
        assert engine.recognize(samples, ROUTE) == state.original
    else:
        result = engine.interpret(samples, ROUTE, 7)
        assert result is not None and result.generation == 7
        assert (result.speaker, result.listener) == (0, 1)
    actual = next(c[1] for c in calls if c[0] == "asr_samples")
    np.testing.assert_array_equal(actual, samples)
    options = next(c[1] for c in calls if c[0] == "asr")
    assert options["vad_filter"] is True and "word_timestamps" not in options


def test_turn_preserves_route_and_produces_mono_float32_audio(backends):
    from translitter.engine import ModelEngine, TurnResult
    root, calls, _ = backends
    engine = ModelEngine(root)
    engine.load()
    result = engine.interpret(np.ones(16000, dtype=np.float32), ROUTE, 7, lambda: False, lambda message: None)
    assert isinstance(result, TurnResult)
    assert (result.generation, result.speaker, result.listener) == (7, 0, 1)
    assert (result.source_language, result.target_language) == ("fi", "en")
    assert (result.original, result.translation) == ("Hei tänään", "Hello today")
    assert result.audio.dtype == np.float32 and result.audio.ndim == 1
    assert result.rate == 22050 and result.elapsed >= 0
    asr = next(c[1] for c in calls if c[0] == "asr")
    assert asr == dict(language="fi", beam_size=1, vad_filter=True,
                       vad_parameters={"min_silence_duration_ms": 300}, condition_on_previous_text=False)
    generation = next(c[1] for c in calls if c[0] == "translate")
    assert generation["max_new_tokens"] == 256 and generation["num_beams"] == 1
    audio, rate = engine.synthesize_text("Hello", "en")
    assert audio.dtype == np.float32 and rate == 22050


@pytest.mark.parametrize("boundary", ["before", "asr", "translate", "tts", "Recognizing speech", "Translating", "Synthesizing speech"])
def test_cancellation_at_stage_boundaries_discards_turn(backends, boundary):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    engine = ModelEngine(root)
    engine.load()
    cancelled = boundary == "before"
    def cancel_at(stage):
        nonlocal cancelled
        if stage == boundary:
            cancelled = True
    state.hook = cancel_at
    result = engine.interpret(np.ones(16000, dtype=np.float32), ROUTE, 1, lambda: cancelled, cancel_at)
    assert result is None
    stages = [c[0] for c in calls]
    if boundary in ("before", "Recognizing speech"):
        assert "asr" not in stages
    if boundary in ("before", "Recognizing speech", "asr", "Translating"):
        assert "translate" not in stages
    if boundary != "tts":
        assert "tts" not in stages


@pytest.mark.parametrize("original", ["", "   ", "...", "<pad>"])
def test_silence_or_invalid_transcript_never_translates(backends, original):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    state.original = original
    engine = ModelEngine(root)
    engine.load()
    assert engine.interpret(np.ones(16000, dtype=np.float32), ROUTE, 1) is None
    assert not any(c[0] in ("translate", "tts") for c in calls)


@pytest.mark.parametrize("translation", ["", "   ", "...", "<pad>", "\ufffd", None, "bad\x00text"])
def test_empty_or_invalid_translation_never_synthesizes(backends, translation):
    from translitter.engine import ModelEngine
    root, calls, state = backends
    state.translation = translation
    engine = ModelEngine(root)
    engine.load()
    assert engine.interpret(np.ones(16000, dtype=np.float32), ROUTE, 1) is None
    assert not any(c[0] == "tts" for c in calls)


@pytest.mark.parametrize("method", ["interpret", "recognize"])
@pytest.mark.parametrize("samples", [np.array([]), np.zeros((2, 2)), np.array([np.nan]),
                                    np.array([np.inf]), np.array([-np.inf]), np.zeros(16000 * 30 + 1)])
def test_invalid_input_never_reaches_asr(backends, samples, method):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    engine = ModelEngine(root)
    engine.load()
    with pytest.raises(ValueError):
        if method == "interpret":
            engine.interpret(samples, ROUTE, 1)
        else:
            engine.recognize(samples, ROUTE)
    assert not any(c[0] == "asr" for c in calls)


@pytest.mark.parametrize("model,device", [("../outside", "cuda"), ("whisper-turbo", "unknown")])
def test_unrecognized_model_and_device_are_rejected(tmp_path, model, device):
    from translitter.engine import ModelEngine
    with pytest.raises(ValueError):
        ModelEngine(tmp_path, model_name=model, device=device)


def test_missing_assets_fail_before_any_backend_load(backends):
    from translitter.engine import ModelEngine
    root, calls, _ = backends
    (root / "models/whisper-turbo").rmdir()
    with pytest.raises(FileNotFoundError):
        ModelEngine(root).load()
    assert not calls


def test_invalid_or_empty_speech_output_is_rejected(backends):
    from translitter.engine import ModelEngine
    root, _, _ = backends
    engine = ModelEngine(root)
    engine.load()
    engine._voices["en"] = SimpleNamespace(synthesize=lambda text: iter([]))
    with pytest.raises(ValueError):
        engine.synthesize_text("Hello", "en")
    engine._voices["en"] = SimpleNamespace(synthesize=lambda text: iter([
        SimpleNamespace(audio_float_array=np.array([np.nan]), sample_rate=22050)]))
    with pytest.raises(ValueError):
        engine.synthesize_text("Hello", "en")


@pytest.mark.skipif(__import__("os").environ.get("TRANSLITTER_MODEL_TESTS") != "1", reason="Opt-in real offline engines")
def test_real_engine_both_directions_and_silence(tmp_path):
    import json
    import math
    from scipy.signal import resample_poly
    from translitter.engine import ModelEngine
    root = Path(__file__).resolve().parents[1]
    engine = ModelEngine(root)
    engine.load()
    evidence = []
    for route, phrase in [(ROUTE, "Hei, miten voit tänään?"),
                          (Route(1, 0, "en", "fi", "mic1", "out0"), "Hello, how are you today?")]:
        audio, rate = engine.synthesize_text(phrase, route.source_language)
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        samples = np.pad(samples, (3200, 6400))
        result = engine.interpret(samples, route, 1)
        assert result is not None
        assert result.speaker == route.speaker and result.listener == route.listener
        assert result.original and result.translation
        assert result.audio.ndim == 1 and result.audio.dtype == np.float32
        assert np.isfinite(result.audio).all() and len(result.audio) > result.rate
        evidence.append({"input": phrase, "original": result.original, "translation": result.translation,
                         "source": route.source_language, "target": route.target_language,
                         "listener": result.listener, "rate": result.rate,
                         "audio_samples": len(result.audio), "elapsed": result.elapsed})
    assert engine.interpret(np.zeros(16000, dtype=np.float32), ROUTE, 1) is None
    (tmp_path / "engine-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("REAL_ENGINE_EVIDENCE", json.dumps(evidence, ensure_ascii=False))

