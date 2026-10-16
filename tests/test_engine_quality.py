"""Backend selection stays local, explicit, cancellable and reversible."""
from itertools import permutations
from types import SimpleNamespace
import sys
import pytest
from test_engine import backends, install_multilingual_assets
from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def quality(backends, monkeypatch):
    root, calls, state = backends
    state.fail_runtime = False
    state.cancel_runtime = False
    state.cancel_translation = False
    class Client:
        def translate(self, text, source, target, cancelled):
            calls.append(("quality_translate", text, source, target))
            return None if cancelled() or state.cancel_translation else state.translation
    class Runtime:
        def __init__(self, directory):
            assert directory == root
            calls.append(("quality_runtime",))
        def start(self, cancelled):
            calls.append(("quality_start",))
            if state.fail_runtime:
                raise RuntimeError("runtime failed")
            return None if cancelled() or state.cancel_runtime else Client()
        def close(self):
            calls.append(("quality_close",))
    monkeypatch.setitem(sys.modules, "translitter.quality_runtime", SimpleNamespace(QualityRuntime=Runtime))
    engine = ModelEngine(root)
    engine.load()
    calls.clear()
    return engine, calls, state


def test_marian_remains_default_without_optional_runtime(quality):
    engine, calls, _ = quality
    assert engine.translation_backend == "marian"
    turn = engine.interpret_text("Hei", Route(0, 1, "fi", "en", "mic", "out"), 1)
    assert turn is not None and any(c[0] == "translate" for c in calls)
    assert not any(c[0].startswith("quality_") for c in calls)


@pytest.mark.parametrize("source,target", list(permutations(("fi", "en", "ru", "uk"), 2)))
def test_quality_uses_one_direct_translation_and_correct_voice(quality, source, target):
    engine, calls, state = quality
    install_multilingual_assets(engine.root)
    assert engine.set_translation_backend("translategemma")
    turn = engine.interpret_text("Original sentence", Route(0, 1, source, target, "mic", "out"), 2)
    assert turn.translation == state.translation
    assert (turn.source_language, turn.target_language) == (source, target)
    assert [c for c in calls if c[0] == "quality_translate"] == [("quality_translate", "Original sentence", source, target)]
    assert not any(c[0] in ("edge", "translate", "marian_load") for c in calls)
    assert turn.audio.ndim == 1


def test_failed_or_cancelled_quality_start_leaves_fast_backend(quality):
    engine, calls, state = quality
    state.fail_runtime = True
    with pytest.raises(RuntimeError, match="runtime failed"):
        engine.set_translation_backend("translategemma")
    assert engine.translation_backend == "marian"
    assert len([c for c in calls if c[0] == "quality_close"]) == 1
    state.fail_runtime = False
    state.cancel_runtime = True
    assert not engine.set_translation_backend("translategemma")
    assert engine.translation_backend == "marian"


def test_cancelled_translation_cannot_reach_voice(quality):
    engine, calls, state = quality
    engine.set_translation_backend("translategemma")
    state.cancel_translation = True
    assert engine.interpret_text("Hei", Route(0, 1, "fi", "en", "mic", "out"), 1) is None
    assert not any(c[0] == "tts" for c in calls)


@pytest.mark.parametrize("batch", [False, True])
def test_direct_backend_does_not_require_unused_pivot_assets(quality, batch):
    import shutil
    import numpy as np
    engine, calls, _ = quality
    install_multilingual_assets(engine.root)
    for language in ("ru", "uk"):
        for name in (f"translate-{language}-en", f"translate-en-{language}"):
            shutil.rmtree(engine.root / "models" / name)
    engine.set_translation_backend("translategemma")
    route = Route(0, 1, "ru", "fi", "mic", "out")
    turn = (engine.interpret(np.full(16000, .1, dtype=np.float32), route, 1) if batch
            else engine.interpret_text("Сегодня", route, 1))
    assert turn is not None and turn.target_language == "fi"
    assert not any(call[0] == "marian_load" for call in calls)


def test_failed_runtime_close_keeps_ownership_for_retry(quality, monkeypatch):
    engine, calls, _ = quality
    engine.set_translation_backend("translategemma")
    runtime = engine._quality_runtime
    attempted = []
    def close():
        attempted.append(True)
        if len(attempted) == 1:
            raise RuntimeError("could not stop")
    monkeypatch.setattr(runtime, "close", close)
    with pytest.raises(RuntimeError, match="could not stop"):
        engine.close()
    assert engine._quality_runtime is runtime
    engine.close()
    assert len(attempted) == 2 and engine._quality_runtime is None


@pytest.mark.parametrize("startup", ["failure", "cancelled"])
@pytest.mark.parametrize("retry", ["close", "marian", "translategemma"])
def test_failed_start_retains_runtime_until_repeated_cleanup_succeeds(quality, monkeypatch, startup, retry):
    engine, calls, state = quality
    module = sys.modules["translitter.quality_runtime"]
    original = module.QualityRuntime
    instances, attempts = [], []

    class Runtime(original):
        def __init__(self, root):
            super().__init__(root)
            instances.append(self)

        def close(self):
            attempts.append(self)
            if len(attempts) <= 3:
                raise RuntimeError("could not stop")

    monkeypatch.setattr(module, "QualityRuntime", Runtime)
    state.fail_runtime = startup == "failure"
    state.cancel_runtime = startup == "cancelled"
    with pytest.raises(RuntimeError, match="could not stop"):
        engine.set_translation_backend("translategemma")
    assert engine._quality_runtime is instances[0]
    assert engine._quality_client is None
    assert engine.translation_backend == "marian"
    for _ in range(3 - len(attempts)):
        with pytest.raises(RuntimeError, match="could not stop"):
            if retry == "close":
                engine.close()
            else:
                engine.set_translation_backend(retry)
        assert engine._quality_runtime is instances[0]
        assert len(instances) == 1
    engine.close()
    assert all(runtime is instances[0] for runtime in attempts)
    assert engine._quality_runtime is None
    assert engine._quality_client is None


def test_fast_restore_closes_only_owned_runtime_once(quality):
    engine, calls, _ = quality
    assert engine.set_translation_backend("translategemma")
    assert engine.set_translation_backend("marian")
    engine.close()
    engine.close()
    assert len([c for c in calls if c[0] == "quality_close"]) == 1
    assert engine.translation_backend == "marian"
    with pytest.raises(ValueError):
        engine.set_translation_backend("https://cloud.example")
