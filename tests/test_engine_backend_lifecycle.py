"""Backend ownership after a partially successful optional-runtime shutdown."""
from types import SimpleNamespace
import sys

import pytest

from translitter.engine import ModelEngine


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    state = SimpleNamespace(instances=[], close_failures=0, cancelled=False,
                            startup="success", translations=[])

    class Client:
        def __init__(self, runtime):
            self.runtime = runtime

        def translate(self, text, source, target, cancelled):
            assert self.runtime.live, "retired runtime client was reused"
            state.translations.append((text, source, target))
            return "Hello"

    class Runtime:
        def __init__(self, root):
            assert root == tmp_path
            self.live = False
            self.closes = 0
            self.client = Client(self)
            state.instances.append(self)

        def start(self, cancelled):
            self.live = True
            if state.startup == "error":
                raise RuntimeError("startup failed")
            if state.startup == "cancel":
                state.cancelled = True
            return None if state.startup == "none" else self.client

        def close(self):
            self.closes += 1
            # A child can exit before socket-directory cleanup fails.
            self.live = False
            if state.close_failures:
                state.close_failures -= 1
                raise RuntimeError("cleanup failed")

    monkeypatch.setitem(sys.modules, "translitter.quality_runtime",
                        SimpleNamespace(QualityRuntime=Runtime))
    engine = ModelEngine(tmp_path)
    yield engine, state
    state.close_failures = 0
    engine.close()


@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_reselect_after_partial_shutdown_retries_cleanup_before_new_client(
        lifecycle, failed_attempts):
    engine, state = lifecycle
    assert engine.set_translation_backend("translategemma")
    retired = state.instances[0]
    state.close_failures = failed_attempts
    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.set_translation_backend("marian")
    assert engine._quality_runtime is retired
    assert not retired.live

    for _ in range(failed_attempts - 1):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            engine.set_translation_backend("translategemma")
        assert state.instances == [retired]
        assert engine._quality_runtime is retired

    assert engine.set_translation_backend("translategemma")
    assert len(state.instances) == 2
    replacement = state.instances[1]
    assert retired.closes == failed_attempts + 1
    assert engine._quality_runtime is replacement
    assert engine._quality_client is replacement.client
    assert engine._quality_client.translate("Hei", "fi", "en", lambda: False) == "Hello"
    assert state.translations == [("Hei", "fi", "en")]
    assert engine.set_translation_backend("marian")
    engine.close()
    assert replacement.closes == 1
    assert engine._quality_runtime is engine._quality_client is None
    assert engine.translation_backend == "marian"


@pytest.mark.parametrize("startup", ["none", "cancel", "error"])
def test_unsuccessful_start_releases_returned_client_and_recovers(lifecycle, startup):
    engine, state = lifecycle
    state.startup = startup
    if startup == "error":
        with pytest.raises(RuntimeError, match="startup failed"):
            engine.set_translation_backend("translategemma", lambda: state.cancelled)
    else:
        assert not engine.set_translation_backend("translategemma", lambda: state.cancelled)
    retired = state.instances[0]
    assert retired.closes == 1 and not retired.live
    assert engine._quality_runtime is engine._quality_client is None
    assert engine.translation_backend == "marian"
    state.startup = "success"
    state.cancelled = False
    assert engine.set_translation_backend("translategemma", lambda: state.cancelled)
    assert len(state.instances) == 2
    assert engine._quality_client is state.instances[1].client


@pytest.mark.parametrize("backend", ["marian", "translategemma"])
def test_pre_cancelled_switch_preserves_live_owned_backend(lifecycle, backend):
    engine, state = lifecycle
    assert engine.set_translation_backend("translategemma")
    runtime = state.instances[0]
    assert not engine.set_translation_backend(backend, cancelled=lambda: True)
    assert engine.translation_backend == "translategemma"
    assert engine._quality_runtime is runtime and runtime.live
    assert engine._quality_client is runtime.client
    assert runtime.closes == 0 and state.instances == [runtime]


def test_progress_failure_cleans_constructed_runtime_before_retry(lifecycle):
    engine, state = lifecycle

    def progress(message):
        raise RuntimeError("progress callback failed")

    with pytest.raises(RuntimeError, match="progress callback failed"):
        engine.set_translation_backend("translategemma", progress=progress)
    assert len(state.instances) == 1
    assert state.instances[0].closes == 1
    assert engine._quality_runtime is engine._quality_client is None
    assert engine.translation_backend == "marian"
    assert engine.set_translation_backend("translategemma")
    assert len(state.instances) == 2


def test_ready_reselection_reuses_only_untouched_runtime(lifecycle):
    engine, state = lifecycle
    assert engine.set_translation_backend("translategemma")
    assert engine.set_translation_backend("translategemma")
    runtime = state.instances[0]
    assert state.instances == [runtime] and runtime.live
    assert runtime.closes == 0
    assert engine._quality_client is runtime.client
