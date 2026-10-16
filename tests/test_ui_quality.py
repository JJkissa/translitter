"""Stopped-only translation selection with fake audio: no devices or models."""
import pytest
from PySide6.QtCore import Signal

from test_ui import FakeAudio, FakeWorker
from translitter.ui import Window


class QualityWorker(FakeWorker):
    translation_ready = Signal(int, str, bool, str)

    def __init__(self):
        super().__init__()
        self.requests = []

    def set_translation_backend(self, backend):
        request = len(self.requests) + 1
        self.requests.append((request, backend))
        return request


@pytest.fixture
def quality_gui(qtbot, tmp_path):
    worker, audio = QualityWorker(), FakeAudio()
    window = Window(tmp_path, worker=worker, audio=audio, start_worker=False)
    qtbot.addWidget(window)
    window.show()
    return window, audio, worker


def select(window, backend):
    window.backend_selector.setCurrentIndex(window.backend_selector.findData(backend))


def test_default_fast_and_async_selection_gate_start_without_starting_audio(quality_gui):
    w, audio, worker = quality_gui
    assert w.translation_backend.currentData() == "marian"
    assert [(w.backend_selector.itemText(n), w.backend_selector.itemData(n))
            for n in range(w.backend_selector.count())] == [
                ("Marian (fast)", "marian"), ("TranslateGemma 4B", "translategemma")]
    assert not audio.starts and not worker.requests
    worker.ready.emit()
    assert w.start_button.isEnabled()
    select(w, "translategemma")
    assert w.translation_backend.currentData() == "translategemma"
    assert worker.requests == [(1, "translategemma")]
    assert w.models_ready and not w.start_button.isEnabled()
    w.start_session()  # Programmatic start must obey the same gate as the button.
    assert not w.active and not audio.starts and not worker.sessions
    worker.ready.emit()  # Base-model readiness cannot override selected-backend readiness.
    assert not w.start_button.isEnabled()
    worker.translation_ready.emit(1, "marian", True, "")
    worker.translation_ready.emit(0, "translategemma", True, "")
    assert not w.start_button.isEnabled()
    worker.translation_ready.emit(1, "translategemma", True, "")
    assert w.start_button.isEnabled() and not audio.starts
    w.start_session()
    assert w.active and len(audio.starts) == 1


def test_failed_selection_recovers_only_after_latest_marian_confirmation(quality_gui):
    w, audio, worker = quality_gui
    worker.ready.emit()
    select(w, "translategemma")
    worker.translation_ready.emit(1, "translategemma", False, "Optional local runtime not installed; select Marian")
    assert w.models_ready and not w.start_button.isEnabled()
    assert "runtime not installed" in w.status.text()
    worker.ready.emit()
    assert "runtime not installed" in w.status.text()
    w.start_session()
    assert not audio.starts and not worker.sessions
    select(w, "marian")
    worker.translation_ready.emit(1, "translategemma", True, "")
    assert not w.start_button.isEnabled()
    worker.translation_ready.emit(2, "marian", True, "")
    assert w.start_button.isEnabled() and not audio.starts
    select(w, "translategemma")
    worker.translation_ready.emit(1, "translategemma", True, "")
    assert not w.start_button.isEnabled()
    worker.translation_ready.emit(3, "translategemma", True, "")
    assert w.start_button.isEnabled()
    worker.error.emit(-1, "Base models unavailable")
    worker.translation_ready.emit(3, "translategemma", True, "")
    assert not w.models_ready and not w.start_button.isEnabled()


@pytest.mark.parametrize("backend", ["marian", "translategemma"])
def test_active_selector_lock_reverts_programmatic_changes(quality_gui, backend):
    w, audio, worker = quality_gui
    worker.ready.emit()
    if backend == "translategemma":
        select(w, backend)
        worker.translation_ready.emit(1, backend, True, "")
    w.start_session()
    assert w.active and not w.backend_selector.isEnabled()
    before = list(worker.requests)
    select(w, "translategemma" if backend == "marian" else "marian")
    assert w.translation_backend.currentData() == backend and w.backend_selector.currentData() == backend
    assert worker.requests == before and len(audio.starts) == 1
    w.stop_session()
    assert w.backend_selector.isEnabled()


def test_legacy_worker_keeps_default_fast_and_disables_optional_selection(qtbot, tmp_path):
    worker, audio = FakeWorker(), FakeAudio()
    w = Window(tmp_path, worker=worker, audio=audio, start_worker=False)
    qtbot.addWidget(w)
    assert w.translation_backend.currentData() == "marian" and not w.backend_selector.isEnabled()
    select(w, "translategemma")
    assert w.translation_backend.currentData() == "marian" and w.backend_selector.currentData() == "marian"
    worker.ready.emit()
    assert w.start_button.isEnabled() and not audio.starts


def test_late_configuration_signal_cannot_revive_closing_window(quality_gui):
    w, audio, worker = quality_gui
    worker.ready.emit()
    select(w, "translategemma")
    worker.finished = False
    w.close()
    worker.translation_ready.emit(1, "translategemma", True, "")
    w.start_session()
    assert w._closing and not w.start_button.isEnabled() and not audio.starts
    assert worker.shutdowns == 1
    worker.finished = True


def test_quality_warning_and_direct_pair_do_not_require_marian_pivot_assets(quality_gui, tmp_path):
    from translitter.languages import VOICE_FILES
    w, audio, worker = quality_gui
    worker.ready.emit()
    for selector, code in zip(w.language_selectors, ("ru", "uk")):
        selector.setCurrentIndex(selector.findData(code))
        voice = tmp_path / "models" / f"voice-{code}" / VOICE_FILES[code]
        voice.parent.mkdir(parents=True, exist_ok=True)
        voice.touch()
        voice.with_name(voice.name + ".json").touch()
    w._selection_changed()
    assert "Via English" in w.language_status.text()
    assert not w.start_button.isEnabled()  # Marian pivot models are not installed.
    tooltip = w.backend_selector.toolTip().lower()
    assert all(word in tooltip for word in ("local", "slower", "accuracy", "downloads", "direct"))
    assert "does not guarantee" in tooltip
    select(w, "translategemma")
    worker.translation_ready.emit(1, "translategemma", True, "")
    assert "Via English" not in w.language_status.text()
    assert "direct" in w.language_status.text().lower()
    assert w.start_button.isEnabled() and not audio.starts
    # Direct translation still requires the selected local synthesis voices.
    voice.unlink()
    w._selection_changed()
    assert not w.start_button.isEnabled() and "voice-uk" in w.language_status.text()
