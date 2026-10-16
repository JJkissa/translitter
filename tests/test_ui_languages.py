"""Language selectors and immutable session directions; no device/model I/O."""
import pytest

from translitter.languages import LANGUAGE_NAMES, missing_assets
from test_ui import gui, start, turn


def choose(window, first, second):
    for selector, code in zip(window.language_selectors, (first, second)):
        selector.setCurrentIndex(selector.findData(code))


def test_four_spoken_languages_keep_finnish_english_defaults(gui, qtbot):
    window, audio, worker = gui
    assert hasattr(window, "language_selectors"), "Participant language options are missing"
    assert [s.currentData() for s in window.language_selectors] == ["fi", "en"]
    for selector in window.language_selectors:
        assert {selector.itemData(i) for i in range(selector.count())} == {"fi", "en", "ru", "uk"}
    start(window, worker, qtbot)
    assert [(r.source_language, r.target_language) for r in worker.sessions[-1][1]] == [("fi", "en"), ("en", "fi")]
    assert all(not selector.isEnabled() for selector in window.language_selectors)
    window.stop_session()
    assert all(selector.isEnabled() for selector in window.language_selectors)


def test_missing_optional_models_block_start_without_opening_microphones(gui, qtbot):
    window, audio, worker = gui
    assert hasattr(window, "language_selectors"), "Participant language options are missing"
    worker.ready.emit()
    choose(window, "fi", "ru")
    assert not window.start_button.isEnabled()
    assert "not installed" in window.language_status.text().lower()
    assert "translate-en-ru" in window.language_status.text()
    window.start_session()
    assert not window.active and not audio.starts and not worker.sessions
    choose(window, "fi", "en")
    assert window.start_button.isEnabled(), "Missing Russian must not disable Finnish-English"


@pytest.mark.parametrize("first,second", [(a, b) for a in LANGUAGE_NAMES for b in LANGUAGE_NAMES if a != b])
def test_language_pair_controls_routes_captions_and_locked_session(gui, qtbot, first, second):
    window, audio, worker = gui
    # Empty asset markers exercise capability gating, not model inference.
    for relative in missing_assets(window.root, first, second, additional_only=True):
        path = window.root / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    choose(window, first, second)
    assert LANGUAGE_NAMES[first] in window.cards[0].title.text()
    assert window.cards[1].direction.text() == f"{LANGUAGE_NAMES[second].upper()} → {LANGUAGE_NAMES[first].upper()}"
    if "en" not in (first, second):
        assert "via english" in window.language_status.text().lower()
    else:
        assert "direct" in window.language_status.text().lower()
    start(window, worker, qtbot)
    routes = worker.sessions[-1][1]
    assert [(r.source_language, r.target_language) for r in routes] == [(first, second), (second, first)]
    assert audio.starts[-1] == routes
    window.language_selectors[0].setCurrentIndex(window.language_selectors[0].findData(second))
    assert [s.currentData() for s in window.language_selectors] == [first, second]
    for speaker, (source, target) in enumerate(((first, second), (second, first))):
        result = turn(window, speaker)
        result.source_language, result.target_language = source, target
        worker.result.emit(result)
        assert result.original in window.cards[speaker].original.toPlainText()
        assert result.translation in window.cards[speaker].translated.toPlainText()
        assert audio.plays[-1][0] == 1 - speaker
    window.stop_session()
    choose(window, second, first)
    assert all(not card.original.toPlainText() and not card.translated.toPlainText() for card in window.cards)


def test_same_language_cannot_start_capture(gui):
    window, audio, worker = gui
    worker.ready.emit()
    choose(window, "fi", "fi")
    assert not window.start_button.isEnabled()
    window.start_session()
    assert not audio.starts and not worker.sessions


def test_uninstalled_pivot_pair_still_discloses_english_intermediate(gui):
    window, audio, worker = gui
    choose(window, "ru", "uk")
    assert "not installed" in window.language_status.text().lower()
    assert "via english" in window.language_status.text().lower()
    assert len(window.language_status.text()) < 420, "Missing filenames must not swamp the interface"
