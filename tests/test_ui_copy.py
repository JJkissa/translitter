"""Visible UI text is limited to controls, data labels, status and warnings."""
from PySide6.QtWidgets import QLabel
from test_ui import gui


def test_window_has_no_slogans_decorative_badges_or_duplicate_help(gui):
    window, _, _ = gui
    text = "\n".join(item.text() for item in window.findChildren(QLabel))
    for phrase in ("Translitter", "Two people.", "Two headsets.", "One conversation.",
                   "LOCAL ONLY", "audio and conversation stay in memory", "Hold F1 / F2"):
        assert phrase not in text, f"Non-functional UI copy remains: {phrase}"
    assert window.windowTitle().startswith("Translitter"), "Keep OS window identification"
    assert [window.mode.itemText(i) for i in range(window.mode.count())] == ["Push to talk", "Automatic"]
    assert window.language_status.text() == "Direct translation"


def test_empty_transcript_editors_have_labels_without_placeholder_filler(gui):
    window, _, _ = gui
    for card in window.cards:
        assert card.original.placeholderText() == ""
        assert card.translated.placeholderText() == ""
        labels = {item.text() for item in card.findChildren(QLabel)}
        assert {"ORIGINAL", "TRANSLATED"} <= labels
        assert card.hold.text().startswith("Hold to talk")
        assert card.state.text(), "Keep actionable participant state"
    assert window.start_button.text() == "Start conversation"
    assert window.stop_button.text() == "Stop"
    assert window.status.text(), "Keep loading/status feedback"
