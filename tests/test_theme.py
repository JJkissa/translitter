"""Native light-theme regressions; no audio devices or model initialization."""
import re

import pytest
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QPushButton, QStyle, QStyleOptionButton

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from translitter import theme


def contrast(first, second):
    def luminance(value):
        color = QColor(value)
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
                  for c in channels]
        return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + .05) / (dark + .05)


def test_apply_theme_gives_native_widgets_readable_white_black_blue_palette(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    editor = QPlainTextEdit(window)
    theme.apply_theme(window)
    window.show()
    editor.ensurePolished()
    window.ensurePolished()

    assert window.styleSheet() == theme.STYLE
    assert window.font().family() == "Noto Sans"
    assert theme.BLACK == "#111827"
    assert theme.WHITE == "#ffffff"
    assert getattr(theme, "BLUE", None) == "#2563eb"
    assert not hasattr(theme, "PINK")
    palette = window.palette()
    assert palette.color(QPalette.Window).name() == "#f8fafc"
    assert palette.color(QPalette.WindowText).name() == theme.BLACK
    assert contrast(palette.color(QPalette.Window), palette.color(QPalette.WindowText)) >= 7
    assert editor.palette().color(QPalette.Base).name() == theme.WHITE
    assert contrast(editor.palette().color(QPalette.Base), editor.palette().color(QPalette.Text)) >= 7
    colors = re.findall(r"#[0-9a-fA-F]{6}\b", theme.STYLE)
    assert colors
    assert "pink" not in theme.STYLE.lower()
    for value in colors:
        color = QColor(value)
        assert color.blue() >= color.red(), f"Unexpected warm/pink accent: {value}"


def button_background(button, state):
    option = QStyleOptionButton()
    button.initStyleOption(option)
    option.state &= ~(QStyle.State_MouseOver | QStyle.State_Sunken)
    if state == "hover":
        option.state |= QStyle.State_MouseOver
    elif state == "pressed":
        option.state |= QStyle.State_MouseOver | QStyle.State_Sunken
    image = QPixmap(button.size())
    image.fill(QColor("#ff0000"))
    painter = QPainter(image)
    button.style().drawControl(QStyle.CE_PushButton, option, painter, button)
    painter.end()
    return image.toImage().pixelColor(button.width() // 2, 5).name()


@pytest.mark.parametrize("name", ["", "primary", "danger", "hold"])
def test_button_states_are_native_readable_and_visually_distinct(qtbot, name):
    button = QPushButton("Action")
    button.setObjectName(name)
    button.resize(180, 60)
    qtbot.addWidget(button)
    theme.apply_theme(button)
    button.ensurePolished()
    backgrounds = []
    for state in ("normal", "hover", "pressed", "disabled"):
        button.setEnabled(state != "disabled")
        group = QPalette.Active if button.isEnabled() else QPalette.Disabled
        background = button_background(button, state)
        foreground = button.palette().color(group, QPalette.ButtonText)
        assert contrast(foreground, background) >= 4.5, (name, state, background)
        backgrounds.append(background)
        if state == "disabled":
            assert foreground.name() == "#475569", "Disabled labels must use slate, not active blue"
            assert background == "#e2e8f0"
    assert len(set(backgrounds)) == 4, (name, backgrounds)


@pytest.mark.parametrize("widget_type", [QComboBox, QPlainTextEdit])
def test_native_input_selection_and_disabled_states_are_readable(qtbot, widget_type):
    widget = widget_type()
    qtbot.addWidget(widget)
    theme.apply_theme(widget)
    widget.ensurePolished()
    targets = [widget]
    if isinstance(widget, QComboBox):
        widget.addItems(["Finnish", "English"])
        widget.view().ensurePolished()
        targets.append(widget.view())
    for target in targets:
        palette = target.palette()
        foreground = palette.color(QPalette.HighlightedText)
        background = palette.color(QPalette.Highlight)
        assert background.name() == theme.BLUE
        assert foreground.name() == theme.WHITE
        assert contrast(foreground, background) >= 4.5
    widget.setEnabled(False)
    palette = widget.palette()
    assert palette.color(QPalette.Disabled, QPalette.Text).name() == "#475569"
    assert palette.color(QPalette.Disabled, QPalette.Base).name() == "#e2e8f0"
    assert contrast(palette.color(QPalette.Disabled, QPalette.Text),
                    palette.color(QPalette.Disabled, QPalette.Base)) >= 4.5
