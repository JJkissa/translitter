"""Native PinkFlow palette; no browser or external assets."""
from PySide6.QtGui import QFont

BLACK = "#111114"
WHITE = "#f5f5f7"
PINK = "#ff5eaa"
STYLE = """
QWidget { background: #111114; color: #f5f5f7; font-family: 'Noto Sans'; font-size: 13px; }
QLabel { background: transparent; }
QLabel#brand { font-size: 27px; font-weight: 700; }
QLabel#muted { color: #95959f; }
QLabel#badge { color: #ffadd2; background: #33202c; border: 1px solid #5b3048; border-radius: 6px; padding: 6px 12px; }
QLabel#participant { font-size: 21px; font-weight: 600; }
QLabel#section { font-size: 11px; color: #a4a4af; font-weight: 600; }
QFrame#card { background: #18181c; border: 1px solid #34343d; border-radius: 12px; }
QFrame#controls { background: #151518; border: 1px solid #2b2b31; border-radius: 10px; }
QPushButton { background: #222227; border: 1px solid #37373f; border-radius: 7px; padding: 10px 16px; font-weight: 600; }
QPushButton:hover { background: #2e2e35; border-color: #626270; }
QPushButton:pressed { background: #532b40; border-color: #ff5eaa; }
QPushButton:disabled { color: #666671; background: #1b1b1f; border-color: #29292f; }
QPushButton#primary { color: #171018; background: #ff5eaa; border-color: #ff5eaa; }
QPushButton#primary:disabled { background: #5b3046; color: #a78c99; border-color: #5b3046; }
QPushButton#danger { color: #ff91b6; }
QPushButton#hold { padding: 16px; color: #ffadd2; }
QComboBox { background: #202025; border: 1px solid #383840; border-radius: 6px; padding: 9px; selection-background-color: #63314a; }
QComboBox:focus { border-color: #ff5eaa; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: #25252b; selection-background-color: #63314a; }
QPlainTextEdit { background: #101014; border: 1px solid #303038; border-radius: 8px; padding: 12px; font-size: 15px; selection-background-color: #6e3350; }
QCheckBox { background: transparent; spacing: 8px; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #62626e; border-radius: 4px; background: #222227; }
QCheckBox::indicator:checked { background: #ff5eaa; border-color: #ff8bc1; }
QSlider::groove:horizontal { height: 5px; background: #34343d; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #ff5eaa; border-radius: 2px; }
QSlider::handle:horizontal { background: #f5f5f7; width: 14px; margin: -5px 0; border-radius: 7px; }
QProgressBar { background: #101014; border: none; border-radius: 3px; height: 5px; }
QProgressBar::chunk { background: #ff5eaa; border-radius: 3px; }
QToolTip { background: #2b2b33; color: #f5f5f7; border: 1px solid #5b3b4c; padding: 6px; }
QScrollBar:vertical { width: 8px; background: #18181c; }
QScrollBar::handle:vertical { background: #414149; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def apply_theme(window):
    window.setFont(QFont("Noto Sans", 10))
    window.setStyleSheet(STYLE)
