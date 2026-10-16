"""Fixed native light palette: white surfaces, near-black text, blue accents."""
from PySide6.QtGui import QFont

BLACK = "#111827"
WHITE = "#ffffff"
BLUE = "#2563eb"
STYLE = """
QWidget { background: #f8fafc; color: #111827; font-family: 'Noto Sans'; font-size: 13px; }
QLabel { background: transparent; }
QLabel#brand { font-size: 27px; font-weight: 700; }
QLabel#muted { color: #475569; }
QLabel#badge { color: #1e40af; background: #eff6ff; border: 1px solid #93c5fd; border-radius: 6px; padding: 6px 12px; }
QLabel#participant { font-size: 21px; font-weight: 600; }
QLabel#section { font-size: 11px; color: #475569; font-weight: 600; }
QFrame#card { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; }
QFrame#controls { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 10px; }
QPushButton { background: #ffffff; border: 1px solid #94a3b8; border-radius: 7px; padding: 10px 16px; font-weight: 600; }
QPushButton:hover { background: #eff6ff; border-color: #2563eb; }
QPushButton:pressed { background: #dbeafe; border-color: #1d4ed8; }
QPushButton:disabled { color: #475569; background: #e2e8f0; border-color: #cbd5e1; }
QPushButton#primary { color: #ffffff; background: #2563eb; border-color: #2563eb; }
QPushButton#primary:hover:enabled { background: #1d4ed8; border-color: #1d4ed8; }
QPushButton#primary:pressed:enabled { background: #1e40af; border-color: #1e40af; }
QPushButton#primary:disabled { background: #e2e8f0; color: #475569; border-color: #cbd5e1; }
QPushButton#danger { color: #1e40af; }
QPushButton#hold { padding: 16px; color: #1e40af; }
QPushButton#danger:disabled, QPushButton#hold:disabled { color: #475569; }
QComboBox:disabled, QPlainTextEdit:disabled { color: #475569; background: #e2e8f0; border-color: #cbd5e1; }
QComboBox { background: #ffffff; border: 1px solid #94a3b8; border-radius: 6px; padding: 9px; selection-background-color: #2563eb; selection-color: #ffffff; }
QComboBox:focus { border-color: #2563eb; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: #ffffff; color: #111827; selection-background-color: #2563eb; selection-color: #ffffff; }
QPlainTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; font-size: 15px; selection-background-color: #2563eb; selection-color: #ffffff; }
QCheckBox { background: transparent; spacing: 8px; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #64748b; border-radius: 4px; background: #ffffff; }
QCheckBox::indicator:checked { background: #2563eb; border-color: #1d4ed8; }
QSlider::groove:horizontal { height: 5px; background: #cbd5e1; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2563eb; border-radius: 2px; }
QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -5px 0; border-radius: 7px; }
QProgressBar { background: #e2e8f0; border: none; border-radius: 3px; height: 5px; }
QProgressBar::chunk { background: #2563eb; border-radius: 3px; }
QToolTip { background: #ffffff; color: #111827; border: 1px solid #94a3b8; padding: 6px; }
QScrollBar:vertical { width: 8px; background: #f1f5f9; }
QScrollBar::handle:vertical { background: #64748b; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def apply_theme(window):
    window.setFont(QFont("Noto Sans", 10))
    window.setStyleSheet(STYLE)
