"""Native module launcher. Models load in the worker; capture needs Start."""
import argparse
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from .ui import Window
from .worker import ModelWorker


def build_parser():
    parser = argparse.ArgumentParser(description="Local two-headset Finnish ↔ English interpreter")
    parser.add_argument("--model", choices=("whisper-small", "whisper-turbo"), default="whisper-turbo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Translitter")
    app.setOrganizationName("Translitter")
    root = Path(__file__).resolve().parent.parent
    worker = ModelWorker(root, model_name=args.model, device=args.device)
    window = Window(root, worker=worker)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
