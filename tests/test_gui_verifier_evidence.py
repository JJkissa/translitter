"""GUI completion artifacts with real tiny files, no model/audio setup."""
import io
import json
from pathlib import Path

import pytest

from test_gui_verifier_lifecycle import verifier


@pytest.mark.parametrize("stage", ["write", "close", "replace"])
def test_failed_final_save_preserves_previous_evidence_and_never_completes(
        verifier, monkeypatch, capsys, stage):
    path = verifier.root / ".local/gui-integration.json"
    previous = b'{"previous": "successful run"}\n'
    path.write_bytes(previous)
    verifier.state["step"] = 6
    original_open = io.open
    original_replace = Path.replace

    class BrokenStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            if stage == "close" and args[0] is None:
                raise OSError("fixture close")
            return result

        def write(self, text):
            if stage == "write":
                self.stream.write(text[:10])
                raise OSError("fixture write")
            return self.stream.write(text)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def broken_open(file, mode="r", *args, **kwargs):
        stream = original_open(file, mode, *args, **kwargs)
        if "w" in mode and Path(file).parent == path.parent and stage != "replace":
            return BrokenStream(stream)
        return stream

    def broken_replace(source, target):
        if Path(target) == path:
            raise OSError("fixture replace")
        return original_replace(source, target)

    monkeypatch.setattr(io, "open", broken_open)
    monkeypatch.setattr(Path, "replace", broken_replace)
    verifier.ns["check"]()
    assert verifier.state["failed"]
    assert not verifier.state["done"]
    assert path.read_bytes() == previous
    assert sorted(path.parent.iterdir()) == [path]
    assert len(verifier.evidence["errors"]) == 1
    assert "quit" not in verifier.events
    assert ("exit", 1) in verifier.events
    verifier.events.clear()
    verifier.ns["check"]()
    assert verifier.events == []
    assert "PASS_NATIVE_GUI" not in capsys.readouterr().out


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("late_callback", ["check", "abort"])
def test_completed_verification_is_terminal(verifier, capsys, late_callback, expired):
    verifier.state["step"] = 6
    verifier.ns["check"]()
    assert verifier.state["done"] and not verifier.state["failed"]
    path = verifier.root / ".local/gui-integration.json"
    saved = path.read_bytes()
    verifier.events.clear()
    capsys.readouterr()
    # The real timer may already have posted another timeout before stop().
    verifier.ns["time"].monotonic = lambda: 100. if expired else 0.
    if late_callback == "check":
        verifier.ns["check"]()
    else:
        verifier.ns["abort"]("late shutdown notification")
    assert verifier.events == []
    assert verifier.state["done"] and not verifier.state["failed"]
    assert verifier.evidence["errors"] == []
    assert path.read_bytes() == saved
    assert capsys.readouterr().out == ""


def test_serialization_failure_never_sets_done(verifier, capsys):
    verifier.state["step"] = 6
    path = verifier.root / ".local/gui-integration.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)
    verifier.evidence["turns"].append(object())
    verifier.ns["check"]()
    assert verifier.state["failed"]
    assert not verifier.state["done"]
    assert path.read_bytes() == previous
    assert sorted(path.parent.iterdir()) == [path]
    assert "PASS_NATIVE_GUI" not in capsys.readouterr().out


def test_open_failure_preserves_artifact_and_fresh_run_can_recover(verifier, monkeypatch, capsys):
    import tempfile

    path = verifier.root / ".local/gui-integration.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)
    verifier.state["step"] = 6

    def fail_open(**kwargs):
        raise PermissionError("fixture open")

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "NamedTemporaryFile", fail_open)
        verifier.ns["check"]()
    assert verifier.state["failed"] and not verifier.state["done"]
    assert path.read_bytes() == previous
    assert sorted(path.parent.iterdir()) == [path]
    assert "PASS_NATIVE_GUI" not in capsys.readouterr().out
    # A new invocation starts with fresh state; the same artifact path recovers.
    verifier.state.update(failed=False, done=False)
    verifier.evidence["errors"].clear()
    verifier.events.clear()
    verifier.ns["check"]()
    assert verifier.state["done"] and not verifier.state["failed"]
    assert json.loads(path.read_text(encoding="utf-8")) == verifier.evidence
    assert verifier.events == ["timer.stop", "quit"]
    assert "PASS_NATIVE_GUI" in capsys.readouterr().out


def test_multilingual_artifact_uses_utf8_with_ascii_default(verifier, monkeypatch):
    original_open = io.open

    def ascii_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "ascii"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", ascii_open)
    verifier.evidence["turns"].append({"original": "Привіт, ystävä — завтра"})
    verifier.state["step"] = 6
    verifier.ns["check"]()
    path = verifier.root / ".local/gui-integration.json"
    assert verifier.state["done"] and not verifier.state["failed"]
    assert json.loads(path.read_bytes().decode("utf-8")) == verifier.evidence
    assert sorted(path.parent.iterdir()) == [path]
