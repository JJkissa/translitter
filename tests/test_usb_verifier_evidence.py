"""USB verifier artifact/terminal callbacks without hardware or model setup."""
import io
import json
from pathlib import Path

import pytest

from test_usb_verifier_lifecycle import verifier as callback_verifier


@pytest.fixture
def verifier(callback_verifier, tmp_path):
    v = callback_verifier
    v.root = tmp_path
    (tmp_path / ".local").mkdir()
    v.evidence["turns"] = [{"verified_graph": ["fi → en"]},
                           {"verified_graph": ["en → fi"]}]
    v.ns["hub"].capture_bytes = [20, 40]
    v.ns["hub"].played_bytes = [60, 80]
    v.ns["hub"].buffers = {}
    v.ns["worker"].stop_session = lambda generation: v.events.append(("stop_session", generation))
    v.state["playing"] = None
    return v


@pytest.mark.parametrize("stage", ["write", "close", "replace"])
def test_failed_save_preserves_previous_artifact(verifier, monkeypatch, capsys, stage):
    v = verifier
    path = v.root / ".local/usb-pipeline-check.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)
    original_open, original_replace = io.open, Path.replace

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
    v.ns["finish"]()
    assert v.state["failed"] and not v.state["done"]
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
    assert len(v.evidence["errors"]) == 1
    assert "PASS_BOTH" not in capsys.readouterr().out
    v.events.clear()
    v.ns["finish"]()
    assert v.events == []


@pytest.mark.parametrize("stage", ["serialize", "open"])
def test_failed_save_never_sets_done_and_fresh_run_recovers(verifier, monkeypatch, capsys, stage):
    import tempfile

    v = verifier
    path = v.root / ".local/usb-pipeline-check.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)
    with monkeypatch.context() as patch:
        if stage == "serialize":
            v.evidence["unserializable"] = object()
        else:
            def fail_open(**kwargs):
                raise PermissionError("fixture open")
            patch.setattr(tempfile, "NamedTemporaryFile", fail_open)
        v.ns["finish"]()
    assert v.state["failed"] and not v.state["done"]
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
    assert "PASS_BOTH" not in capsys.readouterr().out
    v.evidence.pop("unserializable", None)
    v.evidence["errors"].clear()
    v.state.update(failed=False, done=False)
    v.ns["finish"]()
    assert v.state["done"] and not v.state["failed"]
    assert json.loads(path.read_text(encoding="utf-8")) == v.evidence
    assert "PASS_BOTH" in capsys.readouterr().out


@pytest.mark.parametrize("callback,args", [
    ("finish", ()), ("fail", ("late timeout",)), ("begin", ()),
    ("receive", (None,)), ("inspect_graph", (1,)), ("on_playing", (1, False)),
])
def test_success_is_terminal_for_queued_callbacks(verifier, capsys, callback, args):
    v = verifier
    v.ns["finish"]()
    assert v.state["done"] and not v.state["failed"]
    path = v.root / ".local/usb-pipeline-check.json"
    saved = path.read_bytes()
    evidence = json.dumps(v.evidence)
    v.events.clear()
    v.pending.clear()
    capsys.readouterr()
    v.ns[callback](*args)
    assert v.events == []
    assert v.pending == []
    assert v.state["done"] and not v.state["failed"]
    assert json.dumps(v.evidence) == evidence
    assert path.read_bytes() == saved
    assert capsys.readouterr().out == ""


def test_queued_second_turn_is_cancelled_after_success(verifier):
    v = verifier
    completed = v.evidence["turns"].pop()
    v.state["playing"] = 1
    v.ns["on_playing"](1, False)
    queued = v.pending.pop()
    v.evidence["turns"].append(completed)
    v.ns["finish"]()
    v.events.clear()
    queued()
    assert v.events == []


def test_multilingual_artifact_is_explicit_utf8(verifier, monkeypatch):
    original_open = io.open

    def ascii_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "ascii"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", ascii_open)
    verifier.ns["finish"]()
    path = verifier.root / ".local/usb-pipeline-check.json"
    assert verifier.state["done"] and not verifier.state["failed"]
    assert json.loads(path.read_bytes().decode("utf-8")) == verifier.evidence
    assert list(path.parent.iterdir()) == [path]
