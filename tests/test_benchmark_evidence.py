"""Evidence persistence failures use real files and tiny synthetic models."""
import io
import json
from pathlib import Path

import pytest

from test_benchmark_boundaries import harness


@pytest.mark.parametrize("stage", ["write", "close", "replace"])
@pytest.mark.parametrize("save_number", [2, 3])
def test_failed_evidence_save_preserves_previous_checkpoint_and_allows_retry(
        harness, monkeypatch, capsys, stage, save_number):
    original_open = io.open
    original_replace = Path.replace
    saves = []
    previous = []

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            if stage == "close" and args[0] is None:
                raise OSError("fixture evidence close")
            return result

        def write(self, text):
            if stage == "write":
                self.stream.write(text[:20])
                raise OSError("fixture evidence write")
            return self.stream.write(text)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def failing_open(file, mode="r", *args, **kwargs):
        evidence_save = (isinstance(file, (str, Path)) and "w" in mode
                         and (Path(file).parent == harness.path.parent
                              or Path(file) == harness.path.parent))
        if evidence_save:
            saves.append(Path(file))
            if len(saves) == save_number:
                previous.append(harness.path.read_bytes())
        stream = original_open(file, mode, *args, **kwargs)
        if evidence_save and len(saves) == save_number and stage != "replace":
            return FailingStream(stream)
        return stream

    def failing_replace(path, target):
        if Path(target) == harness.path and len(saves) == save_number:
            raise OSError("fixture evidence replace")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", failing_open)
        patch.setattr(Path, "replace", failing_replace)
        with pytest.raises(OSError, match="fixture evidence " + stage):
            harness.benchmark.main()

    assert harness.events == ["start", "probe", "close"]
    assert len(previous) == 1
    assert harness.path.read_bytes() == previous[0]
    data = json.loads(harness.path.read_text(encoding="utf-8"))
    assert "summary" not in data
    assert len(data["records"]) == save_number - 1
    assert sorted(harness.path.parent.iterdir()) == [harness.path]
    assert "BENCHMARK_SUMMARY" not in capsys.readouterr().out

    harness.calls.clear()
    harness.benchmark.main()
    recovered = json.loads(harness.path.read_text(encoding="utf-8"))
    assert recovered["summary"]["cases"] == len(recovered["records"]) == 2
    assert harness.events == ["start", "probe", "close"] * 2


@pytest.mark.parametrize("stage", ["serialize", "open"])
def test_failure_before_evidence_stream_preserves_existing_file(
        harness, monkeypatch, stage):
    harness.path.parent.mkdir()
    original = b'{"previous": "completed checkpoint"}\n'
    harness.path.write_bytes(original)

    if stage == "open":
        def fail_open(**kwargs):
            raise PermissionError("fixture evidence open")
        monkeypatch.setattr(harness.benchmark, "NamedTemporaryFile", fail_open)
        data = {"records": []}
        exception = PermissionError
    else:
        data = {"records": [object()]}
        exception = TypeError

    with pytest.raises(exception):
        harness.benchmark._save_evidence(harness.path, data)
    assert harness.path.read_bytes() == original
    assert sorted(harness.path.parent.iterdir()) == [harness.path]


def test_non_ascii_evidence_uses_explicit_utf8_even_with_ascii_default(
        harness, monkeypatch):
    original_open = io.open
    text = "Привіт, ystävä — завтра"
    monkeypatch.setattr(harness.benchmark, "CASES", (("uk", "fi", text),))

    def ascii_default_open(file, mode="r", buffering=-1, encoding=None,
                           errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "ascii"
        return original_open(file, mode, buffering, encoding, errors, newline,
                             closefd, opener)

    monkeypatch.setattr(io, "open", ascii_default_open)
    harness.benchmark.main()
    data = json.loads(harness.path.read_bytes().decode("utf-8"))
    assert data["records"][0]["input"] == text
    assert data["records"][0]["candidate"] == "candidate " + text
    assert data["summary"]["cases"] == 1
    assert harness.events == ["start", "probe", "close"]
