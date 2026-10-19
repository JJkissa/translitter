"""Small synthetic benchmark probes; no models, audio, or network."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from scripts import benchmark_quality as benchmark

    state = SimpleNamespace(events=[], fail=None, calls=[], loads=[], decode=[])
    cases = (("ru", "fi", "first fixture"), ("en", "fi", "second fixture"))

    def fail_at(stage):
        if state.fail == stage:
            raise RuntimeError("fixture " + stage)
        if state.fail == "interrupt" and stage == "translate-2":
            raise KeyboardInterrupt("fixture interrupt")

    class Client:
        last_metrics = {}

        def probe(self):
            state.events.append("probe")
            fail_at("probe")
            return True

        def translate(self, text, source, target):
            state.calls.append((text, source, target))
            fail_at(f"translate-{len(state.calls)}")
            self.last_metrics = {"seconds": len(state.calls) / 10,
                                 "first_token_seconds": len(state.calls) / 100}
            return "candidate " + text

    client = Client()

    class Runtime:
        def __init__(self, root):
            assert root == tmp_path

        def start(self):
            state.events.append("start")
            fail_at("start")
            return client

        def close(self):
            state.events.append("close")

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, directory, **kwargs):
            assert kwargs == {"local_files_only": True}
            state.loads.append(directory.name)
            fail_at("load")
            return cls()

        def __call__(self, values, **kwargs):
            state.decode.append(values[0])
            return {"text": values[0]}

        def decode(self, output, **kwargs):
            return output[0]

    class Model:
        @classmethod
        def from_pretrained(cls, directory, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False,
                              "weights_only": True}
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def generate(self, text, **kwargs):
            fail_at("generate")
            return [["baseline " + text]]

    clock = iter(range(100))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda value: None, inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        MarianMTModel=Model, MarianTokenizer=Tokenizer))
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "QualityRuntime", Runtime)
    monkeypatch.setattr(benchmark, "CASES", cases)
    monkeypatch.setattr(benchmark, "monotonic", lambda: next(clock))
    state.benchmark = benchmark
    state.path = tmp_path / ".local/quality-benchmark.json"
    return state


def test_first_benchmark_creates_its_evidence_directory(harness):
    assert not harness.path.parent.exists()
    harness.benchmark.main()
    data = json.loads(harness.path.read_text())
    assert data["summary"]["cases"] == len(data["records"]) == 2
    assert harness.events == ["start", "probe", "close"]


def test_pivot_metrics_and_model_cache_are_attributed_to_each_case(harness):
    harness.path.parent.mkdir()
    harness.benchmark.main()
    data = json.loads(harness.path.read_text())
    first, second = data["records"]
    assert first["baseline_steps"] == [["ru", "en"], ["en", "fi"]]
    assert first["baseline"] == "baseline baseline first fixture"
    assert second["baseline"] == "baseline second fixture"
    assert harness.loads == ["translate-ru-en", "translate-en-fi"]
    assert harness.calls == [("first fixture", "ru", "fi"),
                             ("second fixture", "en", "fi")]
    assert [(r["case"], r["seconds"], r["first_token_seconds"],
             r["baseline_seconds"], r["candidate_direct"]) for r in data["records"]] == [
        (1, 0.1, 0.01, 1, True), (2, 0.2, 0.02, 1, True)]
    assert data["summary"]["baseline_median_seconds"] == 1
    assert data["summary"]["candidate_median_seconds"] == pytest.approx(0.15)
    assert data["summary"]["candidate_median_first_token_seconds"] == pytest.approx(0.015)
    assert data["summary"]["human_accuracy_verdict"] == "not yet evaluated"


@pytest.mark.parametrize("stage", ["start", "probe", "load", "generate",
                                   "translate-1", "translate-2", "interrupt"])
def test_failure_closes_runtime_and_only_checkpoints_complete_cases(harness, stage):
    harness.path.parent.mkdir()
    harness.fail = stage
    exception = KeyboardInterrupt if stage == "interrupt" else RuntimeError
    with pytest.raises(exception, match="fixture"):
        harness.benchmark.main()
    assert harness.events.count("close") == 1
    assert harness.events[-1] == "close"
    if stage in ("translate-2", "interrupt"):
        data = json.loads(harness.path.read_text())
        assert "summary" not in data
        assert len(data["records"]) == 1
        assert data["records"][0]["candidate"] == "candidate first fixture"
        assert data["records"][0]["seconds"] == 0.1
    else:
        assert not harness.path.exists()
