"""Benchmark ownership is exercised with synthetic in-memory model stubs."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fails", [False, True])
def test_benchmark_owns_runtime_and_closes_on_success_or_failure(tmp_path, monkeypatch, fails):
    from scripts import benchmark_quality as benchmark
    events = []
    class Client:
        last_metrics = {"seconds": 0.1, "first_token_seconds": 0.01}
        def probe(self):
            return True
        def translate(self, *args):
            if fails:
                raise RuntimeError("benchmark fixture failure")
            return "synthetic candidate"
    client = Client()
    class Runtime:
        def __init__(self, root):
            assert root == tmp_path
        def start(self):
            events.append("start")
            return client
        def close(self):
            events.append("close")
    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, *args, **kwargs):
            return {}
        def decode(self, *args, **kwargs):
            return "synthetic baseline"
    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, *args):
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            return [[1]]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda *a: None,
                                                              inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(MarianMTModel=Model,
                                                                    MarianTokenizer=Tokenizer))
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "QualityRuntime", Runtime, raising=False)
    monkeypatch.setattr(benchmark, "LocalTranslator", lambda: client, raising=False)
    (tmp_path / ".local").mkdir()
    if fails:
        with pytest.raises(RuntimeError, match="benchmark fixture failure"):
            benchmark.main()
    else:
        benchmark.main()
        evidence = json.loads((tmp_path / ".local/quality-benchmark.json").read_text())
        assert len(benchmark.CASES) == len(evidence["records"]) == evidence["summary"]["cases"] == 14
    assert events == ["start", "close"]
