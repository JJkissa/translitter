"""Marian generation/decode and pivot isolation, without models or audio I/O."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def marian(tmp_path):
    engine = ModelEngine(tmp_path)
    engine._loaded = True  # Prepared fake models; initialization is tested separately.
    engine._torch = SimpleNamespace(inference_mode=nullcontext)
    state = SimpleNamespace(hook=lambda edge, stage: None,
                            outputs={}, decoded={})
    calls, spoken = [], []
    edges = (("ru", "en"), ("en", "uk"))

    class Tokenizer:
        def __init__(self, edge):
            self.edge = edge

        def __call__(self, texts, **kwargs):
            calls.append((self.edge, "tokenize", texts, kwargs))
            state.hook(self.edge, "tokenize")
            return {"input_ids": [[1]]}

        def decode(self, tokens, **kwargs):
            calls.append((self.edge, "decode", tokens, kwargs))
            state.hook(self.edge, "decode")
            if not len(tokens):
                return ""  # An empty token sequence decodes to no text.
            return state.decoded.get(self.edge, "English pivot" if self.edge == edges[0] else "Final translation")

    def model_for(edge):
        def generate(**kwargs):
            calls.append((edge, "generate", kwargs))
            state.hook(edge, "generate")
            return state.outputs.get(edge, [[1, 2]])
        return SimpleNamespace(generate=generate)

    engine._translations = {edge: (Tokenizer(edge), model_for(edge)) for edge in edges}
    engine._synthesize = lambda text, language, cancelled: (
        spoken.append((text, language)) or (np.array([0.2, -0.1], dtype=np.float32), 16000))
    route = Route(0, 1, "ru", "uk", "mic", "out")
    return engine, route, state, calls, spoken, edges


def interpret(marian, cancelled=lambda: False, generation=1):
    engine, route, *_ = marian
    return engine.interpret_text("Original speech", route, generation,
                                 cancelled=cancelled, prepared=True)


def recover(marian):
    _, _, state, calls, spoken, edges = marian
    state.hook = lambda edge, stage: None
    state.outputs.clear()
    state.decoded.clear()
    calls.clear()
    result = interpret(marian, generation=2)
    assert result.generation == 2
    assert result.original == "Original speech"
    assert result.translation == "Final translation"
    assert spoken == [("Final translation", "uk")]
    assert [(call[0], call[1]) for call in calls] == [
        (edge, stage) for edge in edges for stage in ("tokenize", "generate", "decode")]
    assert [call[2] for call in calls if call[1] == "tokenize"] == [
        ["Original speech"], ["English pivot"]]
    for call in calls:
        if call[1] == "generate":
            assert call[2] == {"input_ids": [[1]], "max_new_tokens": 256, "num_beams": 1}
        elif call[1] == "decode":
            assert call[3] == {"skip_special_tokens": True}


@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("stage", ["tokenize", "generate", "decode"])
def test_cancellation_at_each_pivot_boundary_stops_downstream_and_recovers(marian, leg, stage):
    _, _, state, calls, spoken, edges = marian
    stopped = False

    def hook(edge, current):
        nonlocal stopped
        if (edge, current) == (edges[leg], stage):
            stopped = True

    state.hook = hook
    assert interpret(marian, lambda: stopped) is None
    expected = [(edge, current) for edge in edges for current in ("tokenize", "generate", "decode")]
    assert [(call[0], call[1]) for call in calls] == expected[:expected.index((edges[leg], stage)) + 1]
    assert spoken == []
    recover(marian)


@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("stage", ["tokenize", "generate", "decode"])
def test_exception_at_each_pivot_boundary_rejects_whole_turn_and_recovers(marian, leg, stage):
    _, _, state, calls, spoken, edges = marian

    def hook(edge, current):
        if (edge, current) == (edges[leg], stage):
            raise RuntimeError("Backend interrupted")

    state.hook = hook
    with pytest.raises(RuntimeError, match="Backend interrupted"):
        interpret(marian)
    assert (calls[-1][0], calls[-1][1]) == (edges[leg], stage)
    assert spoken == []
    recover(marian)


@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("output,error", [(None, TypeError), ([], IndexError), ({}, KeyError)])
def test_malformed_generated_batch_does_not_publish_pivot_or_speech(marian, leg, output, error):
    _, _, state, calls, spoken, edges = marian
    state.outputs[edges[leg]] = output
    with pytest.raises(error):
        interpret(marian)
    assert (calls[-1][0], calls[-1][1]) == (edges[leg], "generate")
    assert spoken == []
    recover(marian)


@pytest.mark.parametrize("leg", [0, 1])
def test_empty_generated_sequence_rejects_whole_turn_and_recovers(marian, leg):
    _, _, state, calls, spoken, edges = marian
    state.outputs[edges[leg]] = [[]]
    assert interpret(marian) is None
    assert (calls[-1][0], calls[-1][1]) == (edges[leg], "decode")
    assert spoken == []
    recover(marian)


@pytest.mark.parametrize("leg", [0, 1])
@pytest.mark.parametrize("text", ["", " \t\n", "<pad></s>", "\ufffd", "Valid\x1c", None, 123])
def test_invalid_decoded_leg_never_reaches_next_leg_or_synthesis(marian, leg, text):
    _, _, state, calls, spoken, edges = marian
    state.decoded[edges[leg]] = text
    assert interpret(marian) is None
    assert (calls[-1][0], calls[-1][1]) == (edges[leg], "decode")
    assert spoken == []
    recover(marian)


@pytest.mark.parametrize("source,target", [("ru", "en"), ("en", "uk")])
def test_direct_route_uses_only_its_edge_and_normalizes_valid_decode(marian, source, target):
    engine, _, state, calls, spoken, _ = marian
    state.decoded[source, target] = " \tValid translation\n "
    route = Route(0, 1, source, target, "mic", "out")
    result = engine.interpret_text("Original speech", route, 3, prepared=True)
    assert result.translation == "Valid translation"
    assert [call[0] for call in calls] == [(source, target)] * 3
    assert spoken == [("Valid translation", target)]
