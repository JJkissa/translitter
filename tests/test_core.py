"""Direction and validation contracts for the two physical headset routes."""
import importlib.util
import pytest


def core():
    assert importlib.util.find_spec("translitter") is not None, "Pipeline core is missing"
    from translitter import core as module
    return module


def test_finnish_input_is_spoken_in_english_only_to_second_headset():
    m = core()
    first = m.Headset("A", "microphone-a", "output-a", "Headset A")
    second = m.Headset("B", "microphone-b", "output-b", "Headset B")
    routes = m.make_routes(first, second)
    assert routes[0].source_language == "fi"
    assert routes[0].target_language == "en"
    assert routes[0].microphone_id == "microphone-a"
    assert routes[0].output_id == "output-b"
    assert routes[0].listener == 1
    assert routes[1].source_language == "en"
    assert routes[1].target_language == "fi"
    assert routes[1].microphone_id == "microphone-b"
    assert routes[1].output_id == "output-a"
    assert routes[1].listener == 0


@pytest.mark.parametrize("second", [
    ("A", "microphone-b", "output-b"),
    ("B", "microphone-a", "output-b"),
    ("B", "microphone-b", "output-a"),
    ("B", "", "output-b"),
])
def test_shared_or_missing_audio_endpoints_are_rejected(second):
    m = core()
    first = m.Headset("A", "microphone-a", "output-a", "Headset A")
    with pytest.raises(ValueError):
        m.make_routes(first, m.Headset(*second, "Headset B"))
