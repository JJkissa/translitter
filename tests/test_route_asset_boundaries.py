"""Bounded offline routing/asset sweep; no models or physical endpoints."""
from dataclasses import replace
from itertools import permutations

import pytest

from translitter.core import Headset, Route, make_routes
from translitter.languages import missing_assets
from test_languages import EXPECTED_VOICES, TRANSLATOR_FILES, install_assets

PAIRS = tuple(permutations(("fi", "en", "ru", "uk"), 2))
FIRST = Headset("usb-device.a", "alsa_input.usb-device.a.capture", "alsa_output.usb-device.a.playback", "Same label")
SECOND = Headset("usb-device.b", "alsa_input.usb-device.b.capture", "alsa_output.usb-device.b.playback", "Same label")


@pytest.mark.parametrize("first,second", PAIRS)
def test_additional_inventory_is_exact_symmetric_and_recovers(tmp_path, first, second):
    # Independent expected inventory: each non-English participant contributes
    # both directed English edges; baseline Finnish edges/voices are loaded.
    edges = {(a, b) for language in (first, second) if language not in ("fi", "en")
             for a, b in ((language, "en"), ("en", language))}
    expected = [f"translate-{a}-{b}/{filename}"
                for a, b in sorted(edges) for filename in TRANSLATOR_FILES]
    for language in sorted({first, second} - {"fi", "en"}):
        voice = f"voice-{language}/{EXPECTED_VOICES[language]}"
        expected.extend((voice, voice + ".json"))
    assert missing_assets(tmp_path, first, second, additional_only=True) == tuple(expected)
    assert missing_assets(tmp_path, second, first, additional_only=True) == tuple(expected)
    install_assets(tmp_path, first, second)
    assert missing_assets(tmp_path, first, second, additional_only=True) == ()
    # Baseline removal must remain ignored; every optional removed file must be
    # reported alone, without cached readiness or reverse-direction omissions.
    for relative in install_assets(tmp_path, first, second):
        path = tmp_path / "models" / relative
        path.unlink()
        want = (relative,) if relative in expected else ()
        assert missing_assets(tmp_path, first, second, additional_only=True) == want
        path.touch()


@pytest.mark.parametrize("first,second", PAIRS)
def test_directories_cannot_impersonate_model_files(tmp_path, first, second):
    paths = install_assets(tmp_path, first, second)
    for relative in paths:
        path = tmp_path / "models" / relative
        path.unlink()
        path.mkdir()
        assert missing_assets(tmp_path, first, second) == (relative,)
        path.rmdir()
        path.touch()
    assert missing_assets(tmp_path, first, second) == ()


@pytest.mark.parametrize("source,target", PAIRS)
@pytest.mark.parametrize("swapped", (False, True))
def test_distinct_ids_survive_identical_labels_and_participant_swap(source, target, swapped):
    first, second = (SECOND, FIRST) if swapped else (FIRST, SECOND)
    assert make_routes(first, second, source, target) == (
        Route(0, 1, source, target, first.microphone_id, second.output_id),
        Route(1, 0, target, source, second.microphone_id, first.output_id),
    )
    assert (first, second) == ((SECOND, FIRST) if swapped else (FIRST, SECOND))


@pytest.mark.parametrize("speaker", (0, 1))
@pytest.mark.parametrize("field", ("key", "microphone_id", "output_id"))
@pytest.mark.parametrize("value", ("", None))
def test_incomplete_endpoint_identity_rejects_both_participants(speaker, field, value):
    pairs = [FIRST, SECOND]
    pairs[speaker] = replace(pairs[speaker], **{field: value})
    with pytest.raises(ValueError, match="complete microphone/headphone pairs"):
        make_routes(pairs[0], pairs[1])
    assert make_routes(FIRST, SECOND)[0].output_id == SECOND.output_id


@pytest.mark.parametrize("speaker", (0, 1))
@pytest.mark.parametrize("field", ("key", "microphone_id", "output_id"))
def test_one_shared_identity_rejects_otherwise_distinct_pairs(speaker, field):
    pairs = [FIRST, SECOND]
    pairs[speaker] = replace(pairs[speaker], **{field: getattr(pairs[1 - speaker], field)})
    with pytest.raises(ValueError, match="different USB headset"):
        make_routes(pairs[0], pairs[1])
    assert make_routes(FIRST, SECOND)[1].output_id == FIRST.output_id
