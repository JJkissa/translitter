"""Offline language graph and resource contracts."""
from itertools import permutations

import pytest

PAIRS = tuple(permutations(("fi", "en", "ru", "uk"), 2))
TRANSLATOR_FILES = ("pytorch_model.bin", "config.json", "source.spm", "target.spm", "vocab.json")
EXPECTED_VOICES = {
    "fi": "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "ru": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    "uk": "uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
}


def install_assets(root, first, second):
    edges = {(a, b) for language in (first, second) if language != "en"
             for a, b in ((language, "en"), ("en", language))}
    paths = [f"translate-{a}-{b}/{name}" for a, b in sorted(edges) for name in TRANSLATOR_FILES]
    for language in (first, second):
        voice = f"voice-{language}/{EXPECTED_VOICES[language]}"
        paths.extend((voice, voice + ".json"))
    for relative in paths:
        path = root / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return paths


@pytest.mark.parametrize("first,second", PAIRS)
def test_missing_assets_validate_every_file_for_both_directions(tmp_path, first, second):
    from translitter.languages import VOICE_FILES, missing_assets
    assert VOICE_FILES == EXPECTED_VOICES
    paths = install_assets(tmp_path, first, second)
    assert missing_assets(tmp_path, first, second) == ()
    for relative in paths:
        path = tmp_path / "models" / relative
        path.unlink()
        missing = missing_assets(tmp_path, first, second)
        assert isinstance(missing, tuple) and len(missing) == 1
        assert relative in missing[0]
        path.touch()


def test_additional_assets_ignore_loaded_baseline(tmp_path):
    from translitter.languages import missing_assets
    assert missing_assets(tmp_path, "fi", "en", additional_only=True) == ()
    missing = missing_assets(tmp_path, "fi", "ru", additional_only=True)
    assert missing and all(not item.startswith(("translate-fi-en/", "translate-en-fi/", "voice-fi/", "voice-en/")) for item in missing)
    assert any("translate-ru-en" in item for item in missing)
    assert any("translate-en-ru" in item for item in missing)
    assert any("voice-ru" in item for item in missing)


def test_missing_asset_checks_reject_unknown_pairs(tmp_path):
    from translitter.languages import missing_assets
    with pytest.raises(ValueError):
        missing_assets(tmp_path, "fi", "../outside")


@pytest.mark.parametrize("source,target", PAIRS)
def test_translation_steps_use_only_direct_english_edges(source, target):
    from translitter.languages import LANGUAGE_NAMES, translation_steps
    assert LANGUAGE_NAMES == {"fi": "Finnish", "en": "English", "ru": "Russian", "uk": "Ukrainian"}
    expected = ((source, target),) if "en" in (source, target) else ((source, "en"), ("en", target))
    assert translation_steps(source, target) == expected


@pytest.mark.parametrize("source,target", [("fi", "fi"), ("xx", "en"), ("en", "../ru"), (None, "en")])
def test_invalid_language_pair_is_rejected(source, target):
    from translitter.languages import translation_steps
    with pytest.raises(ValueError):
        translation_steps(source, target)
