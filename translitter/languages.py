"""Supported spoken languages and explicit English-pivot translation graph."""

from pathlib import Path

LANGUAGE_NAMES = {"fi": "Finnish", "en": "English", "ru": "Russian", "uk": "Ukrainian"}

VOICE_FILES = {
    "fi": "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "ru": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    "uk": "uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
}
TRANSLATOR_FILES = ("pytorch_model.bin", "config.json", "source.spm", "target.spm", "vocab.json")


def missing_assets(root, first, second, additional_only=False) -> tuple[str, ...]:
    """Report missing files for both session directions without loading models."""
    edges = set(translation_steps(first, second) + translation_steps(second, first))
    voices = {first, second}
    if additional_only:
        edges -= {("fi", "en"), ("en", "fi")}
        voices -= {"fi", "en"}
    paths = [f"translate-{source}-{target}/{filename}"
             for source, target in sorted(edges) for filename in TRANSLATOR_FILES]
    for language in sorted(voices):
        voice = f"voice-{language}/{VOICE_FILES[language]}"
        paths.extend((voice, voice + ".json"))
    return tuple(path for path in paths if not (Path(root) / "models" / path).is_file())


def translation_steps(source: str, target: str) -> tuple[tuple[str, str], ...]:
    if source not in LANGUAGE_NAMES or target not in LANGUAGE_NAMES or source == target:
        raise ValueError("Select two distinct supported spoken languages")
    if "en" in (source, target):
        return ((source, target),)
    return ((source, "en"), ("en", target))
