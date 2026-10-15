# Historical demo environment preparation

## Status: dependency preparation completed; application now implemented

This document preserves the initial dependency-only checkpoint, not current application status. The implemented demo and real-person feedback are summarized in `STATUS.md`; requested improvements are in `ROADMAP.md`.

Preparation completed on October 7, 2026. All six planned model assets are installed and verified. The download queue is empty; no job is running in the background.

## Installed runtime

- Python 3.12.3 isolated under `.venv/`.
- PySide6 6.11.2 and Qt Multimedia.
- faster-whisper 1.2.1 / CTranslate2 4.8.2.
- Transformers 4.57.6, CPU PyTorch 2.14.1, SentencePiece and sacremoses.
- Piper 1.8.0 and ONNX Runtime 1.30.0.
- NumPy, SciPy, SoundFile, PyAV and test dependencies.
- NVIDIA cuBLAS 12.9.2.10, cuDNN 9.27.0.42 and NVRTC 12.9.86 as project-local Python wheel libraries.
- Native `libxcb-cursor0` 0.1.4-1build1 from configured Ubuntu package sources, extracted under `.native/` without root.

There are 54 installed Python packages including transitive dependencies. `requirements.lock.txt` exactly matches the actual environment's freeze output; `uv pip check` reports compatible installed dependencies.

Whisper runs through CTranslate2's CUDA backend. CPU PyTorch is deliberately separate and runs text translation; a CPU PyTorch build does not mean Whisper is restricted to CPU.

## Installed model assets

| Key | Purpose | Actual verification |
| --- | --- | --- |
| `whisper-turbo` | Multilingual large-v3-turbo ASR | Finnish and English generated speech recognized on CUDA |
| `whisper-small` | Smaller multilingual fallback | Both languages recognized on CUDA and CPU |
| `translate-en-fi` | English → Finnish translation | Real English text translated offline |
| `translate-fi-en` | Finnish → English translation | Real Finnish text translated offline |
| `voice-fi` | Finnish Harri medium voice | Actual Finnish WAV speech synthesized offline |
| `voice-en` | English Lessac medium voice | Actual English WAV speech synthesized offline |

All six assets have pinned repository revisions, required filenames, sizes and SHA-256 values recorded in `model-assets.lock.json`. Large-file hashes were checked against publisher metadata. The integration suite independently reread and hashed every required file. There are 34 files totaling 2,853,752,747 bytes (2.658 GiB), excluding download metadata/cache.

The VAD model and English/Finnish phonemization resources are supplied by the installed faster-whisper/Piper packages. Both paths executed without additional runtime downloads. Voice model cards are stored alongside their weights; distribution/license review is outside this local setup checkpoint.

## Real inference results

Text translation returned:

| Input | Actual output |
| --- | --- |
| `Hei, miten voit tänään?` | `Hey, how are you today?` |
| `Hello, how are you today?` | `Hei, kuinka voit tänään?` |

Piper generated Finnish and English speech at 22,050 Hz. The resulting sample audio lasted 2.009 seconds and 1.544 seconds respectively in the recorded run. These were generated test fixtures, not microphone recordings.

Whisper small and turbo recognized the Finnish sample as `Hei, miten voit tänään?` and the English sample as `Hello, how are you today?`. CUDA used `int8_float16`; small's CPU fallback used `int8`.

These samples prove the local engines run together, not broad conversational accuracy or live microphone-to-headphone latency. The final UI, segmentation settings and simultaneous-device behavior still need real-person trials.

## Devices and drivers

Both connected USB headsets expose separate input/output pairs in PipeWire and Qt. Their human-readable names can be identical, so the application uses device IDs and explicit pairing. Physical assignment to User 1 versus User 2 requires identifying the actual headset.

`snd_usb_audio` is loaded, and both capture/playback devices appear in ALSA/PipeWire. The existing NVIDIA driver executes both Whisper models. No driver replacement, reboot, system audio configuration change or administrative installation was necessary.

Qt's native XCB plugin initially failed because `libxcb-cursor.so.0` was absent. `ldd` isolated that dependency. Supplying the distribution library project-locally made native Qt initialization pass.

## Project-local Python launcher

Run project Python through the verified launcher:

```bash
./run-python.sh your_script.py
./run-python.sh -m pytest tests -q
```

The launcher supplies project-local XCB and NVIDIA wheel library paths. It defaults Hugging Face/Transformers to offline operation and disables Hub telemetry. `run-python.sh` launches the project Python environment; the implemented GUI is launched through `./start.sh`.

Full real-model validation is explicitly opt-in:

```bash
TRANSLITTER_MODEL_TESTS=1 ./run-python.sh -m pytest tests -q -s --basetemp .local/model-test-tmp
```

Actual result: **17 passed in 10.45 seconds**. The normal quick suite intentionally skips the heavier model tests unless that flag is supplied. Integration tests generate speech only in temporary test storage and never capture microphones or play audio to devices.

## Recreate Python libraries

Use an isolated environment; do not change PinkFlow or the global interpreter. Install the exact CPU torch wheel from its official index before the full lock:

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python 'torch==2.14.1+cpu' --index https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip check --python .venv/bin/python
```

The native cursor library remains a separate distribution dependency under `.native/`. Weights live under ignored `models/`; reinstalling Python libraries alone does not reinstall model assets. Use the repository/revision/file list in `model-assets.lock.json` to reproduce those downloads rather than resolving a rolling model alias.

## Exact verification boundary

Verified: compatible libraries, native Qt platform initialization, two-device enumeration, pinned asset hashes, real translation, real voice synthesis, both Whisper GPU backends, small's CPU fallback and VAD during recognition.

Not verified: the application GUI, real-person recognition quality, simultaneous live microphone capture, physical participant assignment, destination-specific headphone playback, interruption/echo control or a sustained conversation. No microphones were opened and no audible output was played.

Machine-specific evidence is under ignored `.local/`: `setup-report.json`, `model-integration.json`, `translation-smoke.json` and generated test files. No GitHub publication was performed.
