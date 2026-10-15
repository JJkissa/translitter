# Checkpoint status

## Completed

Verified October 7, 2026:

- Product plan and native design direction documented.
- Isolated Python 3.12 runtime with 54 locked packages and project-local native/CUDA libraries.
- Six revision-pinned assets, 34 verified files, 2,853,752,747 model bytes. Download queue complete.
- Offline Whisper recognition, Finnish↔English Marian translation and Piper Finnish/English synthesis.
- Explicit paired USB endpoints with no default-device fallback, simultaneous input capture and cross-routed target-language playback.
- Fair bounded per-speaker queues, cancellation and stale-generation guards, input validation, thread/load-error handling and safe shutdown.
- Native black/white/pink GUI with headset selectors, swap/refresh, F1/F2/button push-to-talk, opt-in automatic segmentation, original/translated captions, volume and text-only controls.
- Native GUI integration exercised real model worker and actual USB streams/headphones, generated inputs through F1/F2 events, Stop/restart, silence completion and asynchronous close.
- Playback guards and 300 ms cooldown are implemented and unit-tested; they are not full acoustic echo cancellation.
- Installed Qt audio enum/signal binding issue fixed with QtAudio enums and state-query timer; completed output streams are released and Stop resets buffers.
- Latest complete suite: **114 passed in 42.04 seconds** with real model/hardware integrations enabled. Latest native quick suite: **95 passed, 19 skipped in 2.56 seconds**. Launch from outside the project directory also passed.
- Fixed independently reproduced active-PTT speech loss: translated playback defers until recipient release, with at most two FIFO deferred translations per listener. Stop, restart, Text only and close clear them. Overflow retains captions and explicitly warns.
- Extended real GUI/model/USB verification preserved 31,149 held generated English samples while a Finnish translation arrived, submitted them on release and completed both target-headset playbacks. Fixture keyboard events are dispatched through the actual Qt event filter; physical keyboard acceptance remains unverified.
- Focus-loss cancellation is now explicit: unsubmitted held microphone input is discarded, participant states update, and deferred translated playback continues. Queued worker turns and generation are preserved. Real GUI/headset verification confirmed the cancelled generated input was not submitted and queued playback completed.
- Real native window captured and visually inspected. Standard `start.sh` launch verified through desktop capture: models ready, both headset IDs selected, Start enabled, microphone capture off.

Actual generated-speech examples:
- `Hei, miten voit tänään?` → `Hey, how are you today?` → English voice on User 2's headset.
- `Hello, how are you today?` → `Hei, kuinka voit tänään?` → Finnish voice on User 1's headset.

## Evidence

- `model-assets.lock.json`, `requirements.lock.txt`, `.local/setup-report.json`.
- `.local/model-integration.json`, `.local/full-demo-tests/`.
- `.local/usb-pipeline-check.json`: real simultaneous capture and actual PipeWire destination links.
- `.local/gui-integration.json`, `.local/gui-live-check.png`: real GUI + models + headsets with generated PTT input.
- `.local/review-tree.json`: frozen SHA-256 source manifest and clean static scan.
- Repeatable explicit hardware scripts: `scripts/verify_usb.py`, `scripts/verify_gui.py`.

## Independent review — passed for supervised local demo

The initial active-PTT speech-loss and focus-loss playback-stall bugs are fixed and independently confirmed. Final read-only review found no security or logic issues, independently ran 95 quick tests with 19 optional integrations skipped, and exercised both fixes and their bypass variants. All 12 frozen production/verification hashes matched before and after review; parent rechecked them and found no mismatches. Full real-model/USB/native-GUI verification passed all 114 tests. Approval covers the supervised local demo's code and generated-input integration, not human translation or acoustic acceptance. The final verdict is retained in `.local/review-result.json`. The reviewed runnable code is preserved in the repository baseline; model weights and machine-specific verification artifacts are excluded from source control.

## Human acceptance — initial user trial completed; quality improvements required

The user completed a real conversation with another person and reported fast, very good basic-sentence translations. More difficult sentences translated poorly in both directions. Adjacent F1/F2 push-to-talk keys were awkward for two people sharing a keyboard. This is user-reported acceptance feedback, separate from the automated generated-speech tests; no real conversation samples or transcripts were supplied or saved.

The next revision needs configurable separated talk keys, editable custom themes, selectable Finnish/English/Russian/Ukrainian language pairs and stronger evaluated translation quality. These are requirements, not implemented features; see `ROADMAP.md`. Broad complex-sentence quality, sustained use and acoustic feedback behavior remain acceptance gates.

## Running state at this checkpoint

Native launch and model readiness were verified. Microphones open only after explicit Start; current capture state depends on the operator. No cloud inference, driver changes or sudo operations were needed.
