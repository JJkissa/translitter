# Translitter — implementation plan

Status: historical initial design plan. The verified Finnish↔English demo is implemented; see `STATUS.md` for current evidence and `ROADMAP.md` for requested improvements.
Working name: Translitter. The product is a two-person Finnish ↔ English conversation interpreter, not merely a phonetic spelling converter.

## 1. Product contract

One Linux computer, two participants, two independent microphone inputs and two independently selectable headphone outputs.

| Participant | Speaks | Phase 1 sees | Phase 2 hears |
| --- | --- | --- | --- |
| User 1 | Finnish | Their Finnish transcript and User 2's Finnish translation | User 2's words synthesized in Finnish |
| User 2 | English | Their English transcript and User 1's English translation | User 1's words synthesized in English |

The shared screen also offers a chronological bilingual conversation view. Both directions remain visible and distinctly labeled. Original transcription is retained alongside translation so errors can be inspected rather than hidden.

Phase 1 includes translation to text, not only recognition in the original language. Phase 2 adds synthesized playback; it does not replace the text view.

## 2. Important model constraint

Whisper supports multilingual transcription, but its speech-translation task targets English; it does not provide the English → Finnish half of this product. OpenAI also states that turbo is not trained for the translation task.[1]

Proposed default: use Whisper for source-language transcription in both directions, then use an explicit text-translation adapter for each direction. This gives a consistent architecture and preserves both original and translated text.

Initial local text-translation candidates are Helsinki-NLP/opus-mt-en-fi and Helsinki-NLP/opus-mt-fi-en, whose model cards identify the required source and target languages.[3][4] They are baseline candidates, not a claim of adequate conversational quality. Choose the release default only after Finnish/English human review of the trial corpus. Finnish → English direct Whisper translation can be an optional comparison backend later, using an appropriate non-turbo model.[1]

## 3. Architecture

```text
USER 1 / FINNISH
Mic A → capture A → speech segmentation → Whisper ASR (fi)
      → Finnish original + English translation → GUI
      → [Phase 2: English TTS → Headphones B]

USER 2 / ENGLISH
Mic B → capture B → speech segmentation → Whisper ASR (en)
      → English original + Finnish translation → GUI
      → [Phase 2: Finnish TTS → Headphones A]
```

Two independent logical pipelines, with separate buffers, segmentation state, language settings, mute controls, queues and diagnostics. A shared resident ASR model and a fair scheduler are the initial resource strategy; capture remains concurrent while inference jobs may be serialized. Do not load two large models simply because there are two participants.

Proposed modules:

- `domain`: participants, routes, session state and utterance records.
- `audio`: enumeration, explicit input/output selection, capture and playback adapters, format negotiation and resampling.
- `segmentation`: push-to-talk first, then automatic speech start/end detection and bounded utterance chunks.
- `asr`: Whisper backend abstraction and inference worker.
- `translation`: explicit Finnish → English and English → Finnish adapters.
- `tts`: Phase 2 engine adapter and destination-specific playback queues.
- `controller`: lifecycle, fair scheduling, cancellation, queue limits and cross-route validation.
- `ui`: native Qt views and presentation only.
- `settings`: local non-secret configuration and explicit model installation state.

Every utterance carries a session generation, speaker ID, sequence ID, source/target language, timings, original text, translated text and state. Validate the intended destination before playback. Late results from a stopped or superseded session are ignored.

## 4. Proposed technology choices

| Area | Starting choice | Selection gate |
| --- | --- | --- |
| Desktop GUI | Python + PySide6/Qt Widgets | Real native window and visual inspection |
| Capture/playback | Qt Multimedia, explicit QAudioDevice per stream | Two-device simultaneous capture and isolated playback |
| ASR | faster-whisper, multilingual Whisper weights | Real Finnish/English inference and GPU compatibility |
| Translation | Direction-specific local Marian/OPUS models | Human-reviewed conversational corpus |
| Voice output | Piper candidate with Finnish and English voices | Intelligibility, latency and license review |
| Tests | pytest + pytest-qt; separate hardware integration gates | Reproducible unit, native UI and real-audio evidence |

Qt provides constructors that select a specific capture device and a specific playback device, fitting the explicit routing requirement.[5][6] Verify the installed Qt version's audio backend and Python bindings; do not assume features from the latest online documentation are present locally.

faster-whisper provides CTranslate2-backed Whisper inference and CPU/GPU compute options.[2] Start a small multilingual model for wiring checks, then benchmark medium and large-v3-turbo transcription. Do not use an English-only `.en` model on the Finnish pipeline. GPU inference requires compatible CUDA/cuDNN libraries according to the project's requirements.[2] Validate the actual GPU/runtime combination before choosing precision or promising latency; retain a labeled CPU fallback.

Piper is a local neural TTS candidate, and its voice collection includes a Finnish locale.[7][8] The engine repository identifies GPL-3.0 licensing; review engine integration and each chosen voice's own model card before distribution.[7] Keep the TTS adapter replaceable. No voice cloning in the initial scope.

Use pinned, tested dependency versions in the application environment; do not modify PinkFlow or the global Python environment.

## 5. Audio and equipment contract

The required routes are:

- Capture User 1 only from Mic A; translate their words toward User 2.
- Capture User 2 only from Mic B; translate their words toward User 1.
- Play English only to output B and Finnish only to output A.
- Never silently use default speakers or swap a missing headset for another device.

Assumption pending hardware confirmation: both headphones have microphones, or two separate microphones will be supplied. The hardware must expose independently selectable inputs and outputs; a duplicated shared endpoint does not meet this requirement. Two USB headsets or two USB audio adapters are candidate arrangements, not a purchasing instruction.

Persist device identity, not display labels alone or transient enumeration positions. Identical devices and reconnection require explicit verification; if identity is ambiguous, ask the operator to reselect. Validate two distinct capture devices; if a shared multichannel interface is used, make channel assignment explicit rather than accidentally capturing the same microphone twice.

Negotiate a supported native capture format, then normalize for ASR outside the capture path. Provide microphone meters, clipping indicators and a simple route-verification screen. Input test recording is temporary and only starts after an explicit user action. Output tests are opt-in and low-volume.

Device disconnect pauses the affected pipeline and cancels its pending playback. Audio routing changes require a stopped session or an explicit pause/reconfigure step. Application exit stops capture and playback and releases devices.

## 6. GUI direction: PinkFlow visual language

Reuse the visual language, not the automation application's behavior or dependencies. Local inspection of PinkFlow's theme confirms these design anchors:

- Deep near-black workspace: `#111114`.
- Dark raised panels: `#151518`.
- Primary pink: `#ff5eaa`.
- Main text: `#f5f5f7`; muted text: `#95959f`.
- Restrained borders, clear selected/focus states and Noto Sans typography.

Proposed layout:

```text
┌─────────────────────────────────────────────────────────────┐
│ Translitter      Finnish ↔ English      Model / Session state│
├────────────┬───────────────────────┬────────────────────────┤
│ Session    │ USER 1 — FINNISH      │ USER 2 — ENGLISH       │
│ Devices    │ Mic / meter / mute    │ Mic / meter / mute     │
│ Models     │ Finnish original      │ English original       │
│ Settings   │ Finnish received text │ English received text  │
│            ├───────────────────────┴────────────────────────┤
│            │ Chronological bilingual conversation           │
├────────────┴────────────────────────────────────────────────┤
│ Start / Stop    Talk controls    Queue / latency / errors    │
└─────────────────────────────────────────────────────────────┘
```

A device setup view assigns microphones and destinations visibly before Start. Large translated text, secondary original text, selectable/copyable utterances and adjustable font size are priorities. Distinguish listening, processing, translated, speaking and disconnected states in words, not just colors.

Phase 1 voice output controls are absent or clearly marked unavailable. No fake live transcripts or decorative progress that implies inference occurred.

## 7. Phase 1 — text conversation MVP

- Explicit device setup and source/target labels for both participants.
- Start/Stop plus per-participant push-to-talk and mute.
- Source transcription and translated text in both directions.
- Automatic speech segmentation after push-to-talk proves the pipeline.
- Both microphones stay available; simultaneous speech produces separate labeled utterances, not a mixed stream.
- Chronological history with timestamps and visible processing failures.
- Save/load device and model settings; transcript export only by explicit action.
- Silence suppression, bounded queues, clear overload behavior and responsive UI.
- No synthesized translated speech.

This is utterance/chunk-based near-real-time interpretation, not a promise of word-for-word simultaneous interpretation. Any provisional caption is labeled provisional; final text is committed once. Context resets when the session restarts.

## 8. Phase 2 — spoken conversation

- Finnish voice for User 1 and English voice for User 2.
- Synthesis begins only for finalized translated chunks; don't speak speculative text that will later be revised.
- Independent destination playback queues, volume, voice settings and Stop-speaking control.
- Cancel pending or active playback on Stop, disconnect or user interruption.
- Measure time from speaker pause to first translated audio, not just synthesis speed.
- Develop hands-free turn-taking, interruption handling and acoustic-leak protection.

Start with reliable alternating turns, then test overlap and barge-in before calling the experience flowing. Avoid a blanket policy that mutes both users whenever either output speaks. If echo control is not verified, ship push-to-talk as the reliable mode and label hands-free limitations. No infinite translation loops from generated audio leaking into capture.

## 9. Privacy and failure boundaries

- Local inference is the default; no cloud calls or telemetry in the MVP.
- Separate explicit model installation from starting a session; show download size/source and retain model revision and checksums.
- Audio buffers are ephemeral; do not persist conversations or raw recordings automatically.
- Export and any diagnostic recording require visible user action.
- Diagnostic logs exclude conversation text and raw audio by default.
- Render model text as plain text, never executable HTML or commands.
- No administrative privileges, system-wide audio changes or LAN server are required by the design.
- Fail visibly when a model/device is unavailable; never invent output or silently switch routes.
- Provide an untranslated/failed state rather than implying a failed translation succeeded.

## 10. Small sequential implementation checkpoints

| Checkpoint | Deliverable | Verification required before expanding |
| --- | --- | --- |
| C0 — this checkpoint | Plan, project boundaries and researched candidates | Source/citation check; clear unimplemented status |
| C1 — native shell + devices | Branded window, participant panels, device assignment, meters, Start/Stop | Native launch/visual check, enum/selection tests, actual selected capture; dual-headset portion pending until supplied |
| C2 — one real utterance | Finnish recording → Finnish text → English text | Real audio/model run; errors and cancellation tested; measured timing |
| C3 — reverse direction | English recording → English text → Finnish text | Real English input; no participant/direction cross-contamination |
| C4 — dual live text | Both captures, fair queueing, history, mute and auto segmentation | Simultaneous speech, silence, long turns, overload, stop/restart and device unplug tests |
| C5 — Phase 1 release gate | Installable MVP and simple operator guide | Full tests, packaged/native launch, actual two-headset text conversation and Finnish/English quality review |
| C6 — destination TTS | Finnish/English voice synthesis and isolated playback | Real listener checks of both outputs; no output to the wrong route |
| C7 — conversation timing | Turn-taking, interruption and leak protection | Actual two-person headset trial, measured delay, no self-translation loop; Phase 2 acceptance |

At each checkpoint retain code, test results and a short status file. Preserve completed work on provider failures; do not merge unrelated changes to use a large single quota chunk. No background development is implied by a finished checkpoint report.

## 11. Acceptance and performance gates

Phase 1 is complete only when both people can use their own microphones and see the other person's correctly directed translation on the native application. Phase 2 is complete only when both hear those translations on their assigned headphones and can sustain a real conversational trial.

Proposed initial performance targets, not measured promises:

- Translated final text normally available within 2 seconds after a short utterance ends.
- First synthesized audio normally available within 3 seconds after a short utterance ends.
- A 15-minute two-person trial without hangs, runaway queue growth, routing leaks or self-translation loops.

Log end-of-speech, ASR completion, translation completion and first playback timestamps. Report median and p95 delays, model/precision, hardware, utterance lengths, silence threshold and whether both participants were active. Cold-start/model-load time is reported separately. If these targets fail, adjust model size/chunking before claiming conversational readiness.

Build a consented bilingual trial corpus covering everyday phrasing, colloquial Finnish, questions, negation, numbers, names, interruptions, quiet speech and silence. Review meaning preservation in both directions; ASR confidence or an automated metric alone is not proof of correct translation. Don't use synthesized fixture speech as the only real-world evidence.

## 12. Planning checkpoint verification boundary (C0)

This section records the initial planning snapshot. For current dependency and hardware status, see `STATUS.md` and `ENVIRONMENT.md`.

Planning confirmed an existing native PySide6 design reference, a CUDA-capable GPU and a PipeWire audio environment. Machine-specific capacity and device inventory are kept out of source control. These observations were feasibility inputs, not inference benchmarks.

The current graph exposes built-in audio endpoints but does not identify two connected external headsets. Actual independent headset capture/playback is therefore unverified. No microphone was opened, no audio recorded, no model downloaded or loaded, and no application code or GUI was created in C0.

Still to establish during implementation:

- Headset connection types, microphone availability and distinct endpoint identities.
- Tested dependency/GPU compatibility and the default Whisper model.
- Conversational adequacy of both translation models.
- Chosen Finnish/English voice quality and distribution terms.
- Whether reliable hands-free overlap needs additional echo cancellation beyond headset isolation.

First implementation scope: C1 only, then C2, in sequential test-backed slices.

## Sources

[1] https://github.com/openai/whisper
[2] https://github.com/SYSTRAN/faster-whisper
[3] https://huggingface.co/Helsinki-NLP/opus-mt-en-fi
[4] https://huggingface.co/Helsinki-NLP/opus-mt-fi-en
[5] https://doc.qt.io/qt-6/qaudiosource.html
[6] https://doc.qt.io/qt-6/qaudiosink.html
[7] https://github.com/OHF-Voice/piper1-gpl
[8] https://huggingface.co/rhasspy/piper-voices/tree/main/fi/fi_FI
