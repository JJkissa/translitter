# Translitter

Local speech interpreter for two people, one Linux PC and two USB headsets. Finnish, English, Russian and Ukrainian models are installed locally. The native interface uses **white surfaces, near-black text and blue accents**. Finnish ↔ English remains the default.

## Linux AppImage

For another Linux PC, use the AppImage rather than this checkout's development
launcher. Read [end-user requirements](requirements.md) and the illustrated
[installation and weights guide](docs/INSTALLATION.md). The preview bundles
Python, Qt, speech dependencies and CPU/Vulkan translation runtimes; target users
do not need a Python environment, Docker or LM Studio. Model setup explicitly
downloads or imports pinned weights into a separate user-owned data directory.
Download only actual release assets and verify their checksum; source files alone
are not an installable release. The tested target is Ubuntu 24.04.5, x86_64,
glibc 2.39 with native `xcb`; broader compatibility is not established.

## Development-checkout launch

**PTT is release-to-translate.** Hold a talk button or key to record; recognition, translation and speech generation for that turn start only after release. Pauses and the former eight-second boundary do not submit held speech. The Live PTT option has been removed, not merely unchecked. Restart an already-open app to load the change. Verification of this revision is recorded below.

Use the **Translitter** desktop shortcut or application-menu entry, or run:

```bash
./start.sh
```

The shortcut uses this project's launcher, isolated interpreter and native/GPU library paths. It has a matching white/black/blue icon. Models load in a worker thread; microphones remain off until **Start conversation**. The launcher also works from a different directory when called by its full path.

The current desktop shortcut and menu entry were validated, marked executable, and the desktop shortcut was trusted. Launching that exact desktop file reached **Ready** in the native window. Both distinct USB headset entries were detected; a process-owned PipeWire check showed no active audio streams while idle.

## Conversation

1. Assign **User 1** and **User 2** to different headset entries. Their USB suffixes distinguish identical models; tooltips expose the complete endpoint IDs.
2. While stopped, choose each participant's spoken language: **Finnish, English, Russian or Ukrainian**. Select different languages. Finnish/English is the default; the other participant's language determines the translation and voice target.
3. Choose **Marian (fast)** or **TranslateGemma 4B** under **Translation**. Marian remains the default. TranslateGemma starts an owned local model process; wait for readiness before Start becomes available. Leave **Text only** unchecked for spoken output. Click **Start conversation**.
4. User 1 holds **F1** or the left **Hold to talk** button and speaks. Pauses are allowed while holding. **Release** to submit the complete turn for recognition, translation and speech through User 2's selected headphones. Original and translated text remain visible after processing.
5. User 2 uses **F2** or the right button. Their translated voice goes to User 1's headphones.
6. Use alternating turns of at most **30 seconds**, including pauses. Going over the limit discards that recording with a warning; it never starts early translation or silently translates only a prefix. Release and try a shorter turn. Each speaker can queue at most two released turns. The separate opt-in **Automatic** mode still submits after a pause; use **Push to talk** for release-only control.
7. **Stop** immediately closes capture, resets playback buffers and invalidates queued/in-flight results. Stop before changing languages or device assignments. Changing languages clears old visible captions so they are not relabelled.

Keep the window active while holding a talk key. Switching away cancels unsubmitted held audio and shows a notice; accepted worker turns remain, and deferred translated playback drains safely. A late release does not submit cancelled input.

Incoming translated playback waits until a held recipient PTT turn is released, preserving captured speech. At most two translations wait per listener; overflow stays as text with a warning. Speaking while translated audio plays interrupts that participant's playback. The microphone is gated during playback and a short cooldown; these guards are not acoustic echo cancellation.

Volume defaults to 45% and is capped at 80%; system/headset volume also applies. To identify headsets, use the meters with neither PTT button held, then Stop and **Swap** if necessary. This does not submit the tap to recognition in PTT mode.

## Local translator choices

**Marian (fast)** uses the existing local translators. **TranslateGemma 4B** translates directly between the selected languages, including Russian↔Finnish without an English pivot. Both keep Whisper recognition and the selected Piper output voice. Change the translator only while stopped; a missing model/runtime or failed preflight blocks capture rather than silently switching models.

The larger model is optional, not a universal accuracy fix. In the latest 14-case synthetic benchmark using the owned private runtime, median translation time was **0.275 seconds**, versus **0.111 seconds** for Marian. These are text-only observations from this machine, not guaranteed speech latency. It retains “today” in the Russian greeting that Marian drops and translates a problematic Finnish partial as “bread” rather than “sandwich”, but it also changes “4:30” to “16:30” in another fixture. Verify important names, numbers and times with the other speaker.

The previous live-streaming experiment is retired following independently reproduced omissions and duplication; its source/tests are preserved under ignored local evidence rather than exposed in the application. Current PTT supplies the whole released utterance to the local pipeline. Generated speech checks are not a guarantee for human speech, different hardware, continuous overlapping talk or room acoustics. Finnish ASR inflections and repeated-utterance recognition still have known quality limitations.

TranslateGemma uses the separately verified compatible GGUF recorded in `quality-model.lock.json`. The **AppImage bundles its own pinned CPU/Vulkan backend and does not require LM Studio**. An unmarked development checkout can instead use a trusted installed Vulkan/AVX2 `llama-server` backend under LM Studio's backend directory. Translitter launches the selected executable directly; it does not use a cloud service or start/manage LM Studio's API server. Each start creates a private Unix socket inside a mode-0700 directory. Before sending HTTP bytes, the client checks the connected peer's exact process ID and user ID against its owned child; no TCP listener is adopted. Unconfigured clients fail closed. The runtime disables prompt logging, slot inspection, tools, MCP proxy and inherited `LLAMA_*` overrides. Socket-level absolute deadlines prevent byte trickles from extending response limits. No runtime/model downloads occur automatically. It stays warm between conversations; selecting Marian or closing the app releases its owned process and socket directory. Installed runtime binaries and the local OS are trusted software, not a sandbox or independently publisher-authenticated by Translitter.

## Verified language pairs

All **12 ordered directions** among the four languages passed real local text translation, generated-speech recognition/translation/synthesis, and repeated pair switching through the real native Window → model worker → engine → audio hub pipeline. Both participant positions and exact destination IDs were exercised; only the low-level physical Qt endpoints were replaced in these tests.

| Requested pair | Both directions | Translation path |
|---|---|---|
| Ukrainian ↔ Finnish | Passed | Via English |
| Ukrainian ↔ English | Passed | Direct |
| Ukrainian ↔ Russian | Passed | Via English |
| Russian ↔ Finnish | Passed | Via English |

Finnish ↔ English and Russian ↔ English also passed. Ukrainian uses code `uk` internally. Language selectors are locked during an active session. Missing or removed resources still block startup visibly; runtime never downloads models automatically or selects a default audio endpoint.

The table describes the Marian paths. Marian pairs without English explicitly use English as an intermediate language. Two translation steps can compound errors; successful execution is **not** proof of broad linguistic accuracy. The separate TranslateGemma backend passed basic direct text checks for all 12 directions; its realtime native speech checks cover Russian→Finnish and Finnish→English, not an all-pair human conversation trial. The Marian English→Russian translator still omits “today” in “Hello, how are you today?”. A strict expected-failure test preserves that baseline limitation.

**Test boundary:** the speech fixtures are deterministic Piper-generated sentences. No real microphone capture or headphone playback was performed for the four-language matrix. Human Ukrainian/Russian conversation quality, physical keyboard acceptance, headset audibility and room bleed still need a supervised trial. The earlier Finnish/English human trial was fast and good for simple sentences but weak for difficult ones.

## Models and privacy

- Whisper large-v3-turbo provides source-language recognition; installed Whisper small is an alternative.
- Six local Marian translators cover English ↔ Finnish/Russian/Ukrainian.
- Piper voices: Finnish Harri, English Lessac, Russian Irina and Ukrainian Lada.
- The pinned baseline language inventory contains **12 assets, 72 files and 4,179,089,209 bytes**, excluding the optional larger translator. Every file was checked against its recorded size/SHA-256; additional model files also carry publisher LFS or Git-blob evidence. TranslateGemma is recorded separately in `quality-model.lock.json`.
- The Ukrainian installation added three assets, 19 files and 639,169,216 bytes without replacing earlier asset records.
- Model cards/configuration are archived with their weights. Irina's model card labels its dataset license unknown; verification is not a claim of voice redistribution permission. Lada's model card labels its dataset Apache 2.0.
- No cloud inference, automatic conversation export or microphone recording. Captions and audio stay in memory; editors retain at most 100 text blocks.
- Downloaded models, environments, native binaries and machine-specific verification evidence are excluded from source control. Cloning the source alone does not install these assets.

CPU fallback:

```bash
./start.sh --model whisper-small --device cpu
```

CPU conversation speed and human recognition quality have not been established.

## Verification

The pre-publication native-xcb installed-model suite passed **508 tests, 2 physical-hardware skips and 1 known quality expected failure in 90.48 seconds**; the quick suite passed **435 tests, 76 skipped in 5.68 seconds**. The remaining strict expected failure tracks Marian's English→Russian “today” omission. Regressions cover mouse and F1/F2 release, pauses and holds longer than eight seconds, full-utterance submission, over-limit rejection, cancellation, deferred cross-headset output, wrong peers receiving zero HTTP bytes, cleanup and hard deadlines. Additional bug-sweep regressions cover audio polling after synchronous stop/restart and automatic-capture segmentation boundaries. Real-model RU→FI and FI→EN native tests each held PTT for **9.1 seconds**, with zero recognition/results/playback before release, then one complete transcript/translation and exact target-headset dispatch. Only physical endpoints were faked; this is not human-headset acceptance. Historical live-build counts are not counts for this revision. Archived live source/tests and detailed test/review evidence remain under ignored `.local/`, not in the GitHub repository.

The previous fixed-Finnish/English assumption in the audio validator was corrected. Tests now cover all ordered language pairs through the actual audio boundary, malformed-route rejection before any endpoint opens, and all-pair switching with real inference. The expanded test harness also waits for its separately queued result collector and checks session generations rather than reading a stale collected result.

Quick checks:

```bash
./run-python.sh -m pytest tests -q
```

All installed model integrations, **without physical microphone/headphone I/O**:

```bash
TRANSLITTER_MODEL_TESTS=1 TRANSLITTER_RUSSIAN_TESTS=1 TRANSLITTER_MULTILINGUAL_TESTS=1 TRANSLITTER_QUALITY_TESTS=1 ./run-python.sh -m pytest tests -q -s
```

The following is different: it explicitly **opens microphones and plays test audio**:

```bash
TRANSLITTER_MODEL_TESTS=1 TRANSLITTER_HARDWARE_TESTS=1 ./run-python.sh -m pytest tests -q -s
```

Ignored `.local/` evidence stores only generated-test outputs/routing metadata, not real microphone samples. See [status and review checkpoints](docs/STATUS.md), [historical preparation](docs/ENVIRONMENT.md) and [remaining work](docs/ROADMAP.md). Configurable separated PTT keys, editable theme preferences and stronger evaluated difficult-sentence quality remain separate tasks; the new palette is fixed, not a theme editor.
