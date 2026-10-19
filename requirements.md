# Translitter — Linux AppImage requirements

Applies to the **GPU-capable `0.2.0-preview.1` AppImage**, target filename
`Translitter-0.2.0-preview.1-x86_64.AppImage`. This is a preview, not a universal
Linux compatibility or translation-accuracy guarantee.

Use the [artifact verification record](docs/INSTALLATION.md#artifact-verification-record)
to check this preview's tested scope. A passing source test does not establish
release availability; obtain the matching executable and checksum from an actual
authorized GitHub Release.

For installation, model sizes, offline import, pictures and troubleshooting, use
[the installation and weights guide](docs/INSTALLATION.md).

## 1. End-user requirements

### Operating system and desktop

| Requirement | Supported scope / boundary |
|---|---|
| OS / architecture | Linux **x86_64**; preparation and AppDir checks on **Ubuntu 24.04.5**, host **glibc 2.39**. |
| Minimum libc target | **glibc 2.39**, conservatively declared by the package metadata. A newer libc alone does not prove compatibility with another distribution. |
| Desktop session | **X11 or XWayland**, using Qt's `xcb` platform. Native Wayland-only, headless and other-distribution deployment are not accepted targets yet. |
| Audio stack | Working host desktop audio service and Qt-visible input/output endpoints; the prepared environment uses PipeWire/PulseAudio-compatible desktop audio. |
| User permissions | Ordinary signed-in desktop user, writable private data directory, permission to use the selected audio devices. **Do not run Translitter with sudo.** |
| AppImage mounting | Working host FUSE device/access, or use the documented extraction fallback. The pinned type-2 runtime is statically linked and does **not** require a separately installed `libfuse2`.[33] |
| Runtime installation | **No separately installed Python, virtual environment, pip packages, LM Studio, Ollama, Docker, compiler or source checkout is required.** These are not end-user prerequisites. |

The AppImage owns its Python/Qt/speech dependencies and its pinned CPU/Vulkan
llama.cpp translation runtime. Model weights remain outside the AppImage and are
installed separately. Host glibc, kernel, desktop/audio services and optional GPU
drivers are not replaced by the bundle. Older development-launcher descriptions
in the source README are not the AppImage installation procedure.

If mounting fails, first use `APPIMAGE_EXTRACT_AND_RUN=1` or
`--appimage-extract` and `./squashfs-root/AppRun`, not an obsolete `fuse` package
installation recipe. AppImage documents both fallback paths.[2]

### CPU, RAM and disk planning

These are **conservative planning estimates**, not measured minimums or promised
latency. No minimum CPU model, RAM peak or sustained human-conversation speed has
been certified for the final AppImage.

| Configuration | Planning recommendation, not a verified floor |
|---|---|
| Small + Marian, Finnish/English | Modern 64-bit CPU with at least four useful cores; AVX2 recommended. Start with **8 GiB RAM**, preferably **16 GiB** with other desktop programs running. |
| Turbo, additional languages, or TranslateGemma 4B | Prefer **16–32 GiB RAM**. CPU-only quality-model processing may be substantially slower; test short turns before relying on it. |
| Disk | SSD preferred. Budget the AppImage, selected installed weights, temporary partial/replacement files, any retained old model sets, and optional extracted runtime separately. |
| Initial free-space allowance | Plan **10 GiB free** for a Small/Marian trial; **20 GiB or more** for all languages plus optional quality weights and extraction/update headroom. These allowances are estimates, not computed installed sizes. |

Exact model-file totals are calculated from `model-assets.lock.json` and the
**compatible** entries selected by `quality-model.lock.json`:

| Selection | Exact installed model bytes | GiB, rounded |
|---|---:|---:|
| Small + FI/EN baseline | 1,232,085,319 | 1.147 |
| Small + FI/EN/RU/UK | 2,557,421,781 | 2.382 |
| Small + FI/EN/RU/UK + TranslateGemma | 5,047,339,524 | 4.701 |
| Turbo + FI/EN/RU/UK + TranslateGemma | 6,182,792,582 | 5.758 |
| Both speech models retained, all languages + TranslateGemma | 6,669,006,952 | 6.211 |

`1 GiB = 1,073,741,824 bytes`. Totals include the selected pinned cards,
configurations and tokenizers, not just weight tensors; they exclude the
AppImage, filesystem overhead and incomplete downloads. Installed but unselected
models are not automatically removed. The guide lists every setup combination.

### Optional GPU acceleration

- **A GPU is not required.** First establish the Small/CPU/Marian baseline.
- Whisper's optional **NVIDIA CUDA** path requires a compatible NVIDIA GPU and a
  working **host NVIDIA driver**. The selected GPU-capable AppImage includes
  CUDA 12 cuBLAS and cuDNN 9 user-space libraries; the upstream speech engine
  documents those library families.[31] Bundling them does not provide or
  guarantee a compatible host driver; NVIDIA documents driver/runtime
  compatibility separately.[22]
- Setup offers **NVIDIA CUDA** only when its availability probe succeeds. CPU
  remains available. A successful probe is not a complete inference or
  out-of-memory test.
- TranslateGemma's bundled runtime separately probes **Vulkan** GPU devices and
  falls back to its bundled CPU backend. It needs a working host Vulkan driver
  to accelerate. The setup CPU/CUDA selector controls speech recognition, not
  a global GPU switch for every model.
- GPU memory and latency requirements have not been measured for this artifact.
  Do not buy hardware on the basis of an unverified VRAM minimum.

### Physical audio requirements

1. **Two distinct USB headsets**, one per participant, each exposing a paired
   microphone and headphone output that Translitter can identify as the same
   USB device. Two entries for one physical headset do not count as two devices.
2. Enough working USB ports and stable connections; some headsets need the
   duplex/input-and-output profile enabled in the desktop sound settings.
3. Explicit assignment of one pair to **User 1**, the other to **User 2**.
   No default microphone/output fallback, laptop speakers, Bluetooth routing,
   separate unpaired microphones, or speakerphone acceptance is promised.
4. Closed headphones, a quiet enough room and moderate volume. Playback gating
   is not acoustic echo cancellation. Headset audibility, USB-device quirks,
   bleed and two-person keyboard ergonomics require a supervised trial on the
   target PC.

## 2. Installation, access and model consent

- The repository **`JJkissa/translitter` is private**. Preserve that visibility.
  An authorized GitHub account needs repository read access to obtain private
  Releases; GitHub documents read access as the requirement for viewing
  releases.[10] A 404 may mean missing access, not that the project should be
  made public.
- Obtain the actual AppImage and its **`SHA256SUMS`** release asset from the same
  verified release. Verify SHA-256 before making the AppImage executable. A
  checksum checks correspondence with the downloaded manifest, not an
  independently established publisher signature.
- Internet is needed to obtain release files and for **explicitly requested
  setup downloads**. Opening publisher links also uses the browser/network.
  After the selected models are installed, conversation inference is local;
  no hidden runtime model download is allowed. Offline model import is supported.
- First-run setup defaults to **Whisper small, CPU, Finnish and English**.
  Russian, Ukrainian, Turbo and TranslateGemma weights are optional selections.
  FI/EN baseline translators and voices remain required by engine initialization,
  even when the chosen conversation uses other languages or TranslateGemma.
- Review the selected model/voice terms and explicitly approve installation.
  Only **Download selected files** authorizes network model transfer; **Import
  existing models…** copies and verifies local files from a folder containing
  `models/`. Neither choice opens capture or playback devices.
- Installation checks exact byte length and pinned SHA-256 before atomic file
  replacement. Progress, cancellation, reuse of completed valid files and
  resumable network partials are implemented. Resume requires the publisher's
  correct HTTP range response; it is not guaranteed on every network.

### Model and dataset terms are separate from software terms

Review the [per-model table](docs/INSTALLATION.md#model-terms-and-permissions) and
installed publisher cards before consenting. A checksum establishes file identity,
not legal clearance or suitability for a particular deployment.

- TranslateGemma is subject to **Gemma Terms of Use**, including use and
  redistribution conditions; it is not simply an MIT/Apache weight license.[4]
- **Russian Irina's publisher card says `License: Unknown` in its dataset
  section.** This is a real unresolved field, not permission to redistribute
  the voice or a claim that all Piper voices share one license.[14]
- Lessac's linked dataset agreement is a research license with noncommercial
  restrictions on its defined materials.[20] Do not assume that a model-card
  link or the install checkbox resolves the scope of permissions for trained
  weights or a commercial deployment. Obtain appropriate clarification where
  needed. This document reports the sources, not a legal clearance opinion.

## 3. Functional requirements and operating contract

| ID | Required behavior in this preview |
|---|---|
| FR-01 | Native desktop application; no browser UI or separate model-service application needed. |
| FR-02 | Support spoken Finnish (`fi`), English (`en`), Russian (`ru`) and Ukrainian (`uk`) when their pinned resources are installed. Missing resources visibly block the affected route. |
| FR-03 | Select two distinct headset pairs, participant languages, translator and capture mode **while stopped**. Reject equal participant languages and ambiguous/duplicate device routes. |
| FR-04 | Default translator is **Marian (fast)**. For pairs not containing English, Marian explicitly translates through English. TranslateGemma uses direct selected-language translation. |
| FR-05 | Installing the optional quality weights does **not** select TranslateGemma in the main window. Select **TranslateGemma 4B** there while stopped and wait for readiness. |
| FR-06 | Start capture only after **Start conversation** and successful model/audio readiness. Default mode is **Push to talk**. |
| FR-07 | User 1 holds **F1** or the left talk button; User 2 holds **F2** or the right button. **Release before any ASR, translation or TTS for that held turn begins.** Internal pauses do not submit held PTT audio. |
| FR-08 | A whole turn, including pauses, is bounded to **30 seconds**. Overflow discards the **whole recording**, visibly, until release; never auto-submit at the cap or silently translate a prefix. Retry with a shorter turn. |
| FR-09 | A valid released turn is submitted once; show original/translated captions and route synthesized speech to the **other** participant's selected headphones. |
| FR-10 | Keep focus while using held PTT. Focus loss cancels unsubmitted held audio; it must not silently submit it. Previously accepted turns are distinct from that cancelled input. |
| FR-11 | **Stop** closes capture, resets playback and invalidates queued/in-flight results. Stop before reconfiguring. Never change an active route under captured speech. |
| FR-12 | Optional **Automatic** mode has different semantics: it submits on pauses, not on key release. It is not the recommended baseline and must not be mistaken for release-only PTT. |
| FR-13 | Text-only output is optional; leave it unchecked for headset speech. Bound captions/queues in memory; do not automatically save conversation transcripts or human audio recordings. |
| FR-14 | Reopen the AppImage with **`--setup`** to change speech model/device or install language resources. These setup controls are not in the main window. |
| FR-15 | **`--diagnose`** opens no audio. **`--self-test`** uses generated speech in memory and performs real local ASR → translation → TTS without recording or playing human audio. Neither certifies human accuracy. |

## 4. Data, updates and privacy boundaries

Default writable data:

```text
~/.local/share/translitter/
├── settings.json
└── models/
```

An absolute `XDG_DATA_HOME` changes the base to `$XDG_DATA_HOME/translitter`.
`--data-dir DIR` overrides the full application data directory, which then holds
`settings.json` and `models/`. Use a private, user-owned directory, not an
AppImage mount, source checkout dependency or shared/world-writable directory.
Setup settings store speech model, device, installed-language selections and the
optional-quality flag; they are not conversation logs.

The application does not automatically save human recordings or conversation
logs. Speech and captions are processed in memory; this does not make claims
about OS swap, crash dumps, manually captured screenshots, or other programs.
`--screenshot PNG` explicitly writes a startup/setup image at the requested path.
Diagnostic output includes local paths; review it before sharing.

Updates are **manual**; automatic updating is not implemented. Keep the old
AppImage until the new verified one launches and passes your checks. Reuse the
same data directory: replacing the executable must not require deleting models
or settings. Back up settings and any irreplaceable local assets first. Model
lock changes in a future version may require a new explicit setup step.

## 5. Verification boundary and unresolved acceptance

**Verified actual AppImage checks on the stated Ubuntu host:** normal FUSE
launch, manual extraction/AppRun, automatic extract-and-run, relocation into
paths with spaces, clean HOME/PATH without a source venv or LM Studio, explicit
checksum import of 12 asset groups, native setup/main-window readiness, and a real
Small/CPU FI→EN synthetic pipeline passed. The generated fixture “Hei, miten
voit tänään?” produced “Hey, how are you today?”. No capture or playback was
performed. Two USB pairs were discovered, not audibility-tested. The actual
candidate is **1,743,821,304 bytes**, SHA-256
`1346d1978e589a38b3c0e385992cda0d4ba261a4799848807ed5c6358917910a`.
The actual AppImage also completed all 12 Marian language directions and two
TranslateGemma cases using generated speech, including CUDA recognition. Its
bundled Python/CA setup downloaded and verified a missing 2,370-byte pinned model
configuration while reusing the other installed files. These are functional
execution checks, not linguistic scores. Independent review and publication are
tracked separately in the verification record.

Earlier source-level multilingual tests used deterministic generated speech and
fake physical endpoints. They are not final-artifact, human speech, acoustic,
headset audibility or cross-distribution acceptance. Broad recognition and
translation accuracy remains unestablished. In particular, Marian EN→RU has a
known strict regression tracking omission of **“today”** in the greeting “Hello,
how are you today?”. TranslateGemma also changed meaning in a packaged RU→FI
fixture: a greeting asking how the listener was doing became Finnish “how can I
help?”. A larger translator is not a universal accuracy fix.

Required release gates and pending observations live in the
[artifact verification record](docs/INSTALLATION.md#artifact-verification-record).
Do not relabel source/AppDir evidence as artifact evidence. Human tests should
be separately authorized; no recording or physical playback is needed for the
safe packaged self-test.

## 6. Developer prerequisites are not user prerequisites

Building from source is a separate workflow. The source README, Python dependency
locks, runtime-base preparation and `packaging/build_appimage.py` are for
maintainers, not steps required on each user's PC. The package metadata declares
Python 3.12, bundled Qt, speech libraries and pinned llama.cpp **b11669** CPU/Vulkan
backends. The selected build is GPU-capable; a future CPU-only variant must have
its own clearly identified artifact and verified instructions.

### Evidence used for this document

Product behavior and sizes were read from `translitter/packaged.py`,
`setup_ui.py`, `model_setup.py`, `package_paths.py`, `package_runtime.py`,
`languages.py`, `ui.py`, `core.py`, `packaging/build_appimage.py`,
`packaging/tools.lock.json`, `model-assets.lock.json` and
`quality-model.lock.json`. External technical/terms facts below were checked
against primary publishers on **October 19, 2026**.

## Sources

[2] https://docs.appimage.org/user-guide/troubleshooting/fuse.html
[4] https://ai.google.dev/gemma/terms
[10] https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
[14] https://huggingface.co/rhasspy/piper-voices/raw/c10ece1aade47bb51c153c893d14e5bf8e5b7117/ru/ru_RU/irina/medium/MODEL_CARD
[20] https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html
[22] https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
[31] https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md
[33] https://raw.githubusercontent.com/AppImage/type2-runtime/8f39b89e2ac31e1640b3d3f7e9a5108e6ce805fa/README.md
