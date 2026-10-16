# Next revision: conversation quality and usability

## Real two-person feedback

The user tested the demo with another person and reported:

- Basic sentences translated very well and quickly.
- More difficult sentences translated poorly in both directions.
- Adjacent F1/F2 controls are awkward when two people share one keyboard.
- Custom themes are needed.
- Language selection must include at least Finnish, English, Russian and Ukrainian.

No real conversation recordings, transcripts or example sentences were supplied or stored. This report is not a blanket quality certification.

Further user trial with a Russian-speaking friend covered Russian↔Finnish and Russian↔English. The user reports Finnish output is least accurate, Russian is acceptable and English still makes mistakes. A stronger optional local translator has since been installed and tested; Marian remains the fast default. No failing human sentences or source-transcript comparisons were supplied, so ASR versus translation errors are not yet isolated. After trying incremental speech, the user explicitly chose **translation only after releasing PTT** because pauses caused problems. That release-only contract supersedes the earlier streaming preference; there is no pending streaming redesign request. Marian Russian↔Finnish pivots through English; TranslateGemma translates directly.

## Sequential delivery checkpoints

### 1. Configurable talk controls and themes

**Current delivery:** fixed white/black/blue light theme and matching trusted desktop/menu launcher are implemented and natively verified. Editable color preferences and rebinding controls below are still separate work; changing the fixed palette is not a theme editor.

- Separated default talk keys (proposed F1/F12), configurable independently for each participant.
- Update every visible key label/help string; reject duplicate or unsafe bindings.
- Keep hold/release, auto-repeat handling, interruption, focus-loss cancellation and stale-result safeguards.
- Editable local theme colors with presets and persistent non-secret preferences.
- Never save conversation content with settings; keep native Qt presentation and validate theme inputs.
- Exercise real widgets and the standard native launcher, then independently review changes.

### 2. Translation quality investigation

- Collect user-selected failing examples, including whether the original transcript was accurate. Do not assume an ASR error is a translation error.
- Compare baseline and quality decoding on the same text/audio; keep source transcript visible.
- Preserve whole released utterances: the old eight-second PTT auto-split is removed. Capture, queues and inference share a bounded 30-second maximum; an over-limit hold is explicitly discarded rather than partially translated.
- Benchmark a stronger local multilingual translator against the current Marian baseline. Do not promise improvement based only on model size or language coverage.
- Keep the fast baseline available and report measured speed/quality trade-offs.

- TranslateGemma 4B is an explicitly approved, pinned/checksum-verified optional local backend with text benchmarks and direct all-pair smoke tests; it is not universally more accurate. The unapproved incremental experiment is retired, with source/tests and failed reviews retained as local historical evidence. Keep release-only PTT and the separately selected Automatic mode distinct; do not restore while-held processing without a new explicit request.

### 3. Selectable participant languages

**Implemented:** four participant-language options, immutable selected routes, source-ASR/final-voice binding, optional local translators, explicit English pivoting, missing-file gating, native widgets and backend tests. Finnish↔English stays the default and passed real offline model regression tests.

**Russian checkpoint completed:** explicitly approved pinned/checksum-verified Russian translators/voice are installed. Russian↔English and Russian↔Finnish passed real text and generated-speech inference in both directions; Finnish↔English regression tests still pass. See `STATUS.md` for the retained greeting-quality failure and voice-license caveat.

**Ukrainian/all-pair checkpoint completed:** pinned Ukrainian translators and Lada voice installed and verified. All 12 ordered directions passed real text, deterministic generated speech and pair switching through native Window/real worker/engine/audio-hub code, with only the physical Qt endpoints faked. This includes Ukrainian↔Finnish, Ukrainian↔English, Ukrainian↔Russian and Russian↔Finnish in both directions.

**Still pending:** human four-language quality evaluation and physical-headset acceptance for the new pairs. Runtime must still block resources if they are removed; models never download automatically. Broad difficult-sentence accuracy remains a separate gate, including the known EN→RU greeting omission.

- Each participant chooses their spoken language; the other participant's language determines the translation/voice target.
- Required minimum: Finnish (`fi`), English (`en`), Russian (`ru`), Ukrainian (`uk`).
- Validate source/target language throughout routes, worker sessions, ASR, translator, captions and TTS; remove Finnish/English-only assumptions.
- Retain explicit cross-headset routing for every selected pair.
- Unsupported or uninstalled language resources must fail visibly rather than silently pivoting or using the wrong voice.
- Choose and license-check compatible local translation and Russian/Ukrainian voice assets before installation.
- New downloads require explicit approval of sources, disk size and model/license constraints, and must retain pinned revisions/checksums. No automatic runtime downloads or cloud inference.

## Repository boundaries

The repository contains the reviewed Finnish↔English baseline, four-language options/tests, light theme and launcher resources, dependency locks and public model asset manifests. It does not include downloaded weights, virtual environments, native binaries, local device/capture evidence, screenshots, recordings or credentials. The requested next-revision features are not claimed as complete until their respective tests and review gates pass.
