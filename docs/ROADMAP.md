# Next revision: conversation quality and usability

## Real two-person feedback

The user tested the demo with another person and reported:

- Basic sentences translated very well and quickly.
- More difficult sentences translated poorly in both directions.
- Adjacent F1/F2 controls are awkward when two people share one keyboard.
- Custom themes are needed.
- Language selection must include at least Finnish, English, Russian and Ukrainian.

No real conversation recordings, transcripts or example sentences were supplied or stored. This report is not a blanket quality certification.

## Sequential delivery checkpoints

### 1. Configurable talk controls and themes

- Separated default talk keys (proposed F1/F12), configurable independently for each participant.
- Update every visible key label/help string; reject duplicate or unsafe bindings.
- Keep hold/release, auto-repeat handling, interruption, focus-loss cancellation and stale-result safeguards.
- Editable local theme colors with presets and persistent non-secret preferences.
- Never save conversation content with settings; keep native Qt presentation and validate theme inputs.
- Exercise real widgets and the standard native launcher, then independently review changes.

### 2. Translation quality investigation

- Collect user-selected failing examples, including whether the original transcript was accurate. Do not assume an ASR error is a translation error.
- Compare baseline and quality decoding on the same text/audio; keep source transcript visible.
- Evaluate sentence-boundary handling: the current eight-second split can divide longer utterances. Any new limits must remain bounded throughout capture, queues and inference.
- Benchmark a stronger local multilingual translator against the current Marian baseline. Do not promise improvement based only on model size or language coverage.
- Keep the fast baseline available and report measured speed/quality trade-offs.

### 3. Selectable participant languages

- Each participant chooses their spoken language; the other participant's language determines the translation/voice target.
- Required minimum: Finnish (`fi`), English (`en`), Russian (`ru`), Ukrainian (`uk`).
- Validate source/target language throughout routes, worker sessions, ASR, translator, captions and TTS; remove Finnish/English-only assumptions.
- Retain explicit cross-headset routing for every selected pair.
- Unsupported or uninstalled language resources must fail visibly rather than silently pivoting or using the wrong voice.
- Choose and license-check compatible local translation and Russian/Ukrainian voice assets before installation.
- New downloads require explicit approval of sources, disk size and model/license constraints, and must retain pinned revisions/checksums. No automatic runtime downloads or cloud inference.

## Repository boundaries

The repository contains the current reviewed Finnish↔English demo, tests, dependency locks and public model asset manifests. It does not include downloaded weights, virtual environments, native binaries, local device/capture evidence, screenshots, recordings or credentials. The requested next-revision features are not claimed as complete until their respective tests and review gates pass.
