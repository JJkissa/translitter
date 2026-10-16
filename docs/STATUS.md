# Checkpoint status

## Current: GitHub publication checkpoint

The user authorized GitHub publication after the release-only checkpoint. Target repository: `JJkissa/translitter`, private. The current snapshot also includes the subsequent bounded bug-sweep fixes: safe audio polling after synchronous stop/restart, sample-conserving automatic segmentation at unaligned limits, and native automatic-capture cancellation/feedback regressions. PTT remains release-only; these changes do not restore streaming.

Fresh pre-publication verification: **435 quick tests passed, 76 skipped in 5.68 seconds**; **508 installed-model tests passed, 2 hardware tests skipped, 1 known Marian quality xfail in 90.48 seconds**. Downloaded models, environments, private test/review evidence, recordings and local automated work logs are excluded from publication. Publication-specific review/checkpoint evidence is kept under ignored `.local/`; the earlier approval and counts below refer to the frozen release-only revision, not the later bug-sweep delta.

## Historical: approved release-only PTT checkpoint

The user chose release-to-translate instead of another streaming repair. The Live PTT control and UI scheduling paths are removed. PTT retains the whole utterance through pauses and the former eight-second split; only release submits it. Capture is bounded to 30 seconds, and over-limit holds are visibly discarded rather than translated early or silently clipped. Active capture mode is fixed until Stop, including programmatic selector changes. Native UI-focused verification: **78 passed in 2.80 seconds**, including button/F1/F2 release, long pauses/holds, exact cap, overflow, deferred output and focus/cancellation regressions.

The backend is now batch-only: stream APIs, signals, imports and scheduling are removed. UI, worker and ASR share the 30-second cap, and no layer slices accepted input. Queues remain fair/bounded, generations cancel stale work and backend preparation/owned private-runtime cleanup remain in place. Full-turn translation budgets are bounded to 256 Marian tokens and 512 TranslateGemma tokens. The retired live algorithm and its five test files are preserved in `.local/retired-live-ptt-release-only.zip`. Historical streaming defects are not claimed repaired; that feature is withdrawn.

Parent native quick suite: **401 passed, 76 skipped in 5.32 seconds**. Complete installed-model suite without physical audio: **474 passed, 2 hardware skips, 1 known quality xfail in 90.56 seconds**. RU→FI and FI→EN realtime-fed native integration tests each held through **9.1 seconds** of generated input/silence: no ASR, captions, translation or audio before release, then exactly one full result with the final station phrase and correct target endpoint. Evidence: `.local/release-quality-ru-fi.json` and `.local/release-quality-fi-en.json`. The current native screenshot was inspected: no Live PTT control, retained Translation selector, readable unclipped panels and both talk buttons. No physical microphone or headset acceptance is claimed. Current independent frozen-tree gate: `.local/release-only-review-result.json`; implementation checkpoint: `.local/release-only-checkpoint.json`. GitHub publication was paused at that earlier checkpoint.

## Historical: Live PTT blocked after final review

Final review `deleg_6d78e8b5` withheld approval. All 56 frozen hashes matched before/after and the native quick suite passed, but independent worker checks found three silent speech defects: a revised suffix can omit a short new word, a shifted revision can be appended as a duplicate, and committed dedup can suppress a nearby legitimate nonoverlapping repeat. The original partial-prefix/rolling-suffix cases now pass, as do 27 additional positive checks, but those do not override the new failures. Canonical verdict and exact reproductions: `.local/live-upgrade-final-review-6d78e8b5.json`.

Two automatic repair cycles are exhausted; no further automatic source patch is being applied. S1, L2, hard deadlines and benchmark ownership remain cleared through unchanged source hashes. A separate parent check verified the existing nonstreaming fallback through the actual native Window/worker/models/AudioHub with generated Russian speech and fake physical endpoints: **1 passed in 10.08 seconds**, producing `Сегодня я хочу купить хлеб.` → `Tänään haluan ostaa leipää.` using TranslateGemma with Live PTT off. Evidence: `.local/batch-quality-fallback.json`. Physical/human acceptance remains unclaimed. Await a user decision about redesigning sequence/timestamp alignment versus using release-to-translate for now.

### Historical pre-final-review checkpoint

Re-review `deleg_42163b6a` independently cleared S1, L2, byte-trickle deadlines and benchmark ownership, including actual tiny owned-process/Unix endpoint replacement and cleanup-failure variants. All 55 source hashes matched before/after. L1 remains blocked: recognizing only a prefix or rolling-window suffix can erase words already confirmed in two passes. The parent reproduced both failures in permanent `tests/test_streaming_partial.py`: **2 failed, 1 control passed in 0.18 seconds**. Current repair `deleg_cad247df` owns only streaming reconciliation/tests. Canonical verdict: `.local/live-upgrade-rereview-42163b6a.json`; continuation: `.local/live-upgrade-final-repair-checkpoint.json`.

The narrow repair (`deleg_cad247df`) changed only `translitter/streaming.py` and `tests/test_streaming_partial.py`. Two-pass-confirmed pending words are retained across nonempty prefixes, middle/suffix fragments, revisions and rolling truncation. Normalized text/timestamp reconciliation keeps new trailing speech, legitimate repeats and punctuation refinements without overlapping-word duplication. Bounds remain eight seconds PCM, 256 pending words and 32 dedup words; capping a long hypothesis no longer manufactures a quiet tail. The original two reviewer failures are green and expanded focused tests pass. Parent quick suite: **405 passed, 81 skipped in 7.08 seconds**. Parent complete real-model/no-hardware suite: **481 passed, 2 hardware skips, 3 known-quality xfails in 96.73 seconds**. Runtime/privacy source is unchanged from its independent clearance. Final independent review is still required; older sections below are historical checkpoints rather than current totals.

The first independent review (`deleg_b710bf1f`) matched all 51 frozen files before/after, verified the selected GGUF and passed the native quick suite (364 passed, 81 skipped), but withheld approval for three concrete defects: S1, an unowned replacement TCP listener can receive translation text; L1, short stable unpunctuated speech can disappear after more than eight seconds of held silence; L2, repeatedly failing startup cleanup can lose the runtime reference. Canonical verdict: `.local/live-upgrade-review-b710bf1f.json`. Passing model tests below do not override these findings.

Repairs from `deleg_de33e7ee` are implemented and parent-tested. S1 now uses a private mode-0700 Unix socket and checks exact SO_PEERCRED PID/UID before HTTP bytes; replacement peers and dead children get no request data. Unconfigured clients fail closed. L2 keeps engine ownership before startup can allocate resources and retains failed cleanup for retry. L1 preserves bounded two-pass-confirmed pending words and commits short unpunctuated speech after a timestamp quiet tail; tests cover more than eight seconds of held silence, transient empty recognition and no duplicate release. Parent also reproduced a slow-byte-trickle response deadline gap and fixed it with absolute socket receive deadlines plus a post-finish deadline check. The benchmark now starts/closes its own owned runtime on success and failure. No repair is approved until renewed independent review passes.

The approved larger translator is downloaded and publisher-checksum verified. The first Ollama GGUF could not load in the installed llama.cpp runtime; a separately verified compatible GGUF runs locally with generic Gemma turn wrapping around the explicit translation instruction. No model metadata or weights were patched. The original bundled structured-content Jinja template cannot be parsed automatically by this runtime.

A fresh 14-sentence synthetic text benchmark through the repaired private runtime completed: Marian median **0.111 seconds**, TranslateGemma median **0.275 seconds**, candidate first-token median **0.077 seconds**. Earlier runs were slower; these are measured text times, not guaranteed end-to-end speech latency or a human accuracy score. TranslateGemma retains “today” in the Russian greeting but still changes “4:30” to “16:30” in the appointment fixture. Marian remains default; no universal quality improvement is claimed. Evidence: `.local/quality-benchmark.json`.

Live PTT now uses bounded rolling recognition and stable-phrase commits while held, with provisional captions and release-tail flushing. Parent regressions reproduced lost accepted tails on focus loss or a subsequent PTT press; finalized work is now kept separately in bounded queues and those tests pass. The last parent quick suite before the ongoing backend-controls work was **331 passed, 72 skipped**.

Real-model rolling snapshots using generated speech produced translation/TTS before the fixture ended for Finnish→English and Russian→Finnish. Separate functional/quality checks: **3 passed, 1 strict known-quality xfail in 12.26 seconds**. The Finnish case reveals a real issue: partial ASR changes *leipää* to *leipä*, then Marian says “sandwich” instead of “bread”. The same partial text translates as “bread” with the real TranslateGemma backend. This does not establish broad Finnish accuracy. These are offline snapshot-loop checks, **not** actual realtime worker, microphone or physical-headset acceptance. Evidence: `.local/live-ptt-real-model-smoke.json`.

Managed private-runtime startup, real inference and owned-process/socket-directory cleanup pass. The native Translation selector offers Marian (fast) / TranslateGemma 4B; worker-thread preflight gates Start, coalesces obsolete requests and does not open audio. Native/engine guards agree about direct translation without Marian pivot assets. Hardened flags disable slots, tools, MCP proxy and cross-origin response access; inherited `LLAMA_*` overrides are removed. The old TCP/TIME_WAIT reservation design has been replaced entirely, rather than used as endpoint authentication. Installed runtime binaries and the local OS are trusted software, not sandboxed or independently publisher-authenticated by this app.

Real TranslateGemma text checks pass all 12 ordered directions. Realtime-fed generated speech through the actual Window → worker → engine → AudioHub produced early translated/synthesized output while held for RU→FI (about 2.4 seconds after press) and FI→EN (about 3.4 seconds), with final-tail station phrases and exact cross-output dispatch checked. Only physical Qt endpoints are faked. These are synthetic tests, not human conversation latency guarantees. An additional strict quality regression records Whisper collapsing an exactly repeated generated Finnish utterance when decoding the longer window; this is not claimed fixed by the larger text translator.

The repaired complete no-hardware suite passed **467 tests, 2 physical hardware skips, 3 known quality xfails in 96.08 seconds**; the native quick suite passed **391 tests, 81 skipped in 6.96 seconds**. Earlier 440 passed + 2 skipped + 3 xfailed correctly totalled 445 collected tests; an older offscreen artifact's count objection is not a real inconsistency. Both runs remain historical evidence for their respective trees. The desktop shortcut launch reached Ready with distinct USB headsets, Marian default and Live PTT enabled; generated native tests were rerun after the repairs. No physical microphones or headphone playback were exercised during this upgrade.

The canonical first review withheld approval as recorded above; all three findings and the additional hard-deadline concern now have exercised regressions. A repaired source/test/launcher manifest and independent re-review are the remaining release gate. The older/unattributed offscreen artifact `.local/live-upgrade-review-result.json` is retained unchanged for provenance, not used as the current approval record. GitHub publication remains paused.

## Previous reviewed checkpoint: all four languages, desktop shortcut and light blue theme

- Installed and publisher-verified Ukrainian→English, English→Ukrainian and Lada Ukrainian voice. Added **3 assets, 19 files, 639,169,216 bytes**; previous nine asset records unchanged. Current full inventory: **12 assets, 72 files, 4,179,089,209 bytes**. Download queue complete.
- All **12 ordered language directions** passed real text translation and deterministic generated-speech ASR/translation/TTS. All 12 also passed in one real native Window → ModelWorker → engine → AudioHub pair-switch sequence, alternating participant positions and asserting session generation, captions and exact target endpoint. Only low-level Qt discovery/source/sink endpoints were replaced; no physical microphone capture or playback was performed.
- This includes Ukrainian↔Finnish, Ukrainian↔English, Ukrainian↔Russian and Russian↔Finnish, all in both directions. Non-English pairs explicitly pivot through English. Finnish↔English remains the installed default.
- Expanded startup contracts cover all language pairs at the real audio boundary. Corrected an asynchronous test-collector race by waiting for the separately queued result and checking its generation, rather than reading the prior result; production captions and target-audio dispatch were already correct.
- Complete model-enabled suite after UI-copy cleanup: **316 passed, 2 hardware tests skipped, 1 known quality expected failure in 51.59 seconds**. Quick suite: **247 passed, 72 skipped in 3.71 seconds**. The EN→RU greeting temporal omission remains documented, not claimed fixed.
- Removed the decorative in-app title/header, slogan, LOCAL ONLY badge, repeated footer and transcript placeholders. Kept the OS window title, operational labels, states, errors and concise pivot warning. Mode choices are now simply Push to talk / Automatic. The cleaned native view was captured and inspected separately; restart an already-open app to load that copy change.
- New native fixed palette: white/off-white surfaces, near-black text, blue accents and slate disabled states. Theme tests exercise readable normal/hover/pressed/disabled and input selection states; no theme editor or shortcut-rebinding feature is claimed.
- Installed executable trusted **Translitter** desktop shortcut and application-menu entry with a matching self-contained SVG icon. Both entries match `assets/translitter.desktop.in` and pass `desktop-file-validate`; categories were corrected to avoid duplicate menu placement. Launching the actual desktop file through GIO reached Ready, with distinct USB headsets selected and no process-owned PipeWire audio streams. Native screenshot was visually inspected.
- Model cards remain archived: Lada's dataset is labelled Apache 2.0; Irina's dataset license remains unknown. No voice redistribution permission is claimed.
- Evidence: `.local/four-language-checkpoint.json`, `.local/multilingual-model-integration.json`. Final independent review passed with no security/logic findings: 247 quick tests passed, 72 skipped in native xcb; all 34 frozen hashes matched before/after and in the parent's final check. Ukrainian assets and exact desktop/menu targets were independently verified. Verdict: `.local/four-language-review-result.json`. Older sections below are historical checkpoints; their earlier missing-model limits and test totals do not describe this revision.

## Audio route-validation correction

The Russian/English startup error `Invalid cross-headset routing` was reproduced in the production `AudioHub.start` boundary. Its validation reconstructed routes with the old fixed Finnish/English defaults, rejecting every ordered pair except User 1 Finnish/User 2 English. The comparison now uses both selected participant languages while preserving speaker/listener, endpoint-uniqueness, availability and format checks.

Added 27 regressions: all 12 ordered language pairs reach correct fake Qt endpoints, nine malformed routes fail before opening endpoints, and six native Window → real ModelWorker → real AudioHub startup cases cover Russian/English, Russian/Finnish and Finnish/English in both participant orders. Only low-level Qt audio devices are replaced; earlier language-widget tests replaced the whole AudioHub and therefore missed this defect.

Verification: **262 passed, 2 hardware tests skipped, 1 known quality xfailed in 24.03 seconds**, with real baseline/Russian inference enabled. No real microphone capture or playback was performed for this correction. Independent fix review passed with no security or logic findings: 230 quick tests passed, 35 skipped in native xcb, and all 28 frozen hashes matched before/after and in the parent's final check. Verdict: `.local/audio-language-fix-review-result.json`. Restart Translitter to load the corrected Python code; Refresh alone does not reload code.

## Historical Russian-only installation checkpoint

- Explicit approval received to download the assets needed for Russian↔Finnish and Russian↔English. Installed pinned English→Russian and Russian→English translators plus the Russian Irina voice and matching configuration/model card.
- Added **3 assets, 19 files, 686,167,246 bytes**. Full inventory is **9 assets, 53 files, 3,539,919,993 bytes**. Independently reread every installed file and verified size/SHA-256, including publisher weight digests. Baseline six asset records remained identical. Queue is complete.
- Real text and generated-speech recognition/translation/synthesis passed Russian→English, English→Russian, Russian→Finnish and Finnish→Russian. Native widgets enable these installed pairs; tests use fake audio endpoints and never open microphones or play audio.
- Full baseline+Russian model suite: **235 passed, 2 hardware integrations skipped, 1 expected quality failure in 23.57 seconds**.
- Independent read-only installation review passed with no security/logic findings. All 28 frozen hashes matched before/after and in the parent's final check; all 19 Russian files matched pinned publisher LFS or Git blob metadata. Reviewer quick suite: **203 passed, 35 skipped in 3.56 seconds**. Non-blocking suggestion: also check non-LFS publisher Git blobs in the regression test (installer and reviewer already verified them). Verdict: `.local/russian-review-result.json`.
- Known model-quality failure is preserved explicitly: English→Russian drops “today” from “Hello, how are you today?”. Greedy and four-beam decoding reproduce it. The strict expected-failure test keeps this limitation visible; broad accuracy is not certified.
- Irina's archived model card reports an unknown dataset license. No voice redistribution permission is claimed.
- An already-open app retains its last missing-model display until a stopped-session **Refresh**, language change or restart. Automated Refresh was not executed because its desktop-input approval timed out; the user's existing window was not modified or restarted.
- Ukrainian assets remain uninstalled. No GitHub operations occurred. Evidence: `.local/russian-model-integration.json`, `model-assets.lock.json`, `model-downloads.pending.json`.

## Historical options-only checkpoint

- Separate participant-language selectors offer Finnish, English, Russian and Ukrainian. Defaults remain Finnish/English, using the original direct translators and voices.
- Routes, worker validation, ASR language, translation steps, captions and final voice use the selected immutable session configuration. Stop before changing languages; changing a pair clears old visible captions.
- English-linked pairs use direct translators; non-English pairs explicitly pivot through English. No claim of improved complex-sentence accuracy.
- Missing optional translation files or voice/sidecar files disable Start before microphone capture. Backend guards also reject missing resources before ASR. Missing RU/UK assets never disable the installed FI↔EN pair.
- Russian weights are now installed as recorded above; Ukrainian weights still need approval and pinned/checksum-verified installation. Runtime has no automatic downloads.
- Verification: **220 passed, 2 skipped in 16.86 seconds**, including real offline Finnish/English recognition/translation/synthesis; both hardware integrations were disabled. Native default and RU/UK missing-model screenshots were inspected without opening microphones or headphones.
- All distinct language combinations are covered with fake optional model backends and native widgets. Russian additionally passed generated-speech inference as recorded above. Real Ukrainian inference, broader human translation quality and physical-headset operation for these new pairs remain unverified.
- Independent review of this feature delta passed with no security/logic findings: **203 passed, 19 skipped in 3.57 seconds**, all 25 frozen source/test/launcher hashes verified before/after and rechecked by the parent. Verdict: `.local/language-review-result.json`; manifest: `.local/language-review-tree.json`. This approves the implemented options/backend guards, not real RU/UK language quality.

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

Additional user-reported trial: a conversation with a Russian-speaking friend tested Russian↔Finnish and Russian↔English. Finnish was reported least accurate, Russian acceptable, English imperfect. This is human feedback separate from generated fixtures; no recordings or failing sentence/transcript pairs were supplied. It motivated the optional larger translator and the historical Live PTT experiment described above. That while-held feature is now retired; current PTT waits for release. Automatic mode remains a separate pause-triggered option. Broad human accuracy is not certified by the synthetic checks.

The user completed a real conversation with another person and reported fast, very good basic-sentence translations. More difficult sentences translated poorly in both directions. Adjacent F1/F2 push-to-talk keys were awkward for two people sharing a keyboard. This is user-reported acceptance feedback, separate from the automated generated-speech tests; no real conversation samples or transcripts were supplied or saved.

Language selectors and optional multilingual backends are now implemented with the missing-model limitation above. Configurable separated talk keys, editable custom themes and stronger evaluated translation quality are still outstanding; see `ROADMAP.md`. Broad complex-sentence quality, sustained use and acoustic feedback behavior remain acceptance gates.

## Running state at this checkpoint

Native launch and model readiness were verified. Microphones open only after explicit Start; current capture state depends on the operator. No cloud inference, driver changes or sudo operations were needed.
