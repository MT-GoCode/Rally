# Voice v2 — Apple streaming + Parakeet transcribe-after

Decision (user, 2026-07-06): **Streaming submode → Apple SpeechTranscriber (Swift, on-device).
Transcribe-after submode → Parakeet batch (Python, unchanged).** Reason: Apple is streaming-native
with real VAD and finalizes instantly at key-release (no batch re-transcribe latency); Parakeet is
~2 WER more accurate on whole-utterance dictation (matters for technical jargon), and after-mode
tolerates end-latency. parakeet-mlx STREAMING is torn out (offline checkpoint × per-chunk
normalization = filler hallucination; not fixable without a purpose-built streaming checkpoint).

## Architecture — hybrid, minimal rewrite
The global ⌃⌥ hotkey tap + the screen overlay stay in **Python** (they work, they're the hard part,
and they must work from any app). Audio + transcription become **mode-dependent**:

- **Chord detection stays in Python** (single source of truth, reported via /voice/poll state).
- **Transcribe-after**: Python captures the built-in mic (sounddevice) + Parakeet batch — UNCHANGED.
- **Streaming**: Python does NOT open the mic or run Parakeet. It only drives the overlay (red dot)
  and reports chord state. The **Swift app** owns streaming: AVAudioEngine (built-in mic) →
  Apple `SpeechTranscriber` → live volatile partials in the UI → finalized transcript at chord-up.
  No mic conflict because only one process opens the mic per mode.

### Flow (streaming submode)
1. App is in Voice·Text + streaming; posts /voice/config {streaming:true} so Python knows to
   drive-overlay-only (no sounddevice/parakeet) on chord.
2. User presses ⌃⌥ → Python tap sets state=`listening`, overlay red. App polls, sees `listening`
   → starts AppleSpeech (AVAudioEngine + SpeechTranscriber), renders volatile partials live in the
   noneditable transcript box.
3. User presses ⌃⌥ again (toggle) / releases (hold) → Python sets state=`idle`, overlay off. App
   sees state left `listening` → stops AppleSpeech, takes its finalized transcript, sends it through
   the existing **AskPipeline** (pins-on-demand, Gemma answers ~0.5s). ESC → Python cancel + app
   discards Apple result.
4. Auto-send: finishing = sending (interrupts a live generation, per AskPipeline). Empty → no-op.

Prefill (feed Gemma while talking): DROPPED for now — research showed the TTFT win is modest
(reminder floor dominates) and Apple finalizes instantly at release, so AskPipeline TTFT is already
~0.5s. Revisit only if wanted.

## Settings
- Voice·Text section gains a **Streaming language** picker (Apple on-device locales; default the
  system locale or en-US). Persisted in UserDefaults (SK.speechLocale). Only affects streaming mode
  (Parakeet after-mode is English-centric; note that if the user picks another language for
  streaming, after-mode still uses Parakeet/English — acceptable, flag in UI copy).
- Force built-in mic for the AVAudioEngine input (match Python's sounddevice built-in pinning): set
  the input node's device to the built-in mic AudioDeviceID (avoid Bluetooth HFP quality drop).

## Files
- NEW `app/Sources/CIVM/AppleSpeech.swift` — @MainActor class: authorize (Speech), ensure asset,
  start(locale)/stop()->String/cancel(), @Published partial, AVAudioEngine builtin-mic capture,
  SpeechAnalyzer+SpeechTranscriber, volatile/final handling.
- `app/Sources/CIVM/ChatSession.swift` — streaming submode drives AppleSpeech off /voice/poll state
  instead of consuming Python's partial/final. after-mode unchanged (still consumes Python final).
- `app/Sources/CIVM/Settings.swift` — SK.speechLocale + language picker.
- `engine/voice.py` + `engine/serve.py` — in streaming mode, chord drives overlay + state ONLY (no
  mic, no Parakeet stream). Remove the now-dead `Stream`/`transcribe_stream` streaming path. Keep
  the tap, overlay, and Parakeet BATCH (transcribe-after) intact. /voice/poll keeps its shape;
  streaming path just never sets partial/final from Python.

## Memory
Apple models ship with the OS (zero bundled weights), run largely in a system service → small
in-process footprint. Gate unchanged (Gemma 16 + headroom covers Parakeet ~1.2GB + Apple few-hundred-MB).

## Verify
- Headless: after-mode still works (Python /voice/inject path unchanged); engine health.
- Live (user): streaming submode → live words appear as you talk (no fillers), instant answer at
  release, ESC discards, language switch works; after-mode → Parakeet accurate transcript at release.
- Measure Apple's real resident memory during a session; report.
