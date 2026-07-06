# SPEC — modes, interrupts, voice (v1)

Contract between the Swift app (`app/Sources/CIVM/`) and the Python engine (`engine/serve.py`).
Both implementers: read this fully; do not change the contract without updating this file.

## Modes
`YOUR INPUT · AGENT OUTPUT` — four modes, shown as buttons in a bar at the top of the chat pane, in
this order (highlighted = current): **Text·Text**, **Text·Voice (Coming Soon)**, **Voice·Text**,
**Voice·Voice (Coming Soon)**. Coming-soon buttons are visible but disabled with a "(Coming Soon)" caption.
Voice·Text has submodes (dropdown under its button, checkmark on selected):
- `toggle` — press <key> to interrupt & talk, press again when finished
- `hold`   — hold <key> to talk, release to finish
- `vad`    — (Coming Soon), disabled

Settings (gear button on BOTH home and chat pages; global, persisted in UserDefaults):
- Voice·Text section: default submode; the hotkey (friendly key-capture UI; default ⌃⌥ = control+option chord);
  transcription mode: `after` (TRANSCRIBE-AFTER-FINISHED, default) or `stream` (STREAMING —
  "less accurate but faster response times").
- Other modes' sections: placeholder "(Coming Soon)".

## Engine HTTP (port 5177, all JSON; ndjson where noted)

### GET /health
`{loaded: bool, model: str, parakeet: bool, ctxWindow: 262144}` — parakeet=true once the ASR model is loaded.
Both models load at startup (Gemma first, then parakeet).

### POST /pin  (unchanged)
`{system:[block], context:[block], history:[]}` → `{ok, overLimit, tokens}`

### POST /chat   ← HISTORY IS NOW APP-OWNED
`{messages:[{role:"user"|"assistant", content:[block]}], reminder:[block]}`
- `messages` = the full clean transcript INCLUDING the current user question as the last element.
  The engine keeps NO conversation state (drop `st["history"]`); it builds
  [pinned prefix] + messages, appends `reminder` blocks to the tail of the LAST user message,
  tokenizes, feeds only tokens past `pin_len` (trim_kv first), generates, streams ndjson
  `{delta}` lines then `{done, ttft, gen_s, new_tokens, chat_tokens, pinned, gen_tps}`.
- `chat_tokens` = cache offset after generation − pin_len (everything past the pin: history+q+reminder+answer).
- Any user message may contain image blocks (chat-pasted images, reminder images) — vision runs per-turn
  on those (pinned ctx images stay pinned). Order paths: pinned images first, then per-turn images.
- **Cancellation**: the client cancels by closing the connection; the engine notices on the next
  token write (BrokenPipe/ConnectionReset) and stops generation. Nothing is persisted server-side.
- App truncates history itself (keep last 24 messages).

### Interrupts (app-side, mode-agnostic — NO new engine feature)
The app owns an `AskPipeline` used by ALL modes:
- If idle: append user msg (text + pasted-image blocks) → POST /chat with messages+reminder.
- If generating ("Interrupt & Ask" pressed, or a voice transcript auto-sends): cancel the in-flight
  request; keep the partial answer in its bubble (flagged `interrupted`); append the new user message
  whose TEXT sent to the engine is prefixed `"@@INTERRUPTION@@: "` but whose DISPLAY omits the prefix
  and uses an amber/orange bubble tint. Stackable (each interrupt repeats this). Then POST /chat with
  the rebuilt messages (… user q, assistant partial, user interruption …).

### Voice (Voice·Text mode; same engine process)
The engine owns: mic capture, parakeet, the ⌃⌥ CGEventTap (global), ESC handling, and the
red/yellow/green dot overlay at the BOTTOM-CENTER OF THE SCREEN (pyobjc NSWindow, all-Spaces,
never grey: red=listening, yellow=processing, green=final ready, hidden when idle).

- POST /voice/config `{enabled: bool, submode:"toggle"|"hold", streaming: bool, key:"ctrl+alt"}`
  — app posts on launch, on settings change, and when entering/leaving Voice·Text mode
  (enabled=false tears the tap down).
- POST /voice/context `{messages:[…], reminder:[block]}` — app posts the CURRENT clean transcript
  whenever it changes (needed for streaming prefill; cheap no-op when streaming=off).
- GET /voice/poll → `{state:"idle"|"listening"|"processing"|"ready", partial: str, final: str|null, seq: int}`
  — app polls ~10Hz while in Voice·Text. `seq` increments per session; when state=="ready", the app
  consumes `final` (sends it through AskPipeline — auto-send) and POSTs /voice/ack {seq}.
- ESC while listening/processing → engine cancels (discard audio/transcript, overlay hides, state→idle). No-op send.
- Hotkey semantics: `toggle` = chord press starts, next press stops+transcribes; `hold` = chord down
  starts, chord up stops+transcribes. The tap must not swallow unrelated keys; re-enable on timeout.
- **Don't interrupt generation until the transcript actually sends** (the app enforces this — engine
  generation and listening may overlap; MLX calls are serialized internally with a lock, see below).

### Streaming transcription (`streaming: true`)
- While listening, parakeet produces growing partial transcripts. Engine incrementally PREFILLS the
  KV with [messages + user-turn-open + partial-text] tokens past the pin (tracking fed token ids;
  on revision, trim back to the longest common prefix and re-feed).
- On finish: transcribe final; /voice/poll flips to ready; app sends via AskPipeline → /chat.
  /chat reuses the prefill state when its token ids share the fed prefix (longest-common-prefix
  check), so only the reminder + generation prompt remain → near-instant TTFT.
- The app shows the growing `partial` in a NONEDITABLE text area in the input region (no edit/images).

### MLX concurrency (engine-internal)
One process, two models. Serialize ALL MLX work through one global lock at natural boundaries:
per decode-step for Gemma (wrap the stream_generate iterator), per chunk for parakeet, per prefill
batch. Audio capture runs on its own thread (no MLX). The CGEventTap runs its own runloop thread.

## Swift UI additions (all in app/Sources/CIVM/)
1. Modes bar top of chat: "YOUR INPUT · AGENT OUTPUT" caption + 4 mode buttons + Voice·Text submode dropdown.
2. "Interrupt & Ask" button state mid-generation (Text·Text and whenever a textbox is present).
3. Chat input: ⌘V pastes images → thumbnail bar ABOVE the input (à-la claude.ai) with remove-x;
   sent as blocks with the question. Msg model gains image blocks for display in bubbles.
4. Reminder pane: staged edits — BlockStream edits a DRAFT; an "Update reminder" button (à-la Cache:
   greyed when draft == active) applies it (persists to chat). send() uses the ACTIVE reminder.
5. Token line under the input bar: `"{pinned} pinned + {chat} chat = {total} tok"` from /chat's
   done meta (pinned, chat_tokens). Replaces the char-count line.
6. Voice·Text input region: NO textbox — hint line for the submode (e.g. "Press ⌃⌥ to interrupt &
   talk, again when finished · Esc cancels") + the same token/status info; when streaming, a
   noneditable live-transcript view fed from /voice/poll partial.
7. Home: parakeet status line next to the Gemma one (from /health.parakeet). Gear → Settings on home too.
8. Settings window/sheet: sections per mode as above; key-capture control (records a modifier chord,
   default ⌃⌥); submode picker; streaming toggle with the accuracy/speed description.

## Engine internals (addenda — see RESEARCH-voice.md for lift-ready code refs)
- Process layout: MAIN thread = AppKit runloop (overlay panel + tap-queue NSTimer, activation policy
  accessory). HTTP = ThreadingHTTPServer on a background thread; handlers do NO MLX work.
  ONE dedicated Gemma worker thread consumes a job queue (pin, generate, prefill) — handlers enqueue
  and pump token queues back to the response. Parakeet runs on its own voice thread. Audio capture
  callback thread does no MLX.
- Generation cancellation: the /chat handler writes tokens from the job's queue; on BrokenPipe/
  ConnectionReset it sets job.cancelled; the Gemma worker checks the flag each decode step and stops.
- Hotkey = MODIFIER-ONLY chord (default ctrl+opt): detect via kCGEventFlagsChanged (both flags set =
  chord down, either cleared = chord up). toggle: chord-down starts; next chord-down stops. hold:
  chord-down starts; chord-up stops. ESC cancels only while listening/processing (consumed), else passes.
- Overlay colors: red (0.92,0.16,0.22) listening · yellow (0.95,0.75,0.10) processing · green
  (0.18,0.80,0.45) ~1s flash on ready then hide · hidden idle. NEVER grey.
- POST /voice/inject {text} — test/debug endpoint: behaves as if a final transcript arrived
  (state→ready, final=text). Used by automated tests; keep it.
- Testing: the live app owns port 5177 — agents test their own engine instance on PORT 5178
  (serve.py takes port as argv[2]); do NOT kill processes on 5177.
- Engine venv additions: `uv pip install parakeet-mlx==0.5.1 sounddevice scipy pyobjc-framework-Cocoa
  pyobjc-framework-Quartz` (also append to engine/setup.sh). ffmpeg is present (wtalk depends on it).

## Capture shortcuts (screenshot + copy-to-chat)
Two global shortcuts that deliver an image/text INTO the open chat. Active whenever a chat is open in
ANY mode (Text·Text and Voice·Text). The engine owns the tap + screencapture + ⌘C synthesis; the app
owns settings + delivery. The voice ⌃⌥ chord and these bindings share the ONE CGEventTap.

### Extended POST /voice/config
`{voiceEnabled, captureEnabled, submode, streaming, key, shot:{binding,style}, copy:{binding}}`
- `voiceEnabled` — the voice ⌃⌥ chord (Voice·Text chat ONLY). Legacy `enabled` still accepted as an alias.
- `captureEnabled` — the screenshot/copy bindings (ANY chat; app posts captureEnabled = screen==.chat).
- `shot.binding` / `copy.binding` — serialized bindings (below). `shot.style` = `"initiate"` | `"hold"`.
- The tap is created when EITHER voiceEnabled OR captureEnabled; torn down when neither.
- Bindings are parsed engine-side (parse_binding) and matched by the tap; the voice chord logic is unchanged.

### Binding strings (app ⇄ engine)
- Keyboard chord: `"<mods…>+<base>"`, e.g. `"cmd+shift+2"`, `"ctrl+alt+c"`. Mods ∈ {cmd,ctrl,alt,shift},
  matched as a SET (order-independent, EXACT — no extra mods). `<base>` is a canonical key name mapped
  to an ANSI virtual keycode by a table shared by Swift (`keyCodeToName`) and Python (`NAME_TO_KEYCODE`)
  — layout-stable (physical key position). Recorder finalizes on the base keyDown; ESC aborts recording.
- Mouse button: `"mouseN"`, 1-indexed (`mouse3` = middle). Engine button number = N−1. Only "other"
  buttons (≥ middle) are bindable; the screenshot binding may be keyboard OR mouse, copy is keyboard-only.
- Defaults: screenshot `cmd+shift+2` / style `initiate`; copy `ctrl+alt+c`.

### /voice/poll events channel + ack
- `/voice/poll` adds `"captures":[{kind:"image",data:<base64 png>} | {kind:"text",text:…}]` (else the
  existing shape). The app drains all of them each poll, delivers them, then POSTs
  `/voice/captures-ack {count}` — the engine drops the first `count` (FIFO, race-safe). The app polls
  /voice/poll at ~10Hz while screen==.chat in ANY mode (captures land in every mode; voice state only
  in Voice·Text).

### Screenshot styles
- `initiate` (default): one press launches the native `screencapture -i /tmp/…png` on its own thread
  (native crosshair; user ESC ⇒ no file ⇒ no-op). ESC still belongs to voice; screencapture gets it
  because voice isn't listening.
- `hold`: press&hold the binding and DRAG — the engine draws a selection-rectangle overlay (borderless
  non-activating NSPanel, semi-transparent fill + border) following the cursor (polled on the main-thread
  timer via CGEvent location, TOP-LEFT coords; flipped to AppKit for the panel). Release ⇒
  `screencapture -R x,y,w,h /tmp/…png` of that region (a click / <4px drag is ignored).

### Copy-to-chat
Press-only. Synthesizes a BALANCED ⌘C to the frontmost app (Command-key down → C down → C up →
Command-key UP, 5ms gaps, keycode C=8 — wtalk pattern, explicit key-up so Command never sticks), waits
~0.15s, reads NSPasteboard (string preferred, else image→PNG) → a `{kind:"text"|"image"}` capture.

### App delivery
- Text·Text: image captures → the pasted-images thumbnail bar (ride with the next send); text → appended
  to the input box.
- Voice modes: both queued to a `queuedCaptures` strip in the voice input region and attached to the
  NEXT spoken message — images as content blocks, queued text prepended to the transcript (newline).

### Test hook
`POST /voice/inject {capture:{kind:"text"|"image", text|data}}` enqueues a capture exactly as a real
shortcut would — lets app→chat delivery be tested without real presses/perms. (`{text}`/`{partial}` unchanged.)

## Notes
- Engine keeps the pinned KV exactly as today (/pin unchanged; trim_kv(pin_len) before each feed).
- Memory gate: +2GB headroom for parakeet (Mem.modelGB 16 → 18).
- Never show a grey overlay dot. Overlay is screen-bottom-center, not in-window.
- screencapture needs Screen Recording; the tap needs Accessibility + Input Monitoring — all already
  requested by the app (requestAgentPerms); the engine child inherits responsibility.
