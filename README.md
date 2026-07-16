# Rally

**Kill the wait. Talk to a local AI at the speed of conversation.**

Every local-LLM chat app makes you pay the same taxes, over and over: re-processing your reference
material, re-reading the conversation, chewing through your images, *then* finally answering. Rally's
mission is to delete that wait entirely — **every second of compute that can happen before you hit
send, does** — so a 26B multimodal model on your Mac answers like a person who was already listening.

All local, all private: chat+vision via MLX (Gemma 4 26B-A4B or Qwen3.5 9B — switchable), speech via
Parakeet / Apple SpeechAnalyzer. Nothing leaves your machine.

## How fast?

| Moment | Cost |
|---|---|
| Send a message you typed normally | **~5 ms – 0.3 s to first token** |
| Ask about 46 pinned textbook pages | same — the pages were cached once |
| Paste a screenshot and ask about it | same — its vision pass ran while you typed |
| Conversation grows past its budget | same — old turns drop while you read, never at send |

## The tricks (every nontrivial feature)

- **Pinned multimodal context** — your reference material (text + up to ~190K tokens of pages,
  figures, screenshots) is forward-passed into the KV cache **once**. Every question after that reads
  it from cache. The pin survives across turns, model idle time, and app restarts of the chat.
- **Aggressive pre-compute** — while you type, Rally samples the composer every 0.5 s and feeds your
  in-progress message into the cache (**images first**, so their expensive vision pass happens
  mid-compose). By send time, only ~7 framing tokens remain. (`aggressive-pre-compute.md`)
- **Pre-generation** — once your composed message is fully fed and stable, Rally starts **generating
  the reply speculatively** while you're still deciding to hit Enter. Send with unchanged text → the
  reply appears **instantly** (~5 ms) and continues seamlessly. Edit instead → the speculation is
  discarded for free.
- **Non-interruptive cache dropping** — bounded conversation memory with hysteresis: when the chat
  outgrows its token budget, the oldest turns are evicted **during the ready phase** (while you read
  the last answer), never at send time. Gemma evicts by delta-re-rope (keeps the "smear" of dropped
  turns); Qwen by snapshot restore.
- **Always sub-second TTFT** (mostly sub-half-second, often milliseconds) — enforced by design: sends
  never pay for cache maintenance, window slides, image passes, or re-reading anything.
- **Four models, one picker** — Qwen3.6 27B (dense flagship, best reasoning+coding, ~16 GB, 262 K ctx),
  Bonsai 27B ternary (the same brain at 2-bit / 94.6% quality, ~9 GB), Gemma 4 26B-A4B (MoE, ~18 GB),
  Qwen3.5 9B (fast + light, ~7 GB). Cards grey out if they don't fit in free memory (weights + 8 GB)
  or aren't downloaded; switching swaps the engine process live.
- **Reminder prompts** — a per-chat instruction that rides with every question, with a placement
  trade-off you control: *at chat start* (cached once), *before question* (precomputable), *after
  question* (max adherence). The cache machinery keeps all three instant.
- **Voice** — hold-to-talk or toggle dictation (on-device Parakeet, or Apple streaming transcription
  that prefills the model *as you speak*); capture shortcuts send screenshots (⌘⇧2) or any app's
  selected text (⌃⌥C) straight into the chat. Hotkeys are self-contained (no event tap — zero
  interference with other apps) or Karabiner-driven, your choice.
- **A real transcript, not a log** — edit any old user message inline (Enter resends from that point),
  copy anything, view a reply's markdown source, reset the conversation to any reply. ↑/↓ walk every
  message; E/C/S/R keys; ⌘B (configurable) toggles the sidebar; Esc stops generation.
- **Memory safety** — a watchdog keeps the engine under a hard ceiling (40 GB) by reclaiming
  conversation cache, never by killing the pin; an idle keepalive stops macOS from paging the model
  out mid-conversation (frontmost-only, so backgrounding Rally frees the memory).
- **Instant everything else** — markdown+KaTeX rendering in an isolated WebView (the UI can never
  beachball), streaming token-by-token, engine auto-respawn on crash.

## Requirements

- Apple Silicon Mac, macOS 14+. **Qwen3.6 27B / Gemma**: 48 GB unified memory recommended (~24 GB free
  to select). **Bonsai 27B ternary and Qwen3.5 9B run on 16–24 GB Macs.** Skip downloads you don't
  want with `CIVM_SKIP_GEMMA/QWEN/QWEN36/BONSAI=1`.
- ~46 GB disk for all four models (~9 GB bonsai-only)
- Xcode 16+ or Command Line Tools with **Swift 6** (`xcode-select --install`)
- **uv** and **ffmpeg**: `brew install uv ffmpeg`

## Install

> ⚠️ Clone to exactly this path — the app and installer anchor on it:

```bash
git clone https://github.com/MT-GoCode/Rally ~/code/contextualized_instant_voice_models
cd ~/code/contextualized_instant_voice_models

./engine/setup.sh   # 1) Python/MLX env + required library patches + model downloads (FOREGROUND;
                    #    CIVM_SKIP_GEMMA=1 or CIVM_SKIP_QWEN=1 to skip one model)
./install.sh        # 2) build the app (SwiftPM), sign, install → ~/Applications/Rally.app
open -a Rally       # 3) go
```

First launch: approve **Microphone** (voice), **Accessibility** + **Input Monitoring** (talk key +
capture shortcuts), **Screen Recording** (screenshot-to-chat). The "Grant agent permissions…" button
re-asks. The engine loads (~10 s qwen / ~60 s gemma); both home-screen dots go green.

Rebuild after changing code: rerun `./install.sh`, relaunch the app.

### Signing note
`install.sh` signs with your Developer ID if one is in the keychain, else `$CODESIGN_IDENTITY`,
else ad-hoc. Ad-hoc works, but macOS resets the permission grants on every rebuild.

## Using Rally

- **Home** — model picker, engine/Parakeet status, chat list, prompt library (named system prompts &
  reminders), settings ⚙.
- **New chat** — just ask; nothing to configure. To pin reference material: paste/type text and
  images (or `JSON…`/`PDF…` import) into **SYSTEM PROMPT** and **PINNED CONTEXT**, hit **Cache**
  (a live token estimate gates the 200 K limit before you ever wait).
- **Watch the top bar** — it narrates the machine: `caching context…` → `warming conversation
  cache…` → `cache ready` → `precomputing next turn — N tok` → `⚡ pre-generating reply` — and shows
  the live conversation-budget gauge and engine memory.
- **Bottom line** — per-message accounting: what was read from cache, what was precomputed, what the
  send actually paid, and the measured TTFT.

## Repo layout

| Path | What |
|---|---|
| `engine/` | Python MLX server (`serve.py`, `voice.py`) on `127.0.0.1:5177` — models, KV pinning, precompute/pregeneration, ASR, hotkeys, overlay |
| `app/` | SwiftPM app (target `CIVM` → Rally.app) — UI; talks to the engine over HTTP |
| `aggressive-pre-compute.md` | The cache/precompute/pregeneration state machine |
| `SPEC-voice.md` | The engine⇄app wire contract |
| `engine/setup.sh`, `install.sh` | The only two commands you need |

## Troubleshooting

- **Engine issues** — `tail -f engine/serve.log`; the app auto-respawns a dead engine.
- **Parakeet stuck "loading…"** — usually ffmpeg missing (`brew install ffmpeg`).
- **Mic fails right after login** (PortAudio −9986) — the engine retries; press the talk key again.
- **AirPods sound muffled while dictating** — shouldn't happen (Rally pins the built-in mic and
  restores your device after); if it does, check System Settings → Sound → Input.
