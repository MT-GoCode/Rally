# Rally

**Contextualized Consistent Instantaneous Voice-enabled Agent with Noninterruptive Context Truncation.**

A native macOS app that pins large multimodal reference material (text + images) into a local
Gemma-4's KV cache **once**, then answers typed or spoken questions about it in **~0.5 s** —
every model runs on-device via MLX (Gemma 4 26B-A4B for chat+vision, Parakeet for speech).

## Requirements

- Apple Silicon Mac, macOS 14+, **48 GB unified memory recommended** (launch needs ~26 GB free:
  16 GB weights + headroom; Parakeet adds ~1.2 GB)
- ~20 GB disk for the model
- Xcode 16+ or Command Line Tools with **Swift 6** (`xcode-select --install`)
- **uv** and **ffmpeg**: `brew install uv ffmpeg`

## Install from scratch (nuke-and-repull safe)

> ⚠️ Clone to exactly this path — the app and installer anchor on it:

```bash
git clone https://github.com/MT-GoCode/Rally ~/code/contextualized_instant_voice_models
cd ~/code/contextualized_instant_voice_models

# 1) Engine (Python/MLX): creates engine/.venv, installs deps, applies two required
#    library patches, downloads the 16 GB model. Run in a normal terminal, FOREGROUND.
./engine/setup.sh

# 2) App: builds (SwiftPM, fetches pinned deps), signs, installs → ~/Applications/Rally.app
./install.sh

# 3) Launch
open -a Rally        # or Spotlight "Rally"
```

First launch: approve the permission prompts — **Microphone** (voice), **Accessibility** +
**Input Monitoring** (the global ⌃⌥ talk key and capture shortcuts), **Screen Recording**
(screenshot-to-chat). The "Grant agent permissions…" button on the home screen re-asks.
The engine then loads Gemma (~60 s) and Parakeet; both dots on the home screen go green.

Rebuild after changing code: rerun `./install.sh`, relaunch the app.

### Signing note
`install.sh` signs with your Developer ID if one is in the keychain (via
`~/code/minh-mac-utils/sign-identity.sh` when present), else `$CODESIGN_IDENTITY`, else ad-hoc.
Ad-hoc works, but macOS resets the permission grants on every rebuild.

## Using Rally

- **Home** — model + Parakeet status, chat list (rename ✎ / delete), prompt library
  (named system prompts & reminders), settings ⚙.
- **New chat** — just ask; no caching needed. To pin reference material: paste/type text and
  images (or `JSON…` import) into **SYSTEM PROMPT** and **CONTEXT**, hit **Cache**
  (live token estimate gates the 200 K limit before you ever wait). Once pinned, answers
  arrive in ~0.5 s no matter how big the context is.
- **Modes bar** — `Text·Text` and `Voice·Text` live (Text·Voice / Voice·Voice coming soon).
  Voice: press **⌃⌥** to talk, again to send (or hold-to-talk; Esc cancels; dot at the bottom
  of the screen: red = listening, yellow = processing). Streaming transcription (Settings)
  pre-fills the model while you speak for near-instant replies.
- **REMINDER pane** — a short instruction block that rides with *every* question (the terse
  "Rally" persona by default); edit it live or load one from the library. Not part of the
  pinned cache, so changing it is free.
- **Interrupt & Ask** — cut the model off mid-answer, in any mode (in voice, just talk).
- **Capture shortcuts** (Settings → Capture) — screenshot-to-chat (default ⌘⇧2; native
  crosshair or hold-a-mouse-button-and-drag) and copy-to-chat (default ⌃⌥C: sends the
  current selection in any app to the chat).

## Repo layout

| Path | What |
|---|---|
| `engine/` | Python MLX server (`serve.py`, `voice.py`) on `127.0.0.1:5177` — models, KV pinning, ASR, hotkeys, screen overlay |
| `app/` | SwiftPM app (target `CIVM` → Rally.app) — pure UI; talks to the engine over HTTP |
| `SPEC-voice.md` | The engine⇄app wire contract (read before touching either side) |
| `engine/setup.sh`, `install.sh` | The only two commands you need |
| `seed/` | Local-only demo content (gitignored — not in this repo; the "Load Sipser seed" button is a no-op without it) |

## Troubleshooting

- **"Not enough memory" at launch** — free ~26 GB (quit big apps) and relaunch.
- **Parakeet stuck "loading…"** — `tail engine/serve.log`; usually ffmpeg missing (`brew install ffmpeg`).
- **Engine issues generally** — `tail -f engine/serve.log`; the app respawns the engine on next launch (`pkill -f serve.py` first if wedged).
- **Mic fails right after login** (PortAudio −9986) — the engine retries automatically; press the talk key again.
