# Aggressive Pre-compute

Rally answers in ~0.1s because by the time you hit **send**, almost everything you're sending has
already been forward-passed into the model's KV cache. This document explains the state machine that
makes that true — including the newest stage, which precomputes your message **while you are still
composing it**.

## The one cache rail

Every cache event in the app reduces to the same simple loop. There are no special cases per mode —
interrupts, finished turns, reminder switches, and chat opens all funnel into the same rail:

```
                        ┌─────────────────────────────────────────────┐
                        ▼                                             │
   [VLM turn ENDS] ──────────► READY THE CONVERSATION CACHE ────► [READY] ──► PRECOMPUTE LOOP
   (finished, interrupted,      (the /precache job)                            (while you compose)
    or stopped — same event)      1. window pre-slide (+headroom)                     │
                                  2. warm [history (+reminder-before)]                │ send
                                  3. PF/QSNAP anchors updated                         ▼
                                                                              [/chat: feed only
                                                                               the un-typed tail]
```

- **"VLM turn ends"** means *any* of: the reply finished, you interrupted with a new message, or you
  pressed Stop/Esc. All three land in the same place: the engine warms the cache for the next turn.
- **Window pre-slide**: if the conversation is near its token budget, the oldest turns are dropped
  *now* — during the ready step, while you're reading the answer — never at send time. A 256-token
  headroom is reserved for your upcoming question so the boundary crossing can't land on TTFT.
- **Ready** means: the KV physically holds `[pin] + [clean history (+reminder if mode=before/start)]`,
  and the `PF` ledger records exactly which token ids those are.

## The precompute loop (composing)

While the cache is READY and you are composing (typing, dictating, or staging images), the app samples
the composer **every 0.5s** and, when it changed, tells the engine to prefill:

```
open turn = [ staged images (FIRST) ] [ typed/dictated text ] ( + reminder if mode=before/start )
```

1. **Tokenize** `[history + open turn]` (conversation-only; the pin is never re-tokenized).
2. **Longest-common-prefix** against what the KV already holds (`PF`).
3. **Rewind** the cache to the divergence point (Gemma: exact trim; Qwen: snapshot restore) and
   **feed only the new suffix**. Appending characters ⇒ a few-token feed. Editing earlier text ⇒
   rewind + refeed of the tail.
4. Repeat at the next sample. In-flight feeds are chunked (≤512 tok), so a sample is never blocked
   long; divergence discovered on the next sample simply re-rewinds.

The HUD shows this live: `precomputing next turn — N tok ready`.

### Why images go FIRST in the turn

Images are the *expensive, stable* part of a message (a vision-tower forward pass); text is the
*cheap, volatile* part. Ordering the turn `[images][text]` means the image pass happens **once**, at
the first sample after you stage the image — every later keystroke appends *after* the image tokens
and reuses them. Measured: staging an image costs ~1.2s once (mid-compose, invisible); every
subsequent text sample is ~0.06s; the send reuses everything.

### The image-signature guard

All image placeholder tokens have the *same id*, so a token-level prefix match cannot tell a cached
image from a *changed* one. `PF.img_sig` (SHA-1 of the image bytes, in order) is the guard:

- **sig unchanged** → the image tokens in the KV are valid → keep the full text LCP (reuse).
- **sig changed / unknown** → reuse is capped at the first image placeholder and all images are
  re-fed with their pixels (the conservative path).

This one rule (`_plan_feed` in `serve.py`) is shared by the compose prefill **and** `/chat`, which is
why a send after composing skips the image forward pass entirely.

## Send

Hitting **send** is just the final compose sample, forced: the same tokenize → LCP → feed-the-tail,
plus the generation prompt, then decoding starts. With the loop warm, the fed tail is typically
~10–15 tokens (`reminder-after` + turn markers). Measured end-to-end: `reused=204, new=13,
TTFT=0.115s` on an image-bearing message.

Reminder placement interacts cleanly:
- **before / start**: the reminder is part of the precomputed shape (placed by the same
  `_place_reminder` the send uses) — zero cost at send.
- **after**: the reminder follows your text, so it can't be precomputed while the text is still
  growing; it's part of the small send-time tail. (This is the placement's documented trade-off.)

## Composing while the VLM is still generating

If the model is mid-generation, the cache is *by definition* not ready — the precompute loop simply
stays idle (its gate fails). Two ways forward:

- **Default**: your edits are just composing; if you send, that's the normal interrupt path
  (cancel → stash partial answer → ready-cache rail → your message asks fresh).
- **"Stop on message edit to optimize precompute"** (Settings → Default conversation cache): the
  moment you *start* editing — first keystroke, staged image, or opening the mic — the generation is
  stopped immediately, the ready-cache rail runs, and the precompute loop takes over. Choose this if
  you habitually read partial answers and start typing the follow-up.

## State ledger (engine)

| State | Meaning |
|---|---|
| `st.kv`, `st.pin_len` | the physical cache; the frozen `[system+ctx+ACK]` prefix length |
| `PF.ids` | exact conversation token ids currently sitting after the pin |
| `PF.img_sig` | identity of the per-turn images those ids' placeholders refer to |
| `QSNAP.pin / QSNAP.conv` | (Qwen only) restorable anchors: bare pin, and clean next-msg target |
| `st.conv_start` | oldest message index still in the cache window (monotonic; UI divider) |

Invariants: `PF.ids` always describes the KV *exactly* (every feed path updates it atomically with
the feed); rewinds only ever land on anchor boundaries or exact trims, never mid-image; snapshots are
immutable (restores slice, so later writes go to fresh buffers).
