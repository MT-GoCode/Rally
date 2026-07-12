#!/usr/bin/env python3
"""Local Gemma-4 engine for contextualized_instant_voice_models (voice + interrupts).

Loads Gemma 4 26B-A4B (MLX) once, holds ONE pinned prompt cache (system+context, vision tower
run ONCE) and reuses it across turns so a new turn only prefills its own tokens (near-instant over
a pinned reference). History is now APP-OWNED: /chat takes the full clean transcript per request.
Parakeet (ASR) loads after Gemma for Voice·Text mode. See SPEC-voice.md for the contract.

Process layout (SPEC "Engine internals"):
  MAIN thread   — AppKit runloop (accessory policy): overlay dot + tap-queue NSTimer + voice glue.
  HTTP thread   — ThreadingHTTPServer; handlers do NO MLX (enqueue jobs, pump token queues).
  Gemma worker  — ONE dedicated thread; consumes a job queue (pin / generate / prefill). All Gemma
                  MLX work is confined here (avoids the "no Stream(gpu,1) in current thread" crash).
  Voice thread  — parakeet load + transcription (batch & streaming). Audio callback on its own thread.
A single global MLX lock serializes GPU work between the Gemma worker and parakeet at natural
boundaries (per Gemma decode-step, per parakeet chunk, per prefill batch).

Endpoints (localhost; JSON; ndjson where noted):
  GET  /health            -> {loaded, model, parakeet, ctxWindow}
  POST /pin               -> build+pin KV over system+context (unchanged); {tokens, ok, overLimit}
  POST /chat              -> {messages:[{role,content:[block]}], reminder:[block]} ; stream ndjson
                             {delta} lines then {done,ttft,gen_s,new_tokens,chat_tokens,pinned,gen_tps}
  POST /new               -> drop the pinned cache
  POST /voice/config      -> {voiceEnabled, captureEnabled, submode, streaming, key,
                             shot:{binding,style}, copy:{binding}}   (legacy "enabled"==voiceEnabled)
  GET  /voice/poll        -> {state, partial, final, seq, perm, captures:[{kind,data|text}]}
  POST /voice/ack         -> {seq}
  POST /voice/captures-ack-> {count}  (drop the first `count` drained capture events)
  POST /voice/inject      -> {text} (final->ready) | {capture:{kind:"text"|"image", text|data}}  [test hook]

Block shape (interro-verbatim): {"type":"text","text":..} |
  {"type":"image","source":{"type":"base64","media_type":..,"data":..}}
"""
import base64, hashlib, json, os, queue, shutil, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_vlm import load, stream_generate, apply_chat_template, prepare_inputs
from mlx_lm.models.cache import make_prompt_cache

import voice as V

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "models/gemma-4-26b-a4b-4bit")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5177
TOKEN_LIMIT = 200_000
CTX_WINDOW = 262_144
# Hard memory boundary. On Apple Silicon memory is unified, so MLX's active allocation (weights + KV +
# live tensors) is the footprint that matters. A watchdog polls it; crossing the ceiling triggers an
# aggressive reclaim (drop conversation KV to the pin + release the buffer pool) so the process can NEVER
# run the machine into swap/OOM. Env-tunable; default leaves headroom on a 48GB machine.
MEM_CEILING_GB = float(os.environ.get("CIVM_MEM_CEILING_GB", "40"))
MEM = {"active_gb": 0.0, "peak_gb": 0.0, "over": False, "ceiling_gb": MEM_CEILING_GB,
       "reclaims": 0, "reclaim_pending": False}
ACK = "Understood — I have the reference material. Ask me anything."
MAX_TOKENS = 2048         # SAFETY CEILING ONLY — the model stops at <end_of_turn>. Length is
                          # controlled by REMINDER, not this.
TEMPERATURE = 0.5
# MLX buffer-reuse pool cap. UNSET by default → MLX's default (full reuse = fastest decode). Capping the
# pool trims the generation footprint but costs tok/s (more alloc/free per token), so it's opt-in only:
# set CIVM_CACHE_LIMIT_GB=<n> if you ever need to trade speed for memory. (Idle already drops via _free_mem.)
GEN_CACHE_LIMIT_GB = os.environ.get("CIVM_CACHE_LIMIT_GB")
# ---- REMINDER prompt: rides WITH each question (default when /chat sends reminder=None) ----
# Gemma-4 uses sliding-window attention (25/30 layers see only the last ~1024 tokens); a reminder
# appended right after the question sits INSIDE that window, so ALL layers attend to it.
REMINDER = (
    "\n\n——— REMINDER (obey now; restates the rules above, which are far back in context) ———\n"
    "• GROUND strictly in the reference material above (between WIKI START / WIKI END). "
    "If it isn't there, say so in one line and defer — never invent.\n"
    "• LENGTH: under 100 words. Give ONLY the single highest-value core; do NOT write the rest.\n"
    "• DEFER everything you cut — and any training-data note — into the final @@APPENDIX@@ line, never the body.\n"
    "• ANCHOR in what I already know (USER KNOWLEDGE); explain any new term on first use.\n"
    "• Answer directly in clean markdown — no preamble, no restating my question.\n"
    "• Go long ONLY if I explicitly ask for a proof, derivation, or artifact.\n"
    '• End with exactly one line: @@APPENDIX@@ {"deferred":[{"item":"…","tag":"too long|not in wiki"}],"training":["…"]}'
)

model = processor = config = None
gemma_ready = threading.Event()
parakeet = {"model": None}                 # set by the voice thread after load

MLX_LOCK = threading.Lock()                # serialize Gemma <-> parakeet GPU work (SPEC concurrency)
GEMMA_Q = queue.Queue()                    # jobs for the single Gemma worker thread
KEEPALIVE = {"pending": False}             # coalesce idle keepalive pings (never queue a second while one is in flight)

st = {}                                    # pinned chat: kv, pin_len, ctx_content, ctx_paths, tmpdir, system
PF = {"ids": None, "img_sig": None}        # streaming-prefill: ids fed past the pin + identity of open-turn images in KV
PRECACHE = {"state": "idle"}               # post-response KV warm-up for the NEXT message: idle|working|done
                                           # (Apple partials prefill Gemma while you talk; /chat LCP-reuses it)

# Real per-operation progress the UI polls via GET /progress. Written by the single Gemma worker from
# INSIDE its loops (per image batch / per prefill chunk / per turn / per token). PROG_LOCK is a tiny,
# separate lock (NEVER MLX_LOCK) so the HTTP thread reads it while the worker holds the GPU — this works
# precisely because the worker is single-threaded, so it's the sole writer publishing its own progress.
PROG_LOCK = threading.Lock()
PROGRESS = {"op": "idle", "stage": "", "done": 0, "total": 0, "frac": 0.0, "label": "", "seq": 0}

def _prog(op=None, stage=None, done=None, total=None, label=None, bump=False):
    with PROG_LOCK:
        if bump:              PROGRESS["seq"] += 1
        if op    is not None: PROGRESS["op"]    = op
        if stage is not None: PROGRESS["stage"] = stage
        if done  is not None: PROGRESS["done"]  = int(done)
        if total is not None: PROGRESS["total"] = int(total)
        if label is not None: PROGRESS["label"] = label
        d, t = PROGRESS["done"], PROGRESS["total"]
        PROGRESS["frac"] = (d / t) if t else 0.0

# ---- voice state machine (shared; guarded by VLOCK) ----
# Streaming submode = Apple SpeechTranscriber (Swift-side); Python only drives the overlay + reports
# chord state. Transcribe-after submode = Parakeet BATCH here. No Python streaming/prefill anymore.
VLOCK = threading.Lock()
VOICE = {"enabled": False, "submode": "toggle", "streaming": False, "key": "ctrl+alt",
         "state": "idle", "partial": "", "final": None, "seq": 0, "perm": True,
         # hotkeyMode: how the dictation/capture triggers arrive. "tap" = legacy global CGEventTap (intrusive,
         # steals keys system-wide — avoid). "self"/"karabiner" = NO tap; triggers come via the /trigger HTTP
         # endpoint (the Swift app's RegisterEventHotKey, or a Karabiner-invoked CLI). Default: no tap.
         "hotkeyMode": "self"}
# Intended listening state, driven by chord events and decoupled from VOICE["state"] (which lags because
# start/stop are QUEUED and applied by the voice thread). Gating stop on the lagging state dropped the
# stop on a fast press+release → stuck "listening". Touched only on the single tap-drain thread.
VOICE_WANT = {"on": False}

# ---- capture shortcuts (screenshot + copy-to-chat); config guarded by VLOCK ----
# The tap is active whenever a chat is open in ANY mode (voice OR capture enabled). "enabled" here
# tracks captureEnabled; bindings are parse_binding() forms cached alongside the raw strings.
CAPTURE = {"enabled": False, "shot_binding": "cmd+shift+2", "shot_style": "initiate",
           "copy_binding": "ctrl+alt+c", "shot_parsed": None, "copy_parsed": None}
CAPTURES = []            # pending capture events for the app: {kind:"image",data:b64}|{kind:"text",text}
CAP_LOCK = threading.Lock()
# TAP_CFG: lock-free snapshot the tap callback reads (refreshed by the main-thread timer under VLOCK;
# both callback and timer run on the main thread, so no lock is needed on the read side).
TAP_CFG = {"voice": False, "capture": False, "shot": None, "copy": None, "chord_mods": None}
# SHOT: press&hold drag state — touched ONLY on the main thread (timer + tap-event application).
SHOT = {"armed": False, "anchor": (0.0, 0.0), "shown": False}

TAP_Q = queue.Queue()    # raw physical events from the tap (drained on main thread)
OVL_Q = queue.Queue()    # overlay commands, applied on main thread: ("show",rgb)/("hide",)/("flash",rgb,secs)
VOICE_Q = queue.Queue()  # commands for the voice thread: ("start",streaming)/("stop",)/("cancel",)


def log(*a): print(*a, file=sys.stderr, flush=True)


def _free_mem():
    """Release MLX's Metal buffer pool back to the OS. After a big transient (the pin's vision encode +
    prefill), MLX holds the freed buffers in its cache — get_active_memory stays low but the PROCESS
    footprint sits at the peak. clear_cache() returns them so Activity Monitor drops back to ~weights+KV."""
    try: (getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", lambda: None))()
    except Exception: pass


def _mx_get(name):
    return getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)

def _mem_reset_peak():
    f = _mx_get("reset_peak_memory")
    if f:
        try: f()
        except Exception: pass

def _mem():
    """(active_gb, peak_gb) of MLX Metal allocations — captures GPU/KV memory that process RSS misses."""
    a, p = _mx_get("get_active_memory"), _mx_get("get_peak_memory")
    try: return (round((a() if a else 0) / 1e9, 2), round((p() if p else 0) / 1e9, 2))
    except Exception: return (0.0, 0.0)

def mem_watchdog():
    """Enforce the hard memory boundary. This thread ONLY reads a stat (get_active_memory) and, when over
    the ceiling, ENQUEUES a reclaim job — it never touches the KV/allocator itself. That's essential: the
    Gemma worker mutates st/PF/KV between decode steps WITHOUT holding MLX_LOCK the whole time, so trimming
    the cache from here would corrupt an in-flight generation. Running the reclaim as a queued job means it
    executes on the worker thread, serialized with all other KV work — safe by construction."""
    while True:
        time.sleep(2.0)
        active, peak = _mem()                               # read-only MLX stat — safe from any thread
        MEM["active_gb"], MEM["peak_gb"] = active, peak
        if active > MEM_CEILING_GB:
            MEM["over"] = True
            if not MEM["reclaim_pending"]:                  # coalesce: one reclaim in flight at a time
                MEM["reclaim_pending"] = True
                log(f"[mem] OVER CEILING {active:.1f}GB > {MEM_CEILING_GB}GB — queued reclaim (worker thread)")
                GEMMA_Q.put(Job("reclaim"))
        else:
            MEM["over"] = False

def _do_reclaim(job):
    """Runs on the WORKER thread (serialized with pin/generate/reconcile → no concurrent KV mutation).
    Release the buffer pool first; if the resident KV itself is still over the ceiling, drop the
    conversation back to the pin (keep [system+context]; the next message rebuilds the conversation)."""
    try:
        _free_mem()
        if _mem()[0] > MEM_CEILING_GB and st.get("kv") is not None and st.get("pin_len") is not None:
            _rewind_to(0)                       # drop conversation → bare pin (gemma trim / qwen snapshot)
            PF["ids"] = []; PF["img_sig"] = None; st["conv_start"] = 0; st["stream_start"] = 0
            QSNAP["conv"] = None                # the conv snapshot holds conversation KV — release it too
            _free_mem()
            MEM["reclaims"] += 1
            log(f"[mem] reclaimed conversation KV → pin; now {_mem()[0]:.1f}GB")
    finally:
        MEM["reclaim_pending"] = False


def load_model():
    global model, processor, config
    # Only cap MLX's buffer-reuse pool if explicitly asked (CIVM_CACHE_LIMIT_GB). Default = MLX's full
    # reuse pool → fastest decode. Capping trims footprint but slows tok/s, so it's off unless requested.
    if GEN_CACHE_LIMIT_GB:
        try:
            _set = getattr(mx, "set_cache_limit", None) or getattr(getattr(mx, "metal", None), "set_cache_limit", None)
            if _set: _set(int(float(GEN_CACHE_LIMIT_GB) * 1024**3))
        except Exception:
            pass
    log(f"loading {MODEL_PATH} …")
    model, processor = load(MODEL_PATH)
    config = model.config
    global IS_QWEN
    IS_QWEN = str(getattr(config, "model_type", "")).startswith("qwen")
    if IS_QWEN:
        # Cap the dynamic-resolution ViT's per-image token spend via qwen's OFFICIAL knob (max_pixels —
        # the same setting in every Qwen-VL model card; recommended range 256–1280 tok/image; the
        # processor downscales the image to fit, nothing is cropped). Uncapped, a real textbook page
        # costs ~1,500 tokens (46-page Sipser pin → 70K tok/139s/22GB peak). Default 1024: the A/B on
        # the real Sipser pages (256 vs 512 vs 1024) showed 1024 gives the most verbatim page reads
        # (Definition 1.5 quoted exactly) at pin 52s / 9.6GB resident; 256 ≈ gemma's fixed budget and
        # started conflating diagram labels. Tunable: CIVM_QWEN_MAX_PIXELS (official max 1280·28·28).
        try:
            mp = int(os.environ.get("CIVM_QWEN_MAX_PIXELS", 1024 * 28 * 28))
            ip = getattr(processor, "image_processor", None)
            if ip is not None:
                if hasattr(ip, "max_pixels"): ip.max_pixels = mp
                if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
                    ip.size["longest_edge"] = mp
            log(f"qwen vision cap: {mp} px (~{mp // (28 * 28)} tok/image)")
        except Exception as e:
            log(f"qwen vision cap failed: {e}")
    log(f"loaded ({getattr(config, 'model_type', '?')})")


IS_QWEN = False   # set by load_model; qwen3_5 = hybrid SSM+attention → snapshot/restore instead of trim/re-rope


def make_kv():
    """Model-appropriate fresh prompt cache. Qwen3.5 NEEDS its own factory (ArraysCache for the ~75%
    linear-attention layers + KVCache for the rest); mlx_lm's generic make_prompt_cache hands it plain
    KVCaches and the SSM mask construction crashes. Gemma keeps the proven mlx_lm path untouched."""
    lm = getattr(model, "language_model", None)
    if IS_QWEN and lm is not None and hasattr(lm, "make_cache"):
        return lm.make_cache()
    return make_prompt_cache(model)


# ---------------- prompt building (proven helpers, preserved) ----------------
def content_of(blocks, tmpdir, paths):
    """blocks -> mlx_vlm message content; images written to tmpdir ONCE (paths appended in order)."""
    out = []
    for b in blocks or []:
        if b.get("type") == "image":
            p = os.path.join(tmpdir, f"img{len(paths)}.png")
            with open(p, "wb") as f:
                f.write(base64.b64decode(b["source"]["data"]))
            paths.append(p)
            out.append({"type": "image"})
        else:
            out.append({"type": "text", "text": b.get("text", "")})
    return out


def build_messages(system, ctx_content, history):
    """system + (context as first user turn + ack) + history (already mlx-shaped messages)."""
    msgs = []
    sys_txt = "\n".join(b.get("text", "") for b in (system or []) if b.get("type") == "text")
    if sys_txt.strip():
        msgs.append({"role": "system", "content": sys_txt})
    if ctx_content:
        msgs.append({"role": "user", "content": ctx_content})
        msgs.append({"role": "assistant", "content": ACK})
    for m in history or []:
        c = m.get("content")
        if isinstance(c, list):
            msgs.append({"role": m["role"], "content": [
                {"type": "image"} if b.get("type") == "image" else {"type": "text", "text": b.get("text", "")}
                for b in c]})
        else:
            msgs.append({"role": m["role"], "content": c})
    return msgs


def prompt_str(msgs, nimg, add_gen=True):
    # qwen3.5 thinks by default (template opens '<think>\n'); enable_thinking=False pre-closes the block
    # so answers start immediately — matching gemma's instant-answer behavior.
    kw = {"enable_thinking": False} if IS_QWEN else {}
    s = apply_chat_template(processor, config, msgs, num_images=nimg, add_generation_prompt=add_gen, **kw)
    if IS_QWEN and not add_gen:
        # The qwen template gives the LAST assistant message a generation-style '<think>\n\n</think>\n\n'
        # prefix but STRIPS it when the same message is re-rendered deeper in a longer history. That
        # breaks the prefix property all cross-turn reuse relies on (every next-turn render diverged at
        # the previous answer's header → reused=0 → 1.6-3.7s TTFT on real chats). Normalize history
        # renders (add_gen=False) to the thinkless form; generation prompts (add_gen=True) keep theirs.
        s = s.replace("<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n")
    return s


def token_ids(prompt, paths):
    """(input_ids, vision_kwargs). Gemma feeds pixels separately (_per_turn_pixels) so extras stay empty
    for it; qwen must pass everything prepare_inputs produced alongside the ids (pixel_values +
    image_grid_thw) or the vision scatter mismatches."""
    inp = prepare_inputs(processor, images=paths or None, prompts=prompt,
                         image_token_index=getattr(config, "image_token_index", None))
    extras = {}
    if IS_QWEN and paths:
        extras = {k: v for k, v in inp.items()
                  if k not in ("input_ids", "attention_mask") and v is not None}
    return inp["input_ids"], extras


def trim_kv(kv, n):
    """Reset the pinned cache back to exactly its first n tokens (drop prior turn). Unchanged."""
    for c in kv:
        if getattr(c, "keys", None) is not None and c.keys.shape[2] > n:
            c.keys = c.keys[:, :, :n, :]
            c.values = c.values[:, :, :n, :]
            if hasattr(c, "offset"):
                c.offset = n


# ---- qwen (hybrid SSM+attention) cache snapshots — replaces arbitrary-point trim_kv ----------------
# The DeltaNet layers hold recurrent STATE (ArraysCache), which can't be trimmed/re-roped to an arbitrary
# token like a KV cache — but it CAN be snapshotted and restored exactly (probe-verified: restored-cache
# output == fresh-cache output). Two snapshots mirror the engine's two anchor points:
#   QSNAP["pin"]  — right after the pin ([system+ctx+ACK]); conversation length 0.
#   QSNAP["conv"] — (ids, snap) after the last background precache (clean history / next-msg target).
# Snapshots are IMMUTABLE: restore slices the KV back to the snapshot offset, so all subsequent writes
# land in fresh buffers and never touch a snapshot's arrays; SSM states are force-copied both ways.
QSNAP = {"pin": None, "conv": None}


def snap_cache(kv):
    out = []
    for c in kv:
        if getattr(c, "keys", None) is not None:
            out.append(("kv", c.keys, c.values, c.offset))
        elif hasattr(c, "cache"):
            out.append(("arr", [mx.array(x) if x is not None else None for x in c.cache]))
        else:
            out.append(("kv", None, None, getattr(c, "offset", 0)))
    return out


def restore_cache(kv, snap):
    for c, s in zip(kv, snap):
        if s[0] == "kv":
            c.keys = s[1][:, :, :s[3], :] if s[1] is not None else None
            c.values = s[2][:, :, :s[3], :] if s[2] is not None else None
            if hasattr(c, "offset"):
                c.offset = s[3]
        else:
            c.cache[:] = [mx.array(x) if x is not None else None for x in s[1]]


def _rewind_to(want):
    """Make the cache hold exactly the first `want` conversation tokens (past the pin), returning the
    ACHIEVED reuse length — the caller feeds target[achieved:]. Gemma: trim_kv to any point (achieved ==
    want). Qwen: append-only (want == len(PF)) needs nothing; otherwise restore the deepest snapshot
    whose ids are a prefix of the current PF at ≤ want (conv, else pin) — the fed tail is then a bit
    longer than gemma's, but it's the same ~[last reply + question + reminder] scale."""
    pf = PF["ids"] or []
    if not IS_QWEN:
        with MLX_LOCK:
            trim_kv(st["kv"], st["pin_len"] + want)
        return want
    if want == len(pf):
        return want                                   # cache already ends exactly there
    c = QSNAP.get("conv")
    if c is not None and len(c[0]) <= want and pf[:len(c[0])] == c[0]:
        with MLX_LOCK:
            restore_cache(st["kv"], c[1])
        return len(c[0])
    with MLX_LOCK:
        restore_cache(st["kv"], QSNAP["pin"])
    return 0


def _rerope_const(rope, x, delta):
    """Apply a CONSTANT positional rotation `delta` to EVERY key in x [B,H,T,D]. mx.fast.rope needs a 4D
    tensor and rotates axis -2, so reshape to seq=1 per key (validated exact vs a fresh rope, err ~1e-5)."""
    B, H, T, D = x.shape
    return rope(x.reshape(B, H * T, 1, D), offset=int(delta)).reshape(B, H, T, D)


def rerope_drop(kv, pin_len, drop):
    """STREAMING eviction: drop the oldest `drop` conversation tokens from the FULL-ATTENTION layers
    while KEEPING the retained tail's smear — slice out [pin_len : pin_len+drop] and delta-re-rope the
    tail DOWN by `drop` so its RoPE positions stay contiguous right after the pin. Values aren't roped.
    The sliding (RotatingKVCache) layers self-bound at 1024 and are left untouched. Near-instant: a
    slice + one constant rotation per full layer, no forward pass. RoPE composes, so this is EXACT
    (unlike a bare slice, which would leave the tail at wrong absolute positions)."""
    if drop <= 0:
        return
    lm = model.language_model.model
    touched = []
    for i, c in enumerate(kv):
        layer = lm.layers[i]
        if getattr(layer, "layer_type", None) != "full_attention":
            continue                                   # sliding layers self-manage; skip
        if getattr(c, "keys", None) is None:
            continue
        off = c.offset
        if off <= pin_len + drop:                      # not enough conversation past the pin to drop
            continue
        rope = layer.self_attn.rope
        k = c.keys[:, :, :off, :]; v = c.values[:, :, :off, :]
        tail_k = _rerope_const(rope, k[:, :, pin_len + drop:, :], -drop)
        c.keys = mx.concatenate([k[:, :, :pin_len, :], tail_k], axis=2)
        c.values = mx.concatenate([v[:, :, :pin_len, :], v[:, :, pin_len + drop:, :]], axis=2)
        c.offset = off - drop
        touched.append(c)
    if touched:
        mx.eval([c.keys for c in touched] + [c.values for c in touched])


# ---- batched image encoding: cap the pin-time memory peak ----
# Encoding ALL context images through the vision tower at once is the transient memory spike at pin
# time (measured: 18GB steady → 28.7GB peak for a 34-image paper). We encode them a few at a time and
# hand the finished features to the pin via stream_generate's vision_cache, so the pin's forward never
# runs the tower on all images together. The features are identical to encoding-all-at-once, so the
# resulting KV — and all cross-turn caching / precache built on it — is byte-for-byte unchanged.
VISION_BATCH = int(os.environ.get("CIVM_VISION_BATCH", "4"))
# Chunked prefill: process a long pinned prompt this-many tokens at a time instead of one giant
# forward (a 9k-token pin was the memory peak). Caps LM activation to one chunk; KV is identical.
PREFILL_STEP = int(os.environ.get("CIVM_PREFILL_STEP", "512"))


def _batch_encode_images(pv):
    """vision_tower + embed_vision over pixel_values, VISION_BATCH images at a time, concatenated."""
    n = int(pv.shape[0])
    if n <= VISION_BATCH:
        f = model.encode_image(pv); mx.eval(f); return f
    parts = []
    for i in range(0, n, VISION_BATCH):
        f = model.encode_image(pv[i:i + VISION_BATCH]); mx.eval(f); parts.append(f)
        mx.clear_cache()                                    # return this batch's buffers to the OS (pin-only; not decode)
        _prog(done=min(i + VISION_BATCH, n))                # per-batch image progress (caller set op/stage/total)
    # encode_image returns [1, images*tokens_per_image, hidden]; join along the token axis.
    return mx.concatenate(parts, axis=1)


def _img_token_id():
    return getattr(config, "image_token_id", None) or \
           getattr(getattr(config, "text_config", config), "image_token_id", -1)


def _pin_prefill(kv, input_ids, pv, feats):
    """Build the pinned KV by pushing the prompt through the LM in CHUNKS instead of one 9k-token
    forward (that single forward's activations were the ~10GB memory spike). Each whole image is its
    OWN chunk — so its bidirectional attention stays intact — and text goes in PREFILL_STEP-token
    chunks; cross-image/text attention is causal via the growing cache. mx.eval() after each chunk
    frees that chunk's activations, capping the peak at ~one chunk's worth. The KV is identical to a
    single forward (same masks, same order), so caching quality is unchanged."""
    img_id = _img_token_id()
    emb = model.get_input_embeddings(input_ids=input_ids, pixel_values=pv, cached_image_features=feats)
    inputs_embeds, pli = emb.inputs_embeds, emb.per_layer_inputs
    ids = input_ids.flatten().tolist()
    is_img = [t == img_id for t in ids]
    mm_full = mx.array([[1 if b else 0 for b in is_img]], dtype=mx.int32)   # image spans → bidirectional
    lm = model.language_model.model
    N = len(ids); start = 0
    while start < N:
        if is_img[start]:                       # a whole contiguous image run = one chunk (keep it intact)
            end = start + 1
            while end < N and is_img[end]:
                end += 1
        else:                                   # a text run, chunked to PREFILL_STEP
            end = start + 1
            while end < N and not is_img[end] and (end - start) < PREFILL_STEP:
                end += 1
        # full mm ids up to `end` so the bidirectional overlay sees each image's whole block
        h = lm(inputs_embeds=inputs_embeds[:, start:end], per_layer_inputs=pli,
               cache=kv, mm_token_type_ids=mm_full[:, :end])
        mx.eval(h)                              # force this chunk's forward + free its activations
        mx.clear_cache()                        # return the chunk's buffers to the OS so the footprint stays flat
        start = end
        _prog(stage="prefill", done=end, total=N)   # per-chunk prefill progress


def _mlx_messages(messages, tmpdir, paths):
    """App messages [{role,content:[block]}] -> mlx messages; per-turn images -> tmpdir (paths appended)."""
    out = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            out.append({"role": m.get("role", "user"), "content": content_of(c, tmpdir, paths)})
        else:
            out.append({"role": m.get("role", "user"), "content": [{"type": "text", "text": str(c)}]})
    return out


def _append_reminder(mlx_msgs, rem_content):
    """Append reminder blocks to the tail of the LAST user message (recency); SPEC /chat."""
    if not rem_content:
        return mlx_msgs
    for i in range(len(mlx_msgs) - 1, -1, -1):
        if mlx_msgs[i]["role"] == "user":
            mlx_msgs[i] = {"role": "user", "content": list(mlx_msgs[i]["content"]) + list(rem_content)}
            return mlx_msgs
    return mlx_msgs + [{"role": "user", "content": list(rem_content)}]


def _per_turn_pixels(paths):
    """pixel_values for JUST the per-turn images (pinned images already live in the KV).
    Order follows `paths`, matching the per-turn image placeholder tokens in the fed suffix."""
    if not paths:
        return None
    tiny = [{"role": "user", "content": [{"type": "image"} for _ in paths]}]
    prompt = prompt_str(tiny, len(paths), add_gen=False)
    inp = prepare_inputs(processor, images=paths, prompts=prompt,
                         image_token_index=getattr(config, "image_token_index", None))
    return inp.get("pixel_values")


def _reminder_content(reminder, tmpdir, paths):
    """None -> built-in REMINDER text; list -> its blocks (images -> tmpdir); str -> raw text."""
    if reminder is None:
        return [{"type": "text", "text": REMINDER}]
    if isinstance(reminder, list):
        return content_of(reminder, tmpdir, paths)
    t = str(reminder).strip()
    return [{"type": "text", "text": ("\n\n" + t) if t else ""}]


# ---------------- Gemma worker (one thread; all Gemma MLX here) ----------------
class Job:
    __slots__ = ("kind", "params", "q", "done", "cancelled", "result", "error")

    def __init__(self, kind, **params):
        self.kind = kind
        self.params = params
        self.q = queue.Queue()      # worker -> handler: ("delta",str)/("done",meta)/("err",msg)
        self.done = threading.Event()
        self.cancelled = False
        self.result = None          # for pin: {ok,...}
        self.error = None


def _do_keepalive(job):
    """Idle keepalive (the app pings this every ~45s while its window is frontmost + a chat is open).
    READ-ONLY: sums every model weight tensor and every pinned-KV key/value, then mx.eval — this faults
    those pages resident (so macOS won't compress/swap the ~18GB of weights+KV under memory pressure) and
    runs a little GPU work (so the GPU doesn't sit fully downclocked). It NEVER mutates st['kv'] / PF /
    offsets, so it cannot corrupt the pinned cache or an eventual generation. ~tens of ms; runs on the
    worker thread so it's serialized behind any real job. This is what keeps TTFT flat after an idle gap
    (the ~3s spike was those cold pages faulting back in on the first token)."""
    try:
        from mlx.utils import tree_flatten
        reads = []
        for _, w in tree_flatten(model.parameters()):        # every weight page (~15GB) → resident + warm
            if isinstance(w, mx.array) and w.size > 0:
                reads.append(mx.sum(w))
        for c in st.get("kv") or []:                          # pinned KV keys/values (~1-2GB) → resident
            k = getattr(c, "keys", None); v = getattr(c, "values", None)
            if k is not None: reads.append(mx.sum(k))
            if v is not None: reads.append(mx.sum(v))
            for x in getattr(c, "cache", None) or []:         # qwen SSM recurrent states (ArraysCache)
                if x is not None: reads.append(mx.sum(x))
        if reads:
            mx.eval(reads)
    except Exception as e:
        log(f"[keepalive] {e}")
    finally:
        _free_mem()                                           # drop the transient sum buffers back to the OS
        KEEPALIVE["pending"] = False
        job.result = {"ok": True, "mem_gb": _mem()[0]}


def gemma_worker():
    load_model()
    # baseline EMPTY pin (pin_len=0): fresh/new chats are chat-able instantly, no /pin needed
    _do_pin(Job("pin", body={}))
    gemma_ready.set()
    while True:
        job = GEMMA_Q.get()
        try:
            if job.kind == "pin":
                _do_pin(job)
            elif job.kind == "generate":
                _do_generate(job)
            elif job.kind == "prefill":
                _do_prefill(job)
            elif job.kind == "precache":
                _do_precache(job)
            elif job.kind == "reconcile":
                _do_reconcile(job)
            elif job.kind == "reclaim":
                _do_reclaim(job)
            elif job.kind == "keepalive":
                _do_keepalive(job)
        except Exception as e:
            import traceback
            log("worker ERR", traceback.format_exc())
            job.error = e
            try:
                job.q.put(("err", str(e)))
            except Exception:
                pass
        finally:
            _prog(op="idle", stage="", done=0, total=0, label="")   # op→idle between jobs
            job.done.set()


def _drop_pin():
    if st.get("tmpdir"):
        shutil.rmtree(st["tmpdir"], ignore_errors=True)
    st.clear()
    PF["ids"] = None; PF["img_sig"] = None


def _do_pin(job):
    d = job.params["body"]
    _prog(op="pin", stage="reading", done=0, total=0, label="reading context", bump=True)
    _mem_reset_peak()                          # so mem_peak_gb reflects THIS pin's transient (chunking check)
    tmpdir = tempfile.mkdtemp(prefix="civm-")
    paths = []
    # images placed in the SYSTEM box are folded into the visual context so the model sees them.
    sys_imgs = content_of([b for b in (d.get("system") or []) if b.get("type") == "image"], tmpdir, paths)
    ctx_content = sys_imgs + content_of(d.get("context"), tmpdir, paths)
    msgs = build_messages(d.get("system"), ctx_content, [])
    if IS_QWEN and msgs and not any(m.get("role") == "user" for m in msgs):
        # qwen's template raises "No user query found" on a system-only render — give a SYSTEM-ONLY pin
        # the same shape a context pin has (a tiny user turn + ACK) so it renders. Gemma path unchanged.
        msgs.append({"role": "user", "content": [{"type": "text", "text": "(Context loaded — begin.)"}]})
        msgs.append({"role": "assistant", "content": ACK})
    if msgs:
        pin_prompt = prompt_str(msgs, len(paths), add_gen=False)
        input_ids, pin_vis = token_ids(pin_prompt, paths)
        pin_len = int(input_ids.shape[-1])
        if pin_len > TOKEN_LIMIT:
            shutil.rmtree(tmpdir, ignore_errors=True)
            job.result = {"ok": False, "overLimit": True, "tokens": pin_len}
            return
        kv = make_kv()
        if IS_QWEN:
            # qwen: plain chunked feed (stream_generate handles prefill_step_size); no gemma-style
            # bidirectional-image chunking needed — MRoPE handles image spans inside the generic path.
            _prog(stage="prefill", done=0, total=pin_len, label="prefilling context")
            with MLX_LOCK:
                for _ in stream_generate(model, processor, "", input_ids=input_ids, prompt_cache=kv,
                                         max_tokens=0, temperature=0.0, prefill_step_size=PREFILL_STEP,
                                         **pin_vis):
                    pass
        else:
            # gemma: encode images in small batches, then prefill the KV in chunks (one image per chunk,
            # text in PREFILL_STEP pieces) — both cap the memory peak. KV is exactly pin_len.
            with MLX_LOCK:
                # _batch_encode_images is a full vision-tower forward and must be serialized against the
                # Parakeet voice thread's MLX ops (the old one-shot pin ran the vision encode inside
                # stream_generate under the lock), so encode INSIDE the lock, before _pin_prefill.
                pv = _per_turn_pixels(paths) if paths else None
                if pv is not None:
                    _prog(stage="encode", done=0, total=len(paths), label=f"encoding {len(paths)} images")
                feats = _batch_encode_images(pv) if pv is not None else None
                _prog(stage="prefill", done=0, total=pin_len, label="prefilling context")
                _pin_prefill(kv, input_ids, pv, feats)
    else:
        kv, pin_len = make_kv(), 0
    _drop_pin()
    st.update(kv=kv, pin_len=pin_len, ctx_content=ctx_content, ctx_paths=paths,
              tmpdir=tmpdir, system=d.get("system"), conv_start=0, stream_start=0)   # fresh pin → window restarts at turn 0
    PF["ids"] = None; PF["img_sig"] = None
    if IS_QWEN:                                # the frozen anchor every divergence can rewind to
        QSNAP["pin"] = snap_cache(kv); QSNAP["conv"] = None
    peak = _mem()[1]
    _free_mem()                # release the pin's transient buffers so the process drops to ~weights+KV
    job.result = {"ok": True, "overLimit": False, "tokens": pin_len, "mem_peak_gb": peak}


def _bos_id():
    tok = getattr(processor, "tokenizer", processor)
    return getattr(tok, "bos_token_id", None)


def _render_ids(mlx_msgs, paths, add_gen, extras_out=None):
    """Tokenize conversation-only mlx messages (no system/ctx), drop the leading <bos> so the ids are
    exactly the tokens that follow the pinned [system+ctx+ACK] prefix. Deterministic (pure text).
    extras_out (dict): receives qwen vision kwargs (pixel_values + image_grid_thw) when images present."""
    prompt = prompt_str(mlx_msgs, len(paths), add_gen=add_gen)
    ids_arr, extras = token_ids(prompt, paths)
    if extras_out is not None:
        extras_out.update(extras)
    ids = ids_arr.flatten().tolist()
    bos = _bos_id()
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids


def _place_reminder(mlx_msgs, rem_content, mode):
    """Reminder placement — the latency/adherence trade-off:
    'after' (aka 'last'): AFTER the last user msg — best recency, but on the next turn's critical path.
    'before': BEFORE the last user msg — precacheable (precedes the unknown question), still recent.
    'start':  BEFORE the FIRST user msg — cached once for the whole chat, fastest, weakest recency."""
    if not rem_content:
        return mlx_msgs
    if mode in ("before", "start"):
        order = range(len(mlx_msgs) - 1, -1, -1) if mode == "before" else range(len(mlx_msgs))
        sep = [{"type": "text", "text": "\n\n"}]
        for i in order:
            if mlx_msgs[i]["role"] == "user":
                mlx_msgs[i] = {"role": "user", "content": list(rem_content) + sep + list(mlx_msgs[i]["content"])}
                return mlx_msgs
        return mlx_msgs
    return _append_reminder(mlx_msgs, rem_content)


def _conv_ids(messages, reminder, mode="last"):
    """Tokenize ONLY the conversation (messages + reminder placed per `mode`) with turn markers — the
    tokens that FOLLOW the pinned [system+ctx+ACK] prefix in the KV.

    We do NOT re-render the multimodal context here. Its image-token expansion is NOT reproducible
    run-to-run, so re-tokenizing the whole prompt drifts the pin boundary and shatters cross-turn
    reuse (the match kept landing inside the context images). Rendering the conversation alone and
    dropping the leading <bos> yields exactly the tokens that sit after the ACK — pure text (plus any
    per-turn chat images), which tokenizes deterministically, so the prefix match is stable.

    Returns (conv_ids: list[int], vision_kwargs, per_turn_paths, tmpdir)."""
    tmpdir = tempfile.mkdtemp(prefix="civm-turn-")
    per_turn_paths = []
    mlx_msgs = _mlx_messages(messages, tmpdir, per_turn_paths)
    rem_content = _reminder_content(reminder, tmpdir, per_turn_paths)
    mlx_msgs = _place_reminder(mlx_msgs, rem_content, mode)
    extras = {}
    ids = _render_ids(mlx_msgs, per_turn_paths, add_gen=True, extras_out=extras)   # conversation only
    if not IS_QWEN:
        pv = _per_turn_pixels(per_turn_paths)
        if pv is not None: extras = {"pixel_values": pv}
    return ids, extras, per_turn_paths, tmpdir


def _lcp(a, b):
    """Longest common prefix length of two token-id lists."""
    n = min(len(a or []), len(b or [])); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---- bounded conversation window (sliding, with hysteresis) ------------------------------------
def _reminder_tok_len(reminder):
    """Token length of the reminder text — it counts toward the conversation budget."""
    tok = getattr(processor, "tokenizer", processor)
    if reminder is None:      txt = REMINDER
    elif isinstance(reminder, str): txt = reminder
    else: txt = "".join(b.get("text", "") for b in (reminder or []) if isinstance(b, dict) and b.get("type") == "text")
    try: return len(tok.encode(txt)) if txt else 0
    except Exception: return 0


def _turn_lens(messages):
    """Approx per-message conversation token length (text only) for windowing — cheap, no forward pass.
    +5 per message is a rough <start_of_turn>role … <end_of_turn> marker overhead."""
    tok = getattr(processor, "tokenizer", processor)
    out = []
    for m in messages:
        txt = "".join(b.get("text", "") for b in (m.get("content") or []) if b.get("type") == "text")
        try: n = len(tok.encode(txt)) if txt else 0
        except Exception: n = 0
        out.append(n + 5)
    return out


def _window_start(messages, reminder, mode, cur, trigger, target):
    """Hysteresis sliding window over conversation TURNS. Keep `cur` unless the windowed conversation
    exceeds `trigger` (X) tokens; then advance `cur` to the earliest user-turn start whose remaining
    conversation ≤ `target` (Y). Monotonic within a session (never rewinds). trigger<=0 disables it
    (returns 0 → whole conversation, today's behaviour).

    FLOOR: never trim past the last COMPLETE turn — always keep ≥1 full Q&A in cache so follow-ups have
    context, even when `target` is smaller than a single turn. Without this the window would walk all the
    way to the bare current question (dropping the previous answer) whenever target < one turn's tokens."""
    n = len(messages)
    cur = max(0, min(cur or 0, n))
    if trigger <= 0 or n == 0:
        return 0
    lens = _turn_lens(messages)
    rem_n = _reminder_tok_len(reminder)
    def clen(s): return sum(lens[s:]) + rem_n
    # The floor is the start of the last complete turn: if the newest message is the in-flight user
    # question, that's the PREVIOUS user turn (keep prev Q&A + the question); otherwise it's the last
    # user turn (keep the last Q&A). conv_start is never allowed past this index.
    user_starts = [i for i in range(n) if messages[i].get("role") == "user"]
    if messages[-1].get("role") == "user":
        floor = user_starts[-2] if len(user_starts) >= 2 else 0
    else:
        floor = user_starts[-1] if user_starts else 0
    if clen(cur) <= trigger:
        return min(cur, floor)                      # inside the deadband, but never below the floor
    for s in range(cur, n):                         # over X → drop oldest whole turns down to ≤ Y
        if messages[s].get("role") == "user" and clen(s) <= target:
            return min(s, floor)
    return floor                                    # couldn't reach Y (giant turn) → keep the last full turn


def _live_drop(messages, conv_start):
    """LIVE drop (BOTH resume modes — this is what an ONGOING chat always does): when the window slides,
    physically evict the dropped turns from the cache via delta-re-rope. Near-instant (a constant rotation
    on the full-attention keys), NO forward pass, and it keeps the smear — so ongoing chats never recompute
    the cache. SAFE FALLBACK: if the dropped-turn tokens aren't an exact prefix of PF (e.g. reminder placed
    'start', or drift), do nothing → the windowed feed re-prefills instead (correct, just no smear).
    The RECENT-vs-STREAMING choice is a RESUME concern (/reconcile), not a live one. Fires once per drop."""
    if not st.get("kv") or conv_start <= 0:
        return
    if IS_QWEN:
        # No re-rope on SSM state. The window still bounds what gets FED after a snapshot restore; the
        # dropped turns' influence lingers in the recurrent state until the next pin-restore (a free
        # "smear"), and the 8 full-attention layers' extra rows are reclaimed at the next rebuild.
        st["stream_start"] = conv_start
        return
    old = st.get("stream_start", 0)
    if conv_start <= old:
        st["stream_start"] = conv_start
        return
    pf = PF["ids"] or []
    tmpdir = tempfile.mkdtemp(prefix="civm-sd-")
    try:
        paths = []      # thread a REAL paths list so image tokens in dropped turns expand (match PF)
        dropped = _render_ids(_mlx_messages(messages[old:conv_start], tmpdir, paths), paths, add_gen=False)
    except Exception as e:
        log(f"stream drop: render failed ({e}) → reprefill fallback"); st["stream_start"] = conv_start; return
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    D = len(dropped)
    if 0 < D < len(pf) and pf[:D] == dropped:          # exact prefix → safe to re-rope-drop
        with MLX_LOCK:
            rerope_drop(st["kv"], st["pin_len"], D)
        PF["ids"] = pf[D:]
        log(f"stream rerope-drop: {D} tok (turns {old}→{conv_start}), conv now {len(pf) - D} tok")
    else:
        log(f"stream drop fallback→reprefill (no prefix align; D={D}, pf={len(pf)})")
    st["stream_start"] = conv_start


_EOT = {"id": "unset"}
def _eot_id():
    if _EOT["id"] == "unset":
        tok = getattr(processor, "tokenizer", processor)
        try: _EOT["id"] = tok.convert_tokens_to_ids("<|im_end|>" if IS_QWEN else "<end_of_turn>")
        except Exception: _EOT["id"] = None
    return _EOT["id"]


def _strip_open(ids_list):
    """Drop the trailing '<end_of_turn>' so the last user turn stays OPEN — a strict prefix of the
    final turn's tokens (so /chat's LCP reuse matches through the dictated text)."""
    eot = _eot_id()
    if eot is not None:
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == eot:
                return ids_list[:i]
    return ids_list


def _img_sig(paths):
    """Identity of the per-turn images last fed into the KV. Placeholder token ids are all identical, so
    an LCP can silently 'match' STALE image tokens whose pixels changed — the sig is what makes image
    reuse safe: equal sig ⇒ same bytes ⇒ the cached image tokens are valid."""
    if not paths:
        return None
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _plan_feed(ids, want_lcp, paths, vis):
    """THE shared reuse rail for image-bearing feeds (compose-prefill AND /chat): decide how much of
    `ids` is safely reusable, rewind the cache there, and return (achieved, tail, tail_vis).
    - No images → plain rewind (today's text path).
    - sig UNCHANGED (same image bytes, same order as last fed — history or staged alike) → the image
      tokens already in the KV are valid: keep the full text LCP. This is what lets a send reuse the
      compose-prefill's image forward pass.
    - sig CHANGED/unknown → cap reuse at the first placeholder and refeed all images with pixels
      (placeholder ids are identical for every image, so an LCP can't distinguish stale image tokens —
      the sig is the guard).
    Rewind can land before the images (qwen snapshot boundaries) — tail_vis follows the TAIL: pixels are
    attached iff placeholders are actually in it (feeds never split an image run mid-way)."""
    if paths and _img_sig(paths) != PF.get("img_sig"):
        img_id = _img_token_id()
        first = next((i for i, t in enumerate(ids) if t == img_id), len(ids))
        want_lcp = min(want_lcp, first)
    achieved = _rewind_to(want_lcp)
    tail = ids[achieved:]
    tail_vis = vis if (paths and any(t == _img_token_id() for t in tail)) else {}
    return achieved, tail, tail_vis


def _do_prefill(job):
    """Compose/voice prefill: feed [messages + OPEN user(images-first + partial text)] past the pin,
    incrementally — the generalized 'aggressive precompute' (the app samples the composer every 0.5s;
    Apple streaming uses the same path). max_tokens=0 = clean prefill; the user turn stays OPEN (no
    turn-end) so /chat's LCP reuses it exactly. Reminder is placed for before/start modes (so their
    precached shape survives); 'after' rides the /chat render, of which this is a strict prefix."""
    if not st.get("kv"):
        job.result = {"fed": 0}; return
    partial = job.params["partial"]
    messages = job.params["messages"]
    tmpdir = tempfile.mkdtemp(prefix="civm-pf-")
    try:
        per_turn_paths = []
        mlx_msgs = _mlx_messages(messages, tmpdir, per_turn_paths)
        open_content = content_of(job.params.get("images") or [], tmpdir, per_turn_paths)  # staged images FIRST
        open_content += [{"type": "text", "text": partial}] if partial else []
        if not open_content:
            job.result = {"fed": len(PF["ids"] or [])}; return
        mlx_msgs = mlx_msgs + [{"role": "user", "content": open_content}]
        mode = job.params.get("mode") or "last"
        if mode in ("before", "start"):                     # keep the precached reminder shape aligned
            mlx_msgs = _place_reminder(mlx_msgs, _reminder_content(job.params.get("reminder"), tmpdir, per_turn_paths), mode)
        extras = {}
        ids = _render_ids(mlx_msgs, per_turn_paths, add_gen=False, extras_out=extras)
        if not IS_QWEN:
            pv = _per_turn_pixels(per_turn_paths)
            extras = {"pixel_values": pv} if pv is not None else {}
        open_list = _strip_open(ids)               # drop trailing turn-end → user turn stays OPEN
        lcp = _lcp(PF["ids"] or [], open_list)      # conversation-token reuse (past the frozen pin)
        if lcp >= len(open_list):
            PF["ids"] = open_list                   # nothing new to feed (partial unchanged/shorter)
            job.result = {"fed": len(open_list), "reused": lcp}; return
        achieved, tail, tail_vis = _plan_feed(open_list, lcp, per_turn_paths, extras)
        with MLX_LOCK:
            for _ in stream_generate(model, processor, "", input_ids=mx.array([tail]),
                                     prompt_cache=st["kv"], max_tokens=0, temperature=0.0,
                                     prefill_step_size=PREFILL_STEP, **tail_vis):
                pass
        PF["ids"] = open_list
        if per_turn_paths:
            PF["img_sig"] = _img_sig(per_turn_paths)
        job.result = {"fed": len(open_list), "reused": achieved}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _precache_target(messages, reminder, mode):
    """The token sequence the KV should hold BEFORE the next user message arrives (`messages` already
    includes the just-generated answer). 'last' mode → clean history through the answer. 'before'
    mode → clean history + an OPEN user turn holding just the reminder (so only the next question is
    left to feed). Returns (target_ids, per_turn_pixels)."""
    tmpdir = tempfile.mkdtemp(prefix="civm-pc-")
    try:
        paths = []
        extras = {}
        mlx = _mlx_messages(messages, tmpdir, paths)
        if mode == "before":
            rem = _reminder_content(reminder, tmpdir, paths)
            if rem:
                mlx = mlx + [{"role": "user", "content": list(rem) + [{"type": "text", "text": "\n\n"}]}]
            ids = _strip_open(_render_ids(mlx, paths, add_gen=False, extras_out=extras))   # + open user turn
        elif mode == "start":
            mlx = _place_reminder(mlx, _reminder_content(reminder, tmpdir, paths), "start")
            ids = _render_ids(mlx, paths, add_gen=False, extras_out=extras)   # reminder pinned on Q1
        else:  # after/last
            ids = _render_ids(mlx, paths, add_gen=False, extras_out=extras)   # clean history; reminder rides next Q
        if not IS_QWEN:
            pv = _per_turn_pixels(paths)
            if pv is not None: extras = {"pixel_values": pv}
        return ids, extras
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _do_precache(job):
    """Warm the KV for the NEXT message the instant the current answer finishes (idle time). Feeds
    only the divergent tail past what's already cached, so the next /chat re-feeds ~just the question."""
    if not st.get("kv"):
        PRECACHE["state"] = "idle"; return
    PRECACHE["state"] = "working"
    try:
        messages = job.params["messages"]; reminder = job.params["reminder"]; mode = job.params["mode"]
        trigger = int(job.params.get("trigger") or 0); target = int(job.params.get("target") or 0)
        recache = job.params.get("recache") or "recent"
        # PRE-SLIDE with headroom: this runs right after an answer (user is mid-read — free time). If the
        # NEXT question would push the conversation over the trigger, the slide used to happen inside the
        # next /chat and its drop/refeed cost landed on TTFT. Reserve room for a typical question (+ the
        # reminder, which _window_start already counts) so the boundary crossing happens NOW, not then.
        eff_trigger = max(target, trigger - 256) if trigger > 0 else 0
        conv_start = _window_start(messages, reminder, mode, st.get("conv_start", 0), eff_trigger, target)
        st["conv_start"] = conv_start
        _live_drop(messages, conv_start)                 # ongoing chat: always re-rope-drop the front (no recompute)
        tgt_ids, vis = _precache_target(messages[conv_start:], reminder, mode)
        lcp = _lcp(PF["ids"] or [], tgt_ids)
        if vis:
            # pixels must match the image placeholders in the fed tokens: with per-turn images their
            # placeholders sit in the reused prefix, so re-feed the whole target past the pin.
            lcp = 0
        if lcp < len(tgt_ids):
            lcp = _rewind_to(lcp)
            feed = mx.array([tgt_ids[lcp:]])
            with MLX_LOCK:
                for _ in stream_generate(model, processor, "", input_ids=feed,
                                         prompt_cache=st["kv"], max_tokens=0, temperature=0.0,
                                         prefill_step_size=PREFILL_STEP, **vis):
                    pass
        PF["ids"] = tgt_ids; PF["img_sig"] = None   # clean history — no OPEN-turn images in KV
        if IS_QWEN:                             # anchor: clean next-msg target — divergent turns restore here
            QSNAP["conv"] = (list(tgt_ids), snap_cache(st["kv"]))
    finally:
        PRECACHE["state"] = "done"
        _free_mem()


def _stream_replay(messages, reminder, mode, trigger, target):
    """STREAMING cold-resume: replay the whole conversation onto the pin (clean history, turn by turn),
    re-rope-dropping the oldest turns to `target` whenever the physical conversation exceeds `trigger`.
    The final window carries the SMEAR of dropped turns, AND the KV never exceeds ~trigger+one-turn (each
    turn is fed then possibly dropped — bounded, chunked, no spike). Sets PF['ids'] to the retained tail;
    returns the retained window's first message index (conv_start)."""
    pin_len = st["pin_len"]
    with MLX_LOCK:
        trim_kv(st["kv"], pin_len)
    PF["ids"] = []; PF["img_sig"] = None; st["stream_start"] = 0
    starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not starts:
        return 0
    starts.append(len(messages))
    seg_lens, fed, win = [], [], 0                 # per-turn fed length, live tokens past pin, oldest-retained turn
    tmproot = tempfile.mkdtemp(prefix="civm-rp-")
    try:
        for b in range(len(starts) - 1):
            _prog(op="reconcile", stage="replay", done=b, total=len(starts) - 1, label=f"replaying turn {b + 1}/{len(starts) - 1}")
            paths = []
            seg_ids = _render_ids(_mlx_messages(messages[starts[b]:starts[b + 1]], tmproot, paths), paths, add_gen=False)
            pv = _per_turn_pixels(paths)
            with MLX_LOCK:
                for _ in stream_generate(model, processor, "", input_ids=mx.array([seg_ids]),
                                         pixel_values=pv, prompt_cache=st["kv"], max_tokens=0, temperature=0.0, prefill_step_size=PREFILL_STEP):
                    pass
            fed += seg_ids; seg_lens.append(len(seg_ids))
            if len(fed) > trigger:                 # hysteresis: over X → drop oldest whole turns to ≤ Y
                while len(fed) > target and win < b:
                    D = seg_lens[win]
                    with MLX_LOCK:
                        rerope_drop(st["kv"], pin_len, D)
                    fed = fed[D:]; win += 1
        PF["ids"] = fed
        return starts[win]
    finally:
        shutil.rmtree(tmproot, ignore_errors=True)
        _free_mem()


def _do_reconcile(job):
    """Rebuild the conversation cache so it is CORRECT for the current settings — called on chat OPEN and
    after a content/setting change, so the next message lands on a correct cache. recent → prefill the
    window; streaming → replay with smear. The pin ([system+ctx]) is assumed current (/pin runs first)."""
    if not st.get("kv"):
        PRECACHE["state"] = "idle"; job.result = {}; return
    PRECACHE["state"] = "working"
    _prog(op="reconcile", stage="rebuilding", done=0, total=0, label="rebuilding conversation cache", bump=True)
    _mem_reset_peak()
    try:
        messages = job.params["messages"]; reminder = job.params["reminder"]; mode = job.params["mode"]
        trigger = int(job.params.get("trigger") or 0); target = int(job.params.get("target") or 0)
        recache = job.params.get("recache") or "recent"
        pin_len = st["pin_len"]
        # STREAMING mode only needs the expensive turn-by-turn replay (to rebuild the SMEAR of dropped
        # turns) when turns are ACTUALLY dropped. If the whole conversation fits under the trigger nothing
        # drops → no smear → a single-pass prefill produces the identical KV, far faster. (This is why a
        # large budget on a short chat was spending a minute replaying ~20 turns one forward pass at a time
        # for no reason.) recent mode never replays.
        cs0 = _window_start(messages, reminder, mode, 0, trigger, target) if (trigger > 0 and messages) else 0
        if recache == "streaming" and cs0 > 0 and not IS_QWEN:
            conv_start = _stream_replay(messages, reminder, mode, trigger, target)   # dropping → rebuild the smear
        else:
            _rewind_to(0)                                       # back to the bare pin (gemma trim / qwen snapshot)
            PF["ids"] = []; PF["img_sig"] = None; st["stream_start"] = 0
            conv_start = cs0                                    # recent (or qwen, or nothing to drop) → single prefill below
        st["conv_start"] = conv_start; st["stream_start"] = conv_start
        # warm the precache target for the NEXT message on top of the rebuilt window (reminder-aware)
        tgt_ids, vis = _precache_target(messages[conv_start:], reminder, mode) if messages else ([], {})
        lcp = _lcp(PF["ids"] or [], tgt_ids)
        if vis: lcp = 0
        if lcp < len(tgt_ids):
            lcp = _rewind_to(lcp)
            with MLX_LOCK:
                for _ in stream_generate(model, processor, "", input_ids=mx.array([tgt_ids[lcp:]]),
                                         prompt_cache=st["kv"], max_tokens=0, temperature=0.0,
                                         prefill_step_size=PREFILL_STEP, **vis):
                    pass
        PF["ids"] = tgt_ids; PF["img_sig"] = None
        if IS_QWEN:
            QSNAP["conv"] = (list(tgt_ids), snap_cache(st["kv"]))
        job.result = {"conv_start": conv_start, "conv_tokens": len(tgt_ids), "mem_peak_gb": _mem()[1]}
    finally:
        PRECACHE["state"] = "done"
        _free_mem()


def _do_generate(job):
    if not st.get("kv"):
        job.q.put(("err", "nothing pinned — cache first"))
        return
    messages = job.params["messages"]
    reminder = job.params["reminder"]
    mode = job.params.get("mode") or "last"
    trigger = int(job.params.get("trigger") or 0)   # X: trim when conv+reminder cache exceeds this
    target = int(job.params.get("target") or 0)     # Y: …down to this (hysteresis). 0 disables both.
    # Bounded sliding window: drop oldest whole turns so only messages[conv_start:] sit past the pin.
    recache = job.params.get("recache") or "recent"
    _mem_reset_peak()                                # so mem_peak in the meta is THIS turn's peak (spike check)
    conv_start = _window_start(messages, reminder, mode, st.get("conv_start", 0), trigger, target)
    st["conv_start"] = conv_start
    _live_drop(messages, conv_start)                 # ongoing chat: always re-rope-drop the front (no recompute)
    win = messages[conv_start:]
    conv_ids, vis, per_turn_paths, tmpdir = _conv_ids(win, reminder, mode)
    try:
        pin_len = st["pin_len"]              # frozen physical length of the pinned [system+ctx+ACK] KV
        # Cross-turn reuse on the CONVERSATION tokens only (pure text past the pin → deterministic).
        # LCP against last turn's conversation tokens; the reminder rides the last user turn so the
        # match lands right after the previous user message — ~[last ai + new user + reminder] re-feeds.
        pf = PF["ids"] or []
        lcp_conv = _lcp(pf, conv_ids)
        # Images in HISTORY turns (or the reminder) sit in the reused prefix → _plan_feed refeeds from
        # the first placeholder (the old conservative rule). Images in the CURRENT message with an
        # UNCHANGED sig (compose-prefill already forward-passed them) keep the full LCP — the send
        # skips the image forward pass entirely (that's the aggressive-precompute payoff).
        lcp_conv, feed_list, vis = _plan_feed(conv_ids, lcp_conv, per_turn_paths, vis)
        if not feed_list:                    # nothing new (shouldn't happen — genprompt differs each turn)
            lcp_conv = _rewind_to(len(conv_ids) - 1)
            feed_list = conv_ids[lcp_conv:]
        feed_ids = mx.array([feed_list])
        reused = lcp_conv                    # conversation tokens reused from the cross-turn cache
        new_tokens = len(feed_list)
        # Breakdown of what was ACTUALLY forward-passed this turn (the feed tail), IN ORDER, summing
        # to new_tokens. A piece only counts if it's genuinely in the fed tail — a cached reminder or
        # cached last-reply contributes 0 (that's the whole point of the reminder modes).
        def _txt(bl): return "".join(b.get("text", "") for b in (bl or []) if b.get("type") == "text")
        _tok = getattr(processor, "tokenizer", processor)
        def _n(s):
            try: return len(_tok.encode(s)) if s else 0
            except Exception: return 0
        try: fed_text = _tok.decode(feed_list)
        except Exception: fed_text = ""
        def _fed(s): return bool(s.strip()) and s.strip()[:60] in fed_text
        rem_txt = REMINDER if reminder is None else _txt(reminder)
        user_txt = _txt(messages[-1]["content"]) if messages and messages[-1].get("role") == "user" else ""
        ai_txt = _txt(messages[-2]["content"]) if len(messages) >= 2 and messages[-2].get("role") == "assistant" else ""
        rem_n = _n(rem_txt) if _fed(rem_txt) else 0
        ai_n = _n(ai_txt) if _fed(ai_txt) else 0
        user_n = _n(user_txt) if _fed(user_txt) else 0
        parts = []                           # in forward-pass order
        if ai_n: parts.append(("last reply", ai_n))
        if mode == "before" and rem_n: parts.append(("reminder", rem_n))
        if user_n: parts.append(("your msg", user_n))
        if mode not in ("before", "start") and rem_n: parts.append(("reminder", rem_n))
        struct_n = max(0, new_tokens - sum(n for _, n in parts))   # turn markers / gen prompt (NOT the model appendix)
        if struct_n: parts.append(("structure", struct_n))
        anew_parts = [{"label": l, "n": n} for l, n in parts]
        _prog(op="generate", stage="prefill", done=0, total=new_tokens, label="prefilling", bump=True)
        gen = stream_generate(model, processor, "", input_ids=feed_ids,
                              prompt_cache=st["kv"],
                              max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                              prefill_step_size=PREFILL_STEP, **vis)   # vis: gemma pixel_values / qwen pixels+grid
        ttft, last, start = None, None, time.time()
        gen_count = 0
        gen_ids = []                             # sampled token ids (to record the KV's physical tail)
        answer = []                              # delta texts (to reconstruct the answer for precache)
        while True:
            if job.cancelled:
                try: gen.close()
                except Exception: pass
                break
            with MLX_LOCK:                       # serialize per decode-step (SPEC)
                try:
                    chunk = next(gen)
                except StopIteration:
                    break
            if ttft is None:
                ttft = time.time() - start
                _prog(stage="decode", done=0, total=0, label="generating")   # total=0 → indeterminate + live count
            gen_count = chunk.generation_tokens or gen_count   # running total (every chunk)
            _prog(done=gen_count, label=f"{gen_count} tok")
            tok = getattr(chunk, "token", None)
            if tok is not None:
                gen_ids.append(int(tok))
            if chunk.text:
                last = chunk
                answer.append(chunk.text)
                job.q.put(("delta", chunk.text))
        # CACHE ACROSS TURNS: the KV past the pin now physically holds [conv_ids + gen_ids]. Record
        # those conversation tokens so the NEXT /chat reuses all of them and re-feeds only the
        # divergent tail. Preserved on interrupt too (the history up to the interruption stays cached).
        PF["ids"] = conv_ids + gen_ids
        if per_turn_paths:
            PF["img_sig"] = _img_sig(per_turn_paths)
        if not job.cancelled:
            chat_tokens = len(conv_ids) + gen_count
            meta = {"done": True, "ttft": round(ttft or 0, 3), "gen_s": round(time.time() - start, 2),
                    "new_tokens": new_tokens, "chat_tokens": int(chat_tokens), "pinned": pin_len,
                    "reused": int(reused), "gen_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
                    "anew_parts": anew_parts, "mode": mode,
                    "conv_start": conv_start, "conv_tokens": len(conv_ids),   # sliding-window boundary + size
                    "mem_active_gb": _mem()[0], "mem_peak_gb": _mem()[1]}      # MLX Metal mem (spike check)
            job.q.put(("done", meta))
            # PRECACHE: warm the KV for the next message NOW (idle time). Use the RAW full answer —
            # the app stores Msg.text = the unmodified stream (INCLUDING @@APPENDIX@@) and resends it
            # verbatim next turn. The old appendix-strip made PF diverge from the app's history at the
            # FIRST answer's appendix, so every turn refed everything after it (the growing "structure"
            # anew; and on qwen it collapsed reuse to zero → 3-4s TTFT on real chats).
            ans = "".join(answer)
            if ans:
                msgs2 = messages + [{"role": "assistant", "content": [{"type": "text", "text": ans}]}]
                PRECACHE["state"] = "working"    # flip immediately (the job may sit behind others briefly)
                GEMMA_Q.put(Job("precache", messages=msgs2, reminder=reminder, mode=mode,
                                trigger=trigger, target=target, recache=recache))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _free_mem()


# ---------------- voice state machine ----------------
def voice_on_final(text):
    text = (text or "").strip()
    if not text:                      # nothing transcribed → straight back to idle, dot gone
        voice_cancel()
        return
    with VLOCK:
        VOICE["final"] = text
        VOICE["partial"] = text
        VOICE["seq"] += 1
        VOICE["state"] = "ready"
    OVL_Q.put(("hide",))              # dot is ONLY red (listening) or yellow (processing) — else hidden


def voice_processing():
    with VLOCK:
        VOICE["state"] = "processing"
    OVL_Q.put(("show", V.YELLOW))


def voice_cancel():
    VOICE_WANT["on"] = False        # keep intent in sync with the forced-idle so the next chord starts clean
    with VLOCK:
        VOICE["state"] = "idle"
        VOICE["partial"] = ""
        VOICE["final"] = None
    OVL_Q.put(("hide",))


def voice_ack(seq):
    with VLOCK:
        if seq == VOICE["seq"]:
            VOICE["state"] = "idle"
            VOICE["final"] = None
            VOICE["partial"] = ""
            return True
    return False


# ---------------- capture shortcuts (screenshot + copy-to-chat) ----------------
def push_capture(cap):
    """Enqueue a capture event for the app to drain via /voice/poll + /voice/captures-ack."""
    if not cap:
        return
    with CAP_LOCK:
        CAPTURES.append(cap)


def snapshot_captures():
    with CAP_LOCK:
        return list(CAPTURES)


def drain_captures(count):
    """Remove the first `count` captures (the ones the app just consumed — FIFO, race-safe)."""
    with CAP_LOCK:
        del CAPTURES[:max(0, count)]


def _shot_initiate():
    """press-to-initiate: native screencapture -i on its own thread (ESC = no file = no-op)."""
    b64 = V.capture_interactive()
    if b64:
        push_capture({"kind": "image", "data": b64})


def _shot_region(x, y, w, h):
    """press&hold: capture the dragged rectangle (TOP-LEFT global points)."""
    b64 = V.capture_region(x, y, w, h)
    if b64:
        push_capture({"kind": "image", "data": b64})


def _do_copy():
    """copy-to-chat: synthesize ⌘C to the frontmost app, let the pasteboard settle, read it."""
    V.synth_cmd_c()
    time.sleep(0.15)
    cap = V.read_pasteboard_capture()
    if cap:
        push_capture(cap)


# ---------------- voice thread (parakeet: load + transcription) ----------------
def voice_thread():
    gemma_ready.wait()
    parakeet["model"] = V.load_parakeet(log)
    rec = None
    try:
        rec = V.Rec()
    except Exception as e:
        log(f"mic init failed: {e}")
    active_streaming = {"on": False}   # is the current listen session a (Apple-owned) streaming one?
    poll_dt = 0.3
    while True:
        try:
            cmd = VOICE_Q.get(timeout=poll_dt)
        except queue.Empty:
            cmd = None
        m = parakeet["model"]
        if cmd is None:
            continue
        kind = cmd[0]
        if kind == "start":
            streaming = cmd[1]
            # STREAMING submode: the Swift app owns audio + transcription via Apple SpeechTranscriber.
            # Python only drives the overlay + reports the chord state; it does NOT open the mic.
            if streaming:
                active_streaming["on"] = True
                with VLOCK:
                    VOICE["state"] = "listening"; VOICE["partial"] = ""; VOICE["final"] = None
                OVL_Q.put(("show", V.RED))
                continue
            # TRANSCRIBE-AFTER submode: Python captures the built-in mic → Parakeet BATCH at stop.
            active_streaming["on"] = False
            if rec is None or m is None:
                OVL_Q.put(("flash", V.YELLOW, 0.6))   # not ready
                continue
            try:
                rec.start()
                with VLOCK:
                    VOICE["state"] = "listening"; VOICE["partial"] = ""; VOICE["final"] = None
                OVL_Q.put(("show", V.RED))
            except Exception as e:
                log(f"mic start failed: {e}")
                OVL_Q.put(("flash", V.YELLOW, 1.0))
        elif kind == "stop":
            if active_streaming["on"]:
                # streaming chord-up: signal the app to FINALIZE its Apple transcript (state=ready,
                # bumped seq). The app reads Apple's text (not VOICE["final"]); its ack resets us to
                # idle. ESC instead routes to "cancel" → voice_cancel → idle (app discards).
                active_streaming["on"] = False
                OVL_Q.put(("hide",))
                with VLOCK:
                    VOICE["seq"] += 1
                    VOICE["state"] = "ready"; VOICE["partial"] = ""; VOICE["final"] = ""
                continue
            if rec is None or not rec.on:
                continue
            voice_processing()
            audio, sr = rec.stop(), rec.sr
            try:
                text = V.transcribe_batch(m, audio, sr) if m is not None else ""
            except Exception as e:
                log(f"transcribe failed: {e}"); text = ""
            voice_on_final(text)
        elif kind == "cancel":
            active_streaming["on"] = False
            if rec is not None and rec.on:
                rec.stop()
            voice_cancel()


# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("content-length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"loaded": gemma_ready.is_set(),
                                    "model": os.path.basename(MODEL_PATH),
                                    "parakeet": parakeet["model"] is not None,
                                    "precache": PRECACHE["state"],
                                    "ctxWindow": CTX_WINDOW,
                                    "memGb": MEM["active_gb"], "memCeilingGb": MEM["ceiling_gb"],
                                    "memOver": MEM["over"], "memReclaims": MEM["reclaims"]})
        if self.path == "/progress":
            # Real per-op progress for the UI's cache HUD. Runs on a separate HTTP thread, reads the
            # worker's live counter under the tiny PROG_LOCK (never MLX_LOCK) → answers while the GPU is busy.
            with PROG_LOCK:
                snap = dict(PROGRESS)              # copy under the fast lock…
            return self._json(200, snap)           # …serialize/send outside it
        if self.path == "/voice/poll":
            with VLOCK:
                out = {"state": VOICE["state"], "partial": VOICE["partial"],
                       "final": VOICE["final"], "seq": VOICE["seq"], "perm": VOICE["perm"]}
            out["captures"] = snapshot_captures()   # events channel (drained by /voice/captures-ack)
            return self._json(200, out)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/new":
                # drop the pinned cache and install the empty baseline (pin_len=0) in its place,
                # so a fresh chat never needs a /pin round-trip before chatting.
                job = Job("pin", body={})
                GEMMA_Q.put(job)
                job.done.wait()
                return self._json(200, {"ok": True})

            if self.path == "/pin":
                d = self._body()
                job = Job("pin", body=d)
                GEMMA_Q.put(job)
                job.done.wait()
                if job.error:
                    return self._json(500, {"error": str(job.error)})
                return self._json(200, job.result)

            if self.path == "/chat":
                body = self._body()
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    return self._json(400, {"error": "empty messages"})
                if not st.get("kv"):
                    return self._json(409, {"error": "nothing pinned — cache first"})
                self.send_response(200)
                self.send_header("content-type", "application/x-ndjson")
                self.end_headers()
                PRECACHE["state"] = "idle"     # a new turn begins; clear the previous "done" flag
                job = Job("generate", messages=messages, reminder=body.get("reminder"),
                          mode=body.get("reminderMode") or "last",
                          trigger=body.get("trimTrigger"), target=body.get("trimTarget"),
                          recache=body.get("recacheMode"))
                GEMMA_Q.put(job)
                while True:
                    kind, payload = job.q.get()
                    if kind == "delta":
                        try:
                            self.wfile.write((json.dumps({"delta": payload}) + "\n").encode())
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            job.cancelled = True
                            return
                    elif kind == "done":
                        try:
                            self.wfile.write((json.dumps(payload) + "\n").encode())
                            self.wfile.flush()
                        except Exception:
                            pass
                        return
                    elif kind == "err":
                        try:
                            self.wfile.write((json.dumps({"error": payload}) + "\n").encode())
                            self.wfile.flush()
                        except Exception:
                            pass
                        return

            if self.path == "/voice/config":
                d = self._body()
                with VLOCK:
                    # voiceEnabled (legacy "enabled" accepted) gates the voice chord; captureEnabled
                    # gates the screenshot/copy bindings. Either one keeps the tap alive.
                    VOICE["enabled"] = bool(d.get("voiceEnabled", d.get("enabled", VOICE["enabled"])))
                    VOICE["submode"] = d.get("submode", VOICE["submode"])
                    VOICE["streaming"] = bool(d.get("streaming", VOICE["streaming"]))
                    VOICE["key"] = d.get("key", VOICE["key"])
                    VOICE["hotkeyMode"] = d.get("hotkeyMode", VOICE["hotkeyMode"])   # tap|self|karabiner
                    if not VOICE["enabled"]:
                        VOICE_WANT["on"] = False    # voice turned off → clear intent so re-enable starts clean
                        VOICE["state"] = "idle"; VOICE["partial"] = ""; VOICE["final"] = None
                    CAPTURE["enabled"] = bool(d.get("captureEnabled", CAPTURE["enabled"]))
                    shot = d.get("shot") or {}
                    if "binding" in shot: CAPTURE["shot_binding"] = shot.get("binding") or ""
                    if "style" in shot: CAPTURE["shot_style"] = shot.get("style") or "initiate"
                    cp = d.get("copy") or {}
                    if "binding" in cp: CAPTURE["copy_binding"] = cp.get("binding") or ""
                    CAPTURE["shot_parsed"] = V.parse_binding(CAPTURE["shot_binding"])
                    CAPTURE["copy_parsed"] = V.parse_binding(CAPTURE["copy_binding"])
                return self._json(200, {"ok": True, "perm": VOICE["perm"]})

            if self.path == "/trigger":
                # External hotkey trigger — the Swift app's RegisterEventHotKey OR a Karabiner-invoked CLI
                # (no CGEventTap). Pushes the SAME events the tap used onto TAP_Q, so the voice/capture
                # state machines are unchanged. kind: chord_down|chord_up|cancel|shot|copy.
                d = self._body()
                kind = d.get("kind", "")
                # Gate the DICTATION chord on voiceEnabled: in Karabiner mode the app can't unregister the
                # hook, so the CLI keeps firing chord_down even with this chat's voice OFF — without this it
                # would start a real listening session (mic + red overlay). Capture (shot/copy) is separate.
                if kind == "chord_down": (TAP_Q.put("chord_down") if VOICE["enabled"] else None)
                elif kind == "chord_up": (TAP_Q.put("chord_up") if VOICE["enabled"] else None)
                elif kind == "cancel":   TAP_Q.put("esc")
                elif kind == "shot":     TAP_Q.put(("shot_down", float(d.get("x", 0)), float(d.get("y", 0))))
                elif kind == "shot_up":  TAP_Q.put(("shot_up", float(d.get("x", 0)), float(d.get("y", 0))))
                elif kind == "copy":     TAP_Q.put(("copy",))
                return self._json(200, {"ok": True})

            if self.path == "/keepalive":
                # Idle keepalive — the app pings while its window is frontmost + a chat is open, to keep
                # the ~18GB of weights+KV resident and the GPU warm (else the first token after an idle
                # gap pays a ~3s cold-page fault). Fire-and-forget + coalesced: never queue a second while
                # one is pending, and never block the HTTP handler behind a real job.
                if not KEEPALIVE["pending"]:
                    KEEPALIVE["pending"] = True
                    GEMMA_Q.put(Job("keepalive"))
                return self._json(200, {"ok": True})

            if self.path == "/voice/prefill":
                # Aggressive precompute: the app samples the composer (typed text, Apple partial, staged
                # images) and this prefill feeds [history + OPEN user(images-first + text)] past the pin.
                # {messages, partial, images?, reminder?, reminderMode?} → {fed, reused}.
                d = self._body()
                if not st.get("kv"):
                    return self._json(200, {"fed": 0})
                job = Job("prefill", messages=d.get("messages") or [], partial=str(d.get("partial") or ""),
                          images=d.get("images") or [], reminder=d.get("reminder"),
                          mode=d.get("reminderMode") or "last")
                GEMMA_Q.put(job)
                job.done.wait()
                return self._json(200, job.result or {"fed": 0})

            if self.path == "/precache":
                # re-warm the KV for the next message with a (possibly new) reminder mode. Used when
                # the user switches reminder placement mid-chat — the chip shows caching → ready.
                d = self._body()
                if not st.get("kv"):
                    return self._json(200, {"ok": True})
                PRECACHE["state"] = "working"
                job = Job("precache", messages=d.get("messages") or [], reminder=d.get("reminder"),
                          mode=d.get("reminderMode") or "last",
                          trigger=d.get("trimTrigger"), target=d.get("trimTarget"),
                          recache=d.get("recacheMode"))
                GEMMA_Q.put(job); job.done.wait()
                return self._json(200, {"ok": True})

            if self.path == "/reconcile":
                # Rebuild the conversation cache so it's CORRECT for the current settings (chat open, or
                # a content/setting change). recent → prefill the window; streaming → replay with smear.
                d = self._body()
                if not st.get("kv"):
                    return self._json(200, {"ok": True, "conv_start": 0, "conv_tokens": 0})
                PRECACHE["state"] = "working"
                job = Job("reconcile", messages=d.get("messages") or [], reminder=d.get("reminder"),
                          mode=d.get("reminderMode") or "last",
                          trigger=d.get("trimTrigger"), target=d.get("trimTarget"),
                          recache=d.get("recacheMode"))
                GEMMA_Q.put(job); job.done.wait()
                return self._json(200, {"ok": True, **(job.result or {})})

            if self.path == "/voice/ack":
                d = self._body()
                ok = voice_ack(int(d.get("seq", -1)))
                return self._json(200, {"ok": ok})

            if self.path == "/voice/captures-ack":
                d = self._body()
                drain_captures(int(d.get("count", 0)))
                return self._json(200, {"ok": True, "pending": len(snapshot_captures())})

            if self.path == "/voice/inject":
                d = self._body()
                # capture test hook: enqueue as if a real screenshot/copy shortcut fired (no perms needed).
                if "capture" in d:
                    cap = d.get("capture") or {}
                    kind = cap.get("kind")
                    if kind == "text":
                        push_capture({"kind": "text", "text": str(cap.get("text") or "")})
                    elif kind == "image":
                        push_capture({"kind": "image", "data": str(cap.get("data") or "")})
                    else:
                        return self._json(400, {"error": "capture needs kind text|image"})
                    return self._json(200, {"ok": True, "pending": len(snapshot_captures())})
                if "text" in d:
                    voice_on_final(str(d.get("text") or ""))
                    return self._json(200, {"ok": True, "state": "ready"})
                return self._json(400, {"error": "need text or capture"})

            self._json(404, {"error": "not found"})
        except Exception as e:
            import traceback
            log("ERR", traceback.format_exc())
            try:
                self._json(500, {"error": str(e)})
            except Exception:
                pass


class Server(ThreadingHTTPServer):   # ThreadingHTTPServer already mixes in ThreadingMixIn
    daemon_threads = True
    allow_reuse_address = True


def bind_server():
    """Bind :PORT NOW, before the 16GB model loads. If the port is taken (another engine already
    running — e.g. a second app instance), exit IMMEDIATELY rather than loading weights and then
    lingering headless holding memory. Fail-fast is what makes a duplicate launch harmless."""
    try:
        return Server(("127.0.0.1", PORT), H)
    except OSError as e:
        log(f"port {PORT} already in use ({e}) — another engine is running; exiting.")
        os._exit(0)


def serve_http(server):
    log(f"engine HTTP ready on :{PORT}")
    server.serve_forever()


# ---------------- parent watchdog (unchanged intent) ----------------
def watch_parent():
    ppid = os.getppid()

    def loop():
        while True:
            time.sleep(2)
            if os.getppid() != ppid or os.getppid() == 1:
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()


# ---------------- main: AppKit runloop + tap-queue timer ----------------
def _tap_cfg():
    """Lock-free snapshot the tap callback reads (main thread only; refreshed each tick)."""
    return TAP_CFG


def _mouse_loc_topleft():
    """Current cursor position in TOP-LEFT global points (CGEvent / screencapture space)."""
    return V.mouse_loc_topleft()


def _drain_tap_and_overlay(overlay, sel, tap, flash):
    # overlay commands (must touch AppKit on main only)
    while True:
        try:
            cmd = OVL_Q.get_nowait()
        except queue.Empty:
            break
        if overlay is None:
            continue
        try:
            if cmd[0] == "show":
                overlay.show(cmd[1]); flash["until"] = 0
            elif cmd[0] == "hide":
                overlay.hide(); flash["until"] = 0
            elif cmd[0] == "flash":
                overlay.show(cmd[1]); flash["until"] = time.time() + cmd[2]
        except Exception as e:
            log(f"overlay error: {e}")
    if overlay is not None and flash["until"] and time.time() > flash["until"]:
        with VLOCK:
            active = VOICE["state"] in ("listening", "processing")
        if not active:
            try: overlay.hide()
            except Exception: pass
        flash["until"] = 0

    # selection rectangle: follows the cursor while a press&hold screenshot drag is armed
    if sel is not None:
        if SHOT["armed"]:
            cx, cy = _mouse_loc_topleft()
            ax, ay = SHOT["anchor"]
            try:
                sel.show_topleft(min(ax, cx), min(ay, cy), abs(ax - cx), abs(ay - cy))
            except Exception as e:
                log(f"sel overlay error: {e}")
            SHOT["shown"] = True
        elif SHOT["shown"]:
            try: sel.hide()
            except Exception: pass
            SHOT["shown"] = False

    # lazy tap lifecycle: the global CGEventTap is created ONLY in legacy "tap" mode. In "self" /
    # "karabiner" mode we install NO tap (zero interference) — triggers arrive via /trigger instead.
    want_tap = (VOICE.get("hotkeyMode", "self") == "tap") and (VOICE["enabled"] or CAPTURE["enabled"])
    if want_tap and not tap["obj"]:
        t = V.Tap(TAP_Q, _should_consume_esc, cfg=_tap_cfg, log=log, prompt_perms=False)
        ok = t.create()
        tap["obj"] = t
        with VLOCK:
            VOICE["perm"] = ok
    elif not want_tap and tap["obj"]:
        tap["obj"].destroy(); tap["obj"] = None
    if tap["obj"]:
        tap["obj"].tick()

    # refresh the lock-free snapshot the tap callback reads (parsed bindings + enable flags)
    with VLOCK:
        TAP_CFG["voice"] = VOICE["enabled"]
        TAP_CFG["capture"] = CAPTURE["enabled"]
        TAP_CFG["shot"] = CAPTURE["shot_parsed"]
        TAP_CFG["copy"] = CAPTURE["copy_parsed"]
        TAP_CFG["chord_mods"] = V.parse_chord_mods(VOICE["key"])   # user-set dictation chord (was hardcoded ctrl+alt)

    # drain raw physical events -> apply submode / capture-style / state semantics
    while True:
        try:
            ev = TAP_Q.get_nowait()
        except queue.Empty:
            break
        _apply_tap_event(ev)


def _should_consume_esc():
    with VLOCK:
        return VOICE["state"] in ("listening", "processing")


def _apply_capture_event(ev):
    """Screenshot / copy-to-chat tap events (main thread). Blocking work spawns a daemon thread."""
    kind = ev[0]
    if kind == "shot_down":
        x, y = ev[1], ev[2]
        with VLOCK:
            style = CAPTURE["shot_style"]
        if style == "hold":
            SHOT["armed"] = True
            SHOT["anchor"] = (x, y)          # rect overlay drawn by the timer while armed
        else:                                # press to initiate — native crosshair
            threading.Thread(target=_shot_initiate, daemon=True).start()
    elif kind == "shot_up":
        if SHOT["armed"]:                    # press&hold release → capture the dragged region
            SHOT["armed"] = False
            x, y = ev[1], ev[2]
            ax, ay = SHOT["anchor"]
            rx, ry, rw, rh = min(ax, x), min(ay, y), abs(ax - x), abs(ay - y)
            if rw >= 4 and rh >= 4:          # ignore a click / zero-size drag
                threading.Thread(target=_shot_region, args=(rx, ry, rw, rh), daemon=True).start()
    elif kind == "copy":
        threading.Thread(target=_do_copy, daemon=True).start()


def _apply_tap_event(ev):
    if isinstance(ev, tuple):                # capture events (shot_down/shot_up/copy)
        _apply_capture_event(ev)
        return
    with VLOCK:
        state = VOICE["state"]
        submode = VOICE["submode"]
        streaming = VOICE["streaming"]
    if ev == "esc":
        if VOICE_WANT["on"] or state in ("listening", "processing"):
            VOICE_WANT["on"] = False
            VOICE_Q.put(("cancel",))
        return
    if submode == "toggle":
        # a COMPLETE press+release (chord_up) toggles listening on/off — "start when both keys pressed
        # and released, and again to stop". chord_down is ignored so it doesn't fire mid-press.
        if ev == "chord_up":
            VOICE_WANT["on"] = not VOICE_WANT["on"]
            VOICE_Q.put(("start", streaming) if VOICE_WANT["on"] else ("stop",))
    else:  # hold: listen only WHILE the chord is held. Gate on intent, not the lagging state, so a fast
           # press+release still queues both start and stop (FIFO → no stuck "listening").
        if ev == "chord_down" and not VOICE_WANT["on"]:
            VOICE_WANT["on"] = True
            VOICE_Q.put(("start", streaming))
        elif ev == "chord_up" and VOICE_WANT["on"]:
            VOICE_WANT["on"] = False
            VOICE_Q.put(("stop",))


def main():
    watch_parent()
    server = bind_server()   # claim the port BEFORE loading 16GB — a duplicate launch exits here, instantly
    # background workers (Gemma loads inside its worker; parakeet after, in the voice thread)
    threading.Thread(target=gemma_worker, daemon=True).start()
    threading.Thread(target=voice_thread, daemon=True).start()
    threading.Thread(target=lambda: serve_http(server), daemon=True).start()
    threading.Thread(target=mem_watchdog, daemon=True).start()   # hard memory-boundary enforcement

    # AppKit main runloop (accessory app): overlay + tap-queue NSTimer. Degrade gracefully headless.
    try:
        from AppKit import NSApplication, NSTimer
        from PyObjCTools import AppHelper
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(2)  # accessory (no Dock icon)
        try:
            overlay = V.Overlay()
        except Exception as e:
            log(f"overlay init failed (headless?): {e}")
            overlay = None
        try:
            sel = V.SelOverlay()
        except Exception as e:
            log(f"selection overlay init failed (headless?): {e}")
            sel = None
        tap = {"obj": None}
        flash = {"until": 0}
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.03, True, lambda _t: _drain_tap_and_overlay(overlay, sel, tap, flash))
        log("AppKit runloop up (accessory)")
        AppHelper.runEventLoop()
    except Exception as e:
        log(f"AppKit unavailable ({e}); running headless service loop")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
