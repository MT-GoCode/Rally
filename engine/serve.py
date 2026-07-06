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
  POST /voice/context     -> {messages, reminder}  (base transcript for streaming prefill)
  GET  /voice/poll        -> {state, partial, final, seq, perm, captures:[{kind,data|text}]}
  POST /voice/ack         -> {seq}
  POST /voice/captures-ack-> {count}  (drop the first `count` drained capture events)
  POST /voice/inject      -> {text} (final->ready) | {partial} (streaming prefill)
                             | {capture:{kind:"text"|"image", text|data}}  [test hook]

Block shape (interro-verbatim): {"type":"text","text":..} |
  {"type":"image","source":{"type":"base64","media_type":..,"data":..}}
"""
import base64, json, os, queue, shutil, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_vlm import load, stream_generate, apply_chat_template, prepare_inputs
from mlx_lm.models.cache import make_prompt_cache

import voice as V

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "models/gemma-4-26b-a4b-4bit")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5177
TOKEN_LIMIT = 200_000
CTX_WINDOW = 262_144
ACK = "Understood — I have the reference material. Ask me anything."
MAX_TOKENS = 2048         # SAFETY CEILING ONLY — the model stops at <end_of_turn>. Length is
                          # controlled by REMINDER, not this.
TEMPERATURE = 0.5
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

st = {}                                    # pinned chat: kv, pin_len, ctx_content, ctx_paths, tmpdir, system
PF = {"ids": None}                         # streaming-prefill state (Gemma-worker-owned): token id
                                           #   list currently in the KV past the pin, or None (pin only)

# ---- voice state machine (shared; guarded by VLOCK) ----
VLOCK = threading.Lock()
VOICE = {"enabled": False, "submode": "toggle", "streaming": False, "key": "ctrl+alt",
         "state": "idle", "partial": "", "final": None, "seq": 0, "perm": True}
VOICE_CTX = {"messages": [], "reminder": None}   # base transcript for streaming prefill

# ---- capture shortcuts (screenshot + copy-to-chat); config guarded by VLOCK ----
# The tap is active whenever a chat is open in ANY mode (voice OR capture enabled). "enabled" here
# tracks captureEnabled; bindings are parse_binding() forms cached alongside the raw strings.
CAPTURE = {"enabled": False, "shot_binding": "cmd+shift+2", "shot_style": "initiate",
           "copy_binding": "ctrl+alt+c", "shot_parsed": None, "copy_parsed": None}
CAPTURES = []            # pending capture events for the app: {kind:"image",data:b64}|{kind:"text",text}
CAP_LOCK = threading.Lock()
# TAP_CFG: lock-free snapshot the tap callback reads (refreshed by the main-thread timer under VLOCK;
# both callback and timer run on the main thread, so no lock is needed on the read side).
TAP_CFG = {"voice": False, "capture": False, "shot": None, "copy": None}
# SHOT: press&hold drag state — touched ONLY on the main thread (timer + tap-event application).
SHOT = {"armed": False, "anchor": (0.0, 0.0), "shown": False}

TAP_Q = queue.Queue()    # raw physical events from the tap (drained on main thread)
OVL_Q = queue.Queue()    # overlay commands, applied on main thread: ("show",rgb)/("hide",)/("flash",rgb,secs)
VOICE_Q = queue.Queue()  # commands for the voice thread: ("start",streaming)/("stop",)/("cancel",)


def log(*a): print(*a, file=sys.stderr, flush=True)


def load_model():
    global model, processor, config
    log(f"loading {MODEL_PATH} …")
    model, processor = load(MODEL_PATH)
    config = model.config
    log("Gemma loaded")


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
    return apply_chat_template(processor, config, msgs, num_images=nimg, add_generation_prompt=add_gen)


def token_ids(prompt, paths):
    inp = prepare_inputs(processor, images=paths or None, prompts=prompt,
                         image_token_index=getattr(config, "image_token_index", None))
    return inp["input_ids"]


def trim_kv(kv, n):
    """Reset the pinned cache back to exactly its first n tokens (drop prior turn). Unchanged."""
    for c in kv:
        if getattr(c, "keys", None) is not None and c.keys.shape[2] > n:
            c.keys = c.keys[:, :, :n, :]
            c.values = c.values[:, :, :n, :]
            if hasattr(c, "offset"):
                c.offset = n


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


def _lcp(a, b):
    """Longest common prefix length of two token-id lists."""
    if not a or not b:
        return 0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


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
        except Exception as e:
            import traceback
            log("worker ERR", traceback.format_exc())
            job.error = e
            try:
                job.q.put(("err", str(e)))
            except Exception:
                pass
        finally:
            job.done.set()


def _drop_pin():
    if st.get("tmpdir"):
        shutil.rmtree(st["tmpdir"], ignore_errors=True)
    st.clear()
    PF["ids"] = None


def _do_pin(job):
    d = job.params["body"]
    tmpdir = tempfile.mkdtemp(prefix="civm-")
    paths = []
    # images placed in the SYSTEM box are folded into the visual context so the model sees them.
    sys_imgs = content_of([b for b in (d.get("system") or []) if b.get("type") == "image"], tmpdir, paths)
    ctx_content = sys_imgs + content_of(d.get("context"), tmpdir, paths)
    msgs = build_messages(d.get("system"), ctx_content, [])
    if msgs:
        pin_prompt = prompt_str(msgs, len(paths), add_gen=False)
        pin_len = int(token_ids(pin_prompt, paths).shape[-1])
        if pin_len > TOKEN_LIMIT:
            shutil.rmtree(tmpdir, ignore_errors=True)
            job.result = {"ok": False, "overLimit": True, "tokens": pin_len}
            return
        kv = make_prompt_cache(model)
        with MLX_LOCK:
            for _ in stream_generate(model, processor, pin_prompt, image=paths or None,
                                     prompt_cache=kv, max_tokens=1, temperature=0.0):
                pass
        trim_kv(kv, pin_len)       # drop the 1 primed token -> cache holds exactly the prefix
    else:
        kv, pin_len = make_prompt_cache(model), 0
    _drop_pin()
    st.update(kv=kv, pin_len=pin_len, ctx_content=ctx_content, ctx_paths=paths,
              tmpdir=tmpdir, system=d.get("system"))
    PF["ids"] = None
    job.result = {"ok": True, "overLimit": False, "tokens": pin_len}


def _build_full(messages, reminder):
    """Render [pin prefix + messages(+reminder on last user)] -> (full_ids mx, per_turn_paths, tmpdir).
    per-turn images (in messages OR reminder) written to a per-request tmpdir, ordered AFTER pinned."""
    tmpdir = tempfile.mkdtemp(prefix="civm-turn-")
    per_turn_paths = []
    mlx_msgs = _mlx_messages(messages, tmpdir, per_turn_paths)
    rem_content = _reminder_content(reminder, tmpdir, per_turn_paths)
    mlx_msgs = _append_reminder(mlx_msgs, rem_content)
    all_paths = st["ctx_paths"] + per_turn_paths
    full_msgs = build_messages(st["system"], st["ctx_content"], mlx_msgs)
    prompt = prompt_str(full_msgs, len(all_paths), add_gen=True)
    full_ids = token_ids(prompt, all_paths)
    return full_ids, per_turn_paths, tmpdir


def _do_generate(job):
    if not st.get("kv"):
        job.q.put(("err", "nothing pinned — cache first"))
        return
    messages = job.params["messages"]
    reminder = job.params["reminder"]
    full_ids, per_turn_paths, tmpdir = _build_full(messages, reminder)
    try:
        full_list = full_ids.flatten().tolist()
        pin_len = st["pin_len"]
        # prefill reuse: trim to the longest common prefix with what's already fed (>= pin).
        lcp = max(_lcp(PF["ids"] or [], full_list), pin_len)
        trim_kv(st["kv"], lcp)
        feed_ids = full_ids[:, lcp:]
        per_pv = _per_turn_pixels(per_turn_paths)
        reused = lcp - pin_len       # tokens past pin reused from streaming prefill
        new_tokens = int(feed_ids.shape[-1])
        gen = stream_generate(model, processor, "", input_ids=feed_ids,
                              pixel_values=per_pv, prompt_cache=st["kv"],
                              max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        ttft, last, start = None, None, time.time()
        gen_count = 0
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
            gen_count = chunk.generation_tokens or gen_count   # running total (every chunk)
            if chunk.text:
                last = chunk
                job.q.put(("delta", chunk.text))
        if not job.cancelled:
            chat_tokens = (len(full_list) - pin_len) + gen_count
            meta = {"done": True, "ttft": round(ttft or 0, 3), "gen_s": round(time.time() - start, 2),
                    "new_tokens": new_tokens, "chat_tokens": int(chat_tokens), "pinned": pin_len,
                    "reused": int(reused), "gen_tps": round(getattr(last, "generation_tps", 0) or 0, 1)}
            job.q.put(("done", meta))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        PF["ids"] = None            # a full turn was fed; next utterance re-prefills cleanly from pin


def _do_prefill(job):
    """Streaming prefill: feed [messages + OPEN user(partial)] tokens past the pin, incrementally.
    Uses max_tokens=0 for a clean prefill (no spurious generated token in the sliding-window cache)
    and keeps the user turn OPEN (strips the trailing <end_of_turn>) so /chat's LCP reuse is exact."""
    if not st.get("kv"):
        return
    partial = job.params["partial"]
    messages = job.params["messages"]
    tmpdir = tempfile.mkdtemp(prefix="civm-pf-")
    try:
        per_turn_paths = []
        mlx_msgs = _mlx_messages(messages, tmpdir, per_turn_paths)
        mlx_msgs = mlx_msgs + [{"role": "user", "content": [{"type": "text", "text": partial}]}]
        all_paths = st["ctx_paths"] + per_turn_paths
        full_msgs = build_messages(st["system"], st["ctx_content"], mlx_msgs)
        prompt = prompt_str(full_msgs, len(all_paths), add_gen=False)
        ids_list = token_ids(prompt, all_paths).flatten().tolist()
        open_list = _strip_open(ids_list)
        pin_len = st["pin_len"]
        lcp = max(_lcp(PF["ids"] or [], open_list), pin_len)
        if lcp >= len(open_list):
            PF["ids"] = open_list          # nothing new to feed
            return
        trim_kv(st["kv"], lcp)
        feed = mx.array([open_list[lcp:]])
        per_pv = _per_turn_pixels(per_turn_paths)
        t0 = time.time()
        with MLX_LOCK:                      # per prefill batch (SPEC)
            for _ in stream_generate(model, processor, "", input_ids=feed, pixel_values=per_pv,
                                     prompt_cache=st["kv"], max_tokens=0, temperature=0.0):
                pass
        PF["ids"] = open_list
        # live-computation proof: tail -f serve.log while dictating to watch tokens land in the KV
        log(f"prefill +{len(open_list) - lcp} tok ({len(open_list) - pin_len} past pin) in {time.time() - t0:.2f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _strip_open(ids_list):
    """Drop the trailing '<end_of_turn>\\n' so the last user turn stays OPEN (rotating-cache-safe)."""
    eot = _eot_id()
    if eot is not None:
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == eot:
                return ids_list[:i]
    return ids_list


_EOT = {"id": "unset"}
def _eot_id():
    if _EOT["id"] == "unset":
        tok = getattr(processor, "tokenizer", processor)
        try:
            _EOT["id"] = tok.convert_tokens_to_ids("<end_of_turn>")
        except Exception:
            _EOT["id"] = None
    return _EOT["id"]


# ---------------- voice state machine (shared logic; mic AND /voice/inject) ----------------
def _enqueue_prefill(partial):
    if not VOICE["streaming"] or not st.get("kv"):
        return
    with VLOCK:
        msgs = list(VOICE_CTX["messages"])
    GEMMA_Q.put(Job("prefill", partial=partial, messages=msgs))


def voice_on_partial(text):
    with VLOCK:
        if VOICE["state"] not in ("listening",):
            VOICE["state"] = "listening"
        VOICE["partial"] = text or ""
    OVL_Q.put(("show", V.RED))
    _enqueue_prefill(text or "")


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
    stream = {"s": None}
    poll_dt = 0.3
    while True:
        try:
            cmd = VOICE_Q.get(timeout=poll_dt)
        except queue.Empty:
            cmd = None
        m = parakeet["model"]
        # streaming: while listening, pull new audio, feed parakeet, publish growing partial.
        # Resample the WHOLE buffer to 16k each poll and feed only the new 16k tail — avoids the
        # per-chunk filter edge artifacts of resampling each increment independently.
        if stream["s"] is not None and rec is not None and rec.on:
            try:
                audio16 = V.resample_16k(rec.snapshot(), rec.sr)
                fed = stream.get("fed", 0)
                if len(audio16) > fed:
                    stream["fed"] = len(audio16)
                    stream["s"].add(audio16[fed:])
                    txt = stream["s"].text()
                    if txt and txt != VOICE.get("partial"):
                        voice_on_partial(txt)
            except Exception as e:
                log(f"stream transcribe error: {e}")
        if cmd is None:
            continue
        kind = cmd[0]
        if kind == "start":
            streaming = cmd[1]
            if rec is None or m is None:
                OVL_Q.put(("flash", V.YELLOW, 0.6))   # not ready
                continue
            try:
                rec.start()
                with VLOCK:
                    VOICE["state"] = "listening"; VOICE["partial"] = ""; VOICE["final"] = None
                OVL_Q.put(("show", V.RED))
                if streaming:
                    stream["s"] = V.Stream(m, MLX_LOCK)
                    stream["fed"] = 0
            except Exception as e:
                log(f"mic start failed: {e}")
                OVL_Q.put(("flash", V.YELLOW, 1.0))
        elif kind == "stop":
            if rec is None or not rec.on:
                continue
            voice_processing()
            audio, sr = rec.stop(), rec.sr
            if stream["s"] is not None:
                try: stream["s"].close()
                except Exception: pass
                stream["s"] = None
            try:
                text = V.transcribe_batch(m, audio, sr) if m is not None else ""
            except Exception as e:
                log(f"transcribe failed: {e}"); text = ""
            voice_on_final(text)
        elif kind == "cancel":
            if rec is not None and rec.on:
                rec.stop()
            if stream["s"] is not None:
                try: stream["s"].close()
                except Exception: pass
                stream["s"] = None
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
                                    "ctxWindow": CTX_WINDOW})
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
                job = Job("generate", messages=messages, reminder=body.get("reminder"))
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
                    if not VOICE["enabled"]:
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

            if self.path == "/voice/context":
                d = self._body()
                with VLOCK:
                    VOICE_CTX["messages"] = d.get("messages") or []
                    VOICE_CTX["reminder"] = d.get("reminder")
                return self._json(200, {"ok": True})

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
                if "partial" in d:
                    voice_on_partial(str(d.get("partial") or ""))
                    return self._json(200, {"ok": True, "state": "listening"})
                return self._json(400, {"error": "need text, partial, or capture"})

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


def serve_http():
    log(f"engine HTTP ready on :{PORT}")
    Server(("127.0.0.1", PORT), H).serve_forever()


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

    # lazy tap lifecycle: create when EITHER voice or capture is enabled (perms belong to the app)
    want_tap = VOICE["enabled"] or CAPTURE["enabled"]
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
        if state in ("listening", "processing"):
            VOICE_Q.put(("cancel",))
        return
    if submode == "toggle":
        if ev == "chord_down":
            if state == "listening":
                VOICE_Q.put(("stop",))
            elif state in ("idle", "ready"):
                VOICE_Q.put(("start", streaming))
        # chord_up ignored in toggle
    else:  # hold
        if ev == "chord_down" and state in ("idle", "ready"):
            VOICE_Q.put(("start", streaming))
        elif ev == "chord_up" and state == "listening":
            VOICE_Q.put(("stop",))


def main():
    watch_parent()
    # background workers (Gemma loads inside its worker; parakeet after, in the voice thread)
    threading.Thread(target=gemma_worker, daemon=True).start()
    threading.Thread(target=voice_thread, daemon=True).start()
    threading.Thread(target=serve_http, daemon=True).start()

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
