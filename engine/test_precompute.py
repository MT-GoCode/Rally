"""Precompute + pre-generation gate — BOTH models. Every case checks OUTPUT TEXT quality, not just
timing (the gemma thinking-token garbage shipped because earlier tests only timed).

Cases per model:
  1 compose-grow      X=Y+Z invariant, Y monotone as text grows
  2 pregen-monotone   speculation climbs across ticks, never resets while unchanged
  3 flush             send identical text -> Z=0, instant, TEXT CORRECT
  4 discard-cycles    speculate x4 drafts (ghost accumulation), send a 5th -> TEXT CLEAN
  5 divergence        speculate A, send B -> answers B, never leaks A
  6 backspace         grow, shrink, regrow different -> no ghost text in answer
  7 empty-clear       empty ping -> phase ready, x=0, pregen=0
  8 pregen-off        pregen:false ticks never speculate
  9 reminder-modes    compose+flush under start/before/after -> correct + fast
 10 image-compose     (qwen) staged image precomputed; flush answers about the image

Run:  .venv/bin/python test_precompute.py [qwen|gemma|both]
"""
import io, os, sys, json, time, base64, signal, subprocess, urllib.request

ENG = os.path.dirname(os.path.abspath(__file__))
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"
GARBAGE = ("<|channel|>", "channel|>", "think silently", "<start_of_turn>", "<|im_start|>")

def post(p, o, timeout=600):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + p, data=json.dumps(o).encode(), headers={"content-type": "application/json"}),
        timeout=timeout).read() or "{}")

def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=5).read() or "{}")

def chat(o):
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/chat", data=json.dumps(o).encode(), headers={"content-type": "application/json"}), timeout=600)
    first, meta, text = None, {}, []
    for line in r:
        line = line.decode().strip()
        if not line: continue
        j = json.loads(line)
        if j.get("delta"):
            if first is None: first = time.time() - t0
            text.append(j["delta"])
        if j.get("done"): meta = j
    return first, meta, "".join(text)

def wait_ready(max_s=90):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if get("/progress").get("phase") in ("ready", "composing", "pregen"): return
        time.sleep(0.4)

def prefill(hist, partial, rem, mode, images=None, pregen=True, trig=0, targ=0):
    return post("/voice/prefill", {"messages": hist, "partial": partial, "images": images or [],
                                   "reminder": rem, "reminderMode": mode, "pregen": pregen,
                                   "trimTrigger": trig, "trimTarget": targ})

def ticks_until(hist, partial, rem, mode, min_pregen=24, max_ticks=10, **kw):
    r = {}
    for _ in range(max_ticks):
        r = prefill(hist, partial, rem, mode, **kw)
        if (r.get("pregen") or 0) >= min_pregen or r.get("pregenDone"): break
        time.sleep(0.35)
    return r

from PIL import Image
def png_b64(c):
    im = Image.new("RGB", (400, 400), c); b = io.BytesIO(); im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64((30, 30, 200))}}

fails = []
def check(name, ok, detail=""):
    print(f"  {'✓' if ok else '✗ FAIL'} {name} {detail}", flush=True)
    if not ok: fails.append(name)

def clean(body):
    return body.strip() and not any(g in body for g in GARBAGE)

def turn(hist, q, rem, mode, **kw):
    hist.append({"role": "user", "content": [{"type": "text", "text": q}]})
    first, meta, text = chat({"messages": hist, "reminder": rem, "reminderMode": mode,
                              "trimTrigger": kw.get("trig", 0), "trimTarget": kw.get("targ", 0),
                              "recacheMode": "recent"})
    hist.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return first, meta, text.split("@@")[0]

def run(model):
    print(f"===== {model} =====", flush=True)
    log = open(f"/tmp/precompute_{model}.log", "w")
    proc = subprocess.Popen([f"{ENG}/.venv/bin/python", "serve.py", f"models/{model}", str(PORT)],
                            cwd=ENG, stdout=log, stderr=log)
    REM = [{"type": "text", "text": "You are Rally. Be concise and exact."}]
    try:
        for _ in range(180):
            try:
                if get("/health").get("loaded"): break
            except Exception: pass
            time.sleep(2)
        post("/pin", {"system": [{"type": "text", "text": "You are a math tutor. Secret code 5252."}],
                      "context": [], "history": []})
        hist = []
        _, _, t = turn(hist, "hello", REM, "start"); wait_ready()

        # 1 compose-grow: X=Y+Z, Y monotone
        ys, ok = [], True
        for p in ("What", "What is the secret", "What is the secret code?"):
            r = prefill(hist, p, REM, "start")
            ys.append(r.get("precomputed", 0))
            if r.get("turnTokens", 0) != r.get("precomputed", 0) + r.get("anewOnSend", 0): ok = False
        check("1 compose-grow X=Y+Z, Y monotone", ok and ys == sorted(ys), f"Y={ys}")

        # 2 pregen-monotone
        ns, prev = [], -1
        mono = True
        for _ in range(6):
            r = prefill(hist, "What is the secret code?", REM, "start")
            n = r.get("pregen") or 0
            ns.append(n)
            if n < prev: mono = False
            prev = n
            if r.get("pregenDone"): break
            time.sleep(0.35)
        check("2 pregen-monotone", mono and ns[-1] > 0, f"{ns}")

        # 3 flush: send identical -> Z=0, instant, correct
        hist.append({"role": "user", "content": [{"type": "text", "text": "What is the secret code?"}]})
        first, meta, text = chat({"messages": hist, "reminder": REM, "reminderMode": "start"})
        body = text.split("@@")[0]
        check("3 flush", meta.get("new_tokens") == 0 and first < 0.3 and "5252" in body and clean(body),
              f"Z={meta.get('new_tokens')} first={first:.3f}s")
        hist.append({"role": "assistant", "content": [{"type": "text", "text": text}]}); wait_ready()

        # 4 discard-cycles: 4 abandoned drafts then a real send — the gemma ghost case
        for d in ("tell me about euler's number", "actually explain limits",
                  "no, what are derivatives", "hmm, integrals maybe"):
            ticks_until(hist, d, REM, "start", min_pregen=40, max_ticks=6)
        ticks_until(hist, "what is the square root of pi", REM, "start")
        first, meta, body = turn(hist, "what is the square root of pi", REM, "start")
        check("4 discard-cycles clean", clean(body) and ("1.77" in body or "77245" in body),
              f"-> {body.strip()[:60]!r}")
        wait_ready()

        # 5 divergence: speculate GRAPE, send PEACH
        ticks_until(hist, "Say GRAPE and nothing else", REM, "start", min_pregen=3)
        first, meta, body = turn(hist, "Say PEACH and nothing else", REM, "start")
        check("5 divergence", "PEACH" in body.upper() and "GRAPE" not in body.upper(), f"-> {body.strip()[:30]!r}")
        wait_ready()

        # 6 backspace ghost: grow, shrink, regrow different, send
        prefill(hist, "Say the word BANANA and nothing else", REM, "start")
        prefill(hist, "Say the word", REM, "start")
        prefill(hist, "Say the word APPLE and nothing else", REM, "start")
        first, meta, body = turn(hist, "Say the word APPLE and nothing else", REM, "start")
        check("6 backspace ghost", "APPLE" in body.upper() and "BANANA" not in body.upper(),
              f"-> {body.strip()[:30]!r}")
        wait_ready()

        # 7 empty-clear
        prefill(hist, "half typed thou", REM, "start"); time.sleep(0.3)
        prefill(hist, "", REM, "start")
        ph = get("/progress")
        check("7 empty-clear", ph.get("phase") == "ready" and not ph.get("x") and not ph.get("pregen"),
              f"phase={ph.get('phase')} x={ph.get('x')} pregen={ph.get('pregen')}")

        # 8 pregen-off: stable ticks with pregen:false never speculate
        prefill(hist, "what is two plus two", REM, "start", pregen=False); time.sleep(0.3)
        r = prefill(hist, "what is two plus two", REM, "start", pregen=False)
        check("8 pregen-off", (r.get("pregen") or 0) == 0 and get("/progress").get("phase") != "pregen",
              f"pregen={r.get('pregen')}")
        first, meta, body = turn(hist, "what is two plus two", REM, "start")
        check("8b pregen-off send ok", clean(body) and "4" in body, f"Z={meta.get('new_tokens')}")
        wait_ready()

        # 9 reminder modes
        for mode in ("before", "after"):
            post("/precache", {"messages": hist, "reminder": REM, "reminderMode": mode}); wait_ready()
            ticks_until(hist, "Repeat the secret code briefly", REM, mode, min_pregen=5)
            first, meta, body = turn(hist, "Repeat the secret code briefly", REM, mode)
            check(f"9 mode={mode} flush", "5252" in body and clean(body) and first < 1.0,
                  f"Z={meta.get('new_tokens')} first={first:.3f}s")
            wait_ready()

        # 10 image compose (qwen only — gemma per-turn images use the classic path)
        if "qwen" in model:
            ticks_until(hist, "What color is this image? Briefly.", REM, "start",
                        images=[IMG], min_pregen=5)
            hist.append({"role": "user", "content": [IMG, {"type": "text", "text": "What color is this image? Briefly."}]})
            first, meta, text = chat({"messages": hist, "reminder": REM, "reminderMode": "start"})
            body = text.split("@@")[0]
            check("10 image compose+flush", "blue" in body.lower() and clean(body),
                  f"Z={meta.get('new_tokens')} first={first:.3f}s")
            hist.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    finally:
        proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill()

which = sys.argv[1] if len(sys.argv) > 1 else "both"
models = {"qwen": ["qwen3.5-9b-4bit"], "gemma": ["gemma-4-26b-a4b-4bit"],
          "both": ["qwen3.5-9b-4bit", "gemma-4-26b-a4b-4bit"]}[which]
for m in models:
    run(m)
print("\nALL PASS ✓" if not fails else f"\nFAILURES: {fails}")
sys.exit(0 if not fails else 1)
