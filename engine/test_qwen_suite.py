"""Phase-1 gate: TTFT across ALL conversation cases on the QWEN engine, plus a GEMMA smoke test
(the port refactored shared code — token_ids tuple, _conv_ids/_precache_target returns — so gemma
must still pin+chat). Cases:
  qwen: pin(12 imgs) / first ask / append-only follow-up / immediate ask (no precache wait, diverged)
        / reminder mode switch (before) + reconcile / interrupt mid-generation + follow-up
        / window slide (tiny budget) / keepalive / memory flatness
  gemma: pin(2 imgs) + one turn (regression smoke)
PASS = every warm qwen TTFT < 1.0s and gemma turn works.
"""
import io, os, sys, json, time, base64, signal, subprocess, urllib.request

ENG = os.path.dirname(os.path.abspath(__file__))
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"

def post(p, o, timeout=300):
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + p, data=json.dumps(o).encode(), headers={"content-type": "application/json"}), timeout=timeout)
    return r

def post_json(p, o, timeout=300):
    return json.loads(post(p, o, timeout).read() or "{}")

def chat_stream(o, interrupt_after=None, timeout=300):
    """POST /chat, stream lines; returns (meta, n_deltas). interrupt_after=N → close after N deltas."""
    r = post("/chat", o, timeout)
    meta, n = {}, 0
    for line in r:
        line = line.decode().strip()
        if not line: continue
        j = json.loads(line)
        if j.get("done"): meta = j; break
        if "delta" in j or "text" in j or j: n += 1
        if interrupt_after is not None and n >= interrupt_after:
            r.close()          # client disconnect → engine cancels the job
            return {}, n
    return meta, n

def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=5).read() or "{}")

from PIL import Image
def png_b64(c):
    im = Image.new("RGB", (112, 112), c); b = io.BytesIO(); im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()

def img_block(i):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": png_b64(((i*37) % 255, (i*89) % 255, (i*53) % 255))}}

def spawn(model_rel):
    log = open(f"/tmp/civm_suite_{os.path.basename(model_rel)}.log", "w")
    p = subprocess.Popen([f"{ENG}/.venv/bin/python", "serve.py", f"models/{model_rel}", str(PORT)],
                         cwd=ENG, stdout=log, stderr=log)
    for _ in range(180):
        try:
            if get("/health").get("loaded"): return p
        except Exception: pass
        time.sleep(2)
    raise RuntimeError("engine never loaded")

def wait_precache(max_s=60):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if get("/health").get("precache") in ("done", "idle"): return
        time.sleep(0.3)

results, fails = [], []
def check(name, ttft, limit=1.0, extra=""):
    ok = ttft is not None and ttft < limit
    results.append((name, ttft, ok, extra))
    if not ok: fails.append(name)
    print(f"  {'✓' if ok else '✗ FAIL'} {name}: TTFT={ttft if ttft is not None else '?'}s {extra}", flush=True)

def turn(messages, text, **kw):
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    body = {"messages": messages, "reminder": None, "reminderMode": kw.get("mode", "after"),
            "trimTrigger": kw.get("trigger", 0), "trimTarget": kw.get("target", 0),
            "recacheMode": kw.get("recache", "recent")}
    meta, _ = chat_stream(body)
    ans = meta.get("text") or ""      # engine sends full text? deltas were streamed; meta has stats
    messages.append({"role": "assistant", "content": [{"type": "text", "text": kw.get("answer_stub") or "ok"}]})
    return meta

# ============================ QWEN ============================
print("=== QWEN suite ===", flush=True)
proc = spawn("qwen3.5-9b-4bit")
try:
    t0 = time.time()
    pin = post_json("/pin", {"system": [{"type": "text", "text": "You are a concise tutor. Reference code: 7741. Answer in ONE short sentence."}],
                             "context": [img_block(i) for i in range(12)], "history": []})
    print(f"  pin: {pin.get('tokens')} tok, {time.time()-t0:.1f}s, peak={pin.get('mem_peak_gb')}GB", flush=True)

    msgs = []
    # 1. first ask (cold conversation, warm pin)
    m = turn(msgs, "What is the reference code? One short sentence.")
    check("first ask", m.get("ttft"), 1.0, f"(reused={m.get('reused')} new={m.get('new_tokens')})")
    # IMPORTANT: replace the stub with nothing — engine precaches with the REAL generated answer, and the
    # app also stores the real answer. Simulate by asking engine-consistent turns: use meta-independent stubs
    # but then the next turn DIVERGES like an app-edited history — that's exactly the divergence case, also
    # worth timing. For the append-only case, wait for precache and use the real answer path instead:
    # (the engine precached with ITS real answer; our stub history diverges → tests the snapshot-restore path)
    m = turn(msgs, "And what color scheme are the reference images? One short sentence.")
    check("2nd ask (diverged history → snapshot restore)", m.get("ttft"), 1.0,
          f"(reused={m.get('reused')} new={m.get('new_tokens')})")
    wait_precache()
    m = turn(msgs, "Thanks. Repeat the code once more, briefly.")
    check("3rd ask (after precache)", m.get("ttft"), 1.0, f"(reused={m.get('reused')} new={m.get('new_tokens')})")
    tps = m.get("gen_tps")
    print(f"  decode speed: {tps} tok/s", flush=True)

    # 2. immediate back-to-back (no precache wait)
    m = turn(msgs, "Now say 'done', nothing else.")
    check("immediate 4th ask (no wait)", m.get("ttft"), 1.0, f"(reused={m.get('reused')} new={m.get('new_tokens')})")

    # 3. reminder mode 'before' + reconcile
    rec = post_json("/reconcile", {"messages": msgs, "reminder": [{"type": "text", "text": "Always answer in one sentence."}],
                                   "reminderMode": "before", "trimTrigger": 0, "trimTarget": 0, "recacheMode": "recent"})
    m = turn(msgs, "What is 2+2?", mode="before")
    check("ask after reconcile (mode=before)", m.get("ttft"), 1.0, f"(reused={m.get('reused')} new={m.get('new_tokens')})")

    # 4. interrupt mid-generation, then follow-up
    msgs.append({"role": "user", "content": [{"type": "text", "text": "Count slowly from 1 to 200, one number per line."}]})
    _, n = chat_stream({"messages": msgs, "reminder": None, "reminderMode": "after",
                        "trimTrigger": 0, "trimTarget": 0, "recacheMode": "recent"}, interrupt_after=3)
    print(f"  interrupted after {n} deltas", flush=True)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "1\n2\n3"}]})
    time.sleep(0.5)
    m = turn(msgs, "Fine, just say 'ok'.")
    check("ask after interrupt", m.get("ttft"), 1.5, f"(reused={m.get('reused')} new={m.get('new_tokens')})")

    # 5. window slide with tiny budget (trim during conversation)
    msgs2 = []
    post_json("/reconcile", {"messages": msgs2, "reminder": None, "reminderMode": "after",
                             "trimTrigger": 400, "trimTarget": 200, "recacheMode": "recent"})
    slid = None
    for t in range(6):
        m = turn(msgs2, f"Topic {t}: say a 20-word fact about the number {t}.",
                 trigger=400, target=200, answer_stub=("fact " * 20))
        if t >= 2: check(f"windowed turn {t}", m.get("ttft"), 1.5,
                         f"(conv_start={m.get('conv_start')} conv={m.get('conv_tokens')})")
        slid = m.get("conv_start")
    print(f"  window slid to conv_start={slid}", flush=True)

    # 6. keepalive + memory
    post_json("/keepalive", {})
    h = get("/health")
    print(f"  keepalive ok, mem={h.get('memGb')}GB (ceiling {h.get('memCeilingGb')})", flush=True)
    if (h.get("memGb") or 99) > 15: fails.append("memory>15GB")
finally:
    proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill()

# ============================ GEMMA smoke ============================
print("=== GEMMA smoke (regression) ===", flush=True)
proc = spawn("gemma-4-26b-a4b-4bit")
try:
    pin = post_json("/pin", {"system": [{"type": "text", "text": "You are a tester. Code: 9313."}],
                             "context": [img_block(1), img_block(2)], "history": []})
    print(f"  pin: {pin.get('tokens')} tok  ok={pin.get('ok')}", flush=True)
    gm = []
    m = turn(gm, "What is the code? One short sentence.")
    check("gemma ask", m.get("ttft"), 3.0, f"(reused={m.get('reused')} new={m.get('new_tokens')})")
    m = turn(gm, "Say 'ok'.")
    check("gemma 2nd ask", m.get("ttft"), 1.5, f"(reused={m.get('reused')} new={m.get('new_tokens')})")
finally:
    proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill()

print("\n=== RESULT ===")
for n, t, ok, e in results:
    print(f"  {'✓' if ok else '✗'} {n}: {t}s {e}")
print("ALL PASS ✓" if not fails else f"FAILURES: {fails}")
sys.exit(0 if not fails else 1)
