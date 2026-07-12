"""Replay a REAL stored Rally chat against an engine, measuring pin / reconcile / per-turn TTFT + tps.
Usage: .venv/bin/python replay_chat.py <model-dir-under-models/> [chat-name-substring]
Flow mirrors the app exactly: /pin {system,context} -> /reconcile {stored history} (what opening the
chat does) -> then replay each stored USER turn in order, accumulating the ENGINE'S OWN full answers
(incl. @@APPENDIX@@) as history like the app does, waiting for precache between turns (user reading
time). Prints per-turn ttft/reused/new/tps and a summary."""
import glob, json, os, signal, subprocess, sys, time, urllib.request

ENG = os.path.dirname(os.path.abspath(__file__))
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-9b-4bit"
NAME = (sys.argv[2] if len(sys.argv) > 2 else "sipser").lower()

def post(p, o, timeout=1200, stream=False):
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + p, data=json.dumps(o).encode(), headers={"content-type": "application/json"}), timeout=timeout)
    if not stream:
        return json.loads(r.read() or "{}")
    meta, text = {}, []
    for line in r:
        line = line.decode().strip()
        if line:
            j = json.loads(line)
            if j.get("done"): meta = j
            elif j.get("delta"): text.append(j["delta"])
    return meta, "".join(text)

def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=5).read() or "{}")

# ---- load the stored chat ----
chats = os.path.expanduser("~/Library/Application Support/contextualized_instant_voice_models/chats")
chat = None
for f in glob.glob(os.path.join(chats, "*.json")):
    j = json.load(open(f))
    if NAME in (j.get("name") or "").lower():
        chat = j; break
assert chat, f"no chat matching {NAME!r}"
msgs_stored = chat.get("messages") or []
user_turns = [m for m in msgs_stored if m.get("role") == "user"]
print(f"chat {chat.get('name')!r}: {len(chat.get('context') or [])} ctx blocks, "
      f"{len(msgs_stored)} msgs ({len(user_turns)} user turns)", flush=True)

def blocks(ms):  # stored message content -> engine blocks (pass-through)
    return ms.get("content") or ([{"type": "text", "text": ms.get("text", "")}] if ms.get("text") else [])

def wire(history):
    out = []
    for m in history:
        t = m.get("text", "")
        c = [{"type": "text", "text": t}] if t else []
        c += [b for b in (m.get("content") or []) if b.get("type") == "image"]
        out.append({"role": m["role"], "content": c})
    return out

# ---- spawn engine ----
log = open(f"/tmp/replay_{MODEL.replace('/', '_')}.log", "w")
proc = subprocess.Popen([f"{ENG}/.venv/bin/python", "serve.py", f"models/{MODEL}", str(PORT)],
                        cwd=ENG, stdout=log, stderr=log)
try:
    t0 = time.time()
    for _ in range(240):
        try:
            if get("/health").get("loaded"): break
        except Exception: pass
        time.sleep(2)
    print(f"engine loaded in {time.time()-t0:.0f}s  ({MODEL})", flush=True)

    # 1. PIN the real system+context (46 real images)
    t0 = time.time()
    pin = post("/pin", {"system": chat.get("system") or [], "context": chat.get("context") or [], "history": []})
    print(f"PIN: ok={pin.get('ok')} {pin.get('tokens')} tok in {time.time()-t0:.1f}s  peak={pin.get('mem_peak_gb')}GB", flush=True)

    # 2. RECONCILE with the full stored history (what opening the chat does)
    settings = chat.get("settings") or {}
    mode = settings.get("reminderMode") or "after"
    reminder = chat.get("reminder") or None
    t0 = time.time()
    rec = post("/reconcile", {"messages": wire(msgs_stored), "reminder": reminder, "reminderMode": mode,
                              "trimTrigger": 5000, "trimTarget": 3000, "recacheMode": "recent"})
    print(f"RECONCILE (chat open): {time.time()-t0:.1f}s  conv_start={rec.get('conv_start')} conv_tokens={rec.get('conv_tokens')}", flush=True)

    # 3. REPLAY every stored user turn with accumulated engine answers
    history, rows = [], []
    for i, ut in enumerate(user_turns):
        history.append({"role": "user", "text": "", "content": blocks(ut)})
        meta, text = post("/chat", {"messages": wire(history), "reminder": reminder, "reminderMode": mode,
                                    "trimTrigger": 5000, "trimTarget": 3000, "recacheMode": "recent"}, stream=True)
        ttft, tps = meta.get("ttft"), meta.get("gen_tps")
        rows.append((ttft, tps, meta.get("reused"), meta.get("new_tokens"), meta.get("mem_peak_gb")))
        q = "".join(b.get("text", "") for b in blocks(ut) if b.get("type") == "text")[:48]
        print(f"  turn {i+1:2d}/{len(user_turns)}: ttft={ttft}s tps={tps} reused={meta.get('reused')} "
              f"new={meta.get('new_tokens')} peak={meta.get('mem_peak_gb')}GB  Q={q!r}", flush=True)
        history.append({"role": "assistant", "text": text, "content": []})
        for _ in range(120):                       # wait for precache (user reading time)
            if get("/health").get("precache") in ("done", "idle"): break
            time.sleep(0.5)

    ttfts = [r[0] for r in rows if r[0] is not None]
    tpss = [r[1] for r in rows if r[1]]
    print(f"\nSUMMARY [{MODEL}] pin={pin.get('tokens')}tok  turns={len(rows)}")
    print(f"  TTFT   avg={sum(ttfts)/len(ttfts):.3f}s  max={max(ttfts):.3f}s  subsec={sum(1 for t in ttfts if t<1.0)}/{len(ttfts)}")
    print(f"  tok/s  avg={sum(tpss)/len(tpss):.1f}  min={min(tpss):.1f}")
    print(f"  mem now={get('/health').get('memGb')}GB", flush=True)
finally:
    proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill()
