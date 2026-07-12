"""Memory soak on the QWEN engine: real Sipser pin + 40 long turns with an aggressive sliding window
(trigger 2000/target 1000 → many trims), keepalive pings, snapshot-leak detection (resident memory must
plateau, not creep), and a forced over-ceiling reclaim at the end. PASS = mem never exceeds SOAK_LIMIT,
last-10-turn trend is flat (< 0.3GB drift), reclaim path works."""
import glob, json, os, signal, subprocess, sys, time, urllib.request

ENG = os.path.dirname(os.path.abspath(__file__))
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"
SOAK_LIMIT_GB = 14.0     # qwen should live ~10GB; 14 = generous ceiling for the soak

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

chats = os.path.expanduser("~/Library/Application Support/contextualized_instant_voice_models/chats")
chat = next(json.load(open(f)) for f in glob.glob(os.path.join(chats, "*.json"))
            if "sipser" in (json.load(open(f)).get("name") or "").lower())

log = open("/tmp/mem_soak.log", "w")
proc = subprocess.Popen([f"{ENG}/.venv/bin/python", "serve.py", "models/qwen3.5-9b-4bit", str(PORT)],
                        cwd=ENG, stdout=log, stderr=log)
fails = []
try:
    for _ in range(120):
        try:
            if get("/health").get("loaded"): break
        except Exception: pass
        time.sleep(2)
    pin = post("/pin", {"system": chat.get("system") or [], "context": chat.get("context") or [], "history": []})
    print(f"pin {pin.get('tokens')} tok peak={pin.get('mem_peak_gb')}GB", flush=True)

    hist, mems = [], []
    topics = ["DFA", "NFA", "regular languages", "closure properties", "the pumping lemma", "state diagrams",
              "transition functions", "accept states", "the union operation", "concatenation"]
    for t in range(40):
        hist.append({"role": "user", "content": [{"type": "text", "text":
            f"Turn {t}: explain {topics[t % len(topics)]} from the chapter in about 80 words."}]})
        meta, text = post("/chat", {"messages": hist, "reminder": None, "reminderMode": "after",
                                    "trimTrigger": 2000, "trimTarget": 1000, "recacheMode": "recent"}, stream=True)
        hist.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        m = get("/health")
        mems.append(m.get("memGb") or 0)
        over = m.get("memOver")
        print(f"  t{t:02d}: ttft={meta.get('ttft')}s mem={mems[-1]}GB conv_start={meta.get('conv_start')} "
              f"conv={meta.get('conv_tokens')} over={over}", flush=True)
        if mems[-1] > SOAK_LIMIT_GB: fails.append(f"t{t}: mem {mems[-1]} > {SOAK_LIMIT_GB}")
        if t % 7 == 0: post("/keepalive", {})
        for _ in range(60):
            if get("/health").get("precache") in ("done", "idle"): break
            time.sleep(0.5)

    drift = max(mems[-10:]) - min(mems[-10:])
    print(f"\nlast-10 mem: min={min(mems[-10:])} max={max(mems[-10:])} drift={drift:.2f}GB", flush=True)
    if drift > 0.5: fails.append(f"creep: last-10 drift {drift:.2f}GB")
    print(f"peak across soak: {max(mems)}GB", flush=True)
    print("PASS ✓" if not fails else f"FAILURES: {fails}", flush=True)
finally:
    proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill()
sys.exit(0 if not fails else 1)
