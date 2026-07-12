"""Image-token-cap quality A/B on the REAL Sipser pages: pin the actual chat context at
max_pixels = 256/512/1024 tokens-per-image, ask OCR-heavy questions whose answers require READING
the pages, print answers side by side + pin cost, so the cap default is evidence-based.
(Gemma's fixed budget is ~256 tok/image — the current quality baseline.)"""
import glob, json, os, signal, subprocess, sys, time, urllib.request

ENG = os.path.dirname(os.path.abspath(__file__))
PORT = 5199
BASE = f"http://127.0.0.1:{PORT}"

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
chat = None
for f in glob.glob(os.path.join(chats, "*.json")):
    j = json.load(open(f))
    if "sipser" in (j.get("name") or "").lower():
        chat = j; break
assert chat

# OCR-heavy questions — answers must come from READING the pinned pages (not general knowledge).
QUESTIONS = [
    "Quote the book's formal definition of a finite automaton (Definition 1.5) as a 5-tuple, exactly as the page states it.",
    "What specific language does the book say machine M1 recognizes in Example 1.9? Use the book's own notation.",
    "In the chapter's figures, what are the state names used in the very first state diagram shown, and which one is the accept state?",
    "What does the book say about the empty string epsilon and the empty language in this chapter? Quote or closely paraphrase the page.",
]

for tok_per_img in (256, 512, 1024):
    mp = tok_per_img * 28 * 28
    log = open(f"/tmp/viscap_{tok_per_img}.log", "w")
    env = dict(os.environ); env["CIVM_QWEN_MAX_PIXELS"] = str(mp)
    proc = subprocess.Popen([f"{ENG}/.venv/bin/python", "serve.py", "models/qwen3.5-9b-4bit", str(PORT)],
                            cwd=ENG, stdout=log, stderr=log, env=env)
    try:
        for _ in range(120):
            try:
                if get("/health").get("loaded"): break
            except Exception: pass
            time.sleep(2)
        t0 = time.time()
        pin = post("/pin", {"system": chat.get("system") or [], "context": chat.get("context") or [], "history": []})
        print(f"\n===== CAP {tok_per_img} tok/img: pin {pin.get('tokens')} tok in {time.time()-t0:.0f}s "
              f"peak={pin.get('mem_peak_gb')}GB resident={get('/health').get('memGb')}GB =====", flush=True)
        for q in QUESTIONS:
            meta, text = post("/chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": q}]}],
                                        "reminder": [{"type": "text", "text": "Answer from the pinned pages only, concisely."}],
                                        "reminderMode": "after"}, stream=True)
            ans = text.split("@@APPENDIX@@")[0].strip().replace("\n", " ")
            print(f"  Q: {q[:70]}", flush=True)
            print(f"  A: {ans[:300]}", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM); time.sleep(1); proc.kill(); time.sleep(1)
print("\nDONE — compare answers across caps for OCR fidelity.", flush=True)
