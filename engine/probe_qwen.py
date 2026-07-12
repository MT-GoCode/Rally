"""Probe Qwen3.5-9B-MLX-4bit under mlx-vlm 0.6.3 — verifies the design assumptions for the qwen
engine port BEFORE touching serve.py:
  1. load + cache structure (which layers are KVCache vs SSM/ArraysCache; what attrs they carry)
  2. what prepare_inputs returns for an image prompt (qwen needs image_grid_thw etc., gemma didn't)
  3. SNAPSHOT/RESTORE exactness: pin prefix -> snap -> feed turn A + decode -> restore -> feed turn B
     + decode, must equal a fresh-cache run of prefix+B  (this replaces trim_kv for qwen)
  4. multi-image (12 images) prompt works
  5. TTFT: warm feed of a small tail over the pinned cache
Run: .venv/bin/python probe_qwen.py 2>&1 | tail -60
"""
import io, os, sys, time, base64, tempfile

import mlx.core as mx
from mlx_vlm import load, stream_generate, apply_chat_template, prepare_inputs
from mlx_lm.models.cache import make_prompt_cache
from PIL import Image

MODEL = os.path.join(os.path.dirname(__file__), "models/qwen3.5-9b-4bit")

def mem(): return round(mx.get_active_memory() / 1e9, 2)

print(f"[probe] loading {MODEL} …", flush=True)
t0 = time.time()
model, processor = load(MODEL)
config = model.config
print(f"[probe] loaded in {time.time()-t0:.1f}s  mem={mem()}GB  model_type={getattr(config,'model_type',None)}", flush=True)

# ---- 1. cache structure ----
def make_kv():
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "make_cache"):
        return lm.make_cache()          # hybrid: ArraysCache (SSM) + KVCache (attention)
    return make_prompt_cache(model)

kv = make_kv()
print(f"[probe] make_cache OK: {len(kv)} layers")
types = {}
for c in kv:
    types[type(c).__name__] = types.get(type(c).__name__, 0) + 1
print(f"[probe] cache layer types: {types}")
c0 = kv[0]
print(f"[probe] first cache attrs: {[a for a in ('keys','values','offset','cache','state') if hasattr(c0, a)]}")

# ---- 2. prepare_inputs shape for an image prompt ----
def png(color):
    im = Image.new("RGB", (112, 112), color)
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()

tmp = tempfile.mkdtemp(prefix="probe-")
paths = []
for i in range(12):
    p = os.path.join(tmp, f"i{i}.png")
    open(p, "wb").write(png(((i*37) % 255, (i*89) % 255, (i*53) % 255)))
    paths.append(p)

msgs2 = [{"role": "system", "content": "You are a tester."},
         {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                      {"type": "text", "text": "Reference images above."}]},
         {"role": "assistant", "content": "Understood."}]
prompt2 = apply_chat_template(processor, config, msgs2, num_images=2, add_generation_prompt=False)
inp = prepare_inputs(processor, images=paths[:2], prompts=prompt2)
print(f"[probe] prepare_inputs type={type(inp).__name__}")
d = inp if isinstance(inp, dict) else vars(inp)
for k, v in d.items():
    if v is None: continue
    shape = getattr(v, "shape", None)
    print(f"        {k}: {type(v).__name__} shape={shape}")

# ---- helpers: feed / decode / snapshot ----
def feed(kv, ids, extra=None):
    kwargs = dict(extra or {})
    for _ in stream_generate(model, processor, "", input_ids=mx.array([ids]), prompt_cache=kv,
                             max_tokens=0, temperature=0.0, prefill_step_size=512, **kwargs):
        pass

def decode(kv, ids, n=12, extra=None):
    out = []
    kwargs = dict(extra or {})
    for r in stream_generate(model, processor, "", input_ids=mx.array([ids]), prompt_cache=kv,
                             max_tokens=n, temperature=0.0, prefill_step_size=512, **kwargs):
        out.append(r.text)
    return "".join(out)

def ids_of(prompt, imgs=None):
    inp = prepare_inputs(processor, images=imgs, prompts=prompt)
    d = inp if isinstance(inp, dict) else vars(inp)
    ids = d["input_ids"]
    extra = {k: v for k, v in d.items() if k not in ("input_ids", "attention_mask") and v is not None}
    return ids.flatten().tolist(), extra

def snap_cache(kv):
    out = []
    for c in kv:
        if getattr(c, "keys", None) is not None:
            out.append(("kv", c.keys, c.values, c.offset))
        elif hasattr(c, "cache"):
            out.append(("arr", [mx.array(x) if x is not None else None for x in c.cache]))
        else:
            out.append(("other", {a: getattr(c, a) for a in ("offset", "state") if hasattr(c, a)}))
    return out

def restore_cache(kv, snap):
    for c, s in zip(kv, snap):
        if s[0] == "kv":
            # slice back to the snapshot offset — later in-place writes may have altered rows past it,
            # but [0:offset] is immutable (writes land at >= offset). Slicing forces a clean view.
            c.keys = s[1][:, :, :s[3], :] if s[1] is not None else None
            c.values = s[2][:, :, :s[3], :] if s[2] is not None else None
            c.offset = s[3]
        elif s[0] == "arr":
            c.cache[:] = [mx.array(x) if x is not None else None for x in s[1]]

# ---- 3. snapshot/restore exactness ----
sys_pin = [{"role": "system", "content": "You are a precise assistant. Reference: the sky is green today, code 4471."},
           {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Context images."}]},
           {"role": "assistant", "content": "Understood — I have the reference."}]
pin_prompt = apply_chat_template(processor, config, sys_pin, num_images=2, add_generation_prompt=False)
pin_ids, pin_extra = ids_of(pin_prompt, paths[:2])
print(f"[probe] pin: {len(pin_ids)} tokens (2 images)  extra_keys={list(pin_extra.keys())}")

t0 = time.time()
kv = make_kv()
feed(kv, pin_ids, pin_extra)
mx.clear_cache()
print(f"[probe] pin fed in {time.time()-t0:.1f}s  mem={mem()}GB")

SNAP = snap_cache(kv)
snap_types = [s[0] for s in SNAP]
print(f"[probe] snapshot kinds: kv={snap_types.count('kv')} arr={snap_types.count('arr')} other={snap_types.count('other')}")

def turn_ids(text):
    m = [{"role": "user", "content": text}]
    p = apply_chat_template(processor, config, m, num_images=0, add_generation_prompt=True, enable_thinking=False)
    ids, _ = ids_of(p, None)
    return ids

tA = turn_ids("What color is the sky per the reference? Answer in 5 words.")
tB = turn_ids("What is the code number in the reference? Answer briefly.")

outA = decode(kv, tA, 12)
print(f"[probe] turn A on pinned cache: {outA!r}")
restore_cache(kv, SNAP)
t0 = time.time()
outB_restored = decode(kv, tB, 12)
ttft_restored = time.time() - t0
print(f"[probe] turn B after RESTORE: {outB_restored!r}")

kv2 = make_kv()
feed(kv2, pin_ids, pin_extra)
outB_fresh = decode(kv2, tB, 12)
print(f"[probe] turn B on FRESH cache: {outB_fresh!r}")
match = outB_restored == outB_fresh
print(f"[probe] SNAPSHOT/RESTORE EXACT: {'PASS ✓' if match else 'FAIL ✗'}")

# ---- 4. multi-image (12) ----
msgs12 = [{"role": "user", "content": [{"type": "image"} for _ in range(12)] +
          [{"type": "text", "text": "How many images do you see? Answer with a number."}]}]
p12 = apply_chat_template(processor, config, msgs12, num_images=12, add_generation_prompt=True, enable_thinking=False)
ids12, extra12 = ids_of(p12, paths)
kv3 = make_kv()
t0 = time.time()
out12 = decode(kv3, ids12, 8, extra12)
print(f"[probe] 12-image prompt ({len(ids12)} tok): {out12!r}  ({time.time()-t0:.1f}s)  mem={mem()}GB")

# ---- 5. TTFT on warm cache ----
restore_cache(kv, SNAP)
tC = turn_ids("Say the single word: hello")
t0 = time.time()
first = None
for r in stream_generate(model, processor, "", input_ids=mx.array([tC]), prompt_cache=kv,
                         max_tokens=3, temperature=0.0, prefill_step_size=512):
    first = time.time() - t0
    break
print(f"[probe] warm TTFT (restore + {len(tC)}-tok tail): {first:.3f}s")
print(f"[probe] final mem={mem()}GB  ALL DONE")
