#!/usr/bin/env bash
# Reproducible setup for Gemma 4 26B-A4B on MLX.
# Working recipe (found empirically): mlx-vlm + transformers 5.x, with a one-line guard on
# mlx_lm's NewlineTokenizer register (breaks against transformers 5.x's stricter register()).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
echo "[1/4] venv";     uv venv --python 3.10 .venv 2>&1 | tail -1
echo "[2/4] deps";     uv pip install -p .venv mlx-vlm 'transformers>=5' huggingface_hub \
                          'parakeet-mlx==0.5.1' sounddevice scipy \
                          pyobjc-framework-Cocoa pyobjc-framework-Quartz 2>&1 | tail -2
# (voice deps: parakeet ASR, sounddevice/scipy for mic+resample, pyobjc for the overlay/CGEventTap;
#  ffmpeg must be present — parakeet's transcribe(path) shells out to it.)
echo "[3/4] patch mlx_lm register (transformers 5.x compat)"
F=.venv/lib/python3.10/site-packages/mlx_lm/tokenizer_utils.py
python3 - "$F" <<'PY'
import sys; f=sys.argv[1]; s=open(f).read()
bad='AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)'
if bad in s and "civm-patch" not in s:
    open(f,"w").write(s.replace(bad,'try:  # civm-patch\n    '+bad+'\nexcept Exception:\n    pass')); print("  patched")
else: print("  already patched")
PY
echo "[3b] patch mlx_vlm gemma4 cached-continuation mask (civm-fix — REQUIRED, crashes every cached turn without it)"
G=.venv/lib/python3.10/site-packages/mlx_vlm/models/gemma4/language.py
python3 - "$G" <<'PY'
import sys; f=sys.argv[1]; s=open(f).read()
old = "        q_blocks = mx.expand_dims(block_sequence_ids, -1)"
new = ("        # civm-fix: on a cached continuation, queries are only the NEW tokens (base_mask's\n"
       "        # query dim), while keys span the full cached+new sequence. Slice q to the last n_q\n"
       "        # so shapes match; for a text-only suffix these q ids are -1 -> overlay is a no-op.\n"
       "        n_q = base_mask.shape[-2]\n"
       "        q_ids = block_sequence_ids[:, -n_q:]\n"
       "        q_blocks = mx.expand_dims(q_ids, -1)")
if "civm-fix" in s:
    print("  already patched")
elif old in s:
    open(f, "w").write(s.replace(old, new, 1)); print("  patched")
else:
    print("  !! pattern not found — mlx_vlm changed; apply the civm-fix by hand (see README)"); sys.exit(1)
PY

echo "[4/4] models — run in FOREGROUND (sandbox blocks backgrounded downloads)"
.venv/bin/hf download unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit --local-dir ./models/gemma-4-26b-a4b-4bit 2>&1 | tail -2   # ~16GB
echo "SETUP DONE"
