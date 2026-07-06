# Research brief: wtalk + voiced.py voice machinery (for engine implementation)

Key facts extracted from ~/code/minh-mac-utils/wtalk and ~/code/agentic-teacher-v9/harness/utils/voiced.py.
Read voiced.py directly too — it is ~320 lines and largely liftable.

## Parakeet
- Model: mlx-community/parakeet-tdt-0.6b-v2 via `from parakeet_mlx import from_pretrained`; bf16 ≈1.2GB resident; load+warmup 3–8s.
- Warmup: transcribe 0.5s of zeros after load.
- Batch: resample to 16k (scipy.signal.resample_poly(audio, 16000, src_sr)), write 16-bit mono WAV tmpfile, `model.transcribe(path).text.strip()`; skip audio < 0.25s. NOTE: transcribe(path) shells out to ffmpeg (present on this machine — wtalk uses it daily).
- STREAMING (parakeet-mlx 0.5.1, unused by prior tools but ready):
  `with model.transcribe_stream(context_size=(256,256), depth=1) as s: s.add_audio(mx16k_float_chunk); s.result.text`
  - add_audio takes 1-D mx.array of 16kHz float samples, any chunk size; `s.finalized_tokens` = stable, `s.draft_tokens` = provisional (re-decoded next call). Use .result.text for partials.

## Audio capture (sounddevice 0.5.5)
- Pick BUILT-IN mic by name substring ("macbook","built-in","imac",…) — never Bluetooth (HFP degradation).
- `sd.InputStream(samplerate=device_native, channels=1, dtype='float32', device=dev, callback=cb)`; cb appends `indata[:,0].copy()` to a list.
- STOP with detached close (slow CoreAudio close must not wedge the next keypress):
  `st, self.stream = self.stream, None; threading.Thread(target=lambda:(st.stop(),st.close()),daemon=True).start(); return np.concatenate(frames)`
- CoreAudio recovery after login: retry open with backoff, `sd._terminate(); sd._initialize()` on host-API errors (-9999/-9986).

## CGEventTap (from voiced.py:182–257 — lift this pattern)
- `CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault, mask, cb, None)`; add to runloop; enable.
- Mask: kCGEventKeyDown | kCGEventKeyUp — PLUS kCGEventFlagsChanged for our modifier-only chord (ctrl+opt with no letter — detect via flagsChanged: both Control+Alternate set = chord down; either cleared = chord up).
- Callback does NO work: push to queue.Queue; drain on main thread via 0.03s NSTimer.
- MUST handle kCGEventTapDisabledByTimeout/ByUserInput → CGEventTapEnable(tap, True).
- Consume matched events (return None); pass everything else. ESC (keycode 53) consumed+cancels ONLY while listening/processing, else passes through.
- Orphan watchdog: periodic `os.getppid()==1` → exit (headless tap kept eating chords — bug seen 2026-07-04).
- Permissions: AXIsProcessTrustedWithOptions prompt first; CGEventTapCreate returning None ⇒ missing perms → report state 'perm'.

## Overlay dot (voiced.py Overlay class, 73–94 — lift)
- Borderless non-activating NSPanel, NSFloatingWindowLevel, clear bg, ignoresMouseEvents, layer cornerRadius; bottom-center (x=screenW/2−D/2, y≈84), DIAM 18–30.
- show(r,g,b) sets layer bg + orderFrontRegardless(); hide() = orderOut_.
- Python process must run AppKit: `app.setActivationPolicy_(2)` (accessory) + AppHelper.runEventLoop on MAIN thread.
- Colors for CIVM (user mandate — NEVER grey; voiced's grey transcribing state at line 292 is the thing to NOT copy):
  red (0.92,0.16,0.22)=listening · yellow (0.95,0.75,0.10)=processing · green (0.18,0.80,0.45) flash ~1s=ready → hide. Hidden when idle.

## Threading/process model
- Single python process; MAIN thread = AppKit event loop + NSTimer draining tap queue; HTTP server on background thread; model load on daemon thread; transcription on ad-hoc daemon threads. voiced/wtalk prove MLX calls work from non-main threads (each thread gets its own default MLX stream).
- For CIVM: confine ALL Gemma MLX work (pin, chat generation, streaming prefill) to ONE dedicated worker thread fed by a job queue (the earlier ThreadingHTTPServer crash "no Stream(gpu,1) in current thread" came from running mlx_vlm generation on ad-hoc handler threads). Parakeet runs on its own voice thread. HTTP handlers do NO MLX — they enqueue jobs and pump token queues.

## Deps to add to engine/.venv (uv pip install)
parakeet-mlx==0.5.1 sounddevice scipy pyobjc-framework-Cocoa pyobjc-framework-Quartz
