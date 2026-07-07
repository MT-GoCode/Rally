#!/usr/bin/env python3
"""Voice primitives for the CIVM engine — lifted from agentic-teacher-v9/harness/utils/voiced.py.

Self-contained, reusable pieces the engine wires together (serve.py owns the state machine):
  Overlay    — borderless non-activating NSPanel dot, screen bottom-center, CIVM colors (NEVER grey).
  SelOverlay — borderless non-activating NSPanel selection RECTANGLE (semi-transparent fill + border),
               driven from the main thread while a press&hold screenshot drag is armed.
  Rec        — built-in-mic capture (avoids Bluetooth HFP), detached CoreAudio close.
  Tap        — Quartz CGEventTap. Voice: MODIFIER-ONLY ctrl+opt chord (kCGEventFlagsChanged) + ESC.
               Capture (when enabled): a screenshot binding (keyboard chord OR mouse button) and a
               copy-to-chat keyboard chord. Emits raw events onto a queue.Queue —
               "chord_down"/"chord_up"/"esc" (voice) and ("shot_down",x,y)/("shot_up",x,y)/("copy",)
               (capture) — serve.py applies the toggle/hold + style + state semantics. Must run on
               the thread owning the runloop (main).
  parakeet   — from_pretrained + zeros warmup; batch transcribe(path); streaming transcribe_stream.
  capture    — screencapture -R / -i to a base64 PNG; balanced ⌘C synthesis + NSPasteboard read.

The overlays/tap must be created on the AppKit main thread; parakeet loads on its own voice thread
AFTER Gemma. All colors per SPEC-voice.md ("Overlay colors").
"""
import base64, os, subprocess, sys, tempfile, threading, time, wave

import numpy as np
from scipy.signal import resample_poly

from AppKit import (NSPanel, NSView, NSColor, NSScreen, NSMakeRect,
                    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
                    NSFloatingWindowLevel, NSBackingStoreBuffered)
import Quartz as Q

# AXIsProcessTrustedWithOptions lives in ApplicationServices/HIServices, which aren't in the
# minimal dep set (Cocoa + Quartz only). It's advisory — CGEventTapCreate returning None already
# signals missing perms — so soft-import and no-op when absent.
try:
    from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
except Exception:  # pragma: no cover
    try:
        from HIServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    except Exception:
        AXIsProcessTrustedWithOptions = None
        kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"

TARGET_SR = 16000
MODEL_ID = 'mlx-community/parakeet-tdt-0.6b-v2'

# ---- Overlay colors: ONLY red or yellow; every other state = hidden (user mandate) ----
RED = (0.92, 0.16, 0.22)     # listening
YELLOW = (0.95, 0.75, 0.10)  # processing (brief flash doubles as the mic-error signal)

# selection-rectangle overlay (press&hold screenshot drag): faint fill + a clear accent border
SEL_FILL = (0.30, 0.55, 0.95, 0.18)
SEL_BORDER = (0.30, 0.55, 0.95, 0.95)

# ANSI keycodes
KC_ESC = 53
KC_C = 8                     # 'c' — used for the synthesized ⌘C copy

# ANSI virtual-keycode table (physical key position; layout-stable). The Swift Settings recorder
# uses the INVERSE of this same table to serialize a chord's base key, so a binding string like
# "cmd+shift+2" round-trips to the right keycode here regardless of the active input source.
NAME_TO_KEYCODE = {
    'a': 0, 's': 1, 'd': 2, 'f': 3, 'h': 4, 'g': 5, 'z': 6, 'x': 7, 'c': 8, 'v': 9, 'b': 11,
    'q': 12, 'w': 13, 'e': 14, 'r': 15, 'y': 16, 't': 17, '1': 18, '2': 19, '3': 20, '4': 21,
    '6': 22, '5': 23, '=': 24, '9': 25, '7': 26, '-': 27, '8': 28, '0': 29, ']': 30, 'o': 31,
    'u': 32, '[': 33, 'i': 34, 'p': 35, 'l': 37, 'j': 38, "'": 39, 'k': 40, ';': 41, '\\': 42,
    ',': 43, '/': 44, 'n': 45, 'm': 46, '.': 47, '`': 50,
    'space': 49, 'return': 36, 'tab': 48, 'delete': 51, 'esc': 53,
}

DIAM = 22.0


def parse_binding(s):
    """Serialize-string -> matchable binding.
      'cmd+shift+2' -> ('key', keycode, frozenset({'cmd','shift'}))
      'ctrl+alt+c'  -> ('key', 8, frozenset({'ctrl','alt'}))
      'mouse3'      -> ('mouse', 2)   # 1-indexed name (mouse3 = middle) -> CGEvent button number 2
      '' / bad      -> None
    Modifier names: cmd, ctrl, alt, shift (order-independent — matched as a set)."""
    s = (s or '').strip().lower()
    if not s:
        return None
    if s.startswith('mouse'):
        try:
            n = int(s[5:])
        except ValueError:
            return None
        return ('mouse', max(0, n - 1))
    parts = s.split('+')
    base = parts[-1]
    kc = NAME_TO_KEYCODE.get(base)
    if kc is None:
        return None
    return ('key', kc, frozenset(parts[:-1]))


# modifier-name -> CGEvent flag mask, for the voice modifier-only chord (push-to-talk).
MOD_MASKS = {
    'ctrl':  Q.kCGEventFlagMaskControl,
    'alt':   Q.kCGEventFlagMaskAlternate,
    'shift': Q.kCGEventFlagMaskShift,
    'cmd':   Q.kCGEventFlagMaskCommand,
}


def parse_chord_mods(s):
    """Voice chord string -> the exact set of modifiers to match. 'shift+alt' -> {'shift','alt'};
    empty/unknown -> the ctrl+alt default. The tap fires the chord only when EXACTLY these modifiers
    are held (no extras), so e.g. a shift+alt chord won't fire while you Shift-type a capital."""
    mods = frozenset(p for p in (s or '').strip().lower().split('+') if p in MOD_MASKS)
    return mods or frozenset({'ctrl', 'alt'})


# ---------- overlay: dot bottom-center (wtalk NSPanel pattern) ----------
class Overlay:
    """Borderless, non-activating, click-through dot at the bottom-center of the main screen.
    Must be constructed on the AppKit main thread."""
    def __init__(self):
        scr = NSScreen.mainScreen().frame()
        # AppKit y=0 is the bottom edge; sit ~84pt up, horizontally centered.
        rect = NSMakeRect(scr.size.width / 2 - DIAM / 2, 84, DIAM, DIAM)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        # show on every Space / full-screen app
        try:
            self.panel.setCollectionBehavior_((1 << 0) | (1 << 8))  # CanJoinAllSpaces | FullScreenAuxiliary
        except Exception:
            pass
        v = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, DIAM, DIAM))
        v.setWantsLayer_(True)
        v.layer().setCornerRadius_(DIAM / 2.0)
        self.panel.setContentView_(v)
        self.view = v

    def show(self, rgb):
        r, g, b = rgb
        self.view.layer().setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 0.95).CGColor())
        self.panel.orderFrontRegardless()

    def hide(self):
        self.panel.orderOut_(None)


# ---------- selection rectangle overlay (press&hold screenshot drag) ----------
class SelOverlay:
    """Borderless, non-activating, click-through rectangle that follows the drag while a press&hold
    screenshot is armed. Constructed on the AppKit main thread; show_topleft/hide are main-thread only.
    show_topleft takes TOP-LEFT global coords (CGEvent / screencapture space) and flips to AppKit."""
    def __init__(self):
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 10, 10),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        try:
            self.panel.setCollectionBehavior_((1 << 0) | (1 << 8))  # CanJoinAllSpaces | FullScreenAuxiliary
        except Exception:
            pass
        v = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        v.setWantsLayer_(True)
        lyr = v.layer()
        lyr.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(*SEL_FILL).CGColor())
        lyr.setBorderColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(*SEL_BORDER).CGColor())
        lyr.setBorderWidth_(1.5)
        self.panel.setContentView_(v)
        self.view = v

    def show_topleft(self, x, y_top, w, h):
        w = max(1.0, float(w)); h = max(1.0, float(h))
        scr_h = NSScreen.mainScreen().frame().size.height
        y = scr_h - float(y_top) - h                 # top-left origin -> AppKit bottom-left origin
        self.panel.setFrame_display_(NSMakeRect(float(x), y, w, h), True)
        self.view.setFrame_(NSMakeRect(0, 0, w, h))
        self.panel.orderFrontRegardless()

    def hide(self):
        self.panel.orderOut_(None)


# ---------- mic (built-in only; detached close) ----------
class Rec:
    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        self.frames, self.stream, self.on, self.sr = [], None, False, 48000
        self.device = self._builtin()

    def _builtin(self):
        # Pin the built-in mic so opening the input never flips AirPods A2DP→HFP (muffled call codec).
        # 1) a device that names itself built-in; 2) otherwise the first NON-Bluetooth input — never
        # fall through to a Bluetooth/AirPods default. Logs the pick to serve.log for diagnosis.
        BUILTIN = ('macbook', 'built-in', 'imac', 'mac mini', 'mac studio')
        BT = ('airpod', 'bluetooth', 'headset', 'headphone', 'wireless', 'beats', 'buds')
        try:
            inputs = [(i, d) for i, d in enumerate(self.sd.query_devices())
                      if d.get('max_input_channels', 0) > 0]
        except Exception:
            return None
        for i, d in inputs:
            if any(k in d['name'].lower() for k in BUILTIN):
                sys.stderr.write(f"[voice] built-in mic: {d['name']} (#{i})\n"); sys.stderr.flush()
                return i
        for i, d in inputs:                                   # avoid AirPods/BT so playback stays A2DP
            if not any(k in d['name'].lower() for k in BT):
                sys.stderr.write(f"[voice] mic (non-BT fallback): {d['name']} (#{i})\n"); sys.stderr.flush()
                return i
        sys.stderr.write("[voice] WARNING: no non-Bluetooth input found; using system default\n"); sys.stderr.flush()
        return None

    def start(self):
        if self.on:
            return
        # CoreAudio can refuse right after login / device churn (PortAudio -9986/-9999).
        # wtalk's proven recovery: reset PortAudio's host API and retry with backoff.
        last = None
        for delay in (0.0, 0.3, 0.6, 1.0, 1.5):
            if delay:
                time.sleep(delay)
                try:
                    self.sd._terminate(); self.sd._initialize()
                except Exception:
                    pass
            try:
                info = (self.sd.query_devices(self.device, 'input') if self.device is not None
                        else self.sd.query_devices(kind='input'))
                self.sr = int(info['default_samplerate'])
                self.frames = []
                self.stream = self.sd.InputStream(samplerate=self.sr, channels=1, dtype='float32',
                                                  device=self.device, callback=self._cb)
                self.stream.start()
                self.on = True
                return
            except Exception as e:
                last = e
        raise last

    def _cb(self, indata, frames, t, status):
        if self.on:
            self.frames.append(indata[:, 0].copy())

    def snapshot(self):
        """Concatenated audio so far WITHOUT stopping (for streaming partials)."""
        return np.concatenate(self.frames) if self.frames else np.zeros(0, dtype='float32')

    def stop(self):
        # detached close: the slow CoreAudio close must never wedge the next keypress
        self.on = False
        st, self.stream = self.stream, None
        if st:
            threading.Thread(target=lambda: (st.stop(), st.close()), daemon=True).start()
        return np.concatenate(self.frames) if self.frames else np.zeros(0, dtype='float32')


def resample_16k(audio, sr):
    audio = np.asarray(audio, dtype='float32')
    if len(audio) == 0 or sr == TARGET_SR:
        return audio
    return np.asarray(resample_poly(audio, TARGET_SR, sr), dtype='float32')


# ---------- parakeet ASR ----------
def _ensure_ffmpeg_path():
    """App-spawned processes get launchd's bare PATH (no homebrew) — parakeet's batch
    transcribe shells out to ffmpeg, so prepend its home if it isn't findable."""
    import shutil
    if shutil.which('ffmpeg'):
        return
    for p in ('/opt/homebrew/bin', '/usr/local/bin'):
        if os.path.exists(os.path.join(p, 'ffmpeg')):
            os.environ['PATH'] = p + ':' + os.environ.get('PATH', '')
            return


def load_parakeet(log=print, retries=3):
    """from_pretrained + 0.5s zeros warmup. Returns the model, or None after retries."""
    _ensure_ffmpeg_path()
    for attempt in range(1, retries + 1):
        try:
            from parakeet_mlx import from_pretrained
            m = from_pretrained(MODEL_ID)
            _ = transcribe_batch(m, np.zeros(TARGET_SR // 2, dtype='float32'), TARGET_SR)  # warmup
            log('parakeet ready')
            return m
        except Exception as e:
            log(f'parakeet load failed (attempt {attempt}/{retries}): {e}')
            time.sleep(4)
    return None


def transcribe_batch(model, audio, sr):
    """resample→16k mono WAV→model.transcribe(path). Skips <0.25s. (transcribe shells out to ffmpeg.)"""
    audio = resample_16k(audio, sr)
    if len(audio) < TARGET_SR // 4:
        return ''
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    w = wave.open(path, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(TARGET_SR)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
    w.close()
    try:
        return model.transcribe(path).text.strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# NOTE: parakeet-mlx `transcribe_stream` is intentionally NOT used. Its per-chunk mel normalization
# on an offline checkpoint hallucinates filler tokens ("mm/yeah/you"); live streaming is now Apple
# SpeechTranscriber (Swift). Python keeps only Parakeet BATCH transcribe (transcribe-after submode).


# ---------- screenshot capture (screencapture) + copy-to-chat (⌘C synth + pasteboard) ----------
def _shot_path():
    return os.path.join(tempfile.gettempdir(), f'civm-shot-{os.getpid()}-{int(time.time() * 1000)}.png')


def _png_b64(path):
    """Read a PNG file to base64, or None if absent/empty (screencapture writes nothing on cancel)."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode() if data else None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def capture_region(x, y, w, h):
    """screencapture -R x,y,w,h (TOP-LEFT global points) -> base64 PNG. Silent (-x)."""
    path = _shot_path()
    try:
        subprocess.run(['screencapture', '-x', '-R', f'{int(x)},{int(y)},{int(w)},{int(h)}', path],
                       check=False, timeout=20)
    except Exception:
        pass
    return _png_b64(path)


def capture_interactive():
    """screencapture -i (native crosshair) -> base64 PNG, or None if the user pressed ESC (no file).
    Blocks until the user finishes — call on its OWN thread."""
    path = _shot_path()
    try:
        subprocess.run(['screencapture', '-i', path], check=False, timeout=120)
    except Exception:
        pass
    return _png_b64(path)


def synth_cmd_c():
    """Synthesize a full, BALANCED ⌘C to the frontmost app (wtalk paste pattern): Command key DOWN →
    C down → C up → Command key UP, 5ms apart, each with explicit flags. The explicit Command key-up
    is the point — setting only the flag can leave Command 'stuck down' in the window server."""
    try:
        src = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    except Exception:
        src = None
    CMD = 0x37  # left Command keycode
    cmd_down = Q.CGEventCreateKeyboardEvent(src, CMD, True); Q.CGEventSetFlags(cmd_down, Q.kCGEventFlagMaskCommand)
    c_down = Q.CGEventCreateKeyboardEvent(src, KC_C, True); Q.CGEventSetFlags(c_down, Q.kCGEventFlagMaskCommand)
    c_up = Q.CGEventCreateKeyboardEvent(src, KC_C, False); Q.CGEventSetFlags(c_up, Q.kCGEventFlagMaskCommand)
    cmd_up = Q.CGEventCreateKeyboardEvent(src, CMD, False); Q.CGEventSetFlags(cmd_up, 0)
    for ev in (cmd_down, c_down, c_up, cmd_up):
        Q.CGEventPost(Q.kCGHIDEventTap, ev)
        time.sleep(0.005)


def read_pasteboard_capture():
    """NSPasteboard -> {'kind':'text','text':…} (preferred) | {'kind':'image','data':<b64 png>} | None."""
    from AppKit import NSPasteboard, NSPasteboardTypeString, NSImage, NSBitmapImageRep, NSBitmapImageFileTypePNG
    pb = NSPasteboard.generalPasteboard()
    s = pb.stringForType_(NSPasteboardTypeString)
    if s is not None and str(s).strip():
        return {'kind': 'text', 'text': str(s)}
    img = NSImage.alloc().initWithPasteboard_(pb)
    if img is not None:
        tiff = img.TIFFRepresentation()
        if tiff is not None:
            rep = NSBitmapImageRep.imageRepWithData_(tiff)
            if rep is not None:
                png = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
                if png is not None:
                    return {'kind': 'image', 'data': base64.b64encode(bytes(png)).decode()}
    return None


# ---------- CGEventTap: voice chord + ESC + capture bindings (screenshot / copy) ----------
def _mods_set(flags):
    m = set()
    if flags & Q.kCGEventFlagMaskCommand: m.add('cmd')
    if flags & Q.kCGEventFlagMaskControl: m.add('ctrl')
    if flags & Q.kCGEventFlagMaskAlternate: m.add('alt')
    if flags & Q.kCGEventFlagMaskShift: m.add('shift')
    return m


def _match_key(binding, keycode, mods):
    """True when a keyboard binding ('key',kc,mods) exactly matches this keyDown (mods as a set)."""
    return binding is not None and binding[0] == 'key' and binding[1] == keycode and mods == set(binding[2])


def mouse_loc_topleft():
    """Current cursor position in TOP-LEFT global points (CGEvent / screencapture space)."""
    p = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return (p.x, p.y)


class Tap:
    """Global CGEventTap. Emits raw physical events onto `out_q` (a queue.Queue).
    VOICE (only when cfg()['voice']):
        "chord_down"  both Control+Alternate became set (and Command NOT held)
        "chord_up"    either Control or Alternate was released
        "esc"         Escape keyDown (consumed only if should_consume_esc() is True)
    CAPTURE (only when cfg()['capture']; bindings from cfg()['shot'] / cfg()['copy'] — parse_binding forms):
        ("shot_down", x, y)  screenshot binding pressed  (x,y = TOP-LEFT global cursor point)
        ("shot_up",   x, y)  screenshot binding released (serve no-ops it unless press&hold armed)
        ("copy",)            copy-to-chat binding pressed
    Matched capture events are CONSUMED (return None); the voice modifier chord is PASSED THROUGH
    (never swallow bare modifiers); unrelated events pass untouched. serve.py applies submode /
    style / state semantics. `cfg` is a 0-arg callable returning the live config snapshot (read on
    the main thread — same thread as this callback — so it needs no lock).

    Create + enable() on the thread that owns the CFRunLoop (the AppKit main thread).
    tick() must be called periodically (from the main NSTimer) to re-enable after a
    macOS-induced disable. Tap creation returning None ⇒ missing Accessibility perms.
    """
    def __init__(self, out_q, should_consume_esc, cfg=None, log=print, prompt_perms=False):
        self.out_q = out_q
        self.should_consume_esc = should_consume_esc
        self.cfg = cfg or (lambda: {})
        self.log = log
        self.tap = None
        self.src = None
        self.chord = False           # last observed voice-chord state
        self.shot_active = False     # screenshot binding currently held (debounces auto-repeat)
        self.ok = False              # tap created & enabled
        self.prompt_perms = prompt_perms

    def _cb(self, proxy, etype, event, refcon):
        # macOS disables a tap whose callback ever stalls — re-enable and move on (wtalk scar)
        if etype in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
            if self.tap is not None:
                Q.CGEventTapEnable(self.tap, True)
            return event
        try:
            cfg = self.cfg()
            flags = Q.CGEventGetFlags(event)

            # ---- voice modifier chord (passed through; emitted only when voice enabled) ----
            # EXACTLY the configured modifiers must be held (no extras) — the chord key is user-set
            # (cfg['chord_mods']); the old code hardcoded ctrl+alt so a changed hotkey never took effect.
            if etype == Q.kCGEventFlagsChanged:
                want = cfg.get('chord_mods') or ('ctrl', 'alt')
                chord_now = all(flags & MOD_MASKS[m] for m in want) and \
                    not any(flags & MOD_MASKS[m] for m in MOD_MASKS if m not in want)
                if chord_now and not self.chord:
                    self.chord = True
                    if cfg.get('voice'):
                        self.out_q.put('chord_down')
                elif not chord_now and self.chord:
                    self.chord = False
                    if cfg.get('voice'):
                        self.out_q.put('chord_up')
                return event  # never swallow bare ctrl/opt

            cap = bool(cfg.get('capture'))
            shot = cfg.get('shot')
            copy = cfg.get('copy')

            if etype == Q.kCGEventKeyDown:
                keycode = Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode)
                # ESC (voice) — consume only while listening/processing
                if keycode == KC_ESC:
                    if self.should_consume_esc():
                        self.out_q.put('esc')
                        return None
                    return event
                if cap:
                    mods = _mods_set(flags)
                    if _match_key(shot, keycode, mods):
                        if not self.shot_active:            # first press; ignore auto-repeat
                            self.shot_active = True
                            loc = Q.CGEventGetLocation(event)
                            self.out_q.put(('shot_down', loc.x, loc.y))
                        return None                          # consume (incl. repeats)
                    if _match_key(copy, keycode, mods):
                        self.out_q.put(('copy',))
                        return None
                return event

            if etype == Q.kCGEventKeyUp:
                if cap and shot is not None and shot[0] == 'key':
                    keycode = Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode)
                    if keycode == shot[1]:
                        if self.shot_active:
                            self.shot_active = False
                            loc = Q.CGEventGetLocation(event)
                            self.out_q.put(('shot_up', loc.x, loc.y))
                        return None
                return event

            if etype == Q.kCGEventOtherMouseDown:
                if cap and shot is not None and shot[0] == 'mouse':
                    btn = Q.CGEventGetIntegerValueField(event, Q.kCGMouseEventButtonNumber)
                    if btn == shot[1]:
                        if not self.shot_active:
                            self.shot_active = True
                            loc = Q.CGEventGetLocation(event)
                            self.out_q.put(('shot_down', loc.x, loc.y))
                        return None
                return event

            if etype == Q.kCGEventOtherMouseUp:
                if cap and shot is not None and shot[0] == 'mouse':
                    btn = Q.CGEventGetIntegerValueField(event, Q.kCGMouseEventButtonNumber)
                    if btn == shot[1]:
                        if self.shot_active:
                            self.shot_active = False
                            loc = Q.CGEventGetLocation(event)
                            self.out_q.put(('shot_up', loc.x, loc.y))
                        return None
                return event
        except Exception as e:
            self.log(f'tap cb error: {e}')
        return event

    def create(self):
        """Create + enable the tap on the CURRENT runloop. Returns True on success.
        Non-fatal on failure (missing perms) — caller marks perm=False."""
        # AX trust check. prompt_perms=False ⇒ never pops a system dialog (the CIVM app owns
        # the permission prompt; the engine is spawned by it). Just observe trust.
        if AXIsProcessTrustedWithOptions is not None:
            try:
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: bool(self.prompt_perms)})
            except Exception:
                pass
        mask = (Q.CGEventMaskBit(Q.kCGEventKeyDown) | Q.CGEventMaskBit(Q.kCGEventKeyUp)
                | Q.CGEventMaskBit(Q.kCGEventFlagsChanged)
                | Q.CGEventMaskBit(Q.kCGEventOtherMouseDown) | Q.CGEventMaskBit(Q.kCGEventOtherMouseUp))
        tap = Q.CGEventTapCreate(Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap,
                                 Q.kCGEventTapOptionDefault, mask, self._cb, None)
        if tap is None:
            self.log('event tap creation FAILED — grant Accessibility + Input Monitoring to this python')
            self.ok = False
            return False
        self.tap = tap
        self.src = Q.CFMachPortCreateRunLoopSource(None, tap, 0)
        Q.CFRunLoopAddSource(Q.CFRunLoopGetCurrent(), self.src, Q.kCFRunLoopCommonModes)
        Q.CGEventTapEnable(tap, True)
        self.ok = True
        self.chord = False
        self.shot_active = False
        self.log('event tap active (voice chord + capture bindings)')
        return True

    def tick(self):
        """Re-enable if macOS disabled the tap (belt-and-suspenders alongside the callback path)."""
        if self.tap is not None and not Q.CGEventTapIsEnabled(self.tap):
            Q.CGEventTapEnable(self.tap, True)

    def destroy(self):
        if self.tap is not None:
            try:
                Q.CGEventTapEnable(self.tap, False)
            except Exception:
                pass
        if self.src is not None:
            try:
                Q.CFRunLoopRemoveSource(Q.CFRunLoopGetCurrent(), self.src, Q.kCFRunLoopCommonModes)
            except Exception:
                pass
        self.tap = None
        self.src = None
        self.ok = False
        self.chord = False
        self.shot_active = False
