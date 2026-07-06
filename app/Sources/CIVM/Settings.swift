import SwiftUI
import AppKit

// ---- modes & submodes (persisted globally in UserDefaults via @AppStorage) ----
// YOUR INPUT · AGENT OUTPUT. Only text-text and voice-text ship; the voice-output pair is coming soon.
enum Mode: String, CaseIterable { case textText = "text-text", textVoice = "text-voice", voiceText = "voice-text", voiceVoice = "voice-voice" }
enum Submode: String { case toggle, hold, vad }           // Voice·Text hotkey behaviour
enum Transcription: String { case after, stream }          // after = accurate; stream = faster
enum ShotStyle: String, CaseIterable { case initiate, hold }  // screenshot: press-to-initiate vs press&hold-drag

// UserDefaults keys (shared by the modes bar, the input regions, and the Settings sheet).
enum SK {
    static let mode = "civm.mode", submode = "civm.voiceSubmode", transcription = "civm.transcription", hotkey = "civm.hotkey"
    static let sidebarCollapsed = "civm.sidebarCollapsed"
    static let defaultHotkey = "ctrl+alt"
    // Capture shortcuts (screenshot + copy-to-chat)
    static let shotBinding = "civm.shotBinding", shotStyle = "civm.shotStyle", copyBinding = "civm.copyBinding"
    static let defaultShotBinding = "cmd+shift+2"    // ⌘⇧2
    static let defaultShotStyle = ShotStyle.initiate.rawValue
    static let defaultCopyBinding = "ctrl+alt+c"     // ⌃⌥C
}

// ---- shared settings accessors — the SINGLE authority for reading the SK keys with the right
// fallbacks. SettingsView writes via @AppStorage; RootView's computed vars delegate here (their
// @AppStorage stays only as a render-invalidation trigger); ChatSession keeps stored copies it
// refreshes from these on UserDefaults.didChangeNotification. One place owns each default. ----
extension Mode         { static func from(_ raw: String) -> Mode { Mode(rawValue: raw) ?? .textText }
                         static var current: Mode { from(UserDefaults.standard.string(forKey: SK.mode) ?? "") } }
extension Submode      { static func from(_ raw: String) -> Submode { Submode(rawValue: raw) ?? .toggle }
                         static var current: Submode { from(UserDefaults.standard.string(forKey: SK.submode) ?? "") } }
extension Transcription { static func from(_ raw: String) -> Transcription { Transcription(rawValue: raw) ?? .after }
                          static var current: Transcription { from(UserDefaults.standard.string(forKey: SK.transcription) ?? "") } }
extension ShotStyle    { static func from(_ raw: String) -> ShotStyle { ShotStyle(rawValue: raw) ?? .initiate } }
extension SK {
    static var hotkeyValue: String     { UserDefaults.standard.string(forKey: hotkey) ?? defaultHotkey }
    static var shotBindingValue: String { UserDefaults.standard.string(forKey: shotBinding) ?? defaultShotBinding }
    static var shotStyleValue: String  { UserDefaults.standard.string(forKey: shotStyle) ?? defaultShotStyle }
    static var copyBindingValue: String { UserDefaults.standard.string(forKey: copyBinding) ?? defaultCopyBinding }
}

// ---- hotkey chord: a modifier-only chord, stored as an engine string ("ctrl+alt") and shown as symbols (⌃⌥) ----
func hotkeyString(_ m: NSEvent.ModifierFlags) -> String {
    var parts: [String] = []
    if m.contains(.control) { parts.append("ctrl") }
    if m.contains(.option)  { parts.append("alt") }
    if m.contains(.shift)   { parts.append("shift") }
    if m.contains(.command) { parts.append("cmd") }
    return parts.joined(separator: "+")
}
func hotkeySymbols(_ s: String) -> String {                // "ctrl+alt" → "⌃⌥"
    let map = ["ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"]
    let sym = s.split(separator: "+").map { map[String($0)] ?? "" }.joined()
    return sym.isEmpty ? "⌃⌥" : sym
}

// ---- capture bindings: a keyboard chord ("cmd+shift+2") OR a mouse button ("mouse3") ----
// ANSI virtual-keycode → canonical base-key name. The engine's NAME_TO_KEYCODE (voice.py) is the
// INVERSE of this table, so a chord serialized here round-trips to the right keycode there.
let keyCodeToName: [UInt16: String] = [
    0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g", 6: "z", 7: "x", 8: "c", 9: "v", 11: "b",
    12: "q", 13: "w", 14: "e", 15: "r", 16: "y", 17: "t", 18: "1", 19: "2", 20: "3", 21: "4",
    22: "6", 23: "5", 24: "=", 25: "9", 26: "7", 27: "-", 28: "8", 29: "0", 30: "]", 31: "o",
    32: "u", 33: "[", 34: "i", 35: "p", 37: "l", 38: "j", 39: "'", 40: "k", 41: ";", 42: "\\",
    43: ",", 44: "/", 45: "n", 46: "m", 47: ".", 50: "`",
    49: "space", 36: "return", 48: "tab", 51: "delete",
]

func modSymbols(_ m: NSEvent.ModifierFlags) -> String {    // ⌃⌥⇧⌘ order
    var out = ""
    if m.contains(.control) { out += "⌃" }
    if m.contains(.option)  { out += "⌥" }
    if m.contains(.shift)   { out += "⇧" }
    if m.contains(.command) { out += "⌘" }
    return out
}
// serialize a keyboard chord → engine string (mods then base key). Order [cmd,ctrl,alt,shift]+base
// so the defaults read naturally ("cmd+shift+2", "ctrl+alt+c"); the engine matches mods as a SET.
func bindingString(_ m: NSEvent.ModifierFlags, base: String) -> String {
    var parts: [String] = []
    if m.contains(.command) { parts.append("cmd") }
    if m.contains(.control) { parts.append("ctrl") }
    if m.contains(.option)  { parts.append("alt") }
    if m.contains(.shift)   { parts.append("shift") }
    parts.append(base)
    return parts.joined(separator: "+")
}
func baseKeyDisplay(_ b: String) -> String {
    switch b {
    case "space": return "␣"
    case "return": return "⏎"
    case "tab": return "⇥"
    case "delete": return "⌫"
    default: return b.uppercased()
    }
}
// engine binding string → friendly symbols. "cmd+shift+2" → "⌘⇧2" ; "mouse3" → "Mouse 3".
func bindingSymbols(_ s: String) -> String {
    if s.isEmpty { return "—" }
    if s.hasPrefix("mouse") { return "Mouse \(Int(s.dropFirst(5)) ?? 0)" }
    let parts = s.split(separator: "+").map(String.init)
    guard let base = parts.last else { return "—" }
    let mods = Set(parts.dropLast())
    var out = ""
    if mods.contains("ctrl") { out += "⌃" }
    if mods.contains("alt") { out += "⌥" }
    if mods.contains("shift") { out += "⇧" }
    if mods.contains("cmd") { out += "⌘" }
    return out + baseKeyDisplay(base)
}

// Records a modifier chord live: on each flagsChanged we hold the modifiers pressed, and finalize when
// they're all released — so pressing ⌃⌥ then letting go captures "ctrl+alt". Monitor is main-thread only.
@MainActor final class HotkeyRecorder: ObservableObject {
    @Published var recording = false
    @Published var captured: NSEvent.ModifierFlags = []
    private var monitor: Any?
    var onFinish: ((String) -> Void)?

    func start() {
        cancel()
        recording = true; captured = []
        monitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] e in
            MainActor.assumeIsolated {
                guard let self else { return }
                let mods = e.modifierFlags.intersection([.control, .option, .command, .shift])
                if !mods.isEmpty { self.captured = mods }        // still holding — grow the chord
                else if !self.captured.isEmpty { self.finish() }  // released → lock it in
            }
            return e
        }
    }
    func finish() {
        let final = captured
        cancel()
        if !final.isEmpty { onFinish?(hotkeyString(final)) }
    }
    func cancel() {
        if let m = monitor { NSEvent.removeMonitor(m); monitor = nil }
        recording = false
    }
}

// Records a capture binding: a keyboard chord (modifiers + a base key) OR — when allowMouse — a
// mouse button. Finalizes on the base keyDown (mods captured from live flagsChanged) or on
// otherMouseDown; ESC aborts. Monitors are main-thread only (mirrors HotkeyRecorder).
@MainActor final class BindingRecorder: ObservableObject {
    @Published var recording = false
    @Published var live = ""                          // modifiers held so far, as symbols
    var allowMouse = true
    var onFinish: ((String) -> Void)?
    private var mods: NSEvent.ModifierFlags = []
    private var monitors: [Any] = []

    func start() {
        cancel()
        recording = true; mods = []; live = ""
        var mons: [Any] = []
        if let m = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged, handler: { [weak self] e in
            MainActor.assumeIsolated {
                guard let self, self.recording else { return }
                self.mods = e.modifierFlags.intersection([.command, .control, .option, .shift])
                self.live = modSymbols(self.mods)
            }
            return e
        }) { mons.append(m) }
        if let m = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] e in
            var swallow = false
            MainActor.assumeIsolated {
                guard let self, self.recording else { return }
                if e.keyCode == 53 { self.cancel(); swallow = true; return }   // ESC aborts recording
                if let base = keyCodeToName[e.keyCode] {
                    self.finish(bindingString(self.mods, base: base)); swallow = true
                }
            }
            return swallow ? nil : e
        }) { mons.append(m) }
        if allowMouse, let m = NSEvent.addLocalMonitorForEvents(matching: .otherMouseDown, handler: { [weak self] e in
            MainActor.assumeIsolated {
                guard let self, self.recording else { return }
                self.finish("mouse\(e.buttonNumber + 1)")     // 1-indexed name (middle button = mouse3)
            }
            return nil
        }) { mons.append(m) }
        monitors = mons
    }
    private func finish(_ s: String) { cancel(); onFinish?(s) }
    func cancel() {
        for m in monitors { NSEvent.removeMonitor(m) }
        monitors = []; recording = false
    }
}

// ---- Settings sheet (opened from the gear on BOTH home and chat) — one section per mode ----
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(SK.submode) private var submodeRaw = Submode.toggle.rawValue
    @AppStorage(SK.transcription) private var transcriptionRaw = Transcription.after.rawValue
    @AppStorage(SK.hotkey) private var hotkey = SK.defaultHotkey
    @AppStorage(SK.shotBinding) private var shotBinding = SK.defaultShotBinding
    @AppStorage(SK.shotStyle) private var shotStyleRaw = SK.defaultShotStyle
    @AppStorage(SK.copyBinding) private var copyBinding = SK.defaultCopyBinding
    @StateObject private var rec = HotkeyRecorder()
    @StateObject private var shotRec = BindingRecorder()
    @StateObject private var copyRec = BindingRecorder()

    private var submode: Binding<Submode> { Binding(get: { .from(submodeRaw) }, set: { submodeRaw = $0.rawValue }) }
    private var transcription: Binding<Transcription> { Binding(get: { .from(transcriptionRaw) }, set: { transcriptionRaw = $0.rawValue }) }
    private var shotStyle: Binding<ShotStyle> { Binding(get: { .from(shotStyleRaw) }, set: { shotStyleRaw = $0.rawValue }) }
    private var sym: String { hotkeySymbols(hotkey) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Settings").font(.title2.bold())
            Text("YOUR INPUT · AGENT OUTPUT").font(.caption2.bold()).foregroundStyle(.secondary).tracking(1)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
            GroupBox("Text · Text") {
                Text("Type your question, read the answer. No settings.")
                    .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Text · Voice") { comingSoon }

            GroupBox("Voice · Text") {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Default submode").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: submode) {
                            Text("Press \(sym) to interrupt & talk, again when finished").tag(Submode.toggle)
                            Text("Hold \(sym) to talk").tag(Submode.hold)
                        }.pickerStyle(.radioGroup).labelsHidden()
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Transcription").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: transcription) {
                            Text("Transcribe after finished — more accurate").tag(Transcription.after)
                            Text("Streaming — less accurate but faster response times").tag(Transcription.stream)
                        }.pickerStyle(.radioGroup).labelsHidden()
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Hotkey").font(.caption.bold()).foregroundStyle(.secondary)
                        HStack(spacing: 10) {
                            Button {
                                if rec.recording { rec.cancel() } else { rec.start() }
                            } label: {
                                Text(rec.recording ? "Recording… press your chord" : sym)
                                    .font(rec.recording ? .caption : .body.weight(.semibold))
                                    .frame(minWidth: 60).padding(.vertical, 4).padding(.horizontal, 10)
                                    .background(RoundedRectangle(cornerRadius: 6).fill(rec.recording ? Color.red.opacity(0.18) : Color.gray.opacity(0.14)))
                            }.buttonStyle(.plain)
                            if rec.recording, !rec.captured.isEmpty { Text(hotkeySymbols(hotkeyString(rec.captured))).font(.body.weight(.semibold)) }
                            Text("a modifier chord — default ⌃⌥").font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Voice · Voice") { comingSoon }

            captureBox
                }
            }
            HStack { Spacer(); Button("Done") { rec.cancel(); shotRec.cancel(); copyRec.cancel(); dismiss() }.keyboardShortcut(.defaultAction) }
        }
        .padding(20).frame(width: 480, height: 640)
        .onAppear {
            rec.onFinish = { hotkey = $0 }
            copyRec.allowMouse = false
            shotRec.onFinish = { shotBinding = $0 }
            copyRec.onFinish = { copyBinding = $0 }
        }
        .onDisappear { rec.cancel(); shotRec.cancel(); copyRec.cancel() }
    }

    // ---- Capture: screenshot + copy-to-chat shortcuts (work in any mode while a chat is open) ----
    private var captureBox: some View {
        GroupBox("Capture") {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Screenshot shortcut").font(.caption.bold()).foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        bindingButton(shotRec, current: shotBinding, prompt: "press a key chord or mouse button")
                        Text("a key chord (⌘⇧2) or a mouse button (e.g. middle) → sends the shot to the chat")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Invocation").font(.caption.bold()).foregroundStyle(.secondary)
                    Picker("", selection: shotStyle) {
                        Text("Press to initiate — native crosshair (⎋ cancels)").tag(ShotStyle.initiate)
                        Text("Press & hold — drag a rectangle, release to capture").tag(ShotStyle.hold)
                    }.pickerStyle(.radioGroup).labelsHidden()
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Copy-to-chat shortcut").font(.caption.bold()).foregroundStyle(.secondary)
                    HStack(spacing: 10) {
                        bindingButton(copyRec, current: copyBinding, prompt: "press a key chord")
                        Text("a key chord — copies the frontmost selection into the chat").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Text("Active whenever a chat is open, in any mode.").font(.caption2).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func bindingButton(_ r: BindingRecorder, current: String, prompt: String) -> some View {
        Button {
            if r.recording { r.cancel() } else { r.start() }
        } label: {
            Text(r.recording ? (r.live.isEmpty ? "Recording… \(prompt)" : r.live) : bindingSymbols(current))
                .font(r.recording && r.live.isEmpty ? .caption : .body.weight(.semibold))
                .frame(minWidth: 60).padding(.vertical, 4).padding(.horizontal, 10)
                .background(RoundedRectangle(cornerRadius: 6).fill(r.recording ? Color.red.opacity(0.18) : Color.gray.opacity(0.14)))
        }.buttonStyle(.plain)
    }

    private var comingSoon: some View {
        Text("(Coming Soon)").font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
    }
}
