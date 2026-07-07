import SwiftUI
import AppKit

// ---- input/output config (persisted globally in UserDefaults via @AppStorage) ----
// The old 4-way Mode matrix is gone: input is now ONE hybrid text box with a voice toggle on top
// (type OR dictate at the cursor), and output is Text (Voice coming soon).
enum Submode: String { case toggle, hold, vad }           // voice-input hotkey behaviour
enum Transcription: String { case after, stream }          // after = accurate; stream = faster
// What happens to a finished dictation. The field is a normal text box, so the transcript is inserted
// at the caret; this decides whether we also press send.
enum AutoSend: String, CaseIterable {
    case never, always, ifAlone
    var label: String { self == .never ? "Never" : self == .always ? "Always" : "If box empty" }
    var blurb: String {
        switch self {
        case .never:   return "Insert the transcript at the cursor; you press Enter to send."
        case .always:  return "Send immediately after every dictation (like the old voice mode)."
        case .ifAlone: return "Send only when the box was empty at dictation start; otherwise insert and wait."
        }
    }
}
// AGENT OUTPUT half. Only Text ships today; Voice is coming soon (disabled in the picker).
enum AgentOutput: String, CaseIterable {
    case text, voice
    var label: String { self == .text ? "Text" : "Voice" }
    var comingSoon: Bool { self == .voice }
}
enum ShotStyle: String, CaseIterable { case initiate, hold }  // screenshot: press-to-initiate vs press&hold-drag
// Where the reminder sits each turn — the latency ⇄ adherence trade-off (engine pre-caches accordingly).
enum ReminderMode: String, CaseIterable {
    case start, before, after
    var label: String { self == .start ? "At chat start" : self == .before ? "Before question" : "After question" }
    var blurb: String {
        switch self {
        case .start:  return "Fastest. Reminder cached once at the start — weakest adherence late in a chat."
        case .before: return "Fast. Reminder pre-sent before each question, stays recent."
        case .after:  return "Best adherence. Reminder rides after the question — on the response's critical path."
        }
    }
}

// UserDefaults keys (shared by the input/output bar, the input region, and the Settings sheet).
enum SK {
    static let submode = "civm.voiceSubmode", transcription = "civm.transcription", hotkey = "civm.hotkey"
    static let sidebarCollapsed = "civm.sidebarCollapsed"
    static let defaultHotkey = "ctrl+alt"
    static let defaultSubmode = Submode.hold.rawValue
    // Default system / reminder PROMPTS filled on new-chat creation — a SavedPrompt id, or "" = none.
    static let defaultSystemPrompt = "civm.defaultSystemPrompt", defaultReminderPrompt = "civm.defaultReminderPrompt"
    // Hybrid input: the voice toggle (arms the chord), what a finished dictation does, agent output mode,
    // and whether to mute system audio while dictating.
    static let voiceInput = "civm.voiceInput", autoSend = "civm.autoSend"
    static let agentOutput = "civm.agentOutput", muteDictation = "civm.muteDictation"
    static let defaultVoiceInput = true
    static let defaultAutoSend = AutoSend.ifAlone.rawValue
    static let defaultAgentOutput = AgentOutput.text.rawValue
    static let defaultMuteDictation = true
    // Capture shortcuts (screenshot + copy-to-chat)
    static let shotBinding = "civm.shotBinding", shotStyle = "civm.shotStyle", copyBinding = "civm.copyBinding"
    static let defaultShotBinding = "cmd+shift+2"    // ⌘⇧2
    static let defaultShotStyle = ShotStyle.initiate.rawValue
    static let defaultCopyBinding = "ctrl+alt+c"     // ⌃⌥C
    static let speechLocale = "civm.speechLocale"    // Apple streaming-transcription language (BCP-47)
    static let defaultSpeechLocale = "en-US"
    static let reminderMode = "civm.reminderMode"    // reminder placement (latency ⇄ adherence)
    static let defaultReminderMode = ReminderMode.start.rawValue
}

// ---- shared settings accessors — the SINGLE authority for reading the SK keys with the right
// fallbacks. SettingsView writes via @AppStorage; RootView's computed vars delegate here (their
// @AppStorage stays only as a render-invalidation trigger); ChatSession keeps stored copies it
// refreshes from these on UserDefaults.didChangeNotification. One place owns each default. ----
extension AutoSend     { static var current: AutoSend { AutoSend(rawValue: UserDefaults.standard.string(forKey: SK.autoSend) ?? SK.defaultAutoSend) ?? .ifAlone } }
extension AgentOutput  { static var current: AgentOutput { AgentOutput(rawValue: UserDefaults.standard.string(forKey: SK.agentOutput) ?? SK.defaultAgentOutput) ?? .text } }
extension Submode      { static func from(_ raw: String) -> Submode { Submode(rawValue: raw) ?? .hold }
                         static var current: Submode { from(UserDefaults.standard.string(forKey: SK.submode) ?? SK.defaultSubmode) } }
extension Transcription { static func from(_ raw: String) -> Transcription { Transcription(rawValue: raw) ?? .after }
                          static var current: Transcription { from(UserDefaults.standard.string(forKey: SK.transcription) ?? "") } }
extension ShotStyle    { static func from(_ raw: String) -> ShotStyle { ShotStyle(rawValue: raw) ?? .initiate } }
extension ReminderMode { static var current: ReminderMode { ReminderMode(rawValue: UserDefaults.standard.string(forKey: SK.reminderMode) ?? SK.defaultReminderMode) ?? .start } }
extension SK {
    // Bool keys: absent → the default (object(forKey:) is nil until the toggle is first written).
    static var voiceInputOn: Bool   { (UserDefaults.standard.object(forKey: voiceInput) as? Bool) ?? defaultVoiceInput }
    static var muteDictationOn: Bool { (UserDefaults.standard.object(forKey: muteDictation) as? Bool) ?? defaultMuteDictation }
    static var hotkeyValue: String     { UserDefaults.standard.string(forKey: hotkey) ?? defaultHotkey }
    static var speechLocaleValue: String { UserDefaults.standard.string(forKey: speechLocale) ?? defaultSpeechLocale }
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
                // UNION, never assign: releasing a 2-modifier chord fires flagsChanged for each key,
                // so `= mods` would shrink the set to whichever modifier is released last (⌥⌘ → "alt").
                // formUnion keeps the chord monotonically growing across the whole hold.
                if !mods.isEmpty { self.captured.formUnion(mods) }  // still holding — grow the chord
                else if !self.captured.isEmpty { self.finish() }    // all released → lock it in
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
    @EnvironmentObject var promptLib: PromptLib
    @AppStorage(SK.submode) private var submodeRaw = SK.defaultSubmode
    @AppStorage(SK.transcription) private var transcriptionRaw = Transcription.after.rawValue
    @AppStorage(SK.hotkey) private var hotkey = SK.defaultHotkey
    @AppStorage(SK.shotBinding) private var shotBinding = SK.defaultShotBinding
    @AppStorage(SK.shotStyle) private var shotStyleRaw = SK.defaultShotStyle
    @AppStorage(SK.copyBinding) private var copyBinding = SK.defaultCopyBinding
    @AppStorage(SK.speechLocale) private var speechLocaleID = SK.defaultSpeechLocale
    @AppStorage(SK.reminderMode) private var reminderModeRaw = SK.defaultReminderMode
    @AppStorage(SK.voiceInput) private var voiceInputOn = SK.defaultVoiceInput
    @AppStorage(SK.autoSend) private var autoSendRaw = SK.defaultAutoSend
    @AppStorage(SK.muteDictation) private var muteDictating = SK.defaultMuteDictation
    @AppStorage(SK.agentOutput) private var agentOutputRaw = SK.defaultAgentOutput
    @AppStorage(SK.defaultSystemPrompt) private var defaultSystemPromptID = ""
    @AppStorage(SK.defaultReminderPrompt) private var defaultReminderPromptID = ""
    @State private var speechLocales: [String] = []
    @State private var confirmReset = false
    @StateObject private var rec = HotkeyRecorder()
    @StateObject private var shotRec = BindingRecorder()
    @StateObject private var copyRec = BindingRecorder()

    private var submode: Binding<Submode> { Binding(get: { .from(submodeRaw) }, set: { submodeRaw = $0.rawValue }) }
    private var transcription: Binding<Transcription> { Binding(get: { .from(transcriptionRaw) }, set: { transcriptionRaw = $0.rawValue }) }
    private var autoSend: Binding<AutoSend> { Binding(get: { AutoSend(rawValue: autoSendRaw) ?? .ifAlone }, set: { autoSendRaw = $0.rawValue }) }
    private var shotStyle: Binding<ShotStyle> { Binding(get: { .from(shotStyleRaw) }, set: { shotStyleRaw = $0.rawValue }) }
    private var sym: String { hotkeySymbols(hotkey) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Settings").font(.title2.bold())
            Text("DEFAULTS FOR NEW CHATS — each new chat starts with these; change any of them per-chat inside the chat itself (that won't affect these defaults). The shortcuts at the bottom are global.")
                .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
            GroupBox("Default prompts") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Pre-fill a new chat's system prompt / reminder from your library. Deleting the chosen prompt reverts that to None.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    promptDefaultPicker("System prompt", selection: $defaultSystemPromptID, kind: "system")
                    promptDefaultPicker("Reminder", selection: $defaultReminderPromptID, kind: "reminder")
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Default agent output") {
                VStack(alignment: .leading, spacing: 6) {
                    Picker("", selection: Binding(get: { AgentOutput(rawValue: agentOutputRaw) ?? .text },
                                                  set: { if !$0.comingSoon { agentOutputRaw = $0.rawValue } })) {
                        ForEach(AgentOutput.allCases, id: \.self) { o in
                            Text(o.comingSoon ? "\(o.label) (soon)" : o.label).tag(o)
                        }
                    }.pickerStyle(.segmented).frame(maxWidth: 240)
                    Text("How the tutor replies. Voice output is coming soon.")
                        .font(.caption2).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Default reminder placement") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Where the reminder sits each turn. Nearer the answer = better adherence; earlier = faster (the engine pre-caches it while idle).")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Picker("", selection: Binding(get: { ReminderMode(rawValue: reminderModeRaw) ?? .start },
                                                  set: { reminderModeRaw = $0.rawValue })) {
                        ForEach(ReminderMode.allCases, id: \.self) { m in
                            Text("\(m.label) — \(m.blurb)").tag(m)
                        }
                    }.pickerStyle(.radioGroup).labelsHidden()
                }.frame(maxWidth: .infinity, alignment: .leading)
            }

            GroupBox("Default voice input") {
                VStack(alignment: .leading, spacing: 14) {
                    Toggle("Enable voice input — arms the \(sym) chord (you can still type)", isOn: $voiceInputOn)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Submode").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: submode) {
                            Text("Press \(sym) to interrupt & talk, again when finished").tag(Submode.toggle)
                            Text("Hold \(sym) to talk").tag(Submode.hold)
                        }.pickerStyle(.radioGroup).labelsHidden()
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("When a dictation finishes").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: autoSend) {
                            ForEach(AutoSend.allCases, id: \.self) { Text("\($0.label) — \($0.blurb)").tag($0) }
                        }.pickerStyle(.radioGroup).labelsHidden()
                    }
                    Toggle("Mute system audio while dictating", isOn: $muteDictating)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Transcription").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: transcription) {
                            Text("Transcribe after finished — Parakeet, most accurate").tag(Transcription.after)
                            Text("Streaming — Apple on-device, live as you speak").tag(Transcription.stream)
                        }.pickerStyle(.radioGroup).labelsHidden()
                    }
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("Streaming language").font(.caption.bold()).foregroundStyle(.secondary)
                        Picker("", selection: $speechLocaleID) {
                            ForEach(speechLocales, id: \.self) { Text($0).tag($0) }
                        }.labelsHidden().pickerStyle(.menu).frame(width: 140)
                        Spacer()
                    }
                    .task {
                        var locs = await AppleSpeech.supportedLocaleIDs()
                        if !locs.contains(speechLocaleID) { locs.insert(speechLocaleID, at: 0) }
                        speechLocales = locs
                    }
                    Text("Apple on-device (streaming mode only). Transcribe-after uses Parakeet (English).")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }

            globalShortcutsBox
            captureBox
                }
            }
            HStack {
                Button("Reset all settings…", role: .destructive) { confirmReset = true }
                    .confirmationDialog("Reset every setting to its factory default?", isPresented: $confirmReset, titleVisibility: .visible) {
                        Button("Reset all settings", role: .destructive) { resetAllSettings() }
                        Button("Cancel", role: .cancel) {}
                    } message: { Text("This resets the new-chat defaults, the dictation hotkey, and the capture shortcuts. It does NOT change your existing chats or saved prompts.") }
                Spacer()
                Button("Done") { rec.cancel(); shotRec.cancel(); copyRec.cancel(); dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20).frame(minWidth: 460, idealWidth: 480, minHeight: 420, idealHeight: 660)
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

    // Dictation hotkey — GLOBAL (drives the one CGEventTap), so it's not a per-chat default.
    private var globalShortcutsBox: some View {
        GroupBox("Dictation hotkey — all chats") {
            VStack(alignment: .leading, spacing: 6) {
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
                    Spacer()
                }
                Text("A modifier chord (default ⌃⌥) — same in every chat. Press-to-toggle vs hold is the per-chat submode above.")
                    .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // A "None + your saved prompts" menu bound to a SavedPrompt-id default key.
    private func promptDefaultPicker(_ label: String, selection: Binding<String>, kind: String) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.caption.bold()).foregroundStyle(.secondary).frame(width: 96, alignment: .leading)
            Picker("", selection: selection) {
                Text("None").tag("")
                ForEach(promptLib.prompts.filter { $0.kind == kind }) { p in Text(p.name).tag(p.id.uuidString) }
            }.labelsHidden().pickerStyle(.menu).frame(maxWidth: 220)
            Spacer()
        }
    }

    // Reset the DEFAULTS store + the global hotkey/capture bindings to their factory values. Existing
    // chats (their own per-chat settings) and saved prompts are left untouched.
    private func resetAllSettings() {
        rec.cancel(); shotRec.cancel(); copyRec.cancel()
        submodeRaw = SK.defaultSubmode
        transcriptionRaw = Transcription.after.rawValue
        hotkey = SK.defaultHotkey
        shotBinding = SK.defaultShotBinding
        shotStyleRaw = SK.defaultShotStyle
        copyBinding = SK.defaultCopyBinding
        speechLocaleID = SK.defaultSpeechLocale
        reminderModeRaw = SK.defaultReminderMode
        voiceInputOn = SK.defaultVoiceInput
        autoSendRaw = SK.defaultAutoSend
        muteDictating = SK.defaultMuteDictation
        agentOutputRaw = SK.defaultAgentOutput
        defaultSystemPromptID = ""
        defaultReminderPromptID = ""
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
}
