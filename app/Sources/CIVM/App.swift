import SwiftUI
import AppKit
import Combine
import Observation
import UniformTypeIdentifiers
import AVFoundation
import ApplicationServices
import IOKit.hid
import Speech

// One-shot TCC prompts for everything the voice-agent future needs. Safe to re-run —
// macOS only shows each dialog while the permission is undecided.
func requestAgentPerms() {
    AVCaptureDevice.requestAccess(for: .audio) { _ in }                                   // Microphone
    SFSpeechRecognizer.requestAuthorization { _ in }                                      // Speech recognition (Apple streaming)
    // literal key == kAXTrustedCheckOptionPrompt (the CFString global isn't Swift-6 concurrency-safe)
    _ = AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary) // Accessibility
    _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)                                  // Input Monitoring
    _ = CGRequestScreenCaptureAccess()                                                    // Screen Recording
}

struct Msg: Codable, Identifiable, Equatable {
    var id = UUID()
    var role: String                 // "user" | "assistant"
    var text: String                 // DISPLAY text (the "@@INTERRUPTION@@: " prefix is engine-only, never stored here)
    var content: [Block] = []        // image blocks pasted with this message — shown in the bubble, sent to the engine
    var interrupted: Bool = false    // assistant partial that a mid-generation interrupt cut off
    var isInterruption: Bool = false // user message sent mid-generation (amber bubble; engine text gets the prefix)

    init(role: String, text: String, content: [Block] = [], interrupted: Bool = false, isInterruption: Bool = false) {
        self.role = role; self.text = text; self.content = content
        self.interrupted = interrupted; self.isInterruption = isInterruption
    }
    // Lenient decode (as Chat): messages saved by OLDER builds (no content/flag keys) still open.
    enum CodingKeys: String, CodingKey { case id, role, text, content, interrupted, isInterruption }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(UUID.self, forKey: .id)) ?? UUID()
        role = (try? c.decode(String.self, forKey: .role)) ?? "user"
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        content = (try? c.decode([Block].self, forKey: .content)) ?? []
        interrupted = (try? c.decode(Bool.self, forKey: .interrupted)) ?? false
        isInterruption = (try? c.decode(Bool.self, forKey: .isInterruption)) ?? false
    }
}

// Per-chat settings: SNAPSHOTTED from the DEFAULTS store (SK.*) when a chat is CREATED, then owned by
// that chat. Editing them in the chat UI changes only THIS chat; editing the defaults in Settings
// changes only FUTURE new chats. Lenient-decoded, so chats saved by older builds (no settings key)
// open with the factory values below. Hotkey + capture bindings are NOT here — those are process-global.
struct ChatSettings: Codable, Equatable {
    var agentOutput = AgentOutput.text.rawValue
    var reminderMode = ReminderMode.start.rawValue
    var voiceInput = true
    var submode = Submode.hold.rawValue
    var autoSend = AutoSend.ifAlone.rawValue
    var muteDictation = true
    var transcription = Transcription.after.rawValue
    var speechLocale = "en-US"
    // NOTE: the conversation-cache budget/mode are GLOBAL, not per-chat — they live in SK.* and are read
    // live via ChatSession.trimTrigger/trimTarget/recacheMode. They are intentionally NOT stored here.

    init() {}
    enum CodingKeys: String, CodingKey { case agentOutput, reminderMode, voiceInput, submode, autoSend, muteDictation, transcription, speechLocale }
    init(from d: Decoder) throws {   // per-field lenient: a new field added later won't wipe saved settings
        let c = try d.container(keyedBy: CodingKeys.self)
        if let v = try? c.decode(String.self, forKey: .agentOutput)   { agentOutput = v }
        if let v = try? c.decode(String.self, forKey: .reminderMode)  { reminderMode = v }
        if let v = try? c.decode(Bool.self,   forKey: .voiceInput)    { voiceInput = v }
        if let v = try? c.decode(String.self, forKey: .submode)       { submode = v }
        if let v = try? c.decode(String.self, forKey: .autoSend)      { autoSend = v }
        if let v = try? c.decode(Bool.self,   forKey: .muteDictation) { muteDictation = v }
        if let v = try? c.decode(String.self, forKey: .transcription) { transcription = v }
        if let v = try? c.decode(String.self, forKey: .speechLocale)  { speechLocale = v }
    }

    var agentOutputV: AgentOutput   { AgentOutput(rawValue: agentOutput) ?? .text }
    var reminderModeV: ReminderMode { ReminderMode(rawValue: reminderMode) ?? .start }
    var submodeV: Submode           { Submode(rawValue: submode) ?? .hold }
    var autoSendV: AutoSend         { AutoSend(rawValue: autoSend) ?? .ifAlone }
    var transcriptionV: Transcription { Transcription(rawValue: transcription) ?? .after }

    // Snapshot the current DEFAULTS store (SK.*) — used when a NEW chat is created.
    static func fromDefaults() -> ChatSettings {
        var s = ChatSettings()
        s.agentOutput = AgentOutput.current.rawValue
        s.reminderMode = ReminderMode.current.rawValue
        s.voiceInput = SK.voiceInputOn
        s.submode = Submode.current.rawValue
        s.autoSend = AutoSend.current.rawValue
        s.muteDictation = SK.muteDictationOn
        s.transcription = Transcription.current.rawValue
        s.speechLocale = SK.speechLocaleValue
        return s
    }
}

// A chat = an ordered content-block SYSTEM prompt + an ordered content-block CONTEXT
// (text + images interleaved, exactly the shape the engine pins), plus the transcript.
struct Chat: Codable, Identifiable, Equatable {
    var id = UUID()
    var name = "New chat"
    var system: [Block] = [Block(text: "")]   // start with one typeable text block
    var context: [Block] = [Block(text: "")]
    var messages: [Msg] = []
    var pinnedTokens: Int? = nil
    var chatTokens: Int = 0                                     // last /chat done meta chat_tokens (history+q+reminder+answer)
    var reminder: [Block] = [Block(text: "")]                   // text+image, sent after every question; not pinned. blank by default
    var settings = ChatSettings()                              // per-chat input/output settings (snapshotted from defaults on create)

    init(name: String = "New chat") { self.name = name }
    // Lenient decode: every field falls back to its default, so chats saved by OLDER builds
    // (no reminder key, or reminder-as-String, no chatTokens, no settings) still open instead of failing.
    enum CodingKeys: String, CodingKey { case id, name, system, context, messages, pinnedTokens, chatTokens, reminder, settings }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(UUID.self, forKey: .id)) ?? UUID()
        name = (try? c.decode(String.self, forKey: .name)) ?? "New chat"
        system = (try? c.decode([Block].self, forKey: .system)) ?? [Block(text: "")]
        context = (try? c.decode([Block].self, forKey: .context)) ?? [Block(text: "")]
        messages = (try? c.decode([Msg].self, forKey: .messages)) ?? []
        pinnedTokens = try? c.decode(Int.self, forKey: .pinnedTokens)
        chatTokens = (try? c.decode(Int.self, forKey: .chatTokens)) ?? 0
        if let b = try? c.decode([Block].self, forKey: .reminder) { reminder = b }
        else if let s = try? c.decode(String.self, forKey: .reminder) { reminder = [Block(text: s)] }  // legacy String
        settings = (try? c.decode(ChatSettings.self, forKey: .settings)) ?? ChatSettings()
    }

    // cheap fingerprint of what Cache would pin — drives the "cache unchanged → grey" logic
    var contentHash: Int {
        var h = Hasher()
        for b in nonEmpty(system) + nonEmpty(context) { h.combine(b.type); h.combine(b.text); h.combine(b.data?.count) }
        return h.finalize()
    }
}

// light row for the home list (decodes only these keys from the saved chat JSON)
struct ChatStub: Codable, Identifiable { var id: UUID; var name: String; var pinnedTokens: Int?; var mtime: Date? }

// real content = any image, or any non-blank text block
func nonEmpty(_ blocks: [Block]) -> [Block] {
    blocks.filter { $0.type == "image" || !($0.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

@MainActor @Observable final class Store {
    enum Screen { case home, chat }
    // How the current chat was activated — the session listens on `activated` and reacts once per
    // activation (opened → re-pin; created/seeded → no pin), keeping today's per-path behavior.
    enum Activation { case opened, created, seeded }
    let activated = PassthroughSubject<Activation, Never>()
    var screen: Screen = .home
    var chat = Chat()
    var stubs: [ChatStub] = []
    var loadingChat = false        // decoding a chat off-main → UI shows a spinner, never freezes
    // what the engine's ONE global KV currently holds (chat + content version)
    var enginePinnedChat: UUID? = nil
    var enginePinnedHash: Int? = nil

    private var dir: URL {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("contextualized_instant_voice_models/chats")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    // Serial queue so encodes/writes land in ORDER (fire-and-forget Task.detached could finish out of
    // order and clobber newer content with stale). Off the main thread so a big chat never hitches the UI.
    private let writeQueue = DispatchQueue(label: "civm.chat.save")
    func save(thenRefresh refresh: Bool = false) {
        // don't litter home with pristine empty chats
        guard !(chat.messages.isEmpty && nonEmpty(chat.system).isEmpty && nonEmpty(chat.context).isEmpty) else {
            if refresh { refreshStubs() }; return
        }
        let snapshot = chat, url = dir.appendingPathComponent("\(chat.id).json")
        writeQueue.async { [weak self] in
            if let d = try? JSONEncoder().encode(snapshot) { try? d.write(to: url) }
            if refresh { Task { @MainActor in self?.refreshStubs() } }   // refresh AFTER the write lands (not before)
        }
    }

    func refreshStubs() {
        let fm = FileManager.default
        stubs = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> ChatStub? in
                guard let d = try? Data(contentsOf: url), var s = try? JSONDecoder().decode(ChatStub.self, from: d) else { return nil }
                s.mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
                return s
            }
            .sorted { ($0.mtime ?? .distantPast) > ($1.mtime ?? .distantPast) }
    }
    func open(_ id: UUID) {
        // Decode OFF the main thread — a large saved chat (many messages + base64 images) can take a
        // noticeable moment, which must NEVER freeze the UI. Transition immediately + show a spinner;
        // the real content swaps in when the decode finishes.
        let url = dir.appendingPathComponent("\(id).json")
        loadingChat = true
        chat = Chat(name: "Loading…")        // placeholder so the chat pane isn't showing the previous chat
        screen = .chat
        Task.detached(priority: .userInitiated) { [weak self] in
            let c = (try? Data(contentsOf: url)).flatMap { try? JSONDecoder().decode(Chat.self, from: $0) }
            await MainActor.run {
                guard let self else { return }
                self.loadingChat = false
                guard let c else { self.screen = .home; return }
                self.chat = c
                self.activated.send(.opened)
            }
        }
    }
    func goHome() { save(thenRefresh: true); screen = .home }   // refresh the sidebar AFTER the save write lands
    func startNew() {
        var c = Chat()
        c.settings = .fromDefaults()          // snapshot the current DEFAULTS store into the new chat
        chat = c; screen = .chat; activated.send(.created)
        // default system/reminder prompts (if set) are filled by RootView.newChat, which has the prompt lib
    }

    // Names are USER-SET ONLY (never auto-named from content). Rename rewrites the saved chat's name
    // in place; if it's the open chat, keep store.chat in sync. Delete removes the JSON + refreshes.
    func rename(_ id: UUID, to name: String) {
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !n.isEmpty else { return }
        if chat.id == id { chat.name = n }
        let url = dir.appendingPathComponent("\(id).json")
        if let d = try? Data(contentsOf: url), var c = try? JSONDecoder().decode(Chat.self, from: d) {
            c.name = n
            if let e = try? JSONEncoder().encode(c) { try? e.write(to: url) }
        }
        refreshStubs()
    }
    func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("\(id).json"))
        refreshStubs()
    }
}

// Home chat row: open (name area) + rename (pencil → inline field) + two-click Delete confirm.
// Names are user-set only — the pencil is the ONLY way a chat gets named. (Mirrors PromptRow.)
struct ChatRow: View {
    let stub: ChatStub
    let onOpen: () -> Void
    let onRename: (String) -> Void
    let onDelete: () -> Void
    @State private var editing = false
    @State private var name = ""
    @State private var confirmDelete = false
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 8) {
            if editing {
                TextField("chat name", text: $name).textFieldStyle(.roundedBorder).focused($focused)
                    .onSubmit { commit() }
                    .onAppear { focused = true }
                Button("Save") { commit() }.font(.caption)
                Button("Cancel") { editing = false }.font(.caption)
            } else {
                Button(action: onOpen) {
                    HStack {
                        Text(stub.name.isEmpty ? "(unnamed)" : stub.name)
                        Spacer()
                        if let t = stub.pinnedTokens { Text("\(t) tok").font(.caption2).foregroundStyle(.secondary) }
                        if let m = stub.mtime { Text(m, format: .dateTime.month().day().hour().minute()).font(.caption2).foregroundStyle(.secondary) }
                    }.contentShape(Rectangle())
                }.buttonStyle(.plain)
                Button { name = stub.name; editing = true } label: { Image(systemName: "pencil") }
                    .buttonStyle(.plain).foregroundStyle(.secondary).help("Rename")
                if confirmDelete {
                    Button("Confirm") { onDelete() }.font(.caption).foregroundStyle(.red)
                    Button("Cancel") { confirmDelete = false }.font(.caption)
                } else {
                    Button { confirmDelete = true } label: { Image(systemName: "trash") }
                        .buttonStyle(.plain).foregroundStyle(.red.opacity(0.8)).help("Delete")
                }
            }
        }
        .padding(10).frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.12)))
    }
    private func commit() { onRename(name); editing = false }
}

// Non-publishing dependency container: holds the app's long-lived objects so the App scene can
// own them via a SINGLE @StateObject without re-rendering per token/keystroke/tick. It has NO
// members, so it never fires objectWillChange; the three runtimes are still observed by
// the views that need them, injected below via environmentObject.
@MainActor final class AppDeps: ObservableObject {
    let engine: Engine
    let store: Store
    let promptLib: PromptLib
    let session: ChatSession
    init() {
        let e = Engine(); let s = Store()
        engine = e; store = s
        promptLib = PromptLib()
        session = ChatSession(engine: e, store: s)
    }
}

@main struct CIVMApp: App {
    @StateObject private var deps = AppDeps()
    let memOK: Bool
    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        // Disable the press-and-hold accent popup (ś š ș…) so held keys repeat like a normal text
        // editor — a chat input shouldn't pop the accent menu when you rest on a key. `register` only
        // seeds the lowest-priority domain, so a value in NSGlobalDomain overrides it (which is why it
        // kept coming back). `set` writes the app domain, which wins over global — force it off.
        UserDefaults.standard.set(false, forKey: "ApplePressAndHoldEnabled")
        memOK = Mem.enough
    }
    var body: some Scene {
        WindowGroup("Rally") {
            Group {
                if !memOK {
                    VStack(spacing: 10) {
                        Text("Not enough memory").font(.title2.bold())
                        Text(String(format: "Need ~%.0fGB free (model %.0f + %.0f headroom); have %.1fGB of %.1fGB.",
                                    Mem.modelGB + Mem.headroomGB, Mem.modelGB, Mem.headroomGB, Mem.availableGB, Mem.totalGB))
                            .foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }.padding(40).frame(minWidth: 460, minHeight: 240)
                } else {
                    RootView().environment(deps.engine).environment(deps.store).environment(deps.promptLib).environment(deps.session)
                        .onAppear {
                            deps.engine.start()
                            if !UserDefaults.standard.bool(forKey: "permsRequested") {
                                UserDefaults.standard.set(true, forKey: "permsRequested")
                                requestAgentPerms()
                            }
                        }.frame(minWidth: 260, minHeight: 200)   // shrinks to a thin sliver (collapse the sidebar for the narrowest)
                }
            }
        }
        .commands {
            // ⌃B toggles the sidebar. A MENU key equivalent (not a local key monitor) so it works even
            // while the chat WebView owns the keyboard — key equivalents are checked before first-responder
            // dispatch. State is the same @AppStorage key RootView binds.
            CommandGroup(after: .sidebar) {
                Button("Toggle Sidebar") {
                    UserDefaults.standard.set(!UserDefaults.standard.bool(forKey: SK.sidebarCollapsed), forKey: SK.sidebarCollapsed)
                }
                .keyboardShortcut("b", modifiers: .control)
            }
        }
    }
}

// ---- interleaved text+image content-block stream (pure SwiftUI, visible & selectable) ----
// Fills whatever vertical space its parent gives it (so collapsible panes share height equally).
struct BlockStream: View {
    @Binding var blocks: [Block]
    var jsonKey: String? = nil   // preferred key when loading a {"system":…,"context":…} style file

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(blocks) { blk in row(blk.id) }
                    if blocks.isEmpty {
                        Text("empty — type, paste, or drop images").font(.caption).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }.padding(8)
            }
            .frame(maxWidth: .infinity, minHeight: 54, maxHeight: .infinity)
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.3)))
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                for p in providers { _ = p.loadObject(ofClass: URL.self) { url, _ in
                    if let url { DispatchQueue.main.async { addImage(url) } } } }
                return true
            }
            .onPasteCommand(of: [.image, .png, .tiff, .plainText]) { _ in paste() }
            HStack(spacing: 10) {
                Button("+ Text") { blocks.append(Block(text: "")) }.font(.caption2)
                Button("+ Image…") { pickImages() }.font(.caption2)
                Button("Paste") { paste() }.font(.caption2)
                Button("PDF…") { loadPDF() }.font(.caption2)
                Button("JSON…") { loadJSON() }.font(.caption2)
                if !blocks.isEmpty { Button("clear") { blocks = [] }.font(.caption2) }
                Spacer()
                let imgs = blocks.filter { $0.type == "image" }.count
                Text("\(imgs) image\(imgs == 1 ? "" : "s") · \(textChars) chars")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    // Paste from the clipboard: image → image block, else text → text block. (cmd+V or the button.)
    func paste() {
        if let blk = clipboardImageBlock() { blocks.append(blk); return }
        let pb = NSPasteboard.general
        if let s = pb.string(forType: .string), !s.isEmpty { blocks.append(Block(text: s)) }
    }

    var textChars: Int { blocks.reduce(0) { $0 + ($1.text?.count ?? 0) } }

    // Row is addressed by ID and looks the block up each access — deleting it returns a safe default
    // instead of index-crashing (the ForEach($blocks) element-binding read a freed index → SIGTRAP).
    @ViewBuilder func row(_ id: UUID) -> some View {
        let blk = blocks.first(where: { $0.id == id })
        let textBinding = Binding<String>(
            get: { blocks.first(where: { $0.id == id })?.text ?? "" },
            set: { nv in if let i = blocks.firstIndex(where: { $0.id == id }) { blocks[i].text = nv } })
        HStack(alignment: .top, spacing: 6) {
            if let blk {
                if blk.type == "text" {
                    TextField("text…", text: textBinding, axis: .vertical)
                        .textFieldStyle(.plain).font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.primary).lineLimit(1...)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if blk.type == "image" {
                    AsyncBlockImage(block: blk) { $0.resizable().scaledToFit() }
                        .frame(maxWidth: .infinity, maxHeight: 220, alignment: .leading)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                }
            }
            Button { blocks.removeAll { $0.id == id } } label: {
                Image(systemName: "xmark.circle.fill")
            }.buttonStyle(.plain).foregroundStyle(.secondary.opacity(0.6))
        }
    }

    func addImage(_ url: URL) {
        if let data = try? Data(contentsOf: url), let (b64, mt) = imageToBase64(data) {
            blocks.append(Block(mediaType: mt, data: b64))
        }
    }
    func pickImages() {
        let p = NSOpenPanel(); p.allowsMultipleSelection = true; p.canChooseDirectories = false
        p.allowedContentTypes = [.image]
        if p.runModal() == .OK { p.urls.forEach { addImage($0) } }
    }

    // Load a PDF → one "Page N" text + page-image block per page, replacing this box's contents.
    func loadPDF() {
        let p = NSOpenPanel(); p.allowsMultipleSelection = false; p.canChooseDirectories = false
        p.allowedContentTypes = [.pdf]
        guard p.runModal() == .OK, let u = p.url else { return }
        let pages = pdfToBlocks(u)
        if !pages.isEmpty { blocks = pages }
    }

    // Load THIS box from a JSON file: a bare [blocks] array, or a {"system":…,"context":…,"reminder":…}
    // file (takes this box's key; if the file has exactly one key, takes that).
    func loadJSON() {
        let p = NSOpenPanel(); p.allowsMultipleSelection = false; p.canChooseDirectories = false
        p.allowedContentTypes = [.json]
        guard p.runModal() == .OK, let u = p.url, let d = try? Data(contentsOf: u) else { return }
        if let arr = try? JSONDecoder().decode([Block].self, from: d), !arr.isEmpty { blocks = arr; return }
        if let dict = try? JSONDecoder().decode([String: [Block]].self, from: d) {
            if let k = jsonKey, let arr = dict[k], !arr.isEmpty { blocks = arr }
            else if dict.count == 1, let arr = dict.values.first, !arr.isEmpty { blocks = arr }
        }
    }
}

struct RootView: View {
    @Environment(Engine.self) var engine
    @Environment(Store.self) var store
    @Environment(PromptLib.self) var promptLib
    @Environment(ChatSession.self) var session    // conversation runtime — all behavior lives here (@Observable)
    @AppStorage(SK.sidebarCollapsed) private var sidebarCollapsed = false
    @State private var lastSidebarW: CGFloat = 0   // direction tracking for auto-collapse (shrink-only)
    @State private var saveTarget: SaveTarget? = nil     // chat sidebar "Save current as…" sheet
    private let chatBottomID = "civm.chatBottom"          // stable auto-scroll anchor id
    @State private var editingChatName = false        // chat-screen inline rename
    @State private var chatNameDraft = ""
    @FocusState private var chatNameFocused: Bool
    @State private var systemOpen = true
    @State private var contextOpen = true
    @State private var reminderOpen = true

    // ---- input / voice settings ----
    // These are now PER-CHAT: they live on the open chat (store.chat.settings) and the chat UI writes
    // them via ChatSession setters (which save + re-post engine config). The Settings sheet edits the
    // separate DEFAULTS store (SK.*), consulted only when a new chat is created. Only the hotkey +
    // capture bindings are process-global (they drive the one CGEventTap), so hotkey stays @AppStorage.
    @AppStorage(SK.hotkey) private var hotkey = SK.defaultHotkey
    // Conversation-cache budget/mode are GLOBAL (apply to every chat live) — the ⋯ menu edits the same keys as Settings.
    @AppStorage(SK.cacheTrigger) private var gCacheTrigger = SK.defaultCacheTrigger
    @AppStorage(SK.cacheTarget) private var gCacheTarget = SK.defaultCacheTarget
    @AppStorage(SK.recacheMode) private var gRecacheMode = SK.defaultRecacheMode
    var cs: ChatSettings { store.chat.settings }         // the open chat's live settings (read by the input UI)
    var submode: Submode { cs.submodeV }
    var transcription: Transcription { cs.transcriptionV }
    var hk: String { bindingSymbols(hotkey) }   // full key+mods (e.g. ⌃⌥␣) — hotkeySymbols dropped the base key

    @State private var inputOptsOpen = false            // "Your Input" sub-options popover
    @State private var speechLocales: [String] = []     // Apple streaming locales (loaded lazily for the popover)
    @State private var showSettings = false
    @State private var pasteMonitor: Any? = nil         // ⌘V image-paste interceptor (see inputRegion)
    @FocusState private var inputFocused: Bool
    @State private var reminderDraft: [Block] = []     // staged reminder edits; applied to chat.reminder on "Update reminder"
    @State private var reminderConfirm = false

    var reminderDirty: Bool { reminderDraft != store.chat.reminder }

    // ---- cache status / estimate wording (the view owns the strings; the session owns the state) ----
    var cacheStatusIsError: Bool {
        switch session.cacheState { case .overLimit, .failed: return true; default: return false }
    }
    var estLabel: String {
        if session.contentEmpty { return "nothing to cache" }
        let n = session.estTokens.formatted()
        return session.estOverLimit ? "est. ~\(n) tok — over the 200K limit" : "est. ~\(n) tok"
    }

    // Two lines: the running context size, and an explicit per-message cache-vs-new breakdown from
    // the ACTUAL last run (pinned + history reused = read from cache; new = processed anew; + TTFT).
    // Quick reminder-placement switch, right in the chat (mirrors the Settings picker). Changing it
    // re-warms the cache with the new mode (the precache chip shows caching → ready). Default lives
    // in Settings; this is the fast in-chat toggle.
    @ViewBuilder var reminderModeControl: some View {
        Menu {
            ForEach(ReminderMode.allCases, id: \.self) { m in
                Button {
                    session.setReminderMode(m)
                } label: {
                    if cs.reminderModeV == m { Label(m.label, systemImage: "checkmark") }
                    else { Text(m.label) }
                }
            }
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "text.insert").font(.system(size: 9))
                Text("reminder: \(cs.reminderModeV.label)").font(.caption2)
            }.foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton).fixedSize().menuIndicator(.hidden)
    }

    // Shared chrome for BOTH input modes (text·text and voice·text): the token/cache breakdown line
    // plus the reminder-placement switch. Text and voice differ ONLY in the input widget above this.
    var chatMetaBar: some View {
        HStack(alignment: .bottom) {
            tokenLine.frame(maxWidth: .infinity, alignment: .leading)
            reminderModeControl
        }
    }

    var tokenLine: some View {
        let p = store.chat.pinnedTokens ?? 0, c = store.chat.chatTokens
        let cached = session.lastPinned + session.lastReused
        return VStack(alignment: .leading, spacing: 1) {
            Text("\(p) pinned + \(c) chat = \(p + c) tok\(session.preSent > 0 ? "  ·  ⚡\(session.preSent) of next turn precomputed" : "")")
            if session.lastNew > 0 || session.lastTtft > 0 {
                // X = the turn's notional cost, Y = precomputed while composing, Z = actually processed.
                let x = max(session.lastTurnTokens, session.lastNew), z = session.lastNew
                let y = max(0, x - z)
                Text("last msg: \(cached) read from cache · turn was \(x) tok\(y > 0 ? " · ⚡\(y) precomputed" : "") · \(z) processed anew · TTFT \(String(format: "%.2f", session.lastTtft))s")
                    .foregroundStyle(session.lastTtft > 1.5 ? .orange : .secondary)
                if !session.anewParts.isEmpty {
                    Text("  processed = " + session.anewParts.map { "\($0.n) \($0.label)" }.joined(separator: " + "))
                        .help("What the send itself had to feed. \"structure\" = the turn's framing tokens — closing your message (<|im_end|>) + the model's generation prompt — which can never be precomputed, because your turn must stay OPEN while you compose.")
                }
            }
        }.font(.caption2).foregroundStyle(.secondary)
    }

    var body: some View {
        Group {
            if store.screen == .home { homeBody } else { chatBody }
        }
        .sheet(isPresented: $showSettings) { SettingsView().environment(promptLib) }
        // Voice-config lifecycle (launch, engine-ready, entering/leaving a chat, settings changes) is
        // owned by the session itself — it subscribes to engine.ready / store.screen / the SK keys and
        // re-posts /voice/config on its own. No view-side relays remain.
    }

    private var gearButton: some View {
        Button { showSettings = true } label: { Image(systemName: "gearshape") }
            .help("Settings")
    }

    // ---------- home: model status + chats ----------
    var homeBody: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Rally").font(.title2.bold())
                    Text("Contextualized Consistent Instantaneous Voice-enabled Agent with Noninterruptive Context Truncation")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    gearButton
                }
                HStack(spacing: 8) {
                    Circle().fill(engine.ready ? Color.green : Color.orange).frame(width: 9, height: 9)
                    Text(engine.status).font(.callout).foregroundStyle(engine.ready ? .primary : .secondary)
                }
                HStack(spacing: 8) {
                    Circle().fill(engine.parakeet ? Color.green : Color.orange).frame(width: 8, height: 8)
                    Text(engine.parakeet ? "parakeet ready" : "parakeet loading…")
                        .font(.caption).foregroundStyle(engine.parakeet ? .primary : .secondary)
                }
                Text("MODEL").font(.caption.bold()).foregroundStyle(.secondary).padding(.top, 8)
                HStack(spacing: 10) {
                    ForEach(EngineModel.allCases) { m in modelCard(m) }
                }
                HStack {
                    Button("＋ New chat") { newChat() }   // chat-able immediately (engine holds an empty baseline; first ask resets the pin)
                    Button("Grant agent permissions…") { requestAgentPerms() }.font(.caption)
                }
                Text("CHATS").font(.caption.bold()).foregroundStyle(.secondary).padding(.top, 8)
                if store.stubs.isEmpty { Text("no chats yet").font(.caption).foregroundStyle(.secondary) }
                LazyVStack(spacing: 6) {
                    ForEach(store.stubs) { s in
                        ChatRow(stub: s,
                                onOpen: { store.open(s.id) },   // session re-pins on the .opened activation
                                onRename: { store.rename(s.id, to: $0) },
                                onDelete: { store.delete(s.id) })
                    }
                }
                PromptLibrarySection().padding(.top, 8)
            }
            .padding(24).frame(maxWidth: 620, alignment: .topLeading).frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { store.refreshStubs() }
    }

    // One selectable model card. Greyed out when there isn't enough free memory to run it (free-ish +
    // whatever the currently-running engine would release on switch). Selecting the other model unloads
    // the engine and relaunches with the new one; pins are invalidated so every chat re-pins on open.
    @ViewBuilder func modelCard(_ m: EngineModel) -> some View {
        let selected = engine.model == m
        let runnable = selected || Mem.canRun(m, runningEngineGB: engine.memGb)
        Button {
            guard !selected, runnable else { return }
            store.enginePinnedChat = nil; store.enginePinnedHash = nil   // engine restart voids every pin
            session.resetForModelSwitch()
            engine.switchModel(to: m)
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                        .font(.system(size: 11)).foregroundStyle(selected ? Color.accentColor : .secondary)
                    Text(m.label).font(.callout.weight(.medium))
                }
                Text(m.detail).font(.caption2).foregroundStyle(.secondary)
                if !runnable {
                    Text("needs ~\(Int(m.neededGB)) GB free — close some apps")
                        .font(.caption2).foregroundStyle(.orange)
                }
            }
            .padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 9)
                .fill(selected ? Color.accentColor.opacity(0.12) : Color.gray.opacity(0.08)))
            .overlay(RoundedRectangle(cornerRadius: 9)
                .stroke(selected ? Color.accentColor.opacity(0.5) : Color.gray.opacity(0.2)))
            .opacity(runnable ? 1 : 0.5)
        }
        .buttonStyle(.plain)
        .disabled(!runnable)
    }

    var chatBody: some View {
        HSplitView {
            if !sidebarCollapsed {
                sidebar
                    .frame(minWidth: 200, idealWidth: 480)
                    // Auto-collapse ONLY while the user is shrinking the pane (w decreasing through
                    // ~280pt). Reopen inserts the pane small and grows it — growth must never
                    // re-trigger collapse (that was the reopen "bounce" glitch). No withAnimation
                    // on the structural change: HSplitView renders half-laid-out mid-animation.
                    .background(GeometryReader { g in
                        Color.clear.onChange(of: g.size.width) { _, w in
                            if w > 0, w < 280, w < lastSidebarW { sidebarCollapsed = true }
                            if w > 0 { lastSidebarW = w }
                        }
                    })
            }
            chatPane
                .frame(minWidth: 240)
        }
        .sheet(item: $saveTarget) { t in
            SavePromptSheet(kind: t.rawValue) { name in
                let blocks = t == .system ? store.chat.system : reminderDraft
                promptLib.save(SavedPrompt(name: name, kind: t.rawValue, blocks: blocks))
            }
        }
        // Chat-screen lifecycle: the session owns the voice/capture poll loop, driven by `.task`
        // (runs on appear, auto-cancels on disappear). Config/context posts and the per-chat reset
        // are the session's own subscriptions now; the view only tracks its own draft/edit state.
        .onAppear { reminderDraft = store.chat.reminder }
        .task { await session.pollLoop() }
        .task { await session.pollProgress() }        // real cache-progress poll (drives the isolated HUD)
        .onChange(of: store.chat.id, initial: true) { _, _ in
            reminderDraft = store.chat.reminder
            editingChatName = false
            // Panes stay OPEN on chat open (image decode is off-main via AsyncBlockImage, so an open
            // pinned-context pane no longer threatens the open path; the old auto-collapse read as a bug).
            systemOpen = true
            contextOpen = true
        }
    }

    // ---------- left: collapsible SYSTEM / CONTEXT / REMINDER panes (share height) ----------
    var sidebar: some View {
        @Bindable var store = store   // @Observable → local bindable shadow for $store.chat.* pane bindings
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button { sidebarCollapsed = true } label: {
                    Image(systemName: "sidebar.left")
                }.help("Collapse sidebar")
                Button("⌂ Home") { store.goHome() }
                Spacer()
                gearButton
            }
            chatNameField
            pane("SYSTEM PROMPT", $systemOpen, $store.chat.system, key: "system") { systemPromptMenu }
            pane("PINNED CONTEXT", $contextOpen, $store.chat.context, key: "context")

            HStack(spacing: 8) {
                Button(session.caching ? "Caching…" : "Cache") { session.cache() }
                    .disabled(!session.canCache)
                if session.progBusy { CacheProgressBar(session: session) }
                Text(cacheStatusText(session.cacheState)).font(.caption).foregroundStyle(cacheStatusIsError ? .red : .secondary)
                Spacer()
            }
            HStack(spacing: 6) {
                Text("cache above").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(estLabel).font(.caption2).foregroundStyle(session.estOverLimit ? .red : .secondary)
            }
            Text(engine.status).font(.caption2).foregroundStyle(engine.ready ? .green : .secondary)

            // REMINDER edits a DRAFT — "Update reminder" applies draft → active chat.reminder (send() always uses ACTIVE).
            // The header "Load" menu applies Default / a saved reminder DIRECTLY (explicit user action, not a stray edit).
            pane("REMINDER — rides with each question, not pinned", $reminderOpen, $reminderDraft, key: "reminder") { reminderMenu }
            if reminderOpen {
                let remEst = ChatSession.estBlockTokens(reminderDraft)
                let remOver = remEst > 10_000
                HStack(spacing: 10) {
                    Button("Update reminder") { applyReminder(reminderDraft) }
                        .disabled(!reminderDirty || remOver)
                    if remOver { Text("est. ~\(remEst) tok — over the 10K reminder limit").font(.caption2).foregroundStyle(.red) }
                    else if reminderConfirm { Text("reminder updated ✓").font(.caption2).foregroundStyle(.green) }
                    else if reminderDirty { Text("edited — not applied").font(.caption2).foregroundStyle(.orange) }
                    Spacer()
                    Text("est. ~\(remEst) tok").font(.caption2).foregroundStyle(remOver ? .red : .secondary)
                }
            }
        }
        .padding(12)
    }

    // slim reopen affordance seated at the chat's leading edge (modes-bar row) while the sidebar is collapsed
    var reopenTab: some View {
        Button { lastSidebarW = 0; sidebarCollapsed = false } label: {
            Image(systemName: "sidebar.left").font(.system(size: 13, weight: .medium)).foregroundStyle(.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6)
        }.buttonStyle(.plain).help("Show sidebar")
    }

    // Chat-screen rename: click the name → inline TextField, commit on Enter or blur. Rename ONLY (no delete here).
    var chatNameField: some View {
        HStack(spacing: 6) {
            if editingChatName {
                TextField("chat name", text: $chatNameDraft).textFieldStyle(.roundedBorder).focused($chatNameFocused)
                    .onSubmit { commitChatName() }
                    .onChange(of: chatNameFocused) { _, f in if !f { commitChatName() } }   // commit on blur
                    .onAppear { chatNameFocused = true }
            } else {
                Button { chatNameDraft = store.chat.name; editingChatName = true } label: {
                    HStack(spacing: 5) {
                        Text(store.chat.name.isEmpty ? "(unnamed chat)" : store.chat.name).font(.callout.weight(.semibold))
                        Image(systemName: "pencil").font(.system(size: 10)).foregroundStyle(.secondary)
                    }.contentShape(Rectangle())
                }.buttonStyle(.plain).help("Rename this chat")
                Spacer()
            }
        }
    }
    func commitChatName() {
        let n = chatNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !n.isEmpty { store.chat.name = n; store.save() }
        editingChatName = false
    }

    // SYSTEM pane header menu — load a saved system prompt (replaces chat.system with a copy) or save the current one.
    var systemPromptMenu: some View {
        Menu("Load") {
            let sys = promptLib.prompts.filter { $0.kind == "system" }
            if sys.isEmpty { Text("no saved system prompts") }
            ForEach(sys) { p in Button(p.name) { store.chat.system = freshCopy(p.blocks) } }
            Divider()
            Button("Save current as…") { saveTarget = .system }
        }.menuStyle(.borderlessButton).font(.caption2).fixedSize()
    }

    // REMINDER pane header menu — saved reminders apply DIRECTLY; plus save-current, and clear.
    var reminderMenu: some View {
        Menu("Load") {
            Button("Clear (no reminder)") { applyReminder([Block(text: "")]) }
            let rems = promptLib.prompts.filter { $0.kind == "reminder" }
            if !rems.isEmpty {
                Divider()
                ForEach(rems) { p in Button(p.name) { applyReminder(freshCopy(p.blocks)) } }
            }
            Divider()
            Button("Save current as…") { saveTarget = .reminder }
        }.menuStyle(.borderlessButton).font(.caption2).fixedSize()
    }

    // ---------- right: modes bar + chat + input region (auto-scrolls to follow streaming) ----------
    var chatPane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                if sidebarCollapsed { reopenTab }   // reopen handle sits where the sidebar's collapse button was
                inputOutputBar
            }
            Divider()
            CacheHUDView(session: session, engine: engine, store: store)   // isolated: progress ticks re-render only this
            // The transcript is now a single WKWebView (markdown-it/KaTeX/highlight, all off the main
            // thread). `.id(store.chat.id)` recreates it (fresh web process) per chat and tears down the old
            // one → bounded memory, no beachball. Reads session.streaming/busy so the live reply updates —
            // cheap now, since the heavy render happens in the web process, not here.
            // Messages + composer both live in this one WebView (so ↑/↓ moves seamlessly between them);
            // it owns all conversation keyboard handling in JS. No SwiftUI keyMonitor → no double-handling.
            ChatWebView(session: session,
                        messagesJSON: conversationJSON(store.chat.messages),
                        convStart: session.convStart ?? -1)
                .id(store.chat.id)
                // (No bottom "caching…" chip — the top cache HUD is the single status line; a second
                //  indicator on the same `precache` flag just read as "two loaders for the same thing".)
            Divider()
            chatMetaBar.padding(.horizontal, 12).padding(.vertical, 6)   // token counts + reminder — OUTSIDE the WebView
        }
        .overlay(alignment: .top) { copiedToast }
    }

    // Transient "Copied" confirmation — shown briefly after any copy (C shortcut, ⋯ Copy, right-click).
    @ViewBuilder var copiedToast: some View {
        if session.justCopied {
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text("Copied").font(.caption.weight(.medium))
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().stroke(.secondary.opacity(0.2)))
            .padding(.top, 8)
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    // (cache pipeline HUD + progress bar moved to CacheUI.swift as isolated leaf views — CacheHUDView /
    // CacheProgressBar — so their real-time progress updates never re-evaluate this monolithic RootView.)

    // Boundary between turns that are OUT of the conversation cache (above) and in-context (below).
    // Position is engine-determined (session.convStart); hidden until known (nil on a fresh open).
    var contextBoundaryLine: some View {
        HStack(spacing: 8) {
            VStack { Divider() }
            Text("earlier turns are out of context").font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary).fixedSize().tracking(0.5)
            VStack { Divider() }
        }.padding(.vertical, 2).opacity(0.75)
    }

    @ViewBuilder var precacheChip: some View {
        if !session.precache.isEmpty {
            let done = session.precache == "done"
            HStack(spacing: 5) {
                if done { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                else { ProgressView().controlSize(.mini) }
                Text(done ? "cache ready" : "caching most recent…").font(.caption2)
            }
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().stroke(.secondary.opacity(0.2)))
            .foregroundStyle(.secondary)
            .transition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: session.precache)
        }
    }

    func scrollDown(_ proxy: ScrollViewProxy, animated: Bool = true) {
        if animated { withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(chatBottomID, anchor: .bottom) } }
        else { proxy.scrollTo(chatBottomID, anchor: .bottom) }
    }

    // fresh Blocks (new ids) so loading a library prompt copies its content without aliasing the saved prompt
    func freshCopy(_ blocks: [Block]) -> [Block] {
        blocks.map { $0.type == "image" ? Block(mediaType: $0.mediaType ?? "image/png", data: $0.data ?? "") : Block(text: $0.text ?? "") }
    }

    // New chat: snapshot the defaults (done in store.startNew) + fill the chosen DEFAULT system/reminder
    // prompts from the library. A deleted default prompt just resolves to nil → that pane stays blank.
    func newChat() {
        store.startNew()
        if let sys = defaultPrompt(SK.defaultSystemPrompt, kind: "system") { store.chat.system = freshCopy(sys.blocks) }
        if let rem = defaultPrompt(SK.defaultReminderPrompt, kind: "reminder") { store.chat.reminder = freshCopy(rem.blocks) }
    }
    func defaultPrompt(_ key: String, kind: String) -> SavedPrompt? {
        guard let idS = UserDefaults.standard.string(forKey: key), let id = UUID(uuidString: idS) else { return nil }
        return promptLib.prompts.first { $0.id == id && $0.kind == kind }
    }
    // Apply a reminder from the library (or Default) directly: draft + active chat.reminder, then persist.
    func applyReminder(_ blocks: [Block]) {
        reminderDraft = blocks
        session.commitReminder(blocks); confirmReminder()
    }

    // ---- input / output bar (replaces the old 4-way mode tabs) ----
    // TWO halves: YOUR INPUT (one hybrid box — a voice toggle arms the ⌃⌥ chord on top of typing, with
    // a ⋯ popover for submode / auto-send / mute) and AGENT OUTPUT (Text; Voice coming soon).
    @ViewBuilder var inputOutputBar: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("YOUR INPUT").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                HStack(spacing: 8) {
                    Toggle(isOn: Binding(get: { cs.voiceInput }, set: { session.setVoiceInput($0) })) {
                        HStack(spacing: 4) {
                            Image(systemName: "mic.fill").font(.system(size: 10))
                            Text("Voice").font(.caption.weight(.semibold))
                        }
                    }.toggleStyle(.switch).controlSize(.mini)
                    Button { inputOptsOpen.toggle() } label: {
                        Image(systemName: "ellipsis.circle").font(.system(size: 13))
                    }.buttonStyle(.plain).foregroundStyle(.secondary)
                    .popover(isPresented: $inputOptsOpen, arrowEdge: .bottom) { inputOptionsPopover }
                    if cs.voiceInput {
                        Text(submode == .toggle ? "press \(hk) to talk" : "hold \(hk) to talk")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            Divider().frame(height: 30)
            VStack(alignment: .leading, spacing: 4) {
                Text("VLM OUTPUT").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                Picker("", selection: Binding(get: { cs.agentOutputV },
                                              set: { if !$0.comingSoon { session.setAgentOutput($0) } })) {
                    ForEach(AgentOutput.allCases, id: \.self) { o in
                        Text(o.comingSoon ? "\(o.label) (soon)" : o.label).tag(o)
                    }
                }.pickerStyle(.segmented).labelsHidden().controlSize(.small).frame(width: 150)
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    // The "Your Input" sub-options: dictation submode, what a finished dictation does, and mute-while-talking.
    var inputOptionsPopover: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("DICTATION").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                Picker("", selection: Binding(get: { submode }, set: { if $0 != .vad { session.setSubmode($0) } })) {
                    Text("Press \(hk) to interrupt & talk, again when finished").tag(Submode.toggle)
                    Text("Hold \(hk) to talk").tag(Submode.hold)
                    Text("VAD (coming soon)").tag(Submode.vad)
                }.pickerStyle(.radioGroup).labelsHidden().disabled(!cs.voiceInput)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("AUTO-SEND WHEN DICTATION ENDS").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                Picker("", selection: Binding(get: { cs.autoSendV }, set: { session.setAutoSend($0) })) {
                    ForEach(AutoSend.allCases, id: \.self) { Text($0.label).tag($0) }
                }.pickerStyle(.radioGroup).labelsHidden()
                Text(cs.autoSendV.blurb)
                    .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Toggle("Mute system audio while dictating", isOn: Binding(get: { cs.muteDictation }, set: { session.setMuteDictation($0) })).controlSize(.small)
            Divider()
            VStack(alignment: .leading, spacing: 4) {
                Text("TRANSCRIPTION").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                Picker("", selection: Binding(get: { cs.transcriptionV }, set: { session.setTranscription($0) })) {
                    Text("Transcribe after finished — Parakeet, most accurate").tag(Transcription.after)
                    Text("Streaming — Apple on-device, live as you speak").tag(Transcription.stream)
                }.pickerStyle(.radioGroup).labelsHidden().disabled(!cs.voiceInput)
                HStack(spacing: 6) {
                    Text("Streaming language").font(.caption2).foregroundStyle(.secondary)
                    Picker("", selection: Binding(get: { cs.speechLocale }, set: { session.setSpeechLocale($0) })) {
                        ForEach(speechLocales, id: \.self) { Text($0).tag($0) }
                    }.labelsHidden().pickerStyle(.menu).frame(width: 120)
                    .disabled(!cs.voiceInput || cs.transcriptionV != .stream)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 5) {
                Text("CONVERSATION CACHE").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
                // LIVE readout — what's happening right now
                HStack(spacing: 5) {
                    Image(systemName: "gauge.with.dots.needle.33percent").font(.system(size: 10)).foregroundStyle(Color.accentColor)
                    if let ct = session.convTokens {
                        Text("\(ct) / \(session.trimTrigger) tok in cache").font(.caption2.weight(.medium))
                        if let cvs = session.convStart, cvs > 0 {
                            Text("· \(cvs) turn\(cvs == 1 ? "" : "s") dropped").font(.caption2).foregroundStyle(.orange)
                        }
                    } else { Text("bounded to \(session.trimTrigger) tok").font(.caption2).foregroundStyle(.secondary) }
                }
                HStack(spacing: 10) {
                    Stepper("Trim at \(gCacheTrigger)", value: Binding(get: { gCacheTrigger }, set: { session.setCacheTrigger($0) }),
                            in: 200...SK.cacheTriggerCap, step: 200)
                    Stepper("to \(gCacheTarget)", value: Binding(get: { gCacheTarget }, set: { session.setCacheTarget($0) }),
                            in: 100...max(101, gCacheTrigger - 1), step: 100)
                }.font(.caption2).controlSize(.mini)
                Label("Ongoing: oldest turns re-roped out instantly (no recompute).", systemImage: "bolt.fill")
                    .font(.system(size: 9)).foregroundStyle(.green).labelStyle(.titleAndIcon)
                HStack(spacing: 6) {
                    Text("On reopen:").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary)
                    Picker("", selection: Binding(get: { RecacheMode(rawValue: gRecacheMode) ?? .recent }, set: { session.setRecacheMode($0) })) {
                        ForEach(RecacheMode.allCases, id: \.self) { Text($0.label).tag($0) }
                    }.pickerStyle(.segmented).labelsHidden().controlSize(.mini).frame(width: 190)
                }
                Text((RecacheMode(rawValue: gRecacheMode) ?? .recent).blurb).font(.system(size: 9)).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14).frame(width: 340)
        .task {
            var locs = await AppleSpeech.supportedLocaleIDs()
            if !locs.contains(cs.speechLocale) { locs.insert(cs.speechLocale, at: 0) }
            speechLocales = locs
        }
    }

    // ---- input region (ONE hybrid box: type, or dictate at the caret when voice is armed) ----
    var inputRegion: some View {
        @Bindable var session = session   // @Observable → local bindable shadow for $session.input
        return VStack(spacing: 6) {
            if !session.pastedImages.isEmpty { thumbnailBar }
            // Voice adornments: status dot + (while dictating) the live streaming partial preview.
            if cs.voiceInput {
                HStack(spacing: 6) {
                    if let c = voiceDotColor { Circle().fill(c).frame(width: 7, height: 7) }
                    if session.dictating {
                        Text(submode == .toggle ? "listening — press \(hk) again to insert" : "listening — release \(hk) to insert")
                            .font(.caption2).foregroundStyle(.red)
                    } else if session.busy {
                        Text("· talk to interrupt").font(.caption2).foregroundStyle(.orange)
                    }
                    Spacer()
                }
                if session.dictating, transcription == .stream, !session.livePartial.isEmpty {
                    ScrollView {
                        Text(session.livePartial).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 60).padding(6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.08)))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.2)))
                }
            }
            HStack(alignment: .bottom) {
                // Multi-line: vertical axis grows 1…14 lines. Enter sends; Shift+Enter inserts a newline
                // (handled in the keyDown monitor, since a vertical TextField makes Return a newline).
                // Locked (uneditable) while dictating — the transcript lands at the caret on completion.
                TextField(inputFocused ? (session.pinned ? "ask about the pinned context…" : "ask anything…")
                                       : (cs.voiceInput ? "press Space to type · \(submode == .toggle ? "press" : "hold") \(hk) to talk"
                                                        : "press Space to type…"),
                          text: $session.input, axis: .vertical)
                    .textFieldStyle(.roundedBorder).lineLimit(1...14)   // grows to 14 lines, then scrolls
                    .disabled(!session.canCompose || session.dictating)
                    .focused($inputFocused)
                if session.busy {
                    Button { session.stop() } label: { Image(systemName: "stop.fill") }
                        .help("Stop generating")
                }
                Button(session.busy ? "Interrupt & Ask" : "Ask") { session.send() }
                    .disabled(!session.canSend)
            }
            chatMetaBar
        }
        .padding(10)
        .onPasteCommand(of: [.image, .png, .tiff]) { _ in pasteChatImages() }
    }

    // Slack-like keyboard model. Focus is owned by EITHER the input OR one selected agent message.
    // The monitor returns nil to consume an event, or the event to let it flow to the focused field.
    func keyMonitor(_ e: NSEvent) -> NSEvent? {
        // A modal sheet (Settings / save-chat) owns the keyboard: don't let the chat's single-key
        // shortcuts hijack keys the sheet needs — e.g. a hotkey/binding recorder, or plain typing —
        // especially when a chat message is still selected behind it (s/c/r would be consumed).
        if showSettings || saveTarget != nil { return e }
        let focused = session.inputIsFocused
        let editing = isTextEditing()
        // ⌘V with an image on the clipboard, input focused → stage the image (field editor eats paste).
        if e.modifierFlags.contains(.command), e.charactersIgnoringModifiers == "v", focused,
           NSPasteboard.general.availableType(from: [.png, .tiff]) != nil {
            pasteChatImages(); return nil
        }
        switch e.keyCode {
        case 36, 76:                                   // Return / keypad Enter
            if focused {
                if e.modifierFlags.contains(.shift) {  // Shift+Enter → insert a newline at the caret
                    if let tv = NSApp.keyWindow?.firstResponder as? NSTextView {
                        tv.insertText("\n", replacementRange: tv.selectedRange())
                    }
                    return nil                         // consume so SwiftUI doesn't treat it as a submit
                }
                session.send(); return nil             // plain Enter → send
            }
            return e
        case 126:                                      // Up
            if focused { if chatCaretOnFirstLine() { session.navUp(); return nil }; return e }
            if editing { return e }                    // a block/reminder editor owns the caret
            session.navUp(); return nil                // none/selected → older (or select newest)
        case 125:                                      // Down
            if focused || editing { return e }
            session.navDown(); return nil              // newer, or past-newest → input
        case 49:                                       // Space → jump to input (when not editing text)
            if editing { return e }
            session.focusInput(); return nil
        case 53:                                        // Esc → drop the message-nav selection
            if session.selectedMessageID != nil { session.clearSelection(); return nil }
            return e
        default: break
        }
        // Single-key actions on the selected message (only when no text field owns the caret).
        if let sel = session.selectedMessageID, !editing {
            switch e.charactersIgnoringModifiers {
            case "s": session.toggleSource(sel); return nil
            case "c": copyText(session.messageBody(sel)); return nil
            case "r": session.resetToHere(sel); return nil
            default: break
            }
        }
        return e
    }

    // Is a text field (any NSText field editor) the window's first responder? (chat input, block panes,
    // reminder, rename) — used to NOT hijack single-key shortcuts while the user is typing.
    func isTextEditing() -> Bool {
        (NSApp.keyWindow?.firstResponder as? NSView)?.isKind(of: NSText.self) ?? false
    }
    // Is the chat input's caret on its first visual line? (so Up navigates to messages only from the top)
    func chatCaretOnFirstLine() -> Bool {
        guard let tv = NSApp.keyWindow?.firstResponder as? NSTextView else { return !session.input.contains("\n") }
        let loc = tv.selectedRange().location
        let ns = tv.string as NSString
        if loc <= 0 || loc > ns.length { return true }
        return ns.range(of: "\n", options: [], range: NSRange(location: 0, length: loc)).location == NSNotFound
    }

    var voiceDotColor: Color? {
        switch session.voiceState {
        case .listening: return .red
        case .processing: return .yellow
        default: return nil            // ready/idle → no dot (matches the screen overlay)
        }
    }

    var thumbnailBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(session.pastedImages) { b in
                    ZStack(alignment: .topTrailing) {
                        AsyncBlockImage(block: b) { $0.resizable().scaledToFill() }
                            .frame(width: 54, height: 54)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                        Button { session.removePastedImage(id: b.id) } label: {
                            Image(systemName: "xmark.circle.fill").font(.system(size: 14))
                        }.buttonStyle(.plain).foregroundStyle(.white, .black.opacity(0.55)).padding(2)
                    }
                }
            }.padding(.vertical, 2)
        }
        .frame(height: 60).frame(maxWidth: .infinity, alignment: .leading)
    }

    // A collapsible pane: header toggles it; when open its BlockStream flexes to share sidebar height.
    // `accessory` renders a trailing header control (e.g. a Load menu) that clicks independently of the toggle.
    @ViewBuilder func pane<A: View>(_ title: String, _ open: Binding<Bool>, _ blocks: Binding<[Block]>,
                                    key: String? = nil, @ViewBuilder accessory: () -> A = { EmptyView() }) -> some View {
        HStack(spacing: 4) {
            Button { withAnimation(.easeInOut(duration: 0.22)) { open.wrappedValue.toggle() } } label: {
                HStack(spacing: 4) {
                    Image(systemName: open.wrappedValue ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .bold))
                    Text(title).font(.caption.bold())
                }.foregroundStyle(.secondary).contentShape(Rectangle())
            }.buttonStyle(.plain)
            Spacer()
            accessory()
        }
        if open.wrappedValue { BlockStream(blocks: blocks, jsonKey: key) }
    }

    @ViewBuilder func messageBubble(id: UUID? = nil, role: String, text: String, interrupted: Bool = false,
                                    isInterruption: Bool = false, images: [Block] = []) -> some View {
        let parts = text.components(separatedBy: "@@APPENDIX@@")
        let body = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let appendix = parts.count > 1 ? Appendix.parse(parts[1]) : nil
        let isUser = role == "user"
        // amber tint for a mid-generation interruption; accent for a normal user turn; grey for the agent.
        let bg: Color = isUser ? (isInterruption ? Color.orange.opacity(0.20) : Color.accentColor.opacity(0.18))
                               : Color.gray.opacity(0.14)
        HStack(alignment: .top) {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 8) {
                if isInterruption {
                    Text("INTERRUPTED TO ADD").font(.system(size: 8, weight: .bold)).foregroundStyle(.orange).tracking(0.5)
                }
                if !images.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(images.filter { $0.type == "image" }) { b in
                                AsyncBlockImage(block: b) { $0.resizable().scaledToFit() }
                                    .frame(maxHeight: 120)
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                    }
                }
                if !body.isEmpty {
                    // User text is its own source (highlight → copy = source). Agent text is rendered by
                    // Textual (CoreText), whose partial selection can't map back to markdown — so a
                    // per-message "View source" toggle swaps in the raw markdown as selectable monospace
                    // text (highlight any part → copy = exact source). Right-click also copies full source.
                    let showSrc = id.map { session.sourceShownIDs.contains($0) } ?? false
                    Group {
                        if isUser || showSrc {
                            Text(body).font(showSrc ? .system(size: 12, design: .monospaced) : .body)
                        } else {
                            MarkdownText(raw: body).equatable()   // skip Textual re-parse when body is unchanged
                        }
                    }
                    .contextMenu {
                        Button { copyText(body) } label: { Label("Copy source (markdown)", systemImage: "chevron.left.forwardslash.chevron.right") }
                        if let id, !isUser {
                            Button { session.toggleSource(id) } label: {
                                Label(showSrc ? "View rendered" : "View source", systemImage: showSrc ? "eye" : "curlybraces")
                            }
                        }
                    }
                }
                if let a = appendix { AppendixView(a) }              // distinct, collapsible footnote
                if interrupted { Text("⎯ interrupted").font(.caption2).foregroundStyle(.secondary) }
            }
            .textSelection(.enabled).padding(10)
            .background(bg)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            // keyboard-nav highlight: the currently selected agent message gets an accent ring.
            .overlay(RoundedRectangle(cornerRadius: 12)
                .stroke(Color.accentColor, lineWidth: (id != nil && id == session.selectedMessageID) ? 2 : 0))
            // triple-dot on a committed AGENT reply: copy it, or reset the chat back to it.
            if !isUser, let id { bubbleMenu(id: id, body: body) }
            if !isUser { Spacer(minLength: 40) }
        }
    }

    @ViewBuilder func bubbleMenu(id: UUID, body: String) -> some View {
        let src = session.sourceShownIDs.contains(id)
        Menu {
            Button { copyText(body) } label: { Label("Copy  ·  C", systemImage: "doc.on.doc") }
            Button { session.toggleSource(id) } label: {
                Label((src ? "View rendered" : "View source") + "  ·  S", systemImage: src ? "eye" : "curlybraces")
            }
            Button(role: .destructive) { session.resetToHere(id) } label: {
                Label("Reset to here  ·  R", systemImage: "arrow.uturn.backward")
            }
            Divider()
            Section("Keyboard") {
                Text("↑ ↓   move between replies")
                Text("Space   jump to input")
                Text("S source · C copy · R reset")
            }
        } label: {
            Image(systemName: "ellipsis").font(.caption).foregroundStyle(.secondary)
                .frame(width: 22, height: 22).contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
    }

    func copyText(_ s: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        withAnimation(.easeOut(duration: 0.15)) { session.flashCopied() }
    }

    // ---- chat-input image paste (⌘V) → the session's pasted-images staging bar ----
    // (Clipboard intake is a UI concern; it hands finished Blocks to the session, which owns them.)
    func pasteChatImages() {
        if let blk = clipboardImageBlock() { session.attachPastedImages([blk]) }
    }

    func confirmReminder() {
        reminderConfirm = true
        Task { try? await Task.sleep(for: .seconds(2)); reminderConfirm = false }
    }
}
