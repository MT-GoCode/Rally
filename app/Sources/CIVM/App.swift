import SwiftUI
import AppKit
import Combine
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

// Editable "reminder" sent AFTER each question (in the 1024-tok sliding window → all layers see it).
// It is NOT pinned and NOT stored in the transcript, so editing it is free (no re-cache).
let defaultReminderText = """
——— REMINDER (obey this absolutely, over everything above) ———
Your name is Rally. You talk like a sharp, real human in live conversation — NEVER like an assistant writing an essay.
BREVITY IS LAW. Answer in the fewest possible words. If one word answers it (Yes / No / a name / a number), say ONE word. If one sentence answers it, ONE sentence. Two sentences is the normal maximum. No exceptions unless I explicitly ask you to go deep or produce an artifact.
High signal only: no preamble, no restating my question, no hedging, no filler, no "great question", no lists unless I ask.
Use EVERYTHING you know — your training knowledge AND the context above — whichever answers best.
Be helpful and sharp. If I'm wrong, say so plainly. Minimal.
This REMINDER is silent stage direction: NEVER respond to it, mention it, or acknowledge it. Respond ONLY to my message above, as if the reminder were invisible.
"""

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
    var reminder: [Block] = [Block(text: defaultReminderText)]  // text+image, sent after every question; not pinned

    init(name: String = "New chat") { self.name = name }
    // Lenient decode: every field falls back to its default, so chats saved by OLDER builds
    // (no reminder key, or reminder-as-String, no chatTokens) still open instead of silently failing.
    enum CodingKeys: String, CodingKey { case id, name, system, context, messages, pinnedTokens, chatTokens, reminder }
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

@MainActor final class Store: ObservableObject {
    enum Screen { case home, chat }
    // How the current chat was activated — the session listens on `activated` and reacts once per
    // activation (opened → re-pin; created/seeded → no pin), keeping today's per-path behavior.
    enum Activation { case opened, created, seeded }
    let activated = PassthroughSubject<Activation, Never>()
    @Published var screen: Screen = .home
    @Published var chat = Chat()
    @Published var stubs: [ChatStub] = []
    // what the engine's ONE global KV currently holds (chat + content version)
    @Published var enginePinnedChat: UUID? = nil
    @Published var enginePinnedHash: Int? = nil

    private var dir: URL {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("contextualized_instant_voice_models/chats")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    func save() {
        // don't litter home with pristine empty chats
        guard !(chat.messages.isEmpty && nonEmpty(chat.system).isEmpty && nonEmpty(chat.context).isEmpty) else { return }
        if let d = try? JSONEncoder().encode(chat) { try? d.write(to: dir.appendingPathComponent("\(chat.id).json")) }
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
        guard let d = try? Data(contentsOf: dir.appendingPathComponent("\(id).json")),
              let c = try? JSONDecoder().decode(Chat.self, from: d) else { return }
        chat = c; screen = .chat; activated.send(.opened)
    }
    func goHome() { save(); refreshStubs(); screen = .home }
    func startNew() { chat = Chat(); screen = .chat; activated.send(.created) }

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

    func loadSipserSeed() {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("code/contextualized_instant_voice_models/seed/sipser.json")
        if let d = try? Data(contentsOf: url) { loadData(d, name: "Sipser lecture") }
    }

    private func loadData(_ d: Data, name: String) {
        var c = Chat(name: name)
        if let j = try? JSONDecoder().decode([String: [Block]].self, from: d) {
            if let s = j["system"], !s.isEmpty { c.system = s }
            if let ctx = j["context"], !ctx.isEmpty { c.context = ctx }
            if let r = j["reminder"], !r.isEmpty { c.reminder = r }
        } else if let arr = try? JSONDecoder().decode([Block].self, from: d), !arr.isEmpty {
            c.context = arr          // a bare block array → load as the context
        } else { return }
        chat = c; screen = .chat; save(); activated.send(.seeded)
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
// @Published members, so it never fires objectWillChange; the three runtimes are still observed by
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
    init() { NSApplication.shared.setActivationPolicy(.regular); memOK = Mem.enough }
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
                    RootView().environmentObject(deps.engine).environmentObject(deps.store).environmentObject(deps.promptLib).environmentObject(deps.session)
                        .onAppear {
                            deps.engine.start()
                            if !UserDefaults.standard.bool(forKey: "permsRequested") {
                                UserDefaults.standard.set(true, forKey: "permsRequested")
                                requestAgentPerms()
                            }
                        }.frame(minWidth: 1040, minHeight: 680)
                }
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
                } else if let ns = blk.nsImage {
                    Image(nsImage: ns).resizable().scaledToFit()
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
    @EnvironmentObject var engine: Engine
    @EnvironmentObject var store: Store
    @EnvironmentObject var promptLib: PromptLib
    @EnvironmentObject var session: ChatSession   // conversation runtime — all behavior lives here
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

    // ---- modes / voice settings (persisted; shared with the Settings sheet) ----
    // The @AppStorage vars stay purely as render-invalidation triggers; the computed values delegate
    // to the shared accessors so the fallback logic lives in ONE place (Settings.swift).
    @AppStorage(SK.mode) private var modeRaw = Mode.textText.rawValue
    @AppStorage(SK.submode) private var submodeRaw = Submode.toggle.rawValue
    @AppStorage(SK.transcription) private var transcriptionRaw = Transcription.after.rawValue
    @AppStorage(SK.hotkey) private var hotkey = SK.defaultHotkey
    @AppStorage(SK.reminderMode) private var reminderModeRaw = SK.defaultReminderMode
    var mode: Mode { .from(modeRaw) }
    var submode: Submode { .from(submodeRaw) }
    var transcription: Transcription { .from(transcriptionRaw) }
    var hk: String { hotkeySymbols(hotkey) }

    @State private var showSettings = false
    @State private var pasteMonitor: Any? = nil         // ⌘V image-paste interceptor (see textInputRegion)
    @FocusState private var inputFocused: Bool
    @State private var reminderDraft: [Block] = []     // staged reminder edits; applied to chat.reminder on "Update reminder"
    @State private var reminderConfirm = false

    var reminderDirty: Bool { reminderDraft != store.chat.reminder }

    // ---- cache status / estimate wording (the view owns the strings; the session owns the state) ----
    var cacheStatusText: String {
        switch session.cacheState {
        case .idle:                return "not cached"
        case .nothingToCache:      return "nothing to cache — just ask"
        case .caching:             return "caching…"
        case .cached(let t):       return "cached ✓ · \(t) tok"
        case .changed:             return "cache changed"
        case .overLimit(let t):    return "context is \(t) tok — over the 200K limit"
        case .failed(let m):       return "cache failed: \(m)"
        case .notReady:            return "engine not ready yet"
        }
    }
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
                    reminderModeRaw = m.rawValue
                    session.setReminderMode(m)
                } label: {
                    if ReminderMode(rawValue: reminderModeRaw) == m { Label(m.label, systemImage: "checkmark") }
                    else { Text(m.label) }
                }
            }
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "text.insert").font(.system(size: 9))
                Text("reminder: \(ReminderMode(rawValue: reminderModeRaw)?.label ?? "")").font(.caption2)
            }.foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton).fixedSize().menuIndicator(.hidden)
    }

    var tokenLine: some View {
        let p = store.chat.pinnedTokens ?? 0, c = store.chat.chatTokens
        let cached = session.lastPinned + session.lastReused
        return VStack(alignment: .leading, spacing: 1) {
            Text("\(p) pinned + \(c) chat = \(p + c) tok\(session.preSent > 0 ? "  ·  ⚡\(session.preSent) pre-sent" : "")")
            if session.lastNew > 0 || session.lastTtft > 0 {
                Text("last msg: \(cached) read from cache · \(session.lastNew) processed anew · TTFT \(String(format: "%.2f", session.lastTtft))s")
                    .foregroundStyle(session.lastTtft > 1.5 ? .orange : .secondary)
                if !session.anewParts.isEmpty {
                    Text("  anew = " + session.anewParts.map { "\($0.n) \($0.label)" }.joined(separator: " + "))
                }
            }
        }.font(.caption2).foregroundStyle(.secondary)
    }

    var body: some View {
        Group {
            if store.screen == .home { homeBody } else { chatBody }
        }
        .sheet(isPresented: $showSettings) { SettingsView() }
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
                HStack {
                    Button("＋ New chat") { store.startNew() }   // chat-able immediately (engine holds an empty baseline; first ask resets the pin)
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

    var chatBody: some View {
        HSplitView {
            if !sidebarCollapsed {
                sidebar
                    .frame(minWidth: 240, idealWidth: 480)
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
                .frame(minWidth: 520)
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
        .onChange(of: store.chat.id) { _, _ in
            reminderDraft = store.chat.reminder
            editingChatName = false
        }
    }

    // ---------- left: collapsible SYSTEM / CONTEXT / REMINDER panes (share height) ----------
    var sidebar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button { sidebarCollapsed = true } label: {
                    Image(systemName: "sidebar.left")
                }.help("Collapse sidebar")
                Button("⌂ Home") { store.goHome() }
                Button("Load Sipser seed") { store.loadSipserSeed() }
                Spacer()
                gearButton
            }
            chatNameField
            pane("SYSTEM PROMPT", $systemOpen, $store.chat.system, key: "system") { systemPromptMenu }
            pane("CONTEXT", $contextOpen, $store.chat.context, key: "context")

            HStack(spacing: 8) {
                Button(session.caching ? "Caching…" : "Cache") { session.cache() }
                    .disabled(!session.canCache)
                if session.caching || session.cacheProgress > 0 { cacheProgressBar }
                Text(cacheStatusText).font(.caption).foregroundStyle(cacheStatusIsError ? .red : .secondary)
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
                HStack(spacing: 10) {
                    Button("Update reminder") { applyReminder(reminderDraft) }
                        .disabled(!reminderDirty)
                    if reminderConfirm { Text("reminder updated ✓").font(.caption2).foregroundStyle(.green) }
                    else if reminderDirty { Text("edited — not applied").font(.caption2).foregroundStyle(.orange) }
                    Spacer()
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

    // Minimal estimated cache-progress bar: a thin filling capsule (no percentage text).
    var cacheProgressBar: some View {
        ZStack(alignment: .leading) {
            Capsule().fill(Color.secondary.opacity(0.20)).frame(width: 120, height: 4)
            Capsule().fill(Color.accentColor).frame(width: 120 * max(0, min(session.cacheProgress, 1)), height: 4)
        }.frame(width: 120, height: 4)
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

    // REMINDER pane header menu — Default + saved reminders apply DIRECTLY; plus save-current. (Replaces the old reset button.)
    var reminderMenu: some View {
        Menu("Load") {
            Button("Default") { applyReminder([Block(text: defaultReminderText)]) }
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
                modesBar
            }
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(store.chat.messages) { m in
                            messageBubble(id: m.id, role: m.role, text: m.text, interrupted: m.interrupted, isInterruption: m.isInterruption, images: m.content)
                        }
                        if session.busy { messageBubble(role: "assistant", text: session.streaming.isEmpty ? "…" : session.streaming) }
                        Color.clear.frame(height: 1).id(chatBottomID)      // stable bottom anchor for auto-scroll
                    }.padding(14).frame(maxWidth: .infinity, alignment: .leading)
                }
                .onAppear { scrollDown(proxy, animated: false) }
                .onChange(of: store.chat.id) { _, _ in scrollDown(proxy, animated: false) }
                .onChange(of: store.chat.messages.count) { _, _ in scrollDown(proxy) }
                .onChange(of: session.streaming.count / 40) { _, _ in scrollDown(proxy, animated: false) }   // throttle: ~every 40 streamed chars
                .overlay(alignment: .bottomTrailing) { precacheChip.padding(10) }
            }
            Divider()
            if mode == .voiceText { voiceInputRegion } else { textInputRegion }
        }
    }

    // Bottom-right status: the engine warming the KV for the next message after each reply.
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
    // Apply a reminder from the library (or Default) directly: draft + active chat.reminder, then persist.
    func applyReminder(_ blocks: [Block]) {
        reminderDraft = blocks
        session.commitReminder(blocks); confirmReminder()
    }

    // ---- modes bar ----
    @ViewBuilder var modesBar: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("YOUR INPUT · AGENT OUTPUT").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(1)
            HStack(spacing: 6) {
                modeButton(.textText, "Text · Text", caption: nil, enabled: true)
                modeButton(.textVoice, "Text · Voice", caption: "(Coming Soon)", enabled: false)
                modeButton(.voiceText, "Voice · Text", caption: nil, enabled: true)
                modeButton(.voiceVoice, "Voice · Voice", caption: "(Coming Soon)", enabled: false)
            }
            if mode == .voiceText { submodeMenu }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    func modeButton(_ m: Mode, _ title: String, caption: String?, enabled: Bool) -> some View {
        let on = mode == m
        return Button { if enabled { modeRaw = m.rawValue } } label: {
            VStack(spacing: 1) {
                Text(title).font(.caption.weight(.semibold))
                if let caption { Text(caption).font(.system(size: 8)) }
            }
            .foregroundStyle(on ? Color.accentColor : (enabled ? .primary : .secondary))
            .padding(.vertical, 5).padding(.horizontal, 8).frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 7).fill(on ? Color.accentColor.opacity(0.18) : Color.gray.opacity(0.10)))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(on ? Color.accentColor.opacity(0.6) : .clear, lineWidth: 1))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain).disabled(!enabled).opacity(enabled ? 1 : 0.55)
    }

    var submodeMenu: some View {
        Menu {
            Button((submode == .toggle ? "✓ " : "") + "press \(hk) to interrupt & talk, again when finished") { submodeRaw = Submode.toggle.rawValue }
            Button((submode == .hold ? "✓ " : "") + "hold \(hk) to talk") { submodeRaw = Submode.hold.rawValue }
            Button("VAD (Coming Soon)") {}.disabled(true)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "waveform").font(.system(size: 10))
                Text(submode == .toggle ? "press \(hk) to interrupt & talk, again when finished" : "hold \(hk) to talk").font(.caption)
                Image(systemName: "chevron.down").font(.system(size: 8))
            }.foregroundStyle(.secondary)
        }.menuStyle(.borderlessButton).fixedSize()
    }

    // ---- input regions ----
    var textInputRegion: some View {
        VStack(spacing: 6) {
            if !session.pastedImages.isEmpty { thumbnailBar }
            HStack {
                TextField(session.pinned ? "ask about the pinned context…" : "ask anything…", text: $session.input)
                    .textFieldStyle(.roundedBorder).disabled(!session.canCompose).onSubmit { session.send() }
                    .focused($inputFocused)
                Button(session.busy ? "Interrupt & Ask" : "Ask") { session.send() }
                    .disabled(!session.canSend)
            }
            HStack(alignment: .bottom) {
                tokenLine.frame(maxWidth: .infinity, alignment: .leading)
                reminderModeControl
            }
        }
        .padding(10)
        .onPasteCommand(of: [.image, .png, .tiff]) { _ in pasteChatImages() }
        // A focused TextField's field editor swallows ⌘V (and silently drops image-only content),
        // so onPasteCommand never fires. Intercept ⌘V ahead of key equivalents: image on the
        // clipboard + chat input focused → thumbnail; anything else passes through untouched
        // (plain text paste, the BlockStream panes' own paste handling).
        .onAppear {
            guard pasteMonitor == nil else { return }
            pasteMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { e in
                if e.modifierFlags.contains(.command), e.charactersIgnoringModifiers == "v",
                   inputFocused,
                   NSPasteboard.general.availableType(from: [.png, .tiff]) != nil {
                    pasteChatImages(); return nil
                }
                return e
            }
        }
        .onDisappear { if let m = pasteMonitor { NSEvent.removeMonitor(m); pasteMonitor = nil } }
    }

    var voiceInputRegion: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !session.queuedCaptures.isEmpty { queuedStrip }
            HStack(spacing: 8) {
                // dot mirrors the screen overlay: ONLY red (listening) / yellow (processing), else gone
                if let c = voiceDotColor { Circle().fill(c).frame(width: 8, height: 8) }
                Text(submode == .toggle ? "Press \(hk) to interrupt & talk, again when finished · Esc cancels"
                                        : "Hold \(hk) to talk · Esc cancels")
                    .font(.caption).foregroundStyle(.secondary)
                if session.busy { Text("· agent is answering — talk to interrupt").font(.caption2).foregroundStyle(.orange) }
            }
            if transcription == .stream {
                ScrollView {
                    Text(session.livePartial.isEmpty ? "…" : session.livePartial)
                        .font(.system(size: 13)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 84).padding(8)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.08)))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
            }
            tokenLine
        }
        .padding(10).frame(maxWidth: .infinity, alignment: .leading)
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
                        if let ns = b.nsImage {
                            Image(nsImage: ns).resizable().scaledToFill().frame(width: 54, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                        }
                        Button { session.removePastedImage(id: b.id) } label: {
                            Image(systemName: "xmark.circle.fill").font(.system(size: 14))
                        }.buttonStyle(.plain).foregroundStyle(.white, .black.opacity(0.55)).padding(2)
                    }
                }
            }.padding(.vertical, 2)
        }
        .frame(height: 60).frame(maxWidth: .infinity, alignment: .leading)
    }

    // Voice-mode captures queued for the next spoken message: image thumbnails + text chips, each removable.
    var queuedStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Text("attach to next:").font(.caption2).foregroundStyle(.secondary)
                ForEach(session.queuedCaptures) { b in
                    ZStack(alignment: .topTrailing) {
                        if b.type == "image", let ns = b.nsImage {
                            Image(nsImage: ns).resizable().scaledToFill().frame(width: 48, height: 48)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                        } else {
                            Text(b.text ?? "").font(.caption2).lineLimit(2).frame(maxWidth: 140, alignment: .leading)
                                .padding(6).background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.14)))
                        }
                        Button { session.removeQueuedCapture(id: b.id) } label: {
                            Image(systemName: "xmark.circle.fill").font(.system(size: 13))
                        }.buttonStyle(.plain).foregroundStyle(.white, .black.opacity(0.55)).padding(1)
                    }
                }
            }.padding(.vertical, 2)
        }
        .frame(height: 56).frame(maxWidth: .infinity, alignment: .leading)
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
                            ForEach(images) { b in
                                if let ns = b.nsImage {
                                    Image(nsImage: ns).resizable().scaledToFit().frame(maxHeight: 120)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                            }
                        }
                    }
                }
                if !body.isEmpty {
                    if isUser { Text(body) } else { MarkdownText(raw: body) }   // render markdown live
                }
                if let a = appendix { AppendixView(a) }              // distinct, collapsible footnote
                if interrupted { Text("⎯ interrupted").font(.caption2).foregroundStyle(.secondary) }
            }
            .textSelection(.enabled).padding(10)
            .background(bg)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            // triple-dot on a committed AGENT reply: copy it, or reset the chat back to it.
            if !isUser, let id { bubbleMenu(id: id, body: body) }
            if !isUser { Spacer(minLength: 40) }
        }
    }

    @ViewBuilder func bubbleMenu(id: UUID, body: String) -> some View {
        Menu {
            Button { copyText(body) } label: { Label("Copy", systemImage: "doc.on.doc") }
            Button(role: .destructive) { session.resetToHere(id) } label: {
                Label("Reset to here", systemImage: "arrow.uturn.backward")
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
