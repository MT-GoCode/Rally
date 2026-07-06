import SwiftUI
import AppKit
import UniformTypeIdentifiers
import AVFoundation
import ApplicationServices
import IOKit.hid

// One-shot TCC prompts for everything the voice-agent future needs. Safe to re-run —
// macOS only shows each dialog while the permission is undecided.
func requestAgentPerms() {
    AVCaptureDevice.requestAccess(for: .audio) { _ in }                                   // Microphone
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
    func open(_ id: UUID) -> Bool {
        guard let d = try? Data(contentsOf: dir.appendingPathComponent("\(id).json")),
              let c = try? JSONDecoder().decode(Chat.self, from: d) else { return false }
        chat = c; screen = .chat; return true
    }
    func goHome() { save(); refreshStubs(); screen = .home }
    func startNew() { chat = Chat(); screen = .chat }

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
        chat = c; screen = .chat; save()
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

@main struct CIVMApp: App {
    @StateObject var engine = Engine()
    @StateObject var store = Store()
    @StateObject var promptLib = PromptLib()
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
                    RootView().environmentObject(engine).environmentObject(store).environmentObject(promptLib)
                        .onAppear {
                            engine.start()
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
                    ForEach($blocks) { $b in row($b) }
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
        let pb = NSPasteboard.general
        if let img = NSImage(pasteboard: pb), let tiff = img.tiffRepresentation, let (b64, mt) = imageToBase64(tiff) {
            blocks.append(Block(mediaType: mt, data: b64)); return
        }
        for t in [NSPasteboard.PasteboardType.png, .tiff] {
            if let d = pb.data(forType: t), let (b64, mt) = imageToBase64(d) { blocks.append(Block(mediaType: mt, data: b64)); return }
        }
        if let s = pb.string(forType: .string), !s.isEmpty { blocks.append(Block(text: s)) }
    }

    var textChars: Int { blocks.reduce(0) { $0 + ($1.text?.count ?? 0) } }

    @ViewBuilder func row(_ b: Binding<Block>) -> some View {
        HStack(alignment: .top, spacing: 6) {
            if b.wrappedValue.type == "text" {
                TextField("text…", text: Binding(get: { b.wrappedValue.text ?? "" }, set: { b.wrappedValue.text = $0 }), axis: .vertical)
                    .textFieldStyle(.plain).font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.primary).lineLimit(1...)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if let ns = b.wrappedValue.nsImage {
                Image(nsImage: ns).resizable().scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 220, alignment: .leading)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
            }
            Button { blocks.removeAll { $0.id == b.wrappedValue.id } } label: {
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
    @AppStorage(SK.sidebarCollapsed) private var sidebarCollapsed = false
    @State private var lastSidebarW: CGFloat = 0   // direction tracking for auto-collapse (shrink-only)
    @State private var lastReused = 0              // /chat meta "reused" — streaming-prefill proof
    @State private var prevHadOutput = false       // did the turn we just cancelled produce any text?
    @State private var saveTarget: SaveTarget? = nil     // chat sidebar "Save current as…" sheet
    private let chatBottomID = "civm.chatBottom"          // stable auto-scroll anchor id
    @State private var caching = false
    @State private var cacheMsg = ""
    @State private var cacheProgress: Double = 0      // client-side estimated cache progress (0…1), 0 = hidden
    @State private var editingChatName = false        // chat-screen inline rename
    @State private var chatNameDraft = ""
    @FocusState private var chatNameFocused: Bool
    @State private var input = ""
    @State private var busy = false
    @State private var streaming = ""
    @State private var systemOpen = true
    @State private var contextOpen = true
    @State private var reminderOpen = true

    // ---- modes / voice settings (persisted; shared with the Settings sheet) ----
    @AppStorage(SK.mode) private var modeRaw = Mode.textText.rawValue
    @AppStorage(SK.submode) private var submodeRaw = Submode.toggle.rawValue
    @AppStorage(SK.transcription) private var transcriptionRaw = Transcription.after.rawValue
    @AppStorage(SK.hotkey) private var hotkey = SK.defaultHotkey
    // capture shortcuts (screenshot + copy-to-chat) — bindings serialized as engine strings
    @AppStorage(SK.shotBinding) private var shotBinding = SK.defaultShotBinding
    @AppStorage(SK.shotStyle) private var shotStyleRaw = SK.defaultShotStyle
    @AppStorage(SK.copyBinding) private var copyBinding = SK.defaultCopyBinding
    var mode: Mode { Mode(rawValue: modeRaw) ?? .textText }
    var submode: Submode { Submode(rawValue: submodeRaw) ?? .toggle }
    var transcription: Transcription { Transcription(rawValue: transcriptionRaw) ?? .after }
    var hk: String { hotkeySymbols(hotkey) }

    @State private var showSettings = false
    @State private var queuedCaptures: [Block] = []    // voice-mode captures (images + text) attached to the NEXT ask
    @State private var pastedImages: [Block] = []      // chat-input pasted images (thumbnail bar → ride with the question)
    @State private var pasteMonitor: Any? = nil         // ⌘V image-paste interceptor (see textInputRegion)
    @FocusState private var inputFocused: Bool
    @State private var reminderDraft: [Block] = []     // staged reminder edits; applied to chat.reminder on "Update reminder"
    @State private var reminderConfirm = false
    // AskPipeline (mode-agnostic): one in-flight turn; interrupts cancel it and stack a new one.
    @State private var askTask: Task<Void, Never>? = nil
    // Voice·Text live state (from GET /voice/poll)
    @State private var voiceState = "idle"
    @State private var voiceSeq = -1
    @State private var livePartial = ""

    // engine currently holds a pin for THIS chat (maybe a stale version — that still allows chatting)
    var pinned: Bool { store.enginePinnedChat == store.chat.id }
    // …and it's the CURRENT version → Cache greys out
    var cachedCurrent: Bool { pinned && store.enginePinnedHash == store.chat.contentHash }
    var contentEmpty: Bool { nonEmpty(store.chat.system).isEmpty && nonEmpty(store.chat.context).isEmpty }
    var cacheStatus: String {
        if !cacheMsg.isEmpty { return cacheMsg }
        if caching { return "caching…" }
        if contentEmpty { return "nothing to cache — just ask" }
        if cachedCurrent { return "cached ✓ · \(store.chat.pinnedTokens ?? 0) tok" }
        return pinned ? "cache changed" : "not cached"
    }
    var reminderDirty: Bool { reminderDraft != store.chat.reminder }

    // ---- cheap client-side token estimate (recomputed on every edit) — pre-gates Cache ----
    // Calibration: Sipser seed = 18,474 text chars + 20 images → 14,244 real tokens (≈2.1 chars/tok
    // on dense technical text). Tuned to slightly OVERestimate — the gate's job is to block BEFORE
    // the engine errs, so erring high is correct; the engine's exact 200K check stays as backstop.
    var cacheBlocks: [Block] { nonEmpty(store.chat.system) + nonEmpty(store.chat.context) }
    var estTokens: Int {
        let chars = cacheBlocks.reduce(0) { $0 + ($1.type == "text" ? ($1.text?.count ?? 0) : 0) }
        let images = cacheBlocks.filter { $0.type == "image" }.count
        return Int((Double(chars) / 2.2).rounded(.up)) + 300 * images
    }
    // disable a bit below the real 200K ceiling to leave slack for estimate error (engine overLimit is the backstop)
    var estOverLimit: Bool { estTokens > 190_000 }
    var estLabel: String {
        if contentEmpty { return "nothing to cache" }
        let n = estTokens.formatted()
        return estOverLimit ? "est. ~\(n) tok — over the 200K limit" : "est. ~\(n) tok"
    }
    // rough expected cache time (s) for the estimated progress bar — no engine progress endpoint
    var expectedCacheSeconds: Double {
        let images = cacheBlocks.filter { $0.type == "image" }.count
        return Double(estTokens) / 350.0 + Double(images) * 0.7 + 3.0
    }

    // "{pinned} pinned + {chat} chat = {total} tok" — from /pin and the last /chat done meta.
    // "⚡N pre-fed" appears after a streamed dictation: N tokens were already in the KV at send.
    var tokenLine: some View {
        let p = store.chat.pinnedTokens ?? 0, c = store.chat.chatTokens
        return Text("\(p) pinned + \(c) chat = \(p + c) tok\(lastReused > 0 ? "  ·  ⚡\(lastReused) pre-fed while you spoke" : "")")
            .font(.caption2).foregroundStyle(.secondary)
    }

    var body: some View {
        Group {
            if store.screen == .home { homeBody } else { chatBody }
        }
        .sheet(isPresented: $showSettings) { SettingsView() }
        // Voice config lifecycle that must outlive the chat screen: launch, engine-ready, and
        // entering/leaving a chat (voice only enables in a Voice·Text chat — never on home).
        .onAppear { if engine.ready { postVoiceConfig() } }
        .onChange(of: engine.ready) { _, r in if r { postVoiceConfig() } }
        .onChange(of: store.screen) { _, _ in postVoiceConfig() }
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
                                onOpen: { if store.open(s.id) { cache() } },   // opening evicts the old KV, re-pins this chat
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
        .onAppear { reminderDraft = store.chat.reminder }
        .onChange(of: store.chat.id) { _, _ in
            reminderDraft = store.chat.reminder; voiceSeq = -1; livePartial = ""; queuedCaptures = []
            editingChatName = false
        }
        .onChange(of: modeRaw) { _, _ in postVoiceConfig(); if mode == .voiceText { Task { await postVoiceContext() } } }
        .onChange(of: submodeRaw) { _, _ in postVoiceConfig() }
        .onChange(of: transcriptionRaw) { _, _ in postVoiceConfig() }
        .onChange(of: hotkey) { _, _ in postVoiceConfig() }
        .onChange(of: shotBinding) { _, _ in postVoiceConfig() }
        .onChange(of: shotStyleRaw) { _, _ in postVoiceConfig() }
        .onChange(of: copyBinding) { _, _ in postVoiceConfig() }
        .onChange(of: store.chat.messages.count) { _, _ in if mode == .voiceText { Task { await postVoiceContext() } } }
        // Poll GET /voice/poll ~10Hz while a chat is open in ANY mode: capture events are delivered in
        // every mode; voice state (partial/final auto-send) is consumed only in Voice·Text.
        .task(id: modeRaw) {
            if mode == .voiceText { await postVoiceContext() }
            while !Task.isCancelled {
                if let p = try? await engine.voicePoll() { await handleVoicePoll(p) }
                try? await Task.sleep(for: .milliseconds(100))
            }
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
                Button(caching ? "Caching…" : "Cache") { cache() }
                    .disabled(caching || !engine.ready || cachedCurrent || contentEmpty || estOverLimit)
                if caching || cacheProgress > 0 { cacheProgressBar }
                Text(cacheStatus).font(.caption).foregroundStyle(cacheStatus.contains("over") || cacheStatus.contains("fail") ? .red : .secondary)
                Spacer()
            }
            HStack(spacing: 6) {
                Text("cache above").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(estLabel).font(.caption2).foregroundStyle(estOverLimit ? .red : .secondary)
            }
            Text(engine.status).font(.caption2).foregroundStyle(engine.ready ? .green : .secondary)

            // REMINDER edits a DRAFT — "Update reminder" applies draft → active chat.reminder (send() always uses ACTIVE).
            // The header "Load" menu applies Default / a saved reminder DIRECTLY (explicit user action, not a stray edit).
            pane("REMINDER — rides with each question, not pinned", $reminderOpen, $reminderDraft, key: "reminder") { reminderMenu }
            if reminderOpen {
                HStack(spacing: 10) {
                    Button("Update reminder") {
                        store.chat.reminder = reminderDraft; store.save(); confirmReminder()
                        if mode == .voiceText { Task { await postVoiceContext() } }
                    }
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
            Capsule().fill(Color.accentColor).frame(width: 120 * max(0, min(cacheProgress, 1)), height: 4)
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
                            messageBubble(role: m.role, text: m.text, interrupted: m.interrupted, isInterruption: m.isInterruption, images: m.content)
                        }
                        if busy { messageBubble(role: "assistant", text: streaming.isEmpty ? "…" : streaming) }
                        Color.clear.frame(height: 1).id(chatBottomID)      // stable bottom anchor for auto-scroll
                    }.padding(14).frame(maxWidth: .infinity, alignment: .leading)
                }
                .onAppear { scrollDown(proxy, animated: false) }
                .onChange(of: store.chat.id) { _, _ in scrollDown(proxy, animated: false) }
                .onChange(of: store.chat.messages.count) { _, _ in scrollDown(proxy) }
                .onChange(of: streaming.count / 40) { _, _ in scrollDown(proxy, animated: false) }   // throttle: ~every 40 streamed chars
            }
            Divider()
            if mode == .voiceText { voiceInputRegion } else { textInputRegion }
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
        store.chat.reminder = blocks
        store.save(); confirmReminder()
        if mode == .voiceText { Task { await postVoiceContext() } }
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
            if !pastedImages.isEmpty { thumbnailBar }
            HStack {
                TextField(pinned ? "ask about the pinned context…" : "ask anything…", text: $input)
                    .textFieldStyle(.roundedBorder).disabled(!engine.ready).onSubmit { send() }
                    .focused($inputFocused)
                Button(busy ? "Interrupt & Ask" : "Ask") { send() }
                    .disabled(!engine.ready || (input.isEmpty && pastedImages.isEmpty))
            }
            tokenLine.frame(maxWidth: .infinity, alignment: .leading)
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
            if !queuedCaptures.isEmpty { queuedStrip }
            HStack(spacing: 8) {
                // dot mirrors the screen overlay: ONLY red (listening) / yellow (processing), else gone
                if let c = voiceDotColor { Circle().fill(c).frame(width: 8, height: 8) }
                Text(submode == .toggle ? "Press \(hk) to interrupt & talk, again when finished · Esc cancels"
                                        : "Hold \(hk) to talk · Esc cancels")
                    .font(.caption).foregroundStyle(.secondary)
                if busy { Text("· agent is answering — talk to interrupt").font(.caption2).foregroundStyle(.orange) }
            }
            if transcription == .stream {
                ScrollView {
                    Text(livePartial.isEmpty ? "…" : livePartial)
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
        switch voiceState {
        case "listening": return .red
        case "processing": return .yellow
        default: return nil            // ready/idle → no dot (matches the screen overlay)
        }
    }

    var thumbnailBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(pastedImages) { b in
                    ZStack(alignment: .topTrailing) {
                        if let ns = b.nsImage {
                            Image(nsImage: ns).resizable().scaledToFill().frame(width: 54, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                        }
                        Button { pastedImages.removeAll { $0.id == b.id } } label: {
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
                ForEach(queuedCaptures) { b in
                    ZStack(alignment: .topTrailing) {
                        if b.type == "image", let ns = b.nsImage {
                            Image(nsImage: ns).resizable().scaledToFill().frame(width: 48, height: 48)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.25)))
                        } else {
                            Text(b.text ?? "").font(.caption2).lineLimit(2).frame(maxWidth: 140, alignment: .leading)
                                .padding(6).background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.14)))
                        }
                        Button { queuedCaptures.removeAll { $0.id == b.id } } label: {
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

    @ViewBuilder func messageBubble(role: String, text: String, interrupted: Bool = false,
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
            if !isUser { Spacer(minLength: 40) }
        }
    }

    func cache() { Task { await pinNow() } }
    // pin (or re-pin) this chat's system+context into the engine's single KV slot
    func pinNow() async {
        guard engine.ready else { cacheMsg = "engine not ready yet"; return }
        let id = store.chat.id, hash = store.chat.contentHash
        if contentEmpty {
            // nothing to pin: reset the engine to its EMPTY baseline (instant — no vision, no prefill)
            // so a leftover pin from another chat can't leak into this one.
            await engine.reset()
            store.chat.pinnedTokens = 0
            store.enginePinnedChat = id; store.enginePinnedHash = hash
            return
        }
        caching = true; cacheMsg = ""; cacheProgress = 0
        // estimated progress: fill toward 0.97 over the expected duration while caching (no engine endpoint)
        let start = Date(); let expected = max(0.5, expectedCacheSeconds)
        let ticker = Task { @MainActor in
            while !Task.isCancelled && caching {
                cacheProgress = min(Date().timeIntervalSince(start) / expected, 0.97)
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
        let sys = nonEmpty(store.chat.system), ctx = nonEmpty(store.chat.context)
        do {
            let (tokens, over) = try await engine.pin(system: sys, context: ctx)
            if over { cacheMsg = "context is \(tokens) tok — over the 200K limit"; store.chat.pinnedTokens = nil }
            else {
                store.chat.pinnedTokens = tokens
                store.enginePinnedChat = id; store.enginePinnedHash = hash
                store.save()
            }
        } catch { cacheMsg = "cache failed: \(error.localizedDescription)" }
        caching = false
        ticker.cancel()
        if cacheMsg.isEmpty {                       // success → snap full briefly, then hide
            cacheProgress = 1.0
            Task { @MainActor in try? await Task.sleep(for: .milliseconds(350)); if !caching { cacheProgress = 0 } }
        } else {
            cacheProgress = 0
        }
    }
    // Typed input → AskPipeline. Fires an interruption when busy, a normal ask otherwise.
    func send() {
        let q = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let imgs = pastedImages
        guard !q.isEmpty || !imgs.isEmpty else { return }
        input = ""; pastedImages = []
        ask(text: q, images: imgs)
    }

    // ------------------------------------------------------------------
    // AskPipeline — mode-agnostic. Typed input AND voice transcripts flow through here.
    // History is app-owned: each turn POSTs the whole clean transcript (last 24) + reminder.
    // Interrupt: if a turn is in flight, cancel it, keep its partial in an assistant bubble
    // flagged `interrupted`, then append the new user message (amber; engine text prefixed
    // "@@INTERRUPTION@@: ") and start a fresh turn. Stackable — each interrupt repeats this.
    // ------------------------------------------------------------------
    func ask(text: String, images: [Block] = []) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!q.isEmpty || !images.isEmpty), engine.ready else { return }
        let prev = askTask                      // the in-flight turn (if any) that we're interrupting
        let interrupting = busy
        askTask = Task { await runTurn(q, images: images, interrupting: interrupting, prev: prev) }
    }

    private func runTurn(_ q: String, images: [Block], interrupting: Bool, prev: Task<Void, Never>?) async {
        if interrupting {
            prevHadOutput = false               // the cancelled turn sets this if it actually produced text
            prev?.cancel()                      // stop the in-flight /chat …
            await prev?.value                   // … and let it stash its partial as an interrupted bubble
        }
        var um = Msg(role: "user", text: q, content: images)
        // Only a REAL interruption (the model had said something) gets the amber flag + engine prefix.
        // Cutting off a turn that never produced a token (e.g. still pinning) is just a normal ask.
        if interrupting && prevHadOutput { um.isInterruption = true }
        store.chat.messages.append(um)
        busy = true; streaming = ""
        if !pinned { await pinNow() }           // pin on demand (caching first isn't required)
        do {
            let meta = try await engine.chat(messages: buildEngineMessages(), reminder: store.chat.reminder) { streaming += $0 }
            if Task.isCancelled {               // interrupted right as the stream ended
                stashInterrupted(); return
            }
            store.chat.messages.append(Msg(role: "assistant", text: streaming))
            if let ct = meta["chat_tokens"] as? Int { store.chat.chatTokens = ct }
            if let pt = meta["pinned"] as? Int, pt > 0 { store.chat.pinnedTokens = pt }
            lastReused = meta["reused"] as? Int ?? 0   // streaming-prefill proof: tokens already in KV at send
            streaming = ""; busy = false; store.save()
        } catch {
            if Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled {
                stashInterrupted()              // the interrupting turn keeps busy=true
            } else {
                store.chat.messages.append(Msg(role: "assistant", text: "⚠ \(error.localizedDescription)"))
                streaming = ""; busy = false; store.save()
            }
        }
    }

    // Keep a cancelled turn's partial ONLY if it produced text — no phantom "— interrupted" bubbles.
    private func stashInterrupted() {
        prevHadOutput = !streaming.isEmpty
        if !streaming.isEmpty {
            store.chat.messages.append(Msg(role: "assistant", text: streaming, interrupted: true))
        }
        streaming = ""
    }

    // The clean transcript for the engine: last 24 messages, images as blocks, interruption prefix added HERE only.
    func buildEngineMessages() -> [ChatMessage] {
        store.chat.messages.suffix(24).map { m in
            var blocks: [Block] = []
            let t = m.isInterruption ? "@@INTERRUPTION@@: " + m.text : m.text
            if !t.isEmpty { blocks.append(Block(text: t)) }
            blocks.append(contentsOf: m.content)
            return ChatMessage(role: m.role, content: blocks)
        }
    }

    // ---- chat-input image paste (⌘V) → thumbnail bar above the input ----
    func pasteChatImages() {
        let pb = NSPasteboard.general
        if let img = NSImage(pasteboard: pb), let tiff = img.tiffRepresentation, let (b64, mt) = imageToBase64(tiff) {
            pastedImages.append(Block(mediaType: mt, data: b64)); return
        }
        for t in [NSPasteboard.PasteboardType.png, .tiff] {
            if let d = pb.data(forType: t), let (b64, mt) = imageToBase64(d) { pastedImages.append(Block(mediaType: mt, data: b64)); return }
        }
    }

    func confirmReminder() {
        reminderConfirm = true
        Task { try? await Task.sleep(for: .seconds(2)); reminderConfirm = false }
    }

    // ---- Voice·Text + capture control channel ----
    func postVoiceConfig() {
        // voice chord enables inside a Voice·Text chat only; capture bindings are active whenever a
        // chat is open in ANY mode — never on the home screen.
        let inChat = store.screen == .chat
        Task { await engine.voiceConfig(
            voiceEnabled: inChat && mode == .voiceText,
            captureEnabled: inChat,
            submode: submode.rawValue,
            streaming: transcription == .stream,
            key: hotkey,
            shotBinding: shotBinding, shotStyle: shotStyleRaw, copyBinding: copyBinding) }
    }
    func postVoiceContext() async {
        await engine.voiceContext(messages: buildEngineMessages(), reminder: store.chat.reminder)
    }
    func handleVoicePoll(_ p: VoicePoll) async {
        // capture events arrive in ALL modes — deliver, then ack exactly what we consumed (FIFO).
        if let caps = p.captures, !caps.isEmpty {
            deliverCaptures(caps)
            await engine.voiceCapturesAck(count: caps.count)
        }
        guard mode == .voiceText else { return }
        voiceState = p.state
        livePartial = p.partial
        // When a final transcript is ready, auto-send it (interrupts if busy), ack it once by seq.
        if p.state == "ready", p.seq != voiceSeq,
           let f = p.final, !f.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            voiceSeq = p.seq
            let (pre, imgs) = drainQueuedCaptures()   // fold queued screenshots/copies into this turn
            ask(text: pre.isEmpty ? f : pre + "\n" + f, images: imgs)
            Task { await engine.voiceAck(seq: p.seq) }
        }
    }

    // Deliver captures to the chat as if the user had pasted them. Text mode: images → thumbnail bar,
    // text → input box. Voice modes: both → the queued strip, attached to the next spoken message.
    func deliverCaptures(_ caps: [Capture]) {
        for c in caps {
            if c.kind == "image", let d = c.data, !d.isEmpty {
                let blk = Block(mediaType: "image/png", data: d)
                if mode == .voiceText { queuedCaptures.append(blk) } else { pastedImages.append(blk) }
            } else if c.kind == "text", let t = c.text, !t.isEmpty {
                if mode == .voiceText { queuedCaptures.append(Block(text: t)) }
                else { input += input.isEmpty ? t : " " + t }
            }
        }
    }
    // Pull queued captures for a voice send: (joined text prefix, image blocks); clears the queue.
    func drainQueuedCaptures() -> (String, [Block]) {
        let texts = queuedCaptures.compactMap { $0.type == "text" ? $0.text : nil }.filter { !$0.isEmpty }
        let imgs = queuedCaptures.filter { $0.type == "image" }
        queuedCaptures = []
        return (texts.joined(separator: "\n"), imgs)
    }
}
