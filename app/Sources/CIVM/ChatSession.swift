import Foundation
import Combine

// ---------------------------------------------------------------------------
// ChatSession — the conversation RUNTIME, completely walled off from the UI.
//
// Owns every behavior: the ask/interrupt pipeline, pin/cache state, the
// Voice·Text poll loop + capture consumption, and token accounting. It talks
// ONLY to Engine (the HTTP client) and Store (persistence) — never to any View.
// Views render its published state and forward intents through its API
// (ask / send / cache / pollLoop / commitReminder); no URLSession calls,
// pipeline logic, or timers live in the views.
//
// It OBSERVES its own inputs rather than being poked by the views: engine
// readiness (→ post config), the current screen (→ post config), the SK
// settings keys via UserDefaults.didChangeNotification (→ refresh stored
// copies + post config/context), and chat activation (→ reset transient voice
// state, re-pin, re-post context). That kills the ~dozen view-side relays that
// used to forward these changes by hand.
// ---------------------------------------------------------------------------

// Semantic cache status — the view owns the exact wording/colour; this owns the state (fix B).
enum CacheState: Equatable {
    case idle                    // "not cached"
    case nothingToCache          // "nothing to cache — just ask"
    case caching                 // "caching…"
    case cached(tokens: Int)     // "cached ✓ · N tok"
    case changed                 // "cache changed"
    case overLimit(tokens: Int)  // "context is N tok — over the 200K limit"  (red)
    case failed(String)          // "cache failed: …"                          (red)
    case notReady                // "engine not ready yet"
}

// Semantic Voice·Text state mapped from the wire string at the boundary (fix B).
enum VoiceState: Equatable {
    case idle, listening, processing, ready
    init(wire: String) {
        switch wire {
        case "listening": self = .listening
        case "processing": self = .processing
        case "ready": self = .ready
        default: self = .idle
        }
    }
}

@MainActor final class ChatSession: ObservableObject {
    private let engine: Engine
    private let store: Store
    private var cancellables = Set<AnyCancellable>()

    init(engine: Engine, store: Store) {
        self.engine = engine
        self.store = store

        // ---- observe own inputs (fix A) — replaces the view's onChange relays ----
        // engine readiness: post config once it flips ready (mirrors onChange(engine.ready){ if r }).
        // receive(on:) defers the sink past @Published's willSet so it reads the SETTLED engine.ready
        // (symmetry with the screen sink); the emitted `ready` it fires on — rises to true, falls to
        // false — is unchanged.
        engine.$ready.removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] ready in MainActor.assumeIsolated {
                guard let self else { return }
                if ready {
                    if self.pinOutcome == .notReady { self.pinOutcome = .none }   // engine up → drop the stale "not ready" gate
                    self.postVoiceConfig()
                }
            } }
            .store(in: &cancellables)
        // screen changes: post config on every enter/leave (mirrors onChange(store.screen)). dropFirst
        // skips the initial .home emission so launch doesn't fire a stray config post. receive(on:)
        // defers the sink past @Published's willSet so postVoiceConfig reads the SETTLED store.screen
        // (mirrors onChange, which fires after the value lands — not the stale pre-assignment value).
        store.$screen.dropFirst().removeDuplicates()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in MainActor.assumeIsolated { self?.postVoiceConfig() } }
            .store(in: &cancellables)
        // settings writes (SK keys, from the Settings sheet / modes bar): refresh stored copies and
        // post config; a MODE change additionally re-posts context (mirrors onChange(modeRaw)).
        NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in MainActor.assumeIsolated { self?.settingsChanged() } }
            .store(in: &cancellables)
        // chat activation (open / new / seed): the single path for reset + re-pin + re-post context.
        store.activated
            .sink { [weak self] kind in MainActor.assumeIsolated { self?.onChatActivated(kind) } }
            .store(in: &cancellables)
    }

    // ---- in-flight turn (ask pipeline) ----
    @Published private(set) var busy = false
    @Published private(set) var streaming = ""
    @Published private(set) var lastReused = 0   // /chat meta "reused" — streaming-prefill proof
    private var prevHadOutput = false             // did the turn we just cancelled produce any text?
    // AskPipeline (mode-agnostic): one in-flight turn; interrupts cancel it and stack a new one.
    private var askTask: Task<Void, Never>? = nil

    // ---- pin / cache ----
    @Published private(set) var caching = false
    @Published private(set) var cacheProgress: Double = 0   // client-side estimated cache progress (0…1), 0 = hidden
    // sticky outcome of the last pin attempt (cleared when a fresh non-empty pin starts) — supersedes
    // the old free-form cacheMsg string; CacheState is composed from it plus the derived flags.
    private enum PinOutcome: Equatable { case none, notReady, overLimit(Int), failed(String) }
    @Published private var pinOutcome: PinOutcome = .none

    // ---- compose buffer + capture staging (fix E — intent methods mutate; views only read) ----
    @Published var input = ""                             // typed chat input (bound two-way to the TextField)
    @Published private(set) var pastedImages: [Block] = []   // chat-input pasted images (thumbnail bar → ride with the question)
    @Published private(set) var queuedCaptures: [Block] = [] // voice-mode captures (images + text) attached to the NEXT ask

    // ---- Voice·Text live state (from GET /voice/poll) ----
    @Published private(set) var voiceState: VoiceState = .idle
    @Published private(set) var livePartial = ""
    private var voiceSeq = -1

    // ---- settings: STORED copies refreshed from the shared accessors on didChangeNotification
    // (fix C/E). No per-render UserDefaults lookups — the old computed getters were a hot loop. ----
    private var mode: Mode = .current
    private var submode: Submode = .current
    private var transcription: Transcription = .current
    private var hotkey: String = SK.hotkeyValue
    private var shotBinding: String = SK.shotBindingValue
    private var shotStyleRaw: String = SK.shotStyleValue
    private var copyBinding: String = SK.copyBindingValue

    // ---- cache gate / status (derive from store.chat + caching state) ----
    // engine currently holds a pin for THIS chat (maybe a stale version — that still allows chatting)
    var pinned: Bool { store.enginePinnedChat == store.chat.id }
    // …and it's the CURRENT version → Cache greys out
    var cachedCurrent: Bool { pinned && store.enginePinnedHash == store.chat.contentHash }
    var contentEmpty: Bool { nonEmpty(store.chat.system).isEmpty && nonEmpty(store.chat.context).isEmpty }
    // Semantic status the sidebar renders (was the cacheStatus string + its "over"/"fail" colour sniff).
    var cacheState: CacheState {
        switch pinOutcome {
        case .notReady:        return .notReady
        case .overLimit(let t): return .overLimit(tokens: t)
        case .failed(let m):   return .failed(m)
        case .none:            break
        }
        if caching { return .caching }
        if contentEmpty { return .nothingToCache }
        if cachedCurrent { return .cached(tokens: store.chat.pinnedTokens ?? 0) }
        return pinned ? .changed : .idle
    }

    // ---- cheap client-side token estimate (pre-gates Cache) — a plain per-probe scan of the cache
    // blocks; no memo (hashing the content to cache the result cost as much as the scan itself). ----
    // Calibration: Sipser seed = 18,474 text chars + 20 images → 14,244 real tokens (≈2.1 chars/tok
    // on dense technical text). Tuned to slightly OVERestimate — the gate's job is to block BEFORE
    // the engine errs, so erring high is correct; the engine's exact 200K check stays as backstop.
    private var cacheBlocks: [Block] { nonEmpty(store.chat.system) + nonEmpty(store.chat.context) }
    var estTokens: Int {
        let blocks = cacheBlocks
        let chars = blocks.reduce(0) { $0 + ($1.type == "text" ? ($1.text?.count ?? 0) : 0) }
        let images = blocks.filter { $0.type == "image" }.count
        return Int((Double(chars) / 2.2).rounded(.up)) + 300 * images
    }
    // disable a bit below the real 200K ceiling to leave slack for estimate error (engine overLimit is the backstop)
    var estOverLimit: Bool { estTokens > 190_000 }
    // rough expected cache time (s) for the estimated progress bar — no engine progress endpoint
    private var expectedCacheSeconds: Double {
        let images = cacheBlocks.filter { $0.type == "image" }.count
        return Double(estTokens) / 350.0 + Double(images) * 0.7 + 3.0
    }

    // ---- enablement predicates the buttons bind to (fix E) — exactly today's disable expressions ----
    var canCache: Bool { !caching && engine.ready && !cachedCurrent && !contentEmpty && !estOverLimit }
    // NOTE: intentionally uses the UNTRIMMED input (input.isEmpty), matching today's Ask gate.
    var canSend: Bool { engine.ready && !(input.isEmpty && pastedImages.isEmpty) }
    // compose gate: the chat TextField enables as soon as the engine is ready (Ask still uses canSend).
    var canCompose: Bool { engine.ready }

    // ------------------------------------------------------------------
    // Cache — pin (or re-pin) this chat's system+context into the engine's single KV slot.
    // ------------------------------------------------------------------
    private var pinTask: Task<Void, Never>? = nil
    func cache() { Task { await self.pinNow() } }
    // Serialize pin attempts: if one is already in flight, coalesce onto it instead of firing a second
    // overlapping /pin (an on-demand Ask landing during a manual Cache would otherwise race two tickers
    // and deliver out-of-order outcomes).
    private func pinNow() async {
        if let t = pinTask { await t.value; return }
        let t = Task { await self.runPin() }
        pinTask = t
        await t.value
        pinTask = nil
    }
    private func runPin() async {
        guard engine.ready else { pinOutcome = .notReady; return }
        let id = store.chat.id, hash = store.chat.contentHash
        if contentEmpty {
            // nothing to pin: reset the engine to its EMPTY baseline (instant — no vision, no prefill)
            // so a leftover pin from another chat can't leak into this one.
            pinOutcome = .none          // clear any stale failed/overLimit — there's nothing to cache now
            await engine.reset()
            store.chat.pinnedTokens = 0
            store.enginePinnedChat = id; store.enginePinnedHash = hash
            return
        }
        caching = true; pinOutcome = .none; cacheProgress = 0
        // estimated progress: fill toward 0.97 over the expected duration while caching (no engine endpoint)
        let start = Date(); let expected = max(0.5, expectedCacheSeconds)
        let ticker = Task { [self] in
            while !Task.isCancelled && caching {
                let v = min(Date().timeIntervalSince(start) / expected, 0.97)
                if v != cacheProgress { cacheProgress = v }   // skip identical writes (no 12.5Hz invalidation once clamped at 0.97)
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
        let sys = nonEmpty(store.chat.system), ctx = nonEmpty(store.chat.context)
        do {
            let (tokens, over) = try await engine.pin(system: sys, context: ctx)
            if over { pinOutcome = .overLimit(tokens); store.chat.pinnedTokens = nil }
            else {
                store.chat.pinnedTokens = tokens
                store.enginePinnedChat = id; store.enginePinnedHash = hash
                store.save()
            }
        } catch { pinOutcome = .failed(error.localizedDescription) }
        caching = false
        ticker.cancel()
        if pinOutcome == .none {                     // success → snap full briefly, then hide
            cacheProgress = 1.0
            Task { [self] in try? await Task.sleep(for: .milliseconds(350)); if !caching { cacheProgress = 0 } }
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
    private func ask(text: String, images: [Block] = []) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!q.isEmpty || !images.isEmpty), engine.ready else { return }
        let prev = askTask                      // the in-flight turn (if any) that we're interrupting
        let interrupting = busy
        askTask = Task { await self.runTurn(q, images: images, interrupting: interrupting, prev: prev) }
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
        repostContext()                         // transcript changed → re-stage for the next voice turn (SPEC: post whenever it changes)
        busy = true; streaming = ""
        if !pinned { await pinNow() }           // pin on demand (caching first isn't required)
        do {
            let meta = try await engine.chat(messages: buildEngineMessages(), reminder: store.chat.reminder) { self.streaming += $0 }
            if Task.isCancelled {               // interrupted right as the stream ended
                stashInterrupted(); return
            }
            store.chat.messages.append(Msg(role: "assistant", text: streaming))
            if let ct = meta["chat_tokens"] as? Int { store.chat.chatTokens = ct }
            if let pt = meta["pinned"] as? Int, pt > 0 { store.chat.pinnedTokens = pt }
            lastReused = meta["reused"] as? Int ?? 0   // streaming-prefill proof: tokens already in KV at send
            streaming = ""; busy = false; store.save()
            repostContext()                     // transcript changed → re-stage for the next voice turn
        } catch {
            if Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled {
                stashInterrupted()              // the interrupting turn keeps busy=true
            } else {
                store.chat.messages.append(Msg(role: "assistant", text: "⚠ \(error.localizedDescription)"))
                streaming = ""; busy = false; store.save()
                repostContext()
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
        // no repost here: every stash is immediately followed by the successor turn's append + repost.
    }

    // The clean transcript for the engine: last 24 messages, images as blocks, interruption prefix added HERE only.
    private func buildEngineMessages() -> [ChatMessage] {
        store.chat.messages.suffix(24).map { m in
            var blocks: [Block] = []
            let t = m.isInterruption ? "@@INTERRUPTION@@: " + m.text : m.text
            if !t.isEmpty { blocks.append(Block(text: t)) }
            blocks.append(contentsOf: m.content)
            return ChatMessage(role: m.role, content: blocks)
        }
    }

    // Apply a reminder (the pane draft, or a library/Default choice) to the active chat + persist,
    // then re-post the voice context if we're live in Voice·Text. (The pane draft + "updated ✓"
    // feedback stay in the view; this owns the store write and the engine re-post.)
    func commitReminder(_ blocks: [Block]) {
        store.chat.reminder = blocks; store.save()
        repostContext()
    }

    // ---- capture staging intents (fix E) — clipboard/timeline UI hands finished Blocks in ----
    func attachPastedImages(_ blocks: [Block]) { pastedImages.append(contentsOf: blocks) }
    func removePastedImage(id: UUID) { pastedImages.removeAll { $0.id == id } }
    func removeQueuedCapture(id: UUID) { queuedCaptures.removeAll { $0.id == id } }

    // ---- Voice·Text + capture control channel ----
    // Poll GET /voice/poll ~10Hz while a chat is open in ANY mode: capture events are delivered in
    // every mode; voice state (partial/final auto-send) is consumed only in Voice·Text. The chat pane
    // drives this via `.task` (started on appear, auto-cancelled on disappear).
    func pollLoop() async {
        while !Task.isCancelled {
            if !engine.ready { try? await Task.sleep(for: .milliseconds(500)); continue }   // don't churn /voice/poll before the engine is up
            if let p = try? await engine.voicePoll() { await handleVoicePoll(p) }
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    // reset the per-chat transient voice state when the open chat changes
    private func resetForChatSwitch() { voiceSeq = -1; livePartial = ""; queuedCaptures = [] }

    // Single chat-activation path (fix A). Fires once per open / new / seed, AFTER store.chat settled.
    private var lastActivatedChatId: UUID? = nil
    private func onChatActivated(_ kind: Store.Activation) {
        if store.chat.id != lastActivatedChatId {   // a genuine chat switch (mirrors onChange(chat.id))
            lastActivatedChatId = store.chat.id
            resetForChatSwitch()
            pinOutcome = .none                       // chat B must not show chat A's "cache failed: …" (fix 5)
        }
        repostContext()                             // re-stage the transcript on every activation (also covers reopening the same chat)
        if kind == .opened { cache() }              // opening evicts the old KV, re-pins this chat
    }

    private func postVoiceConfig() {
        // voice chord enables inside a Voice·Text chat only; capture bindings are active whenever a
        // chat is open in ANY mode — never on the home screen. Reads the stored setting copies.
        let inChat = store.screen == .chat
        Task { [self] in await engine.voiceConfig(
            voiceEnabled: inChat && mode == .voiceText,
            captureEnabled: inChat,
            submode: submode.rawValue,
            streaming: transcription == .stream,
            key: hotkey,
            shotBinding: shotBinding, shotStyle: shotStyleRaw, copyBinding: copyBinding) }
    }
    // Self-gated (fix A): posts only in Voice·Text, so every caller is a plain call — no mode checks.
    private func postVoiceContext() async {
        guard mode == .voiceText else { return }
        await engine.voiceContext(messages: buildEngineMessages(), reminder: store.chat.reminder)
    }
    private func repostContext() { Task { await self.postVoiceContext() } }

    // Refresh stored setting copies from the shared accessors; post config if any config-relevant key
    // changed, and re-post context on a mode change (self-gated). Diffing keeps unrelated defaults
    // writes (sidebar, perms) from posting anything.
    private func settingsChanged() {
        let m = Mode.current, sm = Submode.current, tr = Transcription.current
        let hk = SK.hotkeyValue, sb = SK.shotBindingValue, ss = SK.shotStyleValue, cb = SK.copyBindingValue
        let configDirty = m != mode || sm != submode || tr != transcription
            || hk != hotkey || sb != shotBinding || ss != shotStyleRaw || cb != copyBinding
        let modeChanged = m != mode
        mode = m; submode = sm; transcription = tr
        hotkey = hk; shotBinding = sb; shotStyleRaw = ss; copyBinding = cb
        if configDirty { postVoiceConfig() }
        if modeChanged { repostContext() }
    }

    private func handleVoicePoll(_ p: VoicePoll) async {
        // capture events arrive in ALL modes — deliver, then ack exactly what we consumed (FIFO).
        if let caps = p.captures, !caps.isEmpty {
            deliverCaptures(caps)
            await engine.voiceCapturesAck(count: caps.count)
        }
        guard Mode.current == .voiceText else { return }    // fresh read: never auto-send a tick after leaving Voice·Text
        let vs = VoiceState(wire: p.state)
        if voiceState != vs { voiceState = vs }             // dedupe: no 10Hz invalidation storm at idle (fix E)
        if livePartial != p.partial { livePartial = p.partial }
        // When a final transcript is ready, auto-send it (interrupts if busy), ack it once by seq.
        if vs == .ready, p.seq != voiceSeq,
           let f = p.final, !f.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            voiceSeq = p.seq
            let (pre, imgs) = drainQueuedCaptures()   // fold queued screenshots/copies into this turn
            ask(text: pre.isEmpty ? f : pre + "\n" + f, images: imgs)
            Task { await self.engine.voiceAck(seq: p.seq) }
        }
    }

    // Deliver captures to the chat as if the user had pasted them. Text mode: images → thumbnail bar,
    // text → input box. Voice modes: both → the queued strip, attached to the next spoken message.
    private func deliverCaptures(_ caps: [Capture]) {
        let live = Mode.current                             // fresh read: route to today's mode, not a stale copy
        for c in caps {
            if c.kind == "image", let d = c.data, !d.isEmpty {
                let blk = Block(mediaType: "image/png", data: d)
                if live == .voiceText { queuedCaptures.append(blk) } else { pastedImages.append(blk) }
            } else if c.kind == "text", let t = c.text, !t.isEmpty {
                if live == .voiceText { queuedCaptures.append(Block(text: t)) }
                else { input += input.isEmpty ? t : " " + t }
            }
        }
    }
    // Pull queued captures for a voice send: (joined text prefix, image blocks); clears the queue.
    private func drainQueuedCaptures() -> (String, [Block]) {
        let texts = queuedCaptures.compactMap { $0.type == "text" ? $0.text : nil }.filter { !$0.isEmpty }
        let imgs = queuedCaptures.filter { $0.type == "image" }
        queuedCaptures = []
        return (texts.joined(separator: "\n"), imgs)
    }
}
