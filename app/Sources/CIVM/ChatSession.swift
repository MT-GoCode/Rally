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

        // mirror Apple's live transcript into the shared livePartial while streaming.
        apple.$partial.receive(on: DispatchQueue.main)
            .sink { [weak self] p in MainActor.assumeIsolated {
                guard let self, self.appleRunning else { return }
                if self.livePartial != p { self.livePartial = p }
            } }.store(in: &cancellables)

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
    // per-message accounting from the last /chat done meta — explicit cache-vs-new breakdown.
    @Published private(set) var lastReused = 0    // history tokens reused from the cross-turn KV cache
    @Published private(set) var lastNew = 0       // tokens actually fed/processed anew this message
    @Published private(set) var lastPinned = 0    // pinned prefix (system+context) — always reused
    @Published private(set) var lastTtft = 0.0    // measured time-to-first-token (s), from the actual run
    // composition of the anew tokens, IN forward-pass order, summing to lastNew (engine-computed).
    @Published private(set) var anewParts: [(label: String, n: Int)] = []
    @Published private(set) var precache = ""     // "" hidden · "working" caching next msg · "done"
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
    @Published private(set) var preSent = 0        // tokens prefilled into Gemma so far this utterance (live)
    private var voiceSeq = -1

    // ---- Apple SpeechTranscriber (streaming submode; on-device, Swift-side). Python drives the
    // chord/overlay and reports state; the app runs Apple while listening and finalizes on chord-up. ----
    private let apple = AppleSpeech()
    private var appleRunning = false
    // Serializes the AppleSpeech lifecycle: every start/stop/cancel chains onto the previous one
    // (await prev.value first) so a fast chord can't interleave them at their await points — which
    // would race a start()'s installTap against a teardown/reset and crash AVFoundation (fix A).
    private var voiceTask: Task<Void, Never>? = nil

    // ---- settings: STORED copies refreshed from the shared accessors on didChangeNotification
    // (fix C/E). No per-render UserDefaults lookups — the old computed getters were a hot loop. ----
    private var mode: Mode = .current
    private var submode: Submode = .current
    private var transcription: Transcription = .current
    private var hotkey: String = SK.hotkeyValue
    private var shotBinding: String = SK.shotBindingValue
    private var shotStyleRaw: String = SK.shotStyleValue
    private var copyBinding: String = SK.copyBindingValue
    private var reminderModeStored: ReminderMode = .current
    private var modeSwitchTask: Task<Void, Never>? = nil   // in-flight re-cache after a mode switch

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
        busy = true; streaming = ""      // new turn: consumes the warmed cache
        await modeSwitchTask?.value       // if a reminder-mode switch is still re-warming the cache, wait
        modeSwitchTask = nil
        precache = ""                     // hide the chip now that the switch re-cache (if any) is done
        if !pinned { await pinNow() }           // pin on demand (caching first isn't required)
        do {
            let meta = try await engine.chat(messages: buildEngineMessages(), reminder: store.chat.reminder,
                                              reminderMode: ReminderMode.current.rawValue) { self.streaming += $0 }
            if Task.isCancelled {               // interrupted right as the stream ended
                stashInterrupted(); return
            }
            store.chat.messages.append(Msg(role: "assistant", text: streaming))
            if let ct = meta["chat_tokens"] as? Int { store.chat.chatTokens = ct }
            if let pt = meta["pinned"] as? Int, pt > 0 { store.chat.pinnedTokens = pt }
            lastReused = meta["reused"] as? Int ?? 0
            lastNew = meta["new_tokens"] as? Int ?? 0
            lastPinned = meta["pinned"] as? Int ?? 0
            lastTtft = meta["ttft"] as? Double ?? 0
            anewParts = (meta["anew_parts"] as? [[String: Any]] ?? [])
                .map { (label: $0["label"] as? String ?? "", n: $0["n"] as? Int ?? 0) }
            streaming = ""; busy = false; store.save()
            watchPrecache()                     // engine is now warming the KV for the next message
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
        // no repost here: every stash is immediately followed by the successor turn's append + repost.
    }

    // The clean transcript for the engine: the FULL conversation (no truncation), images as blocks,
    // interruption prefix added HERE only. Cross-turn KV reuse keeps this cheap regardless of length.
    private func buildEngineMessages() -> [ChatMessage] {
        store.chat.messages.map { m in
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
    }

    // Truncate the transcript to everything up to AND including the given message (a "reset to here"
    // on an agent reply): drop all later turns, persist, and re-stage the streaming-prefill context.
    // Engine chat is app-owned+stateless, so no /pin or /new is needed — the next ask sends the
    // shortened transcript. A turn in flight is cancelled first.
    func resetToHere(_ id: UUID) {
        guard let i = store.chat.messages.firstIndex(where: { $0.id == id }) else { return }
        askTask?.cancel()
        streaming = ""; busy = false
        store.chat.messages = Array(store.chat.messages.prefix(through: i))
        // the truncated turn's per-message accounting no longer describes any live turn — clear it so the
        // tokenLine breakdown/chip reflect the shortened conversation. Leave pinnedTokens: the pin is
        // unchanged (fix C).
        lastReused = 0; lastNew = 0; lastPinned = 0; lastTtft = 0
        anewParts = []
        precache = ""
        store.chat.chatTokens = 0
        store.save()
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

    // Refresh stored setting copies from the shared accessors; post config if any config-relevant key
    // changed, and re-post context on a mode change (self-gated). Diffing keeps unrelated defaults
    // writes (sidebar, perms) from posting anything.
    private func settingsChanged() {
        let m = Mode.current, sm = Submode.current, tr = Transcription.current
        let hk = SK.hotkeyValue, sb = SK.shotBindingValue, ss = SK.shotStyleValue, cb = SK.copyBindingValue
        let configDirty = m != mode || sm != submode || tr != transcription
            || hk != hotkey || sb != shotBinding || ss != shotStyleRaw || cb != copyBinding
        mode = m; submode = sm; transcription = tr
        hotkey = hk; shotBinding = sb; shotStyleRaw = ss; copyBinding = cb
        if configDirty { postVoiceConfig() }
        // reminder placement changed (via the Settings picker OR the in-chat control) → re-warm the
        // cache with the new mode. Idempotent: setReminderMode already bumped reminderModeStored when
        // the in-chat control fired, so this only fires for the Settings-picker path.
        let rm = ReminderMode.current
        if rm != reminderModeStored { setReminderMode(rm) }
    }

    // While Apple streams, prefill Gemma's KV from the growing partial (debounced) so the final ask
    // is near-instant. Reports the live token count fed past the pin. Runs until appleRunning clears.
    private func startPrefillLoop() {
        Task { [weak self] in
            var lastPosted = ""
            while let self, self.appleRunning {
                try? await Task.sleep(for: .milliseconds(350))
                guard self.appleRunning else { break }
                let partial = self.apple.partial
                if partial.isEmpty || partial == lastPosted { continue }
                lastPosted = partial
                let fed = await self.engine.voicePrefill(messages: self.buildEngineMessages(), partial: partial)
                if self.appleRunning { self.preSent = fed }
            }
        }
    }

    // Switch reminder placement mid-chat: persist it, then re-warm the cache with the new mode so the
    // next message stays instant. The chip shows caching → ready. The re-cache is tracked in
    // modeSwitchTask so the NEXT send() waits for it (else the /chat races the /precache and re-feeds
    // the reminder). Idempotent via reminderModeStored so the in-chat control and the Settings picker
    // (which both write SK.reminderMode) don't double-fire.
    func setReminderMode(_ m: ReminderMode) {
        reminderModeStored = m
        UserDefaults.standard.set(m.rawValue, forKey: SK.reminderMode)
        guard engine.ready, !store.chat.messages.isEmpty else { return }
        precache = "working"
        let t = Task { [weak self] in
            guard let self else { return }
            await self.engine.precache(messages: self.buildEngineMessages(), reminder: self.store.chat.reminder, reminderMode: m.rawValue)
        }
        modeSwitchTask = t
        Task { [weak self] in                       // chip: working → ready (separate, so send() waits only on `t`)
            await t.value
            guard let self else { return }
            if self.precache == "working" { self.precache = "done" }
            try? await Task.sleep(for: .seconds(1.4))
            if self.precache == "done" { self.precache = "" }
        }
    }

    // After a reply the engine warms the KV for the next message; mirror it in the bottom-right chip.
    private func watchPrecache() {
        Task { [weak self] in
            guard let self else { return }
            self.precache = "working"
            for _ in 0..<50 {                      // ~5s cap
                try? await Task.sleep(for: .milliseconds(100))
                if self.busy || self.precache.isEmpty { return }   // a new turn took over
                let s = await self.engine.precacheState()
                if s == "done" { self.precache = "done"; break }
                if s == "idle" { self.precache = ""; return }
            }
            try? await Task.sleep(for: .seconds(1.4))
            if self.precache == "done" { self.precache = "" }       // fade the "done" chip
        }
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

        // STREAMING submode: Apple (Swift) owns audio+transcription. Python's poll state is just the
        // chord signal — listening → run Apple; ready (chord-up) → finalize+send Apple's text; idle
        // while running (ESC) → discard.
        if Transcription.current == .stream {
            switch vs {
            case .listening where !appleRunning:
                appleRunning = true; livePartial = ""; preSent = 0
                let loc = SK.speechLocaleValue
                let prev = voiceTask                        // chain onto any in-flight lifecycle op (fix A)
                voiceTask = Task { [weak self] in
                    await prev?.value
                    guard let self else { return }
                    do {
                        try await self.apple.start(localeID: loc)
                    } catch {
                        // start() failed (perm denied / no format / …): don't leave a false running
                        // state. AppleSpeech.start already tore itself down; reset our voice state and
                        // route to the idle/cancel path so the chord doesn't silently no-op (fix B).
                        NSLog("[Rally] AppleSpeech.start failed: \(error.localizedDescription)")
                        self.appleRunning = false; self.livePartial = ""; self.preSent = 0
                        self.voiceState = .idle
                        await self.apple.cancel()           // idempotent belt-and-suspenders teardown
                    }
                }
                startPrefillLoop()                          // prefill Gemma from Apple's partials as you talk
            case .ready where appleRunning && p.seq != voiceSeq:
                voiceSeq = p.seq; appleRunning = false      // stops the prefill loop
                let seq = p.seq
                let prev = voiceTask                        // wait for the pending start() before stopping (fix A)
                voiceTask = Task { [weak self] in
                    await prev?.value
                    guard let self else { return }
                    let text = await self.apple.stop()
                    await self.engine.voiceAck(seq: seq)
                    self.preSent = 0                        // dictation over → clear live count (→ "reused" after /chat)
                    guard !text.isEmpty else { return }
                    let (pre, imgs) = self.drainQueuedCaptures()
                    self.ask(text: pre.isEmpty ? text : pre + "\n" + text, images: imgs)
                }
            case .idle where appleRunning:                  // ESC / cancel
                appleRunning = false; livePartial = ""; preSent = 0
                let prev = voiceTask                        // wait for the pending start() before cancelling (fix A)
                voiceTask = Task { [weak self] in
                    await prev?.value
                    await self?.apple.cancel()
                }
            default: break
            }
            return
        }

        // TRANSCRIBE-AFTER submode: Parakeet (Python) produces the final; consume it.
        if livePartial != p.partial { livePartial = p.partial }
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
