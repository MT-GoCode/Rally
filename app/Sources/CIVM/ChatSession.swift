import Foundation
import Combine
import Observation
import AppKit   // NSApp / NSTextView — read & set the chat input's caret for insert-at-cursor dictation

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

@MainActor @Observable final class ChatSession {
    @ObservationIgnored private let engine: Engine
    @ObservationIgnored private let store: Store
    @ObservationIgnored private var cancellables = Set<AnyCancellable>()

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
        // engine/store are @Observable now (no Combine $publishers), so self-observe via
        // withObservationTracking: onChange fires once when the tracked property changes; the deferred
        // Task reads the SETTLED value (past the mutation), acts, then re-arms. Tracking starts AFTER
        // init, so neither fires for its initial value (the old dropFirst / "act only on change").
        trackReady()
        trackScreen()
        if engine.ready { postVoiceConfig() }   // replay: withObservationTracking only fires on a SUBSEQUENT change
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

    // engine readiness → post config once it flips ready. onChange fires on the willSet; the Task defers
    // to read the settled value, then re-arms (onChange is one-shot).
    // onChange is nonisolated + one-shot; the deferred Task both reads the SETTLED value and re-arms on the
    // main actor. (A change landing between the fire and the re-arm is dropped, but the FINAL state is
    // always handled because each fire re-reads live state — a narrow, self-correcting gap.)
    private func trackReady() {
        withObservationTracking { _ = engine.ready } onChange: { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                if self.engine.ready {
                    if self.pinOutcome == .notReady { self.pinOutcome = .none }   // engine up → drop the stale gate
                    self.postVoiceConfig()
                }
                self.trackReady()
            }
        }
    }
    private func trackScreen() {
        withObservationTracking { _ = store.screen } onChange: { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.postVoiceConfig()
                self.trackScreen()
            }
        }
    }

    // ---- in-flight turn (ask pipeline) ----
    private(set) var busy = false
    private(set) var streaming = ""
    // per-message accounting from the last /chat done meta — explicit cache-vs-new breakdown.
    private(set) var lastReused = 0    // history tokens reused from the cross-turn KV cache
    private(set) var lastNew = 0       // tokens actually fed/processed anew this message
    private(set) var lastPinned = 0    // pinned prefix (system+context) — always reused
    private(set) var lastTtft = 0.0    // measured time-to-first-token (s), from the actual run
    // composition of the anew tokens, IN forward-pass order, summing to lastNew (engine-computed).
    private(set) var anewParts: [(label: String, n: Int)] = []
    private(set) var precache = ""     // "" hidden · "working" caching next msg · "done"
    var convStart: Int? = nil          // oldest in-window message index (out-of-context boundary); nil = unknown
    var convTokens: Int? = nil         // current conversation+reminder tokens in the cache (live usage vs budget)
    // How many trailing messages the view renders NOW (bounded; grown only by loadEarlier so a big chat's
    // markdown never lays out all at once — Textual's per-bubble CoreText layout on the main thread pegged
    // it ~10s. Small batches + a breather between them keep the app responsive the whole time.
    // HARD-bounded render window. Only the last `visibleCount` messages are ever built, so opening a chat
    // lays out at most this many markdown/math bubbles on the main thread — no auto-grow (that piled
    // layouts up and beachballed). The LazyVStack then realizes only the on-screen rows within the window;
    // "Load earlier" widens it on demand (user-paced), and scrolling up realizes those rows lazily.
    private(set) var visibleCount = 6
    func loadEarlier() { visibleCount = min(visibleCount + 10, store.chat.messages.count) }
    private func resetRenderWindow() { visibleCount = 6 }
    // Cache budget + resume mode are GLOBAL (read live from settings), NOT a per-chat snapshot — so
    // changing them in Settings or the ⋯ menu applies to EVERY open chat immediately (recompute on
    // change and on chat visit). This is why an old chat now trims to the current budget on open.
    var trimTrigger: Int { SK.cacheTriggerValue }
    var trimTarget: Int  { SK.cacheTargetValue }
    var recacheMode: String { RecacheMode.current.rawValue }
    private var prevHadOutput = false             // did the turn we just cancelled produce any text?
    // AskPipeline (mode-agnostic): one in-flight turn; interrupts cancel it and stack a new one.
    private var askTask: Task<Void, Never>? = nil

    // ---- pin / cache ----
    private(set) var caching = false
    // Live cache progress — REAL, polled from the engine's /progress (replaces the old estimated 12.5Hz
    // ticker that beachballed the whole tree). Read ONLY inside the isolated CacheHUD/CacheProgressBar
    // leaf views, so ticks never invalidate the message list or the context pane.
    private(set) var progOp = "idle"      // idle|pin|reconcile|precache|generate
    private(set) var progStage = ""       // encode|prefill|replay|feed|decode
    private(set) var progFrac = 0.0       // 0 = indeterminate (decode); else done/total
    private(set) var progDone = 0
    private(set) var progTotal = 0
    private(set) var progLabel = ""
    var progBusy: Bool { progOp != "idle" }
    // Poll the engine's real progress ~8Hz into the fields above (deduped so idle never invalidates).
    func pollProgress() async {
        while !Task.isCancelled {
            if !engine.ready { try? await Task.sleep(for: .milliseconds(400)); continue }
            if let s = await engine.progress() {
                if progOp    != s.op    { progOp = s.op }
                if progStage != s.stage { progStage = s.stage }
                if progFrac  != s.frac  { progFrac = s.frac }
                if progDone  != s.done  { progDone = s.done }
                if progTotal != s.total { progTotal = s.total }
                if progLabel != s.label { progLabel = s.label }
            }
            try? await Task.sleep(for: .milliseconds(120))
        }
    }
    // sticky outcome of the last pin attempt (cleared when a fresh non-empty pin starts) — supersedes
    // the old free-form cacheMsg string; CacheState is composed from it plus the derived flags.
    private enum PinOutcome: Equatable { case none, notReady, overLimit(Int), failed(String) }
    private var pinOutcome: PinOutcome = .none

    // ---- compose buffer + capture staging (fix E — intent methods mutate; views only read) ----
    var input = ""                             // typed chat input (bound two-way to the TextField)
    private(set) var pastedImages: [Block] = []   // chat-input pasted images (thumbnail bar → ride with the question)

    // ---- hybrid dictation: the field locks while talking; the transcript lands at the caret ----
    private(set) var dictating = false         // a dictation is live → the input field is uneditable
    private var dictCaret = 0                              // UTF-16 caret offset captured at dictation start (field is
                                                          // locked while dictating, so it can't move → valid at insert)
    private var dictWasAlone = false                      // input box was empty at dictation start (drives .ifAlone auto-send)
    private let sysAudio = SystemAudio()                  // mutes system output while dictating (if enabled)

    // ---- keyboard navigation (Slack-like): arrow through AGENT messages, act with s/c/r ----
    // selectedMessageID and input focus are MUTUALLY EXCLUSIVE (one owns keyboard focus at a time):
    // selecting a message defocuses the input; focusing the input clears the selection.
    var selectedMessageID: UUID? = nil        // the highlighted agent message (nil = none)
    var sourceShownIDs: Set<UUID> = []        // agent messages flipped to raw-markdown source view
    private(set) var inputFocusToken = 0      // bump → the view moves keyboard focus to the input
    var inputIsFocused = false                           // mirrored from the view's @FocusState (read by the key monitor)
    private(set) var justCopied = false       // brief "Copied" toast after any copy action
    private var copyFlashTask: Task<Void, Never>? = nil

    func flashCopied() {
        justCopied = true
        copyFlashTask?.cancel()
        copyFlashTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1.1))
            if !Task.isCancelled { self?.justCopied = false }
        }
    }

    // AGENT message ids oldest→newest — the navigation order (user turns are skipped).
    var agentMessageIDs: [UUID] { store.chat.messages.filter { $0.role == "assistant" }.map { $0.id } }

    func focusInput() { selectedMessageID = nil; inputFocusToken &+= 1 }
    func clearSelection() { selectedMessageID = nil }
    func navUp() {                                       // older; from input/none → most-recent agent msg
        let ids = agentMessageIDs; guard !ids.isEmpty else { return }
        if let sel = selectedMessageID, let i = ids.firstIndex(of: sel) {
            if i > 0 { selectedMessageID = ids[i - 1] }
        } else { selectedMessageID = ids.last }
    }
    func navDown() {                                     // newer; past the newest → input
        let ids = agentMessageIDs
        if let sel = selectedMessageID, let i = ids.firstIndex(of: sel) {
            if i < ids.count - 1 { selectedMessageID = ids[i + 1] } else { focusInput() }
        } else { focusInput() }
    }
    func toggleSource(_ id: UUID) {
        if sourceShownIDs.contains(id) { sourceShownIDs.remove(id) } else { sourceShownIDs.insert(id) }
    }
    // The display body of a message (text before @@APPENDIX@@, trimmed) — for copy-source.
    func messageBody(_ id: UUID) -> String {
        guard let m = store.chat.messages.first(where: { $0.id == id }) else { return "" }
        return m.text.components(separatedBy: "@@APPENDIX@@")[0].trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ---- Voice·Text live state (from GET /voice/poll) ----
    private(set) var voiceState: VoiceState = .idle
    private(set) var livePartial = ""
    private(set) var preSent = 0        // tokens prefilled into Gemma so far this utterance (live)
    private var voiceSeq = -1

    // ---- Apple SpeechTranscriber (streaming submode; on-device, Swift-side). Python drives the
    // chord/overlay and reports state; the app runs Apple while listening and finalizes on chord-up. ----
    private let apple = AppleSpeech()
    private var appleRunning = false
    // Serializes the AppleSpeech lifecycle: every start/stop/cancel chains onto the previous one
    // (await prev.value first) so a fast chord can't interleave them at their await points — which
    // would race a start()'s installTap against a teardown/reset and crash AVFoundation (fix A).
    private var voiceTask: Task<Void, Never>? = nil

    // ---- GLOBAL settings: stored copies refreshed on didChangeNotification. Voice input / submode /
    // transcription / auto-send / mute / reminder placement are now PER-CHAT (store.chat.settings) —
    // only the hotkey + capture bindings (which drive the one process-wide tap) stay global here. ----
    private var hotkey: String = SK.hotkeyValue
    private var shotBinding: String = SK.shotBindingValue
    private var shotStyleRaw: String = SK.shotStyleValue
    private var copyBinding: String = SK.copyBindingValue
    private var modeSwitchTask: Task<Void, Never>? = nil   // in-flight re-cache after a reminder-mode change

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
        caching = true; pinOutcome = .none        // caching is a low-freq bool for cacheState; the LIVE
        let sys = nonEmpty(store.chat.system), ctx = nonEmpty(store.chat.context)   // progress bar reads the
        do {                                       // real /progress poll (progFrac), not a fake ticker.
            let (tokens, over) = try await engine.pin(system: sys, context: ctx)
            if over { pinOutcome = .overLimit(tokens); store.chat.pinnedTokens = nil }
            else {
                store.chat.pinnedTokens = tokens
                store.enginePinnedChat = id; store.enginePinnedHash = hash
                store.save()
            }
        } catch { pinOutcome = .failed(error.localizedDescription) }
        caching = false
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
        if !cachedCurrent { await pinNow() }    // re-pin if unpinned OR the system/context changed (wait-until-correct)
        do {
            let s = store.chat.settings
            let meta = try await engine.chat(messages: buildEngineMessages(), reminder: store.chat.reminder,
                                              reminderMode: s.reminderMode, trimTrigger: trimTrigger,
                                              trimTarget: trimTarget, recacheMode: recacheMode) { self.streaming += $0 }
            if Task.isCancelled {               // interrupted right as the stream ended
                stashInterrupted(); return
            }
            store.chat.messages.append(Msg(role: "assistant", text: streaming))
            if let cs = meta["conv_start"] as? Int { convStart = cs }   // sliding-window boundary (out-of-context line)
            if let cvt = meta["conv_tokens"] as? Int { convTokens = cvt } // live conversation-cache usage vs budget
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

    // Stop generation with NO successor turn (the ⏹ button): cancel the in-flight /chat, let runTurn
    // stash whatever it produced as an interrupted bubble, then settle back to idle and re-warm the KV
    // for the next message. (send()'s "Interrupt & Ask" cancels-and-replaces; this just cancels.)
    func stop() {
        guard busy else { return }
        let prev = askTask
        askTask = Task { [weak self] in
            prev?.cancel()
            await prev?.value               // runTurn's cancel path stashes the partial
            guard let self else { return }
            self.busy = false               // no successor turn → return to idle
            self.streaming = ""
            self.store.save()
            self.rewarmCache()              // keep the next message instant
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
        store.chat.chatTokens = 0
        selectedMessageID = nil                 // the anchor turn is now the newest; drop any nav highlight
        store.save()
        rewarmCache()                           // recompute the KV up to the (now truncated) transcript
    }

    // ---- capture staging intents (fix E) — clipboard/timeline UI hands finished Blocks in ----
    func attachPastedImages(_ blocks: [Block]) { pastedImages.append(contentsOf: blocks) }
    func removePastedImage(id: UUID) { pastedImages.removeAll { $0.id == id } }

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
    private func resetForChatSwitch() {
        askTask?.cancel(); streaming = ""; busy = false    // cancel any in-flight turn so its reply can't land in the new chat
        voiceSeq = -1; livePartial = ""
        if dictating { endDictation() }                   // don't carry a locked field / muted audio across chats
        selectedMessageID = nil; sourceShownIDs = []      // nav highlight + source toggles are per-chat
        convStart = nil; convTokens = nil                 // boundary + usage unknown until this chat's cache reports
        resetRenderWindow()                                // reset the bounded render-window for the new chat
    }

    // Single chat-activation path (fix A). Fires once per open / new / seed, AFTER store.chat settled.
    private var lastActivatedChatId: UUID? = nil
    private func onChatActivated(_ kind: Store.Activation) {
        if store.chat.id != lastActivatedChatId {   // a genuine chat switch (mirrors onChange(chat.id))
            lastActivatedChatId = store.chat.id
            resetForChatSwitch()
            pinOutcome = .none                       // chat B must not show chat A's "cache failed: …" (fix 5)
        }
        if kind == .opened {                        // opening evicts the old KV → re-pin, THEN rebuild the
            Task { [weak self] in                   // conversation cache (recent window / streaming replay)
                await self?.pinNow()                // so the boundary + smear are correct before the 1st message
                self?.reconcileCache()
            }
        }
        resetRenderWindow()                         // open shows a small window; "Load earlier" widens on demand
        postVoiceConfig()                           // the new chat may have different voice settings → re-arm hotkeys
    }

    // Rebuild the conversation cache so it's correct for THIS chat's settings (called on open; the engine
    // does recent-window prefill or streaming replay). Tracked in modeSwitchTask so the next send() waits
    // on it. Drives the "conversation cache" spinner and the out-of-context boundary (convStart).
    func reconcileCache() {
        guard engine.ready, store.screen == .chat, !store.chat.messages.isEmpty else { convStart = nil; return }
        precache = "working"
        let msgs = buildEngineMessages(), rem = store.chat.reminder, s = store.chat.settings
        let trig = trimTrigger, tgt = trimTarget, mode = recacheMode
        let t = Task { [weak self] in
            let r = await self?.engine.reconcile(messages: msgs, reminder: rem, reminderMode: s.reminderMode,
                                                 trimTrigger: trig, trimTarget: tgt,
                                                 recacheMode: mode)
            if let self, let r { self.convStart = r.convStart; self.convTokens = r.convTokens }  // divider + gauge on open
            return ()
        }
        modeSwitchTask = t
        Task { [weak self] in
            await t.value
            guard let self else { return }
            if self.precache == "working" { self.precache = "done" }
            try? await Task.sleep(for: .seconds(1.4))
            if self.precache == "done" { self.precache = "" }
        }
    }

    // ---- per-chat setting writes: mutate store.chat.settings, save, and fire the right side effect ----
    private func saveSettings() { store.save() }
    func setVoiceInput(_ on: Bool)      { store.chat.settings.voiceInput = on;      saveSettings(); postVoiceConfig() }
    func setSubmode(_ m: Submode)       { store.chat.settings.submode = m.rawValue; saveSettings(); postVoiceConfig() }
    func setTranscription(_ t: Transcription) { store.chat.settings.transcription = t.rawValue; saveSettings(); postVoiceConfig() }
    func setAutoSend(_ a: AutoSend)     { store.chat.settings.autoSend = a.rawValue; saveSettings() }
    func setMuteDictation(_ on: Bool)   { store.chat.settings.muteDictation = on;   saveSettings() }
    func setAgentOutput(_ o: AgentOutput) { store.chat.settings.agentOutput = o.rawValue; saveSettings() }
    func setSpeechLocale(_ id: String)  { store.chat.settings.speechLocale = id;    saveSettings() }
    // Cache budget / resume strategy — per-chat, applied to the LIVE cache immediately (reconcile rebuilds).
    // GLOBAL budget/mode (UserDefaults) — applies to every chat; reconcile the open chat so it applies now.
    func setCacheTrigger(_ v: Int)      { UserDefaults.standard.set(v, forKey: SK.cacheTrigger); reconcileCache() }
    func setCacheTarget(_ v: Int)       { UserDefaults.standard.set(v, forKey: SK.cacheTarget);  reconcileCache() }
    func setRecacheMode(_ m: RecacheMode) { UserDefaults.standard.set(m.rawValue, forKey: SK.recacheMode) }  // resume-only → no live rebuild

    private func postVoiceConfig() {
        // voice chord arms when THIS chat's voice input is on; capture bindings are active whenever a
        // chat is open (voice on or off) — never on the home screen. Voice settings are per-chat; the
        // hotkey + capture bindings are global (stored copies).
        let inChat = store.screen == .chat
        let s = store.chat.settings
        Task { [self] in await engine.voiceConfig(
            voiceEnabled: inChat && s.voiceInput,
            captureEnabled: inChat,
            submode: s.submode,
            streaming: s.transcriptionV == .stream,
            key: hotkey,
            shotBinding: shotBinding, shotStyle: shotStyleRaw, copyBinding: copyBinding,
            hotkeyMode: HotkeyMode.current.rawValue) }
        refreshHotkeys()                            // (re)register the OS hotkeys for self-contained mode
    }

    // Self-contained mode: register OS-level hotkeys (RegisterEventHotKey) that poke the engine's
    // /trigger — no event tap, zero interference. Karabiner mode registers nothing (the CLI drives it).
    private let hotkeys = HotkeyManager()
    func refreshHotkeys() {
        hotkeys.clear()
        guard HotkeyMode.current == .selfContained, store.screen == .chat else { return }
        let s = store.chat.settings
        if s.voiceInput, let c = HotkeyManager.parse(hotkey) ?? HotkeyManager.parse(SK.defaultHotkey) {
            hotkeys.register(c,
                onPress:   { [weak self] in Task { await self?.engine.trigger("chord_down") } },
                onRelease: { [weak self] in Task { await self?.engine.trigger("chord_up") } })
        }
        if let c = HotkeyManager.parse(shotBinding) {
            hotkeys.register(c, onPress: { [weak self] in Task { await self?.engine.trigger("shot") } })
        }
        if let c = HotkeyManager.parse(copyBinding) {
            hotkeys.register(c, onPress: { [weak self] in Task { await self?.engine.trigger("copy") } })
        }
    }

    // Refresh the GLOBAL stored copies (hotkey + capture bindings) on a defaults change; re-post config
    // if any changed so the tap picks up a new chord/binding. Per-chat settings do NOT flow through here
    // (they're written directly via the setters above) — and changing a DEFAULT never touches the open chat.
    private func settingsChanged() {
        let hk = SK.hotkeyValue, sb = SK.shotBindingValue, ss = SK.shotStyleValue, cb = SK.copyBindingValue
        let hm = HotkeyMode.current.rawValue
        let configDirty = hk != hotkey || sb != shotBinding || ss != shotStyleRaw || cb != copyBinding || hm != hotkeyModeCopy
        hotkey = hk; shotBinding = sb; shotStyleRaw = ss; copyBinding = cb; hotkeyModeCopy = hm
        if configDirty { postVoiceConfig() }   // re-posts hotkeyMode + re-registers the OS hotkeys
    }
    private var hotkeyModeCopy = HotkeyMode.current.rawValue

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

    // Switch this CHAT's reminder placement mid-conversation: persist it on the chat, then re-warm the
    // cache with the new mode so the next message stays instant (chip: caching → ready). The re-cache is
    // tracked in modeSwitchTask so the NEXT send() waits for it (else /chat races /precache and re-feeds
    // the reminder). Per-chat now — the Settings picker edits only the DEFAULT and never lands here.
    func setReminderMode(_ m: ReminderMode) {
        store.chat.settings.reminderMode = m.rawValue
        saveSettings()
        rewarmCache()
    }

    // Re-warm the KV for the next message with the CURRENT transcript + reminder mode (used after a
    // reminder-mode switch AND after resetToHere, so the next ask stays instant). Tracked in
    // modeSwitchTask so the next send() waits for it (no /chat-vs-/precache race). Chip: working→ready.
    func rewarmCache() {
        guard engine.ready, !store.chat.messages.isEmpty else { precache = ""; return }
        precache = "working"
        let msgs = buildEngineMessages(), rem = store.chat.reminder, s = store.chat.settings
        let trig = trimTrigger, tgt = trimTarget, mode = recacheMode
        let t = Task { [weak self] in
            await self?.engine.precache(messages: msgs, reminder: rem, reminderMode: s.reminderMode,
                                        trimTrigger: trig, trimTarget: tgt, recacheMode: mode)
            return ()
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
        // capture events arrive whether voice is on or off — deliver, then ack what we consumed (FIFO).
        if let caps = p.captures, !caps.isEmpty {
            deliverCaptures(caps)
            await engine.voiceCapturesAck(count: caps.count)
        }
        guard store.chat.settings.voiceInput else {         // this chat's voice toggle: ignore voice ticks when off
            if dictating { endDictation() }                 // …but if it was toggled off mid-dictation, unlock + unmute
            return
        }
        let vs = VoiceState(wire: p.state)
        if voiceState != vs { voiceState = vs }             // dedupe: no 10Hz invalidation storm at idle (fix E)
        if vs == .listening { beginDictation() }            // lock the field + snapshot the caret (guarded: once)

        // STREAMING submode: Apple (Swift) owns audio+transcription. Python's poll state is just the
        // chord signal — listening → run Apple; ready (chord-up) → finalize+insert Apple's text; idle
        // while running (ESC) → discard.
        if store.chat.settings.transcriptionV == .stream {
            switch vs {
            case .listening where !appleRunning:
                appleRunning = true; livePartial = ""; preSent = 0
                let loc = store.chat.settings.speechLocale
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
                        self.endDictation()                 // unlock the field + restore audio
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
                    self.deliverDictation(text)             // insert at caret (+ auto-send); unlocks field even if empty
                }
            case .idle where appleRunning:                  // ESC / cancel
                appleRunning = false; livePartial = ""; preSent = 0
                endDictation()                              // unlock the field + restore audio, no insert
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
        if vs == .idle, dictating { endDictation() }        // ESC / empty transcript → unlock, no insert
        if livePartial != p.partial { livePartial = p.partial }
        if vs == .ready, p.seq != voiceSeq,
           let f = p.final, !f.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            voiceSeq = p.seq
            deliverDictation(f)                             // insert at caret (+ auto-send)
            Task { await self.engine.voiceAck(seq: p.seq) }
        }
    }

    // ---- hybrid dictation lifecycle ----------------------------------------------------------------
    // A dictation begins when the chord goes down (voiceState → .listening): snapshot where the caret
    // is in the input box (it's locked while talking, so it can't move) and whether the box was empty,
    // then mute system audio if enabled.
    private func beginDictation() {
        guard !dictating else { return }                    // once per utterance
        dictCaret = inputCaretOffset()
        dictWasAlone = input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        dictating = true
        if store.chat.settings.muteDictation { sysAudio.mute() }
    }
    // A dictation ends (successfully-inserted OR cancelled): unlock the field + restore audio.
    private func endDictation() {
        guard dictating else { return }
        dictating = false
        sysAudio.restore()
    }
    // Finished transcript → drop it in at the caret we snapshotted, then auto-send per the setting.
    private func deliverDictation(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let wasAlone = dictWasAlone
        endDictation()                                      // unlock first (so send()/edits see an enabled field)
        guard !t.isEmpty else { return }                    // empty → nothing to insert; field already unlocked
        insertAtCaret(t, offset: dictCaret)
        switch store.chat.settings.autoSendV {
        case .always:  send()
        case .ifAlone: if wasAlone { send() }               // box was empty at start → the transcript is the message
        case .never:   break                                // leave it inserted; the user presses Enter
        }
    }

    // Caret (UTF-16) in the chat input's field editor; end-of-input if it isn't first responder.
    private func inputCaretOffset() -> Int {
        if let tv = NSApp.keyWindow?.firstResponder as? NSTextView {
            return min(tv.selectedRange().location, (tv.string as NSString).length)
        }
        return (input as NSString).length
    }
    // Insert `text` at `offset` in `input`, then refocus the field so the user can keep typing.
    private func insertAtCaret(_ text: String, offset: Int) {
        let ns = input as NSString
        let loc = max(0, min(offset, ns.length))
        input = ns.replacingCharacters(in: NSRange(location: loc, length: 0), with: text)
        focusInput()                                        // re-focus the (now-unlocked) field for continued typing
    }

    // Deliver captures to the chat as if the user had pasted them: images → thumbnail bar, text → input.
    // One path now (the input box is always present), so screenshots/copies ride the next send either way.
    private func deliverCaptures(_ caps: [Capture]) {
        for c in caps {
            if c.kind == "image", let d = c.data, !d.isEmpty {
                pastedImages.append(Block(mediaType: "image/png", data: d))
            } else if c.kind == "text", let t = c.text, !t.isEmpty {
                input += input.isEmpty ? t : " " + t
            }
        }
    }
}
