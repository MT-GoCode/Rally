import SwiftUI

// ---------------------------------------------------------------------------
// Isolated cache-status leaf views. ChatSession is @Observable (per-property tracking), so a view only
// re-renders for the properties it READS in its body. These structs read the HIGH-FREQUENCY progress /
// streaming fields; because chatPane / the message list / the context pane do NOT read those fields
// (they only CONSTRUCT these views, passing a reference), an 8Hz progress tick or a per-token stream
// update re-renders ONLY the small view here — never the expensive trees. This is the freeze fix.
// ---------------------------------------------------------------------------

// Human-readable text for a cache state (shared by the sidebar row + the HUD).
func cacheStatusText(_ s: CacheState) -> String {
    switch s {
    case .idle:              return "not cached"
    case .nothingToCache:    return "nothing to cache — just ask"
    case .caching:           return "caching…"
    case .cached(let t):     return "cached ✓ · \(t) tok"
    case .changed:           return "cache changed"
    case .overLimit(let t):  return "context is \(t) tok — over the 200K limit"
    case .failed(let m):     return "cache failed: \(m)"
    case .notReady:          return "engine not ready yet"
    }
}

// Live progress capsule — reads ONLY session.progFrac. Its own struct → progress ticks invalidate just this.
struct CacheProgressBar: View {
    let session: ChatSession
    var body: some View {
        let f = session.progFrac
        return ZStack(alignment: .leading) {
            Capsule().fill(Color.secondary.opacity(0.20)).frame(width: 120, height: 4)
            if f > 0 { Capsule().fill(Color.accentColor).frame(width: 120 * max(0, min(f, 1)), height: 4) }
        }.frame(width: 120, height: 4)
    }
}

// The always-visible top-bar cache pipeline HUD (engine load → pin → conversation warm → ready), driven
// by the REAL /progress poll. Isolated leaf: reads the hot progress fields, so its updates re-render only
// this HStack. `label` comes straight from the engine ("encoding 8/34", "prefilling 4096/9048", "83 tok").
struct CacheHUDView: View {
    let session: ChatSession
    let engine: Engine
    let store: Store
    private var stage: (icon: String, text: String, busy: Bool, color: Color) {
        let lbl = session.progLabel
        if !engine.ready                 { return ("hourglass", engine.status, true, .secondary) }
        if store.loadingChat             { return ("hourglass", "loading chat…", true, .secondary) }
        // ONE state machine, engine-computed (/progress.phase); this is a pure renderer.
        switch session.progPhase {
        case "pin":
            return ("externaldrive.badge.plus", lbl.isEmpty ? "caching context (system + reference)…" : "caching context — \(lbl)", true, .accentColor)
        case "warm":
            return ("arrow.triangle.2.circlepath", lbl.isEmpty ? "warming conversation cache…" : "warming conversation cache — \(lbl)", true, .accentColor)
        case "generate":
            return ("bubble.left.and.text.bubble.right", lbl.isEmpty ? "generating…" : "generating — \(lbl)", true, .accentColor)
        case "pregen":
            if !session.speculationCurrent {   // engine is speculating for an OLDER draft — never claim
                                               // "instant" for text the user has typed past (truth gate)
                return ("bolt.fill", "precomputing next turn — syncing your latest edits…", true, .accentColor)
            }
            return session.livePregenDone
                ? ("bolt.fill", "next turn processed ⚡ reply pre-generated — instant send", false, .green)
                : ("bolt.fill", "next turn processed ⚡ pre-generating reply — \(session.livePregen) tok", true, .accentColor)
        case "composing":
            return ("bolt.fill", "precomputing next turn — \(session.livePrecomputed)/\(session.liveTurnTokens) tok fed · \(session.liveAnew) anew on send", false, .accentColor)
        default: break
        }
        if session.busy {   // phase can lag a beat behind the app's own send — keep the generating text stable
            return ("bubble.left.and.text.bubble.right", "generating…", true, .accentColor) }
        switch session.cacheState {
        case .overLimit, .failed:        return ("exclamationmark.triangle.fill", cacheStatusText(session.cacheState), false, .red)
        case .cached:                    return ("checkmark.seal.fill", "ready ✓ — cache warm, ask anything", false, .green)
        case .changed:                   return ("pencil.circle", "context changed — re-cache to apply", false, .orange)
        case .nothingToCache:            return ("checkmark.seal", "ready — just ask", false, .secondary)
        default:                         return ("circle", cacheStatusText(session.cacheState), false, .secondary)
        }
    }
    var body: some View {
        let st = stage
        return HStack(spacing: 8) {
            if st.busy { ProgressView().controlSize(.mini).scaleEffect(0.8) }
            else { Image(systemName: st.icon).font(.system(size: 11)).foregroundStyle(st.color) }
            Text(st.text).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
            if session.progBusy && session.progFrac > 0 { CacheProgressBar(session: session) }
            Spacer(minLength: 8)
            if let ct = session.convTokens {              // live conversation-cache usage vs budget
                HStack(spacing: 4) {
                    Image(systemName: "gauge.with.dots.needle.33percent").font(.system(size: 9))
                    Text("conversation context \(ct)/\(session.trimTrigger) tok").font(.system(size: 9, weight: .medium))
                    if let cvs = session.convStart, cvs > 0 { Text("· \(cvs) dropped").font(.system(size: 9)).foregroundStyle(.orange) }
                }.foregroundStyle(.secondary)
                .help("Conversation held live in the model's cache: \(ct) of your \(session.trimTrigger)-token budget. Turns beyond the budget fall out of context"
                      + ((session.convStart ?? 0) > 0 ? " — \(session.convStart!) older messages have dropped so far." : "."))
            }
            if engine.memGb > 0 {                          // the memory watcher: live engine mem vs the hard ceiling
                HStack(spacing: 3) {
                    Image(systemName: engine.memOver ? "exclamationmark.triangle.fill" : "memorychip").font(.system(size: 9))
                    Text(String(format: "engine %.1f/%.0f GB", engine.memGb, engine.memCeilingGb)).font(.system(size: 9, weight: .medium))
                }.foregroundStyle(engine.memOver ? .red : (engine.memGb > engine.memCeilingGb * 0.85 ? .orange : .secondary))
                .help("Engine memory vs the \(Int(engine.memCeilingGb)) GB safety ceiling. As it nears the ceiling, the engine trims the conversation cache to stay under it.")
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 5)
        .background(st.busy ? Color.accentColor.opacity(0.06) : Color.clear)
        .overlay(alignment: .bottom) { Divider() }
        .animation(.easeInOut(duration: 0.2), value: st.text)
    }
}

// The in-progress assistant bubble. Isolated leaf reading session.busy + session.streaming, so a streamed
// token invalidates ONLY this bubble — the committed history above (whose markdown is expensive) is untouched.
struct StreamingBubble<Content: View>: View {
    let session: ChatSession
    var onGrow: () -> Void = {}                    // scroll-follow lives HERE (not chatPane) so per-token
    @ViewBuilder let content: (String) -> Content  // stream updates never re-evaluate the message list
    var body: some View {
        if session.busy {
            content(session.streaming.isEmpty ? "…" : session.streaming)
                .onChange(of: session.streaming.count / 40) { _, _ in onGrow() }   // throttled follow-scroll
        }
    }
}
