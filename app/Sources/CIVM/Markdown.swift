import SwiftUI
import Textual

// ---------------------------------------------------------------------------
// GFM Markdown + LaTeX math → SwiftUI, rendered NATIVELY by Textual (the
// MarkdownUI author's successor) with its swiftui-math CoreText engine. Tables,
// fenced code (syntax-highlighted), lists, blockquotes AND inline/display math
// ($…$, $$…$$) all render without a WKWebView or JavaScript — so no per-message
// WebContent process and no beachball (the WKWebView+KaTeX approach caused both).
// Re-parsed on every streaming delta; math is CoreText-fast and cached.
// The @@APPENDIX@@ split lives in the caller (messageBubble); Appendix below.
// ---------------------------------------------------------------------------

// Equatable so SwiftUI skips re-evaluating (and Textual skips re-parsing markdown+math) when `raw`
// is unchanged. Without this, any parent invalidation — e.g. selecting a different message, which
// every bubble observes for its highlight ring — re-parses EVERY visible message's CoreText, which
// beachballed the app when arrow-key navigation was spammed. Used via `.equatable()` at the call site.
struct MarkdownText: View, Equatable {
    let raw: String
    nonisolated static func == (a: MarkdownText, b: MarkdownText) -> Bool { a.raw == b.raw }
    var body: some View {
        StructuredText(markdown: raw, syntaxExtensions: [.math])
            .textual.structuredTextStyle(.gitHub)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}


// ---------------------------------------------------------------------------
// APPENDIX — the model ends replies with `@@APPENDIX@@ {json}`. Parse it and
// show a distinct, dimmed, collapsible footnote (deferred parts + training
// notes) rather than the raw sentinel line.
// ---------------------------------------------------------------------------

struct Appendix {
    struct Deferred: Identifiable { let id = UUID(); let item: String; let tag: String }
    var deferred: [Deferred]
    var training: [String]
    var isEmpty: Bool { deferred.isEmpty && training.isEmpty }

    static func parse(_ s: String) -> Appendix? {
        guard let start = s.firstIndex(of: "{") else { return nil }        // nil while the JSON is still streaming in
        let json = String(s[start...])
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let defs = (obj["deferred"] as? [[String: Any]] ?? []).compactMap { d -> Deferred? in
            let item = d["item"] as? String ?? ""; return item.isEmpty ? nil : Deferred(item: item, tag: d["tag"] as? String ?? "")
        }
        let train = (obj["training"] as? [String] ?? []).filter { !$0.isEmpty }
        let a = Appendix(deferred: defs, training: train)
        return a.isEmpty ? nil : a
    }
}

struct AppendixView: View {
    let a: Appendix
    @State private var open = false
    init(_ a: Appendix) { self.a = a }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Button { open.toggle() } label: {
                HStack(spacing: 4) {
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .bold))
                    Image(systemName: "tray.full").font(.system(size: 10))
                    Text(summary).font(.caption2.weight(.medium))
                }.foregroundStyle(.secondary)
            }.buttonStyle(.plain)
            if open {
                if !a.deferred.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("DEFERRED").font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary).tracking(0.5)
                        ForEach(a.deferred) { d in
                            HStack(alignment: .top, spacing: 4) {
                                Text("• \(d.item)").font(.caption2).foregroundStyle(.primary.opacity(0.8)).fixedSize(horizontal: false, vertical: true)
                                if !d.tag.isEmpty { Text(d.tag).font(.caption2.italic()).foregroundStyle(.secondary) }
                            }
                        }
                    }
                }
                if !a.training.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("FROM TRAINING (not in wiki)").font(.system(size: 9, weight: .bold)).foregroundStyle(.orange.opacity(0.85)).tracking(0.5)
                        ForEach(Array(a.training.enumerated()), id: \.offset) { _, t in
                            Text("• \(t)").font(.caption2).foregroundStyle(.primary.opacity(0.8)).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
        .padding(8).frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.secondary.opacity(0.2)))
    }
    var summary: String {
        var p: [String] = []
        if !a.deferred.isEmpty { p.append("\(a.deferred.count) deferred") }
        if !a.training.isEmpty { p.append("\(a.training.count) from training") }
        return "appendix — " + p.joined(separator: ", ")
    }
}
