import SwiftUI
import MarkdownUI

// ---------------------------------------------------------------------------
// GFM Markdown → SwiftUI, rendered by swift-markdown-ui (MarkdownUI): tables,
// blockquotes, nested/task lists and fenced code all render properly. Before
// handing text to MarkdownUI we run the LaTeX-lite pass (mathClean) so Gemma's
// `$q_0 \in Q$`-style math reads cleanly without KaTeX — the pass skips fenced
// code so snippets stay verbatim. Re-parsed on every streaming delta; partial
// constructs (an unclosed ``` fence, half a table row) render without crashing.
// The @@APPENDIX@@ split lives in the caller (messageBubble); Appendix below
// is unchanged.
// ---------------------------------------------------------------------------

struct MarkdownText: View {
    let raw: String
    var body: some View { Markdown(MD.preprocess(raw)).markdownTheme(.civm) }
}

// Compact, system-colored GFM theme sized for chat bubbles (13pt body, tight
// spacing). Semantic colors adapt to light/dark; wide tables & code blocks
// scroll horizontally instead of overflowing the bubble.
extension Theme {
    @MainActor static let civm = Theme()
        .text { FontSize(13); ForegroundColor(.primary) }
        .code { FontFamilyVariant(.monospaced); FontSize(.em(0.92)); BackgroundColor(Color.secondary.opacity(0.14)) }
        .link { ForegroundColor(.accentColor) }
        .heading1 { c in c.label.markdownTextStyle { FontSize(16); FontWeight(.bold) }.markdownMargin(top: 8, bottom: 4) }
        .heading2 { c in c.label.markdownTextStyle { FontSize(14); FontWeight(.bold) }.markdownMargin(top: 8, bottom: 4) }
        .heading3 { c in c.label.markdownTextStyle { FontSize(13); FontWeight(.bold) }.markdownMargin(top: 6, bottom: 2) }
        .heading4 { c in c.label.markdownTextStyle { FontSize(13); FontWeight(.semibold) }.markdownMargin(top: 6, bottom: 2) }
        .paragraph { c in c.label.fixedSize(horizontal: false, vertical: true).markdownMargin(top: 0, bottom: 6) }
        .blockquote { c in
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 1).fill(Color.secondary.opacity(0.4)).frame(width: 3)
                c.label.markdownTextStyle { ForegroundColor(.secondary) }
            }
            .fixedSize(horizontal: false, vertical: true).markdownMargin(top: 0, bottom: 6)
        }
        .codeBlock { c in
            ScrollView(.horizontal, showsIndicators: false) {
                c.label
                    .fixedSize(horizontal: false, vertical: true)
                    .relativeLineSpacing(.em(0.2))
                    .markdownTextStyle { FontFamilyVariant(.monospaced); FontSize(12) }
                    .padding(10)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
            .markdownMargin(top: 2, bottom: 8)
        }
        .table { c in
            ScrollView(.horizontal, showsIndicators: false) {
                c.label
                    .fixedSize(horizontal: false, vertical: true)
                    .markdownTableBorderStyle(.init(color: Color.secondary.opacity(0.3)))
            }
            .markdownMargin(top: 2, bottom: 8)
        }
        .tableCell { c in
            c.label
                .markdownTextStyle { if c.row == 0 { FontWeight(.semibold) } }
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, 4).padding(.horizontal, 8)
        }
}

enum MD {
    // Run the LaTeX-lite cleanup over every line EXCEPT inside fenced code blocks
    // (those reach MarkdownUI verbatim). Safe on partial text: an unclosed fence
    // just leaves the tail treated as code and untouched.
    static func preprocess(_ raw: String) -> String {
        var out: [String] = []; var inFence = false
        for line in raw.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") { inFence.toggle(); out.append(line) }
            else { out.append(inFence ? line : mathClean(line)) }
        }
        return out.joined(separator: "\n")
    }

    static let sym: [String: String] = [
        "\\dots":"…","\\ldots":"…","\\cdots":"⋯","\\cdot":"·","\\bullet":"•",
        "\\rightarrow":"→","\\to":"→","\\Rightarrow":"⇒","\\leftarrow":"←","\\mapsto":"↦","\\implies":"⇒","\\iff":"⇔",
        "\\leq":"≤","\\le":"≤","\\geq":"≥","\\ge":"≥","\\neq":"≠","\\ne":"≠","\\equiv":"≡","\\approx":"≈","\\cong":"≅","\\sim":"∼",
        "\\times":"×","\\ast":"∗","\\star":"⋆","\\circ":"∘","\\cup":"∪","\\cap":"∩","\\setminus":"∖",
        "\\in":"∈","\\notin":"∉","\\subseteq":"⊆","\\subset":"⊂","\\supseteq":"⊇","\\supset":"⊃",
        "\\emptyset":"∅","\\varnothing":"∅","\\forall":"∀","\\exists":"∃","\\neg":"¬","\\wedge":"∧","\\vee":"∨","\\vdash":"⊢","\\models":"⊨",
        "\\infty":"∞","\\partial":"∂","\\nabla":"∇","\\sum":"∑","\\prod":"∏","\\int":"∫","\\pm":"±",
        "\\Sigma":"Σ","\\sigma":"σ","\\Delta":"Δ","\\delta":"δ","\\Gamma":"Γ","\\gamma":"γ","\\lambda":"λ","\\Lambda":"Λ",
        "\\mu":"μ","\\pi":"π","\\Pi":"Π","\\phi":"φ","\\varphi":"φ","\\Phi":"Φ","\\theta":"θ","\\Theta":"Θ",
        "\\alpha":"α","\\beta":"β","\\epsilon":"ε","\\varepsilon":"ε","\\omega":"ω","\\Omega":"Ω","\\rho":"ρ","\\tau":"τ",
        "\\xi":"ξ","\\zeta":"ζ","\\eta":"η","\\kappa":"κ","\\nu":"ν","\\chi":"χ","\\psi":"ψ","\\Psi":"Ψ",
        "\\langle":"⟨","\\rangle":"⟩","\\qquad":"  ","\\quad":" ","\\,":" ","\\;":" ","\\:":" ","\\!":"",
        "\\\\":" ","\\{":"{","\\}":"}","\\%":"%","\\_":"_","\\#":"#","\\&":"&","\\mid":"∣","\\|":"‖"
    ]
    static let subT: [Character: Character] = ["0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉","+":"₊","-":"₋","=":"₌","(":"₍",")":"₎","a":"ₐ","e":"ₑ","i":"ᵢ","j":"ⱼ","o":"ₒ","x":"ₓ","n":"ₙ"]
    static let supT: [Character: Character] = ["0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","+":"⁺","-":"⁻","=":"⁼","(":"⁽",")":"⁾","n":"ⁿ","i":"ⁱ"]

    static func mathClean(_ s: String) -> String {
        var t = s.replacingOccurrences(of: "$$", with: "").replacingOccurrences(of: "$", with: "")
        for (k, v) in sym.sorted(by: { $0.key.count > $1.key.count }) { t = t.replacingOccurrences(of: k, with: v) }
        return scripts(t)
    }

    // convert `_1`, `_{12}`, `^2`, `^{10}` (and a few subscriptable letters) to unicode
    static func scripts(_ s: String) -> String {
        var out = ""; let c = Array(s); var i = 0
        while i < c.count {
            let ch = c[i]
            if (ch == "_" || ch == "^"), i + 1 < c.count {
                let table = ch == "_" ? subT : supT
                if c[i + 1] == "{" {
                    var j = i + 2, inner = ""
                    while j < c.count && c[j] != "}" { inner.append(c[j]); j += 1 }
                    if j < c.count {                                  // matched close brace → map each mappable char
                        out += inner.map { table[$0].map(String.init) ?? String($0) }.joined(); i = j + 1; continue
                    }
                } else if let m = table[c[i + 1]] { out.append(m); i += 2; continue }
            }
            out.append(ch); i += 1
        }
        return out
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
