import SwiftUI
import WebKit

// ---------------------------------------------------------------------------
// The conversation transcript, rendered by a SINGLE WKWebView (markdown-it + KaTeX + highlight.js,
// all bundled offline). One web process per open chat — SwiftUI recreates this view (fresh WKWebView →
// fresh WebContent process) whenever store.chat.id changes via `.id(chat.id)` at the call site, and tears
// the old one down (terminating its process) → bounded memory, no accumulation, and markdown/math/table
// layout runs in the web process, never on the SwiftUI main thread (that was the beachball).
// Stage 1: render committed messages. (Input box + token line + streaming/keyboard bridge come next.)
// ---------------------------------------------------------------------------

// Build the JSON the web page renders: [{role,text,images:[dataURI],interrupted,isInterruption}].
func conversationJSON(_ messages: [Msg]) -> String {
    let arr: [[String: Any]] = messages.map { m in
        var d: [String: Any] = ["role": m.role, "text": m.text,
                                "interrupted": m.interrupted, "isInterruption": m.isInterruption]
        let imgs = m.content.filter { $0.type == "image" }
            .map { "data:\($0.mediaType ?? "image/png");base64,\($0.data ?? "")" }
        if !imgs.isEmpty { d["images"] = imgs }
        return d
    }
    guard let data = try? JSONSerialization.data(withJSONObject: arr),
          let s = String(data: data, encoding: .utf8) else { return "[]" }
    return s
}

struct ChatWebView: NSViewRepresentable {
    let messagesJSON: String
    let convStart: Int
    var streamingText: String = ""    // live in-progress reply (empty when not generating)
    var isBusy: Bool = false

    func makeNSView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        let wv = WKWebView(frame: .zero, configuration: cfg)
        wv.navigationDelegate = context.coordinator
        if #available(macOS 12.0, *) { wv.underPageBackgroundColor = NSColor(red: 0.118, green: 0.118, blue: 0.118, alpha: 1) }
        context.coordinator.webView = wv
        if let url = Bundle.module.url(forResource: "chat", withExtension: "html", subdirectory: "web") {
            wv.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        return wv
    }

    func updateNSView(_ wv: WKWebView, context: Context) {
        context.coordinator.render(messagesJSON, convStart)
        context.coordinator.stream(isBusy ? streamingText : nil)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKNavigationDelegate {
        weak var webView: WKWebView?
        private var loaded = false
        private var pending: (String, Int)?
        private var last: (String, Int)?

        private var lastStream: String? = ""

        func render(_ json: String, _ convStart: Int) {
            if let l = last, l.0 == json, l.1 == convStart { return }   // dedup — don't re-render unchanged
            last = (json, convStart)
            guard loaded, let wv = webView else { pending = (json, convStart); return }
            wv.evaluateJavaScript("window.rally.render(\(json), \(convStart))", completionHandler: nil)
        }

        // nil = not generating (end any live stream); non-nil = the growing reply text.
        func stream(_ text: String?) {
            if text == lastStream { return }
            lastStream = text
            guard loaded, let wv = webView else { return }
            if let t = text, let j = try? String(data: JSONSerialization.data(withJSONObject: [t]), encoding: .utf8) {
                wv.evaluateJavaScript("window.rally.setStream(\(j)[0])", completionHandler: nil)   // JSON-escape the text
            } else {
                wv.evaluateJavaScript("window.rally.endStream()", completionHandler: nil)
            }
        }

        func webView(_ wv: WKWebView, didFinish navigation: WKNavigation!) {
            loaded = true
            if let p = pending { pending = nil; wv.evaluateJavaScript("window.rally.render(\(p.0), \(p.1))", completionHandler: nil) }
            if let t = lastStream { wv.evaluateJavaScript("window.rally.setStream(\(jsonStr(t)))", completionHandler: nil) }
        }
    }
}

private func jsonStr(_ s: String) -> String {
    guard let d = try? JSONSerialization.data(withJSONObject: [s]), let j = String(data: d, encoding: .utf8) else { return "\"\"" }
    return "\(j)[0]"
}
