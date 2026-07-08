import SwiftUI
import WebKit

// ---------------------------------------------------------------------------
// The conversation transcript AND the input composer, in one WKWebView (markdown-it + KaTeX +
// highlight.js, bundled offline). One web process per open chat (`.id(chat.id)` at the call site
// recreates it and tears down the old → bounded memory; all markdown/math layout runs in the web
// process, never the SwiftUI main thread). The composer lives inside so ↑/↓ moves seamlessly between
// it and the messages. Token line, reminder, and cache HUD stay in SwiftUI outside this view.
// ---------------------------------------------------------------------------

// Lets ChatSession drive the live web page (voice insert, focus, compose buffer) without importing WebKit.
@MainActor protocol ChatWebBridge: AnyObject {
    func insertText(_ t: String)
    func focusComposer()
    func setComposer(_ t: String)
    func submitComposer()
    func setThumbs(_ dataURIs: [String])
    func setBusy(_ busy: Bool)       // toggle Send ↔ Stop / Interrupt&Send + placeholder
    func streamSet(_ text: String)   // live reply text (pushed per-token, direct → smooth streaming)
    func streamEnd()                 // reply committed → remove the live bubble
}

// [{id,role,text,images:[dataURI],interrupted,isInterruption}] for the web page to render.
func conversationJSON(_ messages: [Msg]) -> String {
    let arr: [[String: Any]] = messages.map { m in
        var d: [String: Any] = ["id": m.id.uuidString, "role": m.role, "text": m.text,
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

private func jsonStr(_ s: String) -> String {
    guard let d = try? JSONSerialization.data(withJSONObject: [s]), let j = String(data: d, encoding: .utf8) else { return "\"\"" }
    return "\(j)[0]"
}

struct ChatWebView: NSViewRepresentable {
    let session: ChatSession
    let messagesJSON: String
    let convStart: Int

    func makeNSView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.userContentController.add(context.coordinator, name: "rally")
        let wv = WKWebView(frame: .zero, configuration: cfg)
        wv.navigationDelegate = context.coordinator
        if #available(macOS 12.0, *) { wv.underPageBackgroundColor = NSColor(red: 0.118, green: 0.118, blue: 0.118, alpha: 1) }
        context.coordinator.webView = wv
        context.coordinator.session = session
        session.web = context.coordinator            // register for Swift→JS (voice insert / focus / thumbs)
        if let url = Bundle.module.url(forResource: "chat", withExtension: "html", subdirectory: "web") {
            wv.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        return wv
    }

    func updateNSView(_ wv: WKWebView, context: Context) {
        context.coordinator.render(messagesJSON, convStart)
    }

    static func dismantleNSView(_ wv: WKWebView, coordinator: Coordinator) {
        wv.configuration.userContentController.removeScriptMessageHandler(forName: "rally")
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler, ChatWebBridge {
        weak var webView: WKWebView?
        weak var session: ChatSession?
        private var loaded = false
        private var pending: (String, Int)?
        private var last: (String, Int)?
        private var lastStream: String? = nil   // nil = no live stream (was "" → didFinish pushed setStream("") → a phantom "…" bubble on open)

        // ---- Swift → JS ----
        func render(_ json: String, _ convStart: Int) {
            if let l = last, l.0 == json, l.1 == convStart { return }
            last = (json, convStart)
            runJS("window.rally.render(\(json), \(convStart))")
        }
        func streamSet(_ text: String) { lastStream = text; runJS("window.rally.setStream(\(jsonStr(text)))") }
        func streamEnd() { lastStream = nil; runJS("window.rally.endStream()") }
        func insertText(_ t: String)  { runJS("window.rally.insertText(\(jsonStr(t)))") }
        func focusComposer()          { runJS("window.rally.focusComposer()") }
        func setComposer(_ t: String) { runJS("window.rally.setComposer(\(jsonStr(t)))") }
        func submitComposer()         { runJS("window.rally.submitComposer && window.rally.submitComposer()") }
        func setThumbs(_ u: [String]) {
            let arr = (try? String(data: JSONSerialization.data(withJSONObject: u), encoding: .utf8)) ?? "[]"
            runJS("window.rally.setThumbs(\(arr))")
        }
        func setBusy(_ busy: Bool) { runJS("window.rally.setBusy(\(busy))") }
        private func runJS(_ js: String) {
            guard loaded, let wv = webView else { if pending == nil { pending = ("", 0) }; return }
            wv.evaluateJavaScript(js, completionHandler: nil)
        }

        // ---- JS → Swift ----
        func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
            guard let d = m.body as? [String: Any], let action = d["action"] as? String else { return }
            switch action {
            case "send":       session?.sendText(d["text"] as? String ?? "")
            case "input":      session?.input = d["text"] as? String ?? ""
            case "focus":      session?.inputIsFocused = (d["focused"] as? Bool ?? false)
            case "copy":       let s = d["text"] as? String ?? ""; NSPasteboard.general.clearContents(); NSPasteboard.general.setString(s, forType: .string)
            case "reset":       if let id = (d["id"] as? String).flatMap(UUID.init) { session?.resetToHere(id) }
            case "pasteImage":  if let uri = d["data"] as? String { session?.attachDataURIImage(uri) }
            case "removeImage": if let i = d["index"] as? Int { session?.removePastedImageAt(i) }
            case "stop":        session?.stop()
            default: break
            }
        }

        func webView(_ wv: WKWebView, didFinish nav: WKNavigation!) {
            loaded = true
            if let l = last { wv.evaluateJavaScript("window.rally.render(\(l.0), \(l.1))", completionHandler: nil) }
            if let t = lastStream { wv.evaluateJavaScript("window.rally.setStream(\(jsonStr(t)))", completionHandler: nil) }
            pending = nil
            if let s = session { setBusy(s.busy); s.syncThumbs() }   // push current state to the freshly-loaded page
        }
    }
}
