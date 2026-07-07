import Foundation

// Content block — the interro-verbatim stream shape (text | base64 image).
struct Block: Codable, Identifiable, Equatable {
    var id = UUID()
    var type: String            // "text" | "image"
    var text: String?           // for text
    var mediaType: String?      // for image
    var data: String?           // base64, no data: prefix
    enum CodingKeys: String, CodingKey { case type, text, source }
    struct Source: Codable { var type = "base64"; var media_type: String; var data: String }
    init(text: String) { self.type = "text"; self.text = text }
    init(mediaType: String, data: String) { self.type = "image"; self.mediaType = mediaType; self.data = data }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        type = try c.decode(String.self, forKey: .type)
        text = try? c.decode(String.self, forKey: .text)
        if let s = try? c.decode(Source.self, forKey: .source) { mediaType = s.media_type; data = s.data }
    }
    func encode(to e: Encoder) throws {
        var c = e.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        if type == "text" { try c.encode(text ?? "", forKey: .text) }
        else { try c.encode(Source(media_type: mediaType ?? "image/png", data: data ?? ""), forKey: .source) }
    }
}

// Spawns + talks to the local Python MLX engine over localhost.
// A single transcript message on the wire: role + interleaved content blocks (text | image).
struct ChatMessage: Encodable { let role: String; let content: [Block] }
// {messages, reminder} — the shape POST /chat and POST /voice/context both take.
struct ChatReq: Encodable { let messages: [ChatMessage]; let reminder: [Block]; let reminderMode: String }
// A capture event pushed by the engine's screenshot / copy-to-chat shortcuts.
struct Capture: Decodable { var kind: String; var data: String?; var text: String? }   // kind: "image" | "text"
// GET /voice/poll payload (captures = the events channel, drained by /voice/captures-ack).
struct VoicePoll: Decodable {
    var state: String; var partial: String; var final: String?; var seq: Int
    var captures: [Capture]? = nil
}

// The one model the engine runs: Gemma 4 26B-A4B (MoE, 262K ctx).
enum Model {
    static let dirName = "gemma-4-26b-a4b-4bit"
    static let weightsGB = 16.0
}

@MainActor final class Engine: ObservableObject {
    @Published var status = "starting…"     // human-readable engine state
    @Published var ready = false
    @Published var parakeet = false          // ASR model loaded (from /health.parakeet)
    private var proc: Process?
    private let port = 5177
    private var base: String { "http://127.0.0.1:\(port)" }
    private let root = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("code/contextualized_instant_voice_models/engine")

    func start() {
        let py = root.appendingPathComponent(".venv/bin/python")
        let modelPath = root.appendingPathComponent("models/\(Model.dirName)")
        let p = Process()
        p.executableURL = py
        p.arguments = [root.appendingPathComponent("serve.py").path, modelPath.path, String(port)]
        p.standardOutput = FileHandle.nullDevice
        let logURL = root.appendingPathComponent("serve.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        p.standardError = try? FileHandle(forWritingTo: logURL)
        do { try p.run(); proc = p } catch { status = "failed to launch engine: \(error.localizedDescription)"; return }
        Task { await pollHealth() }
    }

    func stop() { proc?.terminate() }

    // POST /precache — re-warm the KV for the next message with a (new) reminder mode.
    func precache(messages: [ChatMessage], reminder: [Block], reminderMode: String) async {
        struct Req: Encodable { let messages: [ChatMessage]; let reminder: [Block]; let reminderMode: String }
        let body = (try? JSONEncoder().encode(Req(messages: messages, reminder: reminder, reminderMode: reminderMode))) ?? Data("{}".utf8)
        _ = try? await post("/precache", body: body, timeout: 30)
    }

    // Read the engine's post-response precache state ("idle"|"working"|"done") from /health.
    func precacheState() async -> String {
        guard let d = try? await get("/health"),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return "idle" }
        return j["precache"] as? String ?? "idle"
    }

    // Poll /health until BOTH models load (Gemma first → ready; parakeet loads after → parakeet).
    private func pollHealth() async {
        for _ in 0..<600 {
            if let d = try? await get("/health"),
               let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                if j["loaded"] as? Bool == true, !ready {
                    status = "model ready — \(j["model"] as? String ?? "")"; ready = true
                }
                parakeet = j["parakeet"] as? Bool ?? false
                if ready && parakeet { return }
            }
            try? await Task.sleep(for: .seconds(1))
        }
        if !ready { status = "engine did not become ready (see serve.log)" }
    }

    // POST /new — drop the current pin; the engine installs its EMPTY baseline (pin_len=0). Instant.
    func reset() async { _ = try? await post("/new", body: Data("{}".utf8), timeout: 30) }

    // POST /pin — returns (tokens, overLimit). Long (vision tower runs once).
    func pin(system: [Block], context: [Block]) async throws -> (Int, Bool) {
        let body = try JSONEncoder().encode(["system": system, "context": context, "history": [Block]()])
        let d = try await post("/pin", body: body, timeout: 900)
        let j = try JSONSerialization.jsonObject(with: d) as? [String: Any] ?? [:]
        return (j["tokens"] as? Int ?? 0, j["overLimit"] as? Bool ?? false)
    }

    // POST /chat — history is APP-OWNED: `messages` is the full clean transcript (incl. the current
    // question as the last element); `reminder` rides on the tail of the last user message. Streams
    // token deltas via onDelta; returns the final `{done,…}` meta line. Cancel by cancelling the calling
    // Task — the bytes loop throws and the closed connection tells the engine to stop generating.
    func chat(messages: [ChatMessage], reminder: [Block], reminderMode: String,
              onDelta: @escaping (String) -> Void) async throws -> [String: Any] {
        var req = URLRequest(url: URL(string: base + "/chat")!)
        req.httpMethod = "POST"; req.timeoutInterval = 120
        req.setValue("application/json", forHTTPHeaderField: "content-type")
        req.httpBody = try JSONEncoder().encode(ChatReq(messages: messages, reminder: reminder, reminderMode: reminderMode))
        var meta: [String: Any] = [:]
        let (bytes, _) = try await URLSession.shared.bytes(for: req)
        for try await line in bytes.lines {
            try Task.checkCancellation()                     // observe interrupt promptly between lines
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
            if let delta = obj["delta"] as? String { onDelta(delta) }
            else if obj["done"] as? Bool == true { meta = obj }
        }
        return meta
    }

    // ---- Voice·Text control channel (engine owns mic/parakeet/tap/overlay; the app just steers it) ----

    // POST /voice/config — post on launch, on settings change, and entering/leaving a chat.
    // voiceEnabled gates the voice chord (Voice·Text only); captureEnabled gates the screenshot/copy
    // bindings (active whenever a chat is open, ANY mode). Bindings are engine strings ("cmd+1"/"mouse3").
    func voiceConfig(voiceEnabled: Bool, captureEnabled: Bool, submode: String, streaming: Bool,
                     key: String, shotBinding: String, shotStyle: String, copyBinding: String) async {
        let body = (try? JSONSerialization.data(withJSONObject: [
            "voiceEnabled": voiceEnabled, "captureEnabled": captureEnabled,
            "submode": submode, "streaming": streaming, "key": key,
            "shot": ["binding": shotBinding, "style": shotStyle],
            "copy": ["binding": copyBinding]])) ?? Data("{}".utf8)
        _ = try? await post("/voice/config", body: body, timeout: 5)
    }
    // POST /voice/captures-ack — drop the first `count` captures we just drained from a poll.
    func voiceCapturesAck(count: Int) async {
        let body = (try? JSONSerialization.data(withJSONObject: ["count": count])) ?? Data("{}".utf8)
        _ = try? await post("/voice/captures-ack", body: body, timeout: 5)
    }
    // POST /voice/prefill — prefill Gemma's KV with the in-progress Apple transcript while the user
    // talks. Returns the token count now sitting past the pin (the live "pre-sent" number).
    func voicePrefill(messages: [ChatMessage], partial: String) async -> Int {
        struct Req: Encodable { let messages: [ChatMessage]; let partial: String }
        let body = (try? JSONEncoder().encode(Req(messages: messages, partial: partial))) ?? Data("{}".utf8)
        guard let d = try? await post("/voice/prefill", body: body, timeout: 10),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return 0 }
        return j["fed"] as? Int ?? 0
    }
    // GET /voice/poll — app polls ~10Hz while in Voice·Text.
    func voicePoll() async throws -> VoicePoll {
        try JSONDecoder().decode(VoicePoll.self, from: try await get("/voice/poll"))
    }
    // POST /voice/ack — acknowledge a consumed final transcript by its seq.
    func voiceAck(seq: Int) async {
        let body = (try? JSONSerialization.data(withJSONObject: ["seq": seq])) ?? Data("{}".utf8)
        _ = try? await post("/voice/ack", body: body, timeout: 5)
    }

    private func get(_ path: String) async throws -> Data {
        var req = URLRequest(url: URL(string: base + path)!); req.timeoutInterval = 3
        return try await URLSession.shared.data(for: req).0
    }
    private func post(_ path: String, body: Data, timeout: TimeInterval) async throws -> Data {
        var req = URLRequest(url: URL(string: base + path)!)
        req.httpMethod = "POST"; req.timeoutInterval = timeout
        req.setValue("application/json", forHTTPHeaderField: "content-type"); req.httpBody = body
        return try await URLSession.shared.data(for: req).0
    }
}

// Available-memory gate (unified memory on Apple Silicon).
enum Mem {
    static var totalGB: Double { Double(ProcessInfo.processInfo.physicalMemory) / 1e9 }
    static var availableGB: Double {
        var stats = vm_statistics64(); var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard kr == KERN_SUCCESS else { return totalGB }
        let page = Double(getpagesize())
        let freeish = Double(stats.free_count + stats.inactive_count + stats.purgeable_count) * page
        return freeish / 1e9
    }
    // gate follows the DEFAULT model (Settings → Model) since that's what start() loads at launch
    static var modelGB: Double { Model.weightsGB }
    static let headroomGB = 10.0
    static var enough: Bool { availableGB >= modelGB + headroomGB }
}
