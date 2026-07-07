import Foundation
import AVFoundation
import Speech
import CoreAudio

// On-device live dictation via Apple's SpeechAnalyzer/SpeechTranscriber (macOS 26+). Used for the
// Voice·Text STREAMING submode — it streams natively with a real VAD and finalizes instantly at
// key-release (no batch re-transcribe latency), unlike parakeet-mlx streaming which hallucinated.
// Transcribe-after submode still uses Parakeet batch in the Python engine.
//
// Flow: start(localeID:) on chord-down → live `partial` updates (finalized + volatile) →
// stop() on chord-up returns the finalized transcript → sent through the AskPipeline.
@available(macOS 26, *)
@MainActor
final class AppleSpeech: ObservableObject {
    @Published private(set) var partial = ""     // finalized + in-progress volatile text, for live display

    private let engine = AVAudioEngine()
    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var builder: AsyncStream<AnalyzerInput>.Continuation?
    private var resultsTask: Task<Void, Never>?
    private var finalized = AttributedString("")
    private var volatileText = AttributedString("")

    static func supportedLocaleIDs() async -> [String] {
        await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }.sorted()
    }

    // Request Speech-recognition authorization (mic is requested elsewhere). `nonisolated` is REQUIRED:
    // requestAuthorization invokes its completion on a background TCC/XPC thread, so the continuation
    // closure must NOT be @MainActor-isolated (else Swift's isolation assert traps → crash on start).
    nonisolated static func authorize() async -> Bool {
        await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0 == .authorized) }
        }
    }

    func start(localeID: String) async throws {
        _ = await Self.authorize()          // prompt for Speech recognition on first use (separate TCC perm)
        finalized = AttributedString(""); volatileText = AttributedString(""); partial = ""
        let locale = Locale(identifier: localeID)
        let t = SpeechTranscriber(locale: locale, transcriptionOptions: [],
                                  reportingOptions: [.volatileResults], attributeOptions: [])
        transcriber = t
        try await Self.ensureModel(t, locale: locale)

        let a = SpeechAnalyzer(modules: [t])
        analyzer = a
        guard let fmt = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [t]) else {
            throw Err.noFormat
        }
        let (seq, b) = AsyncStream<AnalyzerInput>.makeStream()
        builder = b
        try await a.start(inputSequence: seq)

        // Results consumer — Task inherits @MainActor, so it updates @Published directly.
        resultsTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await r in t.results {
                    self.ingest(r.text, isFinal: r.isFinal)
                }
            } catch { /* analyzer finished / cancelled */ }
        }

        Self.pinBuiltInMic(engine)
        let tapFmt = engine.inputNode.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: tapFmt, to: fmt) else { throw Err.noFormat }
        // Install the tap from a NONISOLATED context so the real-time audio-thread callback is NOT
        // @MainActor-isolated (a closure made in this @MainActor method would trap the isolation
        // assert when the audio thread invokes it — that was the crash).
        Self.installTap(on: engine, tapFormat: tapFmt, converter: converter, analyzerFormat: fmt, into: b)
        engine.prepare()
        try engine.start()
    }

    // Chord-up: stop capture, flush finals, return the whole transcript.
    func stop() async -> String {
        teardownAudio()
        builder?.finish()
        try? await analyzer?.finalizeAndFinishThroughEndOfInput()
        await resultsTask?.value                  // let the tail final results land
        let text = String(finalized.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        reset()
        return text
    }

    // ESC: drop everything, no send.
    func cancel() async {
        teardownAudio()
        builder?.finish()
        await analyzer?.cancelAndFinishNow()
        resultsTask?.cancel()
        reset()
    }

    private func ingest(_ text: AttributedString, isFinal: Bool) {
        if isFinal { finalized.append(text); volatileText = AttributedString("") }
        else { volatileText = text }
        partial = (String(finalized.characters) + String(volatileText.characters))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func teardownAudio() {
        if engine.isRunning { engine.stop() }
        engine.inputNode.removeTap(onBus: 0)
    }
    private func reset() { analyzer = nil; transcriber = nil; builder = nil; resultsTask = nil }

    // ---- model asset ----
    private static func ensureModel(_ t: SpeechTranscriber, locale: Locale) async throws {
        let want = locale.identifier(.bcp47)
        guard await SpeechTranscriber.supportedLocales.contains(where: { $0.identifier(.bcp47) == want }) else {
            throw Err.unsupportedLocale
        }
        if await SpeechTranscriber.installedLocales.contains(where: { $0.identifier(.bcp47) == want }) { return }
        if let req = try await AssetInventory.assetInstallationRequest(supporting: [t]) {
            try await req.downloadAndInstall()
        }
    }

    // Install the mic tap from a nonisolated context → the audio-thread callback carries NO actor
    // isolation (the tap only converts + yields to the Sendable continuation).
    nonisolated private static func installTap(on engine: AVAudioEngine, tapFormat: AVAudioFormat,
                                               converter: AVAudioConverter, analyzerFormat: AVAudioFormat,
                                               into builder: AsyncStream<AnalyzerInput>.Continuation) {
        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: tapFormat) { buf, _ in
            if let out = convert(buf, converter, analyzerFormat) {
                builder.yield(AnalyzerInput(buffer: out))
            }
        }
    }

    // ---- audio conversion (tap format → analyzer format); called on the real-time audio thread ----
    nonisolated private static func convert(_ buffer: AVAudioPCMBuffer, _ converter: AVAudioConverter,
                                _ format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let ratio = format.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: cap) else { return nil }
        var fed = false; var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        return err == nil ? out : nil
    }

    // ---- force the built-in mic (else macOS flips AirPods A2DP→HFP → degraded playback) ----
    // Setting the CURRENT DEVICE on the input node's audio unit is what keeps AVAudioEngine off the
    // Bluetooth mic. `@discardableResult` returns whether it stuck (logged for diagnosis).
    @discardableResult
    nonisolated private static func pinBuiltInMic(_ engine: AVAudioEngine) -> Bool {
        guard let dev = builtInMicID() else { NSLog("[Rally] built-in mic not found"); return false }
        guard let unit = engine.inputNode.audioUnit else { NSLog("[Rally] input node has no audioUnit"); return false }
        var id = dev
        let st = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                      &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        NSLog("[Rally] pin built-in mic id=\(dev) status=\(st)")
        return st == noErr
    }
    nonisolated private static func builtInMicID() -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        guard count > 0 else { return nil }
        var ids = [AudioDeviceID](repeating: 0, count: count)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
        for id in ids where hasInputStreams(id) {
            if transportType(id) == kAudioDeviceTransportTypeBuiltIn { return id }   // the MacBook mic
        }
        return nil
    }
    nonisolated private static func hasInputStreams(_ id: AudioDeviceID) -> Bool {
        var a = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
        var s: UInt32 = 0; AudioObjectGetPropertyDataSize(id, &a, 0, nil, &s); return s > 0
    }
    nonisolated private static func transportType(_ id: AudioDeviceID) -> UInt32 {
        var t: UInt32 = 0; var s = UInt32(MemoryLayout<UInt32>.size)
        var a = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        AudioObjectGetPropertyData(id, &a, 0, nil, &s, &t); return t
    }

    enum Err: Error { case noFormat, unsupportedLocale }
}
