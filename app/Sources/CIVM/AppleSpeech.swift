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
    private var tapInstalled = false             // tracks whether a tap is live on bus 0, so teardown is idempotent

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
        // Speech-recognition TCC perm (separate from mic). If denied, fail cleanly rather than proceed
        // and silently produce no transcription.
        guard await Self.authorize() else { throw Err.notAuthorized }
        do {
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

            Self.forceSystemInputToBuiltIn()   // hard guarantee: AirPods stay on A2DP (see note below)
            Self.pinBuiltInMic(engine)
            let tapFmt = engine.inputNode.outputFormat(forBus: 0)
            guard let converter = AVAudioConverter(from: tapFmt, to: fmt) else { throw Err.noFormat }
            // Install the tap from a NONISOLATED context so the real-time audio-thread callback is NOT
            // @MainActor-isolated (a closure made in this @MainActor method would trap the isolation
            // assert when the audio thread invokes it — that was the crash). installTap removes any
            // pre-existing tap first; mark tapInstalled so teardownAudio is idempotent.
            Self.installTap(on: engine, tapFormat: tapFmt, converter: converter, analyzerFormat: fmt, into: b)
            tapInstalled = true
            engine.prepare()
            try engine.start()
        } catch {
            // A superseded or failed start must not leak a running engine, an installed tap, or a live
            // analyzer/results task — tear the whole thing down before rethrowing.
            resultsTask?.cancel()
            builder?.finish()
            teardownAudio()
            reset()
            throw error
        }
    }

    // Chord-up: stop capture, flush finals, return the whole transcript.
    func stop() async -> String {
        teardownAudio()                           // always tear down capture, regardless of finalize outcome
        builder?.finish()
        var finalizeOK = false
        do { try await analyzer?.finalizeAndFinishThroughEndOfInput(); finalizeOK = true }
        catch { finalizeOK = false }
        // Bound the tail wait: if finalize threw it may never close `transcriber.results`, so the
        // consumer's `for try await` would hang forever and wedge the caller. Only wait (with a 2s cap)
        // when finalize succeeded; otherwise skip straight to cancelling the consumer.
        if finalizeOK, let rt = resultsTask {
            await withTaskGroup(of: Void.self) { g in
                g.addTask { await rt.value }                       // the tail final results landing
                g.addTask { try? await Task.sleep(for: .seconds(2)) } // …raced against a timeout
                _ = await g.next()                                // take whichever finishes first
                g.cancelAll()
            }
        }
        resultsTask?.cancel()                     // stop the consumer regardless (no-op if it already ended)
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
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }   // only remove a tap we actually installed
        Self.restoreSystemInput()   // put the user's input device (AirPods) back
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
        engine.inputNode.removeTap(onBus: 0)   // never double-install on bus 0 (AVFoundation traps: 'nullptr == Tap()')
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
    // ---- system default-input override (the reliable guarantee) ----
    // Setting the input node's CurrentDevice alone doesn't reliably keep the AirPods on A2DP — while
    // they're the system Default Input Device macOS holds them in the muffled HFP call profile, and
    // Apple's SpeechAnalyzer may open its own audio path. So for the dictation's duration we point the
    // SYSTEM default input at the built-in mic, then restore the user's device on teardown. Accessed only
    // during serialized start/teardown, so the saved value needs no locking.
    nonisolated(unsafe) private static var savedDefaultInput: AudioDeviceID? = nil
    nonisolated private static func forceSystemInputToBuiltIn() {
        guard let builtin = builtInMicID() else { NSLog("[Rally] built-in mic not found (default-input override)"); return }
        let cur = defaultInputDevice()
        if cur == builtin { return }                     // already built-in → nothing to change/restore
        savedDefaultInput = cur
        setDefaultInputDevice(builtin)
    }
    nonisolated private static func restoreSystemInput() {
        guard let saved = savedDefaultInput else { return }
        setDefaultInputDevice(saved); savedDefaultInput = nil
    }
    nonisolated private static func defaultInputDevice() -> AudioDeviceID {
        var id = AudioDeviceID(0); var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id)
        return id
    }
    nonisolated private static func setDefaultInputDevice(_ id: AudioDeviceID) {
        var dev = id
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let st = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
                                            UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
        NSLog("[Rally] set default input device=\(id) status=\(st)")
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

    enum Err: Error { case noFormat, unsupportedLocale, notAuthorized }
}
