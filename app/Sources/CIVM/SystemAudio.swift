import Foundation
import CoreAudio

// Mutes the system DEFAULT OUTPUT device while dictating, restoring the prior state afterward — so
// whatever is playing doesn't bleed into the room (or the mic) while you talk. Muting OUTPUT never
// touches the INPUT device, so AirPods stay in high-quality A2DP playback (the built-in mic is forced
// separately in AppleSpeech / the engine). Best-effort: if the device exposes no settable master
// mute, both calls are no-ops. Called only from the main actor (ChatSession).
final class SystemAudio {
    private var didMute = false
    private var priorMute: UInt32 = 0

    func mute() {
        guard !didMute, let dev = defaultOutputDevice() else { return }
        var addr = muteAddress()
        guard AudioObjectHasProperty(dev, &addr) else { return }        // device has no master mute → skip
        var settable: DarwinBoolean = false
        guard AudioObjectIsPropertySettable(dev, &addr, &settable) == noErr, settable.boolValue else { return }
        var prior: UInt32 = 0; var sz = UInt32(MemoryLayout<UInt32>.size)
        if AudioObjectGetPropertyData(dev, &addr, 0, nil, &sz, &prior) == noErr { priorMute = prior }
        var on: UInt32 = 1
        if AudioObjectSetPropertyData(dev, &addr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &on) == noErr {
            didMute = true
        }
    }

    func restore() {
        guard didMute else { return }
        didMute = false
        guard let dev = defaultOutputDevice() else { return }
        var addr = muteAddress()
        var val = priorMute                                             // put back whatever it was (usually 0 = unmuted)
        _ = AudioObjectSetPropertyData(dev, &addr, 0, nil, UInt32(MemoryLayout<UInt32>.size), &val)
    }

    private func muteAddress() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyMute,
                                   mScope: kAudioObjectPropertyScopeOutput,
                                   mElement: kAudioObjectPropertyElementMain)   // master mute
    }
    private func defaultOutputDevice() -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var dev = AudioDeviceID(0); var sz = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &sz, &dev) == noErr,
              dev != 0 else { return nil }
        return dev
    }
}
