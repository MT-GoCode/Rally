import AppKit
import Carbon.HIToolbox

// Proper macOS global hotkeys via RegisterEventHotKey (Carbon). The window server routes ONLY the exact
// registered key+modifier combos to us (press AND release), so — unlike a CGEventTap — we never see or
// touch any other keystroke and never interfere with Karabiner or any other app. Requires a REAL key
// (modifier-only chords are impossible here, which is the whole point). Lives in the app; on each event
// it just pokes the engine's /trigger endpoint.
@MainActor
final class HotkeyManager {
    struct Combo { let keycode: UInt32; let mods: UInt32 }   // Carbon keycode + modifier mask
    private struct Entry { let ref: EventHotKeyRef?; let onPress: () -> Void; let onRelease: (() -> Void)? }
    private var entries: [UInt32: Entry] = [:]
    private var nextID: UInt32 = 1
    private var handlerRef: EventHandlerRef?

    // ANSI name → virtual keycode (the inverse of Settings' keyCodeToName).
    static let nameToKeycode: [String: UInt16] =
        Dictionary(uniqueKeysWithValues: keyCodeToName.map { ($1, $0) })

    // Parse a binding string "ctrl+alt+space" / "cmd+shift+2" → Combo (nil if no real base key).
    static func parse(_ s: String) -> Combo? {
        let parts = s.lowercased().split(separator: "+").map(String.init)
        guard let base = parts.last, let kc = nameToKeycode[base] else { return nil }
        var mods: UInt32 = 0
        for p in parts.dropLast() {
            switch p {
            case "cmd":   mods |= UInt32(cmdKey)
            case "ctrl":  mods |= UInt32(controlKey)
            case "alt":   mods |= UInt32(optionKey)
            case "shift": mods |= UInt32(shiftKey)
            default: break
            }
        }
        return Combo(keycode: UInt32(kc), mods: mods)
    }

    func clear() {
        for (_, e) in entries { if let r = e.ref { UnregisterEventHotKey(r) } }
        entries.removeAll()
    }

    // Register a combo. onRelease is optional (only voice hold-to-talk needs the key-up).
    @discardableResult
    func register(_ combo: Combo, onPress: @escaping () -> Void, onRelease: (() -> Void)? = nil) -> Bool {
        installHandler()
        let id = nextID; nextID &+= 1
        var ref: EventHotKeyRef?
        let hkID = EventHotKeyID(signature: OSType(0x52414c59), id: id)   // 'RALY'
        let st = RegisterEventHotKey(combo.keycode, combo.mods, hkID, GetEventDispatcherTarget(), 0, &ref)
        guard st == noErr else { NSLog("[Rally] RegisterEventHotKey failed (\(st)) for kc=\(combo.keycode)"); return false }
        entries[id] = Entry(ref: ref, onPress: onPress, onRelease: onRelease)
        return true
    }

    // One process-wide Carbon handler for pressed + released; dispatches by hotkey id.
    private func installHandler() {
        guard handlerRef == nil else { return }
        var specs = [EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
                     EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))]
        InstallEventHandler(GetEventDispatcherTarget(), { (_, event, userData) -> OSStatus in
            guard let event, let userData else { return noErr }
            var hkID = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hkID)
            let pressed = GetEventKind(event) == UInt32(kEventHotKeyPressed)
            let mgr = Unmanaged<HotkeyManager>.fromOpaque(userData).takeUnretainedValue()
            MainActor.assumeIsolated { mgr.dispatch(hkID.id, pressed) }   // Carbon dispatches on the main runloop
            return noErr
        }, 2, &specs, Unmanaged.passUnretained(self).toOpaque(), &handlerRef)
    }

    private func dispatch(_ id: UInt32, _ pressed: Bool) {
        guard let e = entries[id] else { return }
        if pressed { e.onPress() } else { e.onRelease?() }
    }
}
