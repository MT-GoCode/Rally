import SwiftUI
import Foundation
import Observation

// ---------------------------------------------------------------------------
// Prompt library — reusable SYSTEM prompts and REMINDERs, each an ordered
// content-block stream (the same Block shape the panes edit). Stored one JSON
// file per prompt under Application Support/…/prompts/<id>.json.
// ---------------------------------------------------------------------------

struct SavedPrompt: Codable, Identifiable {
    var id = UUID()
    var name: String
    var kind: String            // "system" | "reminder"
    var blocks: [Block]
}

// which pane's "Save current as…" sheet is open (drives the chat-sidebar save flow)
enum SaveTarget: String, Identifiable { case system, reminder; var id: String { rawValue } }

@MainActor @Observable final class PromptLib {
    var prompts: [SavedPrompt] = []

    private var dir: URL {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("contextualized_instant_voice_models/prompts")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    init() { load() }

    func load() {
        let fm = FileManager.default
        let files = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }
        prompts = files
            .compactMap { u in (try? Data(contentsOf: u)).flatMap { try? JSONDecoder().decode(SavedPrompt.self, from: $0) } }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func write(_ p: SavedPrompt) {
        if let d = try? JSONEncoder().encode(p) { try? d.write(to: dir.appendingPathComponent("\(p.id).json")) }
    }
    func save(_ p: SavedPrompt) { write(p); load() }
    func delete(_ p: SavedPrompt) {
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("\(p.id).json")); load()
    }
}

// ---- HOME: "PROMPT LIBRARY" section (segmented System / Reminders filter) ----
struct PromptLibrarySection: View {
    @Environment(PromptLib.self) var lib
    @State private var kind = "system"
    @State private var editing: SavedPrompt? = nil     // non-nil → editor sheet (new or existing)

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("PROMPT LIBRARY").font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                Picker("", selection: $kind) {
                    Text("System prompts").tag("system")
                    Text("Reminders").tag("reminder")
                }.pickerStyle(.segmented).labelsHidden().fixedSize()
            }
            let rows = lib.prompts.filter { $0.kind == kind }
            if rows.isEmpty {
                Text(kind == "system" ? "no saved system prompts" : "no saved reminders")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ForEach(rows) { p in
                PromptRow(p: p, onEdit: { editing = p }, onDelete: { lib.delete(p) })
            }
            HStack {
                Button("+ New system prompt") { editing = SavedPrompt(name: "", kind: "system", blocks: [Block(text: "")]) }.font(.caption)
                Button("+ New reminder") { editing = SavedPrompt(name: "", kind: "reminder", blocks: [Block(text: "")]) }.font(.caption)
            }.padding(.top, 2)
        }
        .sheet(item: $editing) { p in PromptEditor(prompt: p) { lib.save($0) } }
    }
}

// one library row: name + kind badge + Edit + two-click inline Delete confirm
struct PromptRow: View {
    let p: SavedPrompt
    let onEdit: () -> Void
    let onDelete: () -> Void
    @State private var confirmDelete = false

    var body: some View {
        HStack(spacing: 8) {
            Text(p.name.isEmpty ? "(unnamed)" : p.name).font(.callout)
            Text(p.kind == "system" ? "system" : "reminder")
                .font(.system(size: 9, weight: .bold)).foregroundStyle(.secondary)
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(Capsule().fill(Color.secondary.opacity(0.15)))
            Spacer()
            Button("Edit", action: onEdit).font(.caption)
            if confirmDelete {
                Button("Confirm") { onDelete() }.font(.caption).foregroundStyle(.red)
                Button("Cancel") { confirmDelete = false }.font(.caption)
            } else {
                Button("Delete") { confirmDelete = true }.font(.caption).foregroundStyle(.red)
            }
        }
        .padding(.vertical, 4).padding(.horizontal, 8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.10)))
    }
}

// editor sheet: name + the shared BlockStream editor (reused from App.swift)
struct PromptEditor: View {
    @Environment(\.dismiss) private var dismiss
    let original: SavedPrompt
    let onSave: (SavedPrompt) -> Void
    @State private var name: String
    @State private var blocks: [Block]

    init(prompt: SavedPrompt, onSave: @escaping (SavedPrompt) -> Void) {
        self.original = prompt; self.onSave = onSave
        _name = State(initialValue: prompt.name)
        _blocks = State(initialValue: prompt.blocks)
    }
    private var kindLabel: String { original.kind == "system" ? "system prompt" : "reminder" }
    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(original.name.isEmpty ? "New \(kindLabel)" : "Edit \(kindLabel)").font(.title3.bold())
            TextField("name", text: $name).textFieldStyle(.roundedBorder)
            BlockStream(blocks: $blocks)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") {
                    var p = original; p.name = trimmed; p.blocks = blocks
                    onSave(p); dismiss()
                }.keyboardShortcut(.defaultAction).disabled(trimmed.isEmpty)
            }
        }
        .padding(20).frame(width: 560, height: 480)
    }
}

// tiny name-only sheet for the chat sidebar's "Save current as…" (blocks come from the live pane)
struct SavePromptSheet: View {
    @Environment(\.dismiss) private var dismiss
    let kind: String
    let onSave: (String) -> Void
    @State private var name = ""
    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Save current \(kind == "system" ? "system prompt" : "reminder") as…").font(.title3.bold())
            TextField("name", text: $name).textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { onSave(trimmed); dismiss() }
                    .keyboardShortcut(.defaultAction).disabled(trimmed.isEmpty)
            }
        }.padding(20).frame(width: 380)
    }
}
