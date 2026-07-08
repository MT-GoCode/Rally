import AppKit
import PDFKit
import SwiftUI

// Rasterize a PDF into the content-block stream: "Page 1" text + page image, "Page 2" + image, …
// (same interleaved shape as CONTEXT/the Sipser seed, so the model can see + reference each page).
func pdfToBlocks(_ url: URL) -> [Block] {
    guard let doc = PDFDocument(url: url) else { return [] }
    var out: [Block] = []
    for i in 0..<doc.pageCount {
        guard let page = doc.page(at: i) else { continue }
        let b = page.bounds(for: .mediaBox)
        // render at 2× for legibility; imageToBase64 downscales/JPEGs anything oversized.
        let img = page.thumbnail(of: NSSize(width: b.width * 2, height: b.height * 2), for: .mediaBox)
        guard let tiff = img.tiffRepresentation, let (b64, mt) = imageToBase64(tiff) else { continue }
        out.append(Block(text: "Page \(i + 1)"))
        out.append(Block(mediaType: mt, data: b64))
    }
    return out
}

extension Block {
    // decode an image block's base64 to an NSImage for inline display (nil for text blocks).
    // NOTE: synchronous — DO NOT call in a view body (it beachballs on multi-image contexts). Use
    // AsyncBlockImage, which decodes off the main thread and caches. Kept for non-render callers.
    var nsImage: NSImage? {
        guard type == "image", let d = Data(base64Encoded: data ?? "") else { return nil }
        return NSImage(data: d)
    }
}

// Cross-boundary box: NSImage isn't Sendable, but we only ever read it, so shipping a decoded one back
// from a detached task is safe.
struct SendableImage: @unchecked Sendable { let image: NSImage }

// Decoded-image cache so a given block's base64 is turned into an NSImage AT MOST ONCE.
enum ImageCache {
    // NSCache is internally thread-safe; the compiler can't prove it, so opt out explicitly. Bounded by
    // BYTES (totalCostLimit), not just count, so decoded PDF pages can't grow the app's memory unbounded.
    nonisolated(unsafe) private static let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>(); c.countLimit = 512; c.totalCostLimit = 256 * 1024 * 1024; return c
    }()
    static func get(_ key: String) -> NSImage? { cache.object(forKey: key as NSString) }
    static func set(_ key: String, _ img: NSImage) {
        let s = img.size, cost = max(1, Int(s.width * s.height) * 4)   // ~RGBA bytes
        cache.setObject(img, forKey: key as NSString, cost: cost)
    }
}

// Renders an image block WITHOUT ever decoding base64→NSImage on the main thread. Shows a light
// placeholder until the off-main decode finishes; caches the result so re-renders are instant.
// This is THE fix for the multi-image-context beachball. Caller styles the Image via `content`.
struct AsyncBlockImage<Content: View>: View {
    let block: Block
    @ViewBuilder let content: (Image) -> Content
    @State private var img: NSImage?
    var body: some View {
        Group {
            if let img { content(Image(nsImage: img)) }
            else { RoundedRectangle(cornerRadius: 6).fill(.secondary.opacity(0.08)).overlay(ProgressView().controlSize(.small)) }
        }
        .task(id: block.id) {
            let key = block.id.uuidString
            if let c = ImageCache.get(key) { img = c; return }
            let b64 = block.data ?? ""
            let decoded = await Task.detached(priority: .userInitiated) { () -> SendableImage? in
                guard let d = Data(base64Encoded: b64), let ns = NSImage(data: d) else { return nil }
                return SendableImage(image: ns)
            }.value
            if let decoded { ImageCache.set(key, decoded.image); img = decoded.image }
        }
    }
}

// image file/paste data → (base64, mediaType), downscaling >900KB → ≤1568px JPEG (interro's rule).
func imageToBase64(_ data: Data) -> (String, String)? {
    guard let img = NSImage(data: data) else { return nil }
    if data.count < 900_000 { return (data.base64EncodedString(), "image/png") }
    let maxDim: CGFloat = 1568
    let s = min(1, maxDim / max(img.size.width, img.size.height))
    guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
    let rep = NSBitmapImageRep(cgImage: cg)
    rep.size = NSSize(width: img.size.width * s, height: img.size.height * s)
    guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else { return nil }
    return (jpeg.base64EncodedString(), "image/jpeg")
}

// Decode the clipboard's image (if any) into a Block: NSImage→tiff first, else the raw png/tiff data.
// Shared by the BlockStream panes' paste and the chat-input ⌘V image intake.
func clipboardImageBlock() -> Block? {
    let pb = NSPasteboard.general
    if let img = NSImage(pasteboard: pb), let tiff = img.tiffRepresentation, let (b64, mt) = imageToBase64(tiff) {
        return Block(mediaType: mt, data: b64)
    }
    for t in [NSPasteboard.PasteboardType.png, .tiff] {
        if let d = pb.data(forType: t), let (b64, mt) = imageToBase64(d) { return Block(mediaType: mt, data: b64) }
    }
    return nil
}
