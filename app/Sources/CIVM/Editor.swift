import AppKit
import PDFKit

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
    var nsImage: NSImage? {
        guard type == "image", let d = Data(base64Encoded: data ?? "") else { return nil }
        return NSImage(data: d)
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
