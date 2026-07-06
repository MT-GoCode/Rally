import AppKit

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
