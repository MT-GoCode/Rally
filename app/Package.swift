// swift-tools-version:6.0
import PackageDescription
let package = Package(
    name: "CIVM",
    platforms: [.macOS("26.0")],   // Apple SpeechAnalyzer (live voice) is macOS 26+
    dependencies: [
        // Native GFM markdown + LaTeX math (CoreText, no WebView) — MarkdownUI's successor.
        .package(url: "https://github.com/gonzalezreal/textual", from: "0.5.0")
    ],
    targets: [.executableTarget(name: "CIVM",
        dependencies: [.product(name: "Textual", package: "textual")],
        path: "Sources/CIVM")]
)
