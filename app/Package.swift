// swift-tools-version:6.0
import PackageDescription
let package = Package(
    name: "CIVM",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0")
    ],
    targets: [.executableTarget(name: "CIVM",
        dependencies: [.product(name: "MarkdownUI", package: "swift-markdown-ui")],
        path: "Sources/CIVM")]
)
