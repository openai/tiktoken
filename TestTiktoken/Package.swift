// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TestTiktoken",
    platforms: [
        .macOS(.v10_15)
    ],
    dependencies: [
        .package(path: "/Users/nicholasarner/Development/Active/TiktokenSwift")
    ],
    targets: [
        .executableTarget(
            name: "TestTiktoken",
            dependencies: [
                .product(name: "TiktokenSwift", package: "TiktokenSwift")
            ]
        ),
    ]
)
