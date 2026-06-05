// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BNNSBertEmbeddings",
    platforms: [
        .macOS("26.0"),
        .iOS("26.0"),
    ],
    products: [
        .library(
            name: "BNNSBertEmbeddings",
            targets: ["BNNSBertEmbeddings"]
        ),
        .executable(
            name: "bnns-bert-embeddings",
            targets: ["BNNSBertEmbeddingsCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/jkrukowski/swift-safetensors.git", from: "0.1.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", "1.4.0"..<"1.6.0"),
    ],
    targets: [
        .target(
            name: "BNNSBertEmbeddings",
            dependencies: [
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "BNNSBertEmbeddingsCLI",
            dependencies: [
                "BNNSBertEmbeddings",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
