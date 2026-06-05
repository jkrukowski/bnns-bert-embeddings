// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "benchmarks",
    platforms: [
        .macOS("26.0")
    ],
    dependencies: [
        .package(name: "BNNSBertEmbeddings", path: ".."),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
        .package(url: "https://github.com/jkrukowski/swift-safetensors.git", from: "0.1.1"),
        .package(url: "https://github.com/ordo-one/package-benchmark", from: "1.33.0"),
    ],
    targets: [
        .executableTarget(
            name: "BNNSBertEmbeddingsBenchmarks",
            dependencies: [
                .product(name: "BNNSBertEmbeddings", package: "BNNSBertEmbeddings"),
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Benchmark", package: "package-benchmark"),
            ],
            path: "Benchmarks/BNNSBertEmbeddingsBenchmarks",
            plugins: [
                .plugin(name: "BenchmarkPlugin", package: "package-benchmark")
            ]
        )
    ]
)
