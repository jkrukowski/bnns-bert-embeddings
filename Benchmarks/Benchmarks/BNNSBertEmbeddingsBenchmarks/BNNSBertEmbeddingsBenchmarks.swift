import Accelerate
import BNNSBertEmbeddings
import Benchmark
import Foundation
import Hub
import Safetensors
import Tokenizers

private final class LoadedModel: @unchecked Sendable {
    static let modelId = "sentence-transformers/all-MiniLM-L6-v2"
    static let sampleText = "The quick brown fox jumps over the lazy dog."

    let context: BNNSGraph.Context
    let tokenizer: any Tokenizer
    let config: BertConfig

    init() async throws {
        let hub = HubApi()
        let dir = try await hub.snapshot(
            from: Hub.Repo(id: Self.modelId, type: .models),
            matching: [
                "config.json", "*.safetensors",
                "tokenizer.json", "tokenizer_config.json",
                "vocab.txt", "special_tokens_map.json",
            ]
        )
        let config = try BertConfig.load(from: dir.appendingPathComponent("config.json"))
        let weights = ModelWeights(
            store: try Safetensors.read(at: dir.appendingPathComponent("model.safetensors"))
        )
        self.config = config
        self.tokenizer = try await AutoTokenizer.from(modelFolder: dir)
        self.context = try buildContext(weights: weights, config: config)
    }
}

let benchmarks: @Sendable () -> Void = {
    Benchmark.defaultConfiguration = .init(
        metrics: [
            .wallClock,
            .throughput,
            .peakMemoryResident,
            .mallocCountTotal,
        ],
        maxIterations: 100
    )

    // Measures tokenization + BNNSGraph forward pass + pooling (loading excluded).
    Benchmark("Encode (Bert)") { benchmark, model in
        for _ in benchmark.scaledIterations {
            try await blackHole(
                encode(
                    LoadedModel.sampleText,
                    context: model.context,
                    tokenizer: model.tokenizer,
                    config: model.config
                )
            )
        }
    } setup: {
        try await LoadedModel()
    }

    // Loading on its own (download is cached; measures read + compile + tokenizer).
    Benchmark("Load model") { benchmark in
        for _ in benchmark.scaledIterations {
            try await blackHole(LoadedModel())
        }
    }
}
