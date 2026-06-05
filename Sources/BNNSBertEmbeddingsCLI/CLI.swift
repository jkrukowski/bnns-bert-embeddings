import ArgumentParser
import BNNSBertEmbeddings
import Foundation
import Hub

@main
struct EmbeddingsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bnns-bert-embeddings",
        abstract: "BERT sentence embeddings on the CPU via BNNSGraph."
    )

    @Option(help: "HuggingFace model id to download (config + safetensors + tokenizer).")
    var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"

    @Option(help: "Text to encode.")
    var text: String = "Text to encode"

    func run() async throws {
        let hub = HubApi()
        let modelURL = try await hub.snapshot(
            from: Hub.Repo(id: modelId, type: .models),
            matching: [
                "config.json", "*.safetensors",
                "tokenizer.json", "tokenizer_config.json",
                "vocab.txt", "special_tokens_map.json",
            ]
        )

        let embedding = try await encode(
            text,
            modelURL: modelURL,
            configURL: modelURL.appendingPathComponent("config.json"),
            weightsURL: modelURL.appendingPathComponent("model.safetensors")
        )
        print(embedding)
    }
}
