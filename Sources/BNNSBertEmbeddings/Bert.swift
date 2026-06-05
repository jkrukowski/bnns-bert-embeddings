import Accelerate
import Foundation
import Safetensors
import Tokenizers

// MARK: - Public API

/// Model hyper-parameters decoded from a BERT-family `config.json`.
///
/// JSON keys are mapped from snake_case (e.g. `hidden_size` -> `hiddenSize`).
public struct BertConfig: Codable {
    /// Width of the hidden representation (`hidden_size`).
    public let hiddenSize: Int
    /// Number of stacked transformer encoder blocks (`num_hidden_layers`).
    public let numHiddenLayers: Int
    /// Number of attention heads per block (`num_attention_heads`).
    public let numAttentionHeads: Int
    /// Width of the feed-forward intermediate layer (`intermediate_size`).
    public let intermediateSize: Int
    /// Epsilon added inside LayerNorm for numerical stability (`layer_norm_eps`).
    public let layerNormEps: Float

    /// Per-head width; the hidden axis is split evenly across attention heads.
    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Loads and decodes a `BertConfig` from a `config.json` file.
    ///
    /// - Parameter url: Path to the model's `config.json`.
    /// - Returns: The decoded configuration.
    /// - Throws: An error if the file cannot be read or decoded.
    public static func load(from url: URL) throws -> BertConfig {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(BertConfig.self, from: Data(contentsOf: url))
    }
}

/// A thin wrapper over parsed safetensors that fetches F32 weight tensors by
/// key for graph construction.
public struct ModelWeights {
    let store: ParsedSafetensors

    /// Creates a weight store backed by already-parsed safetensors.
    ///
    /// - Parameter store: The parsed safetensors holding the model's F32 weights.
    public init(store: ParsedSafetensors) {
        self.store = store
    }

    func tensor(_ key: String) -> (shape: [Int], values: [Float]) {
        do {
            return (try store.tensorData(forKey: key).shape, try store.array(forKey: key))
        } catch {
            preconditionFailure("weight '\(key)' missing or not F32 in safetensors: \(error)")
        }
    }
}

/// Builds a BNNSGraph that maps ONE tokenized sentence to an L2-normalized
/// mean-pooled embedding. The batch dimension is fixed at 1; only the sequence
/// length `S` is dynamic (declared as -1), resolved per-call with
/// `setDynamicShapes`. One sentence per execute — no padding, no masking.
///
/// Runtime arguments (all [1, S]):
///   input_ids       Int32
///   position_ids    Int32
///   token_type_ids  Int32
/// Output: [1, hidden]
///
/// Pooling is a plain SUM over tokens followed by L2 normalization. Mean-pooling
/// would divide by S first, but that constant factor cancels under L2
/// normalization, so summing gives the identical unit vector with no need to
/// know S inside the graph.
///
/// Multi-head attention is done by *slicing* the static hidden axis per head
/// (instead of reshaping).
public func buildContext(
    weights: ModelWeights,
    config: BertConfig
) throws -> BNNSGraph.Context {
    try BNNSGraph.makeContext { builder in
        buildGraph(&builder, weights: weights, config: config)
    }
}

/// Loads the model from disk (config, F32 safetensors weights, and tokenizer),
/// compiles the BNNSGraph, and encodes a single sentence into an L2-normalized
/// vector.
///
/// - Parameters:
///   - sentence: The text to embed.
///   - modelURL: Folder holding the tokenizer files (`tokenizer.json`,
///     `tokenizer_config.json`).
///   - configURL: Path to the model's `config.json`.
///   - weightsURL: Path to the model's `model.safetensors`.
public func encode(
    _ sentence: String,
    modelURL: URL,
    configURL: URL,
    weightsURL: URL
) async throws -> [Float] {
    let config = try BertConfig.load(from: configURL)
    let weights = ModelWeights(store: try Safetensors.read(at: weightsURL))
    let tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
    let context = try buildContext(weights: weights, config: config)
    return try await encode(sentence, context: context, tokenizer: tokenizer, config: config)
}

/// Encodes a single sentence into an L2-normalized, sum-pooled embedding using
/// a pre-built graph context and tokenizer.
///
/// This is the pure-compute hot path: it only tokenizes, resolves the dynamic
/// sequence length, executes the graph, and returns the vector. Reuse the same
/// `context`, `tokenizer`, and `config` across calls to avoid reload and
/// recompilation costs.
///
/// - Parameters:
///   - sentence: The text to embed.
///   - context: A compiled graph context from ``buildContext(weights:config:)``.
///   - tokenizer: The tokenizer matching the model.
///   - config: The model configuration (used for the output hidden size).
/// - Returns: The mean-pooled, L2-normalized embedding (length `hiddenSize`).
public func encode(
    _ sentence: String,
    context: BNNSGraph.Context,
    tokenizer: any Tokenizer,
    config: BertConfig
) async throws -> [Float] {
    let ids = tokenizer.encode(text: sentence)
    let S = ids.count

    let inputIds = ids.map(Int32.init)
    let positionIds = (0..<S).map(Int32.init)
    let tokenTypeIds = [Int32](repeating: 0, count: S)

    // Resolve the dynamic S for this call (shapes in argument order; the
    // output is left zero-rank so BNNS infers it).
    let names = context.argumentNames()
    let shapeFor: [String: [Int]] = [
        "input_ids": [1, S], "position_ids": [1, S], "token_type_ids": [1, S],
    ]
    let shapes = names.map { BNNSGraph.Shape(shapeFor[$0, default: []]) }
    _ = try await context.setDynamicShapes(shapes)

    var args = names.map { context.tensor(argument: $0, fillKnownDynamicShapes: true)! }
    defer { args.forEach { $0.deallocate() } }

    args[context.argumentPosition(argument: "input_ids")].allocate(initializingFrom: inputIds)
    args[context.argumentPosition(argument: "position_ids")].allocate(initializingFrom: positionIds)
    args[context.argumentPosition(argument: "token_type_ids")].allocate(
        initializingFrom: tokenTypeIds
    )

    // Names of the graph's runtime input arguments; the remaining argument is the
    // inferred output.
    let inputArgs: Set<String> = ["input_ids", "position_ids", "token_type_ids"]
    let outName = names.first { !inputArgs.contains($0) }!
    let outPosition = context.argumentPosition(argument: outName)
    args[outPosition].allocate(as: Float.self, count: config.hiddenSize)

    try await context.executeFunction(arguments: &args)

    return args[outPosition].makeArray(of: Float.self)
}

// MARK: - Graph construction (internal)

extension BNNSGraph.Builder.Tensor {
    // Slices out attention head `i` (each `dim` wide) along the static last
    // (hidden) axis: [B, S, hidden] -> [B, S, dim].
    @inline(__always)
    func head(_ i: Int, dim: Int) -> Self {
        self[
            BNNSGraph.Builder.SliceRange.fillAll,
            BNNSGraph.Builder.SliceRange.fillAll,
            BNNSGraph.Builder.SliceRange(startIndex: i * dim, endIndex: (i + 1) * dim)
        ]
    }
}

extension BNNSGraph.Builder {
    @inline(__always)
    mutating func constant(_ key: String, from weights: ModelWeights) -> Tensor<Float> {
        let w = weights.tensor(key)
        return constant(name: key, values: w.values, shape: w.shape)
    }
}

private func buildGraph(
    _ builder: inout BNNSGraph.Builder,
    weights: ModelWeights,
    config: BertConfig
) -> [any BNNSGraph.TensorDescriptor] {
    let inputIds = builder.argument(
        name: "input_ids",
        dataType: Int32.self,
        shape: [1, -1],
        intent: .input
    )
    let positionIds = builder.argument(
        name: "position_ids",
        dataType: Int32.self,
        shape: [1, -1],
        intent: .input
    )
    let tokenTypeIds = builder.argument(
        name: "token_type_ids",
        dataType: Int32.self,
        shape: [1, -1],
        intent: .input
    )

    // --- Embeddings: word + position + token_type, then LayerNorm. [1, S, hidden] ---
    let wordEmb = builder.constant("embeddings.word_embeddings.weight", from: weights).gather(
        indices: inputIds,
        axis: 0,
        batchDimensionCount: 0
    )
    let posEmb = builder.constant("embeddings.position_embeddings.weight", from: weights).gather(
        indices: positionIds,
        axis: 0,
        batchDimensionCount: 0
    )
    let typeEmb = builder.constant("embeddings.token_type_embeddings.weight", from: weights).gather(
        indices: tokenTypeIds,
        axis: 0,
        batchDimensionCount: 0
    )
    var h = (wordEmb + posEmb + typeEmb).layerNorm(
        weight: builder.constant("embeddings.LayerNorm.weight", from: weights),
        bias: builder.constant("embeddings.LayerNorm.bias", from: weights),
        axes: [2],
        epsilon: config.layerNormEps
    )

    // --- Encoder blocks ---
    for layer in 0..<config.numHiddenLayers {
        h = encoderBlock(&builder, h, layer: layer, weights: weights, config: config)
    }

    // --- Sum pooling + L2 normalize (see note above on why sum, not mean) ---
    let pooled = h.sum(axes: [1], keepDimensions: false)  // [1, hidden]
    let invNorm = pooled.sumOfSquares(axes: [1], keepDimensions: true).rsqrt(epsilon: 1e-12)
    return [pooled * invNorm]  // [1, hidden]
}

private func encoderBlock(
    _ b: inout BNNSGraph.Builder,
    _ x: BNNSGraph.Builder.Tensor<Float>,
    layer: Int,
    weights: ModelWeights,
    config: BertConfig
) -> BNNSGraph.Builder.Tensor<Float> {
    let p = "encoder.layer.\(layer)"

    // Q, K, V projections -> [B, S, hidden]
    let q = x.linear(
        weight: b.constant("\(p).attention.self.query.weight", from: weights),
        bias: b.constant("\(p).attention.self.query.bias", from: weights)
    )
    let k = x.linear(
        weight: b.constant("\(p).attention.self.key.weight", from: weights),
        bias: b.constant("\(p).attention.self.key.bias", from: weights)
    )
    let v = x.linear(
        weight: b.constant("\(p).attention.self.value.weight", from: weights),
        bias: b.constant("\(p).attention.self.value.bias", from: weights)
    )

    // Per-head scaled dot-product attention (slice the static hidden axis).
    let scale = 1.0 / Float(config.headDim).squareRoot()
    var heads = [BNNSGraph.Builder.Tensor<Float>]()
    for i in 0..<config.numAttentionHeads {
        let qh = q.head(i, dim: config.headDim)
        let kh = k.head(i, dim: config.headDim)
        let vh = v.head(i, dim: config.headDim)
        let scores = qh.matmul(other: kh, transposeOther: true) * scale  // [1, S, S]
        let probs = scores.softmax(axis: 2)
        heads.append(probs.matmul(other: vh))  // [1, S, headDim]
    }
    let merged = b.concatenate(heads, axis: 2)  // [1, S, hidden]

    // Output projection + residual + LayerNorm
    let attnOut = merged.linear(
        weight: b.constant("\(p).attention.output.dense.weight", from: weights),
        bias: b.constant("\(p).attention.output.dense.bias", from: weights)
    )
    let attnNorm = (x + attnOut).layerNorm(
        weight: b.constant("\(p).attention.output.LayerNorm.weight", from: weights),
        bias: b.constant("\(p).attention.output.LayerNorm.bias", from: weights),
        axes: [2],
        epsilon: config.layerNormEps
    )

    // Feed-forward
    let inter = attnNorm.linear(
        weight: b.constant("\(p).intermediate.dense.weight", from: weights),
        bias: b.constant("\(p).intermediate.dense.bias", from: weights)
    ).gelu()
    let ffnOut = inter.linear(
        weight: b.constant("\(p).output.dense.weight", from: weights),
        bias: b.constant("\(p).output.dense.bias", from: weights)
    )

    return (attnNorm + ffnOut).layerNorm(
        weight: b.constant("\(p).output.LayerNorm.weight", from: weights),
        bias: b.constant("\(p).output.LayerNorm.bias", from: weights),
        axes: [2],
        epsilon: config.layerNormEps
    )
}
