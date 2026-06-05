# BNNSGraph Bert Embeddings

Real BERT sentence embeddings on the CPU via [`BNNSGraph`](https://developer.apple.com/documentation/accelerate/bnnsgraph).

> **Note:** This is a proof of concept, written mainly to gauge the efficiency of Apple's new `BNNSGraph` API for real transformer inference on the CPU. It is not a production-ready library.

The model is expressed as a single, statically-compiled `BNNSGraph` with a **dynamic sequence length**, so one compiled context embeds a sentence of any length. Weights are the genuine F32 `safetensors` from the HuggingFace Hub, tokenization uses [`swift-transformers`](https://github.com/huggingface/swift-transformers), and everything runs on CPU/AMX.

## Usage

```swift
import BNNSBertEmbeddings

// Convenience: load everything from disk, then encode one sentence.
let embedding = try await encode(
    "A man is eating food.",
    modelURL: modelDir, // folder with tokenizer files
    configURL: modelDir.appendingPathComponent("config.json"),
    weightsURL: modelDir.appendingPathComponent("model.safetensors"))
```

For repeated calls (e.g. benchmarking), build the model once and reuse it — the
pure-compute `encode` does only **tokenize → graph execute → return vector**:

```swift
import Accelerate
import BNNSBertEmbeddings
import Safetensors
import Tokenizers

// One-time setup (not the hot path).
let config = try BertConfig.load(from: configURL)
let weights = ModelWeights(store: try Safetensors.read(at: weightsURL))
let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
let context = try buildContext(weights: weights, config: config)

// Hot path.
let embedding = try await encode(
    "A man is eating food.", context: context, tokenizer: tokenizer, config: config)
```

## How it works

- **One graph, dynamic `S`.** The batch dimension is fixed at 1 and only the
  sequence length is dynamic (declared `-1`), resolved per call with
  `setDynamicShapes`. No padding, no attention masking.
- **Sum pooling, not mean.** Mean-pooling would divide by `S` first, but that
  constant factor cancels under L2 normalization — so the graph just sums over
  tokens, with no need to know `S`.
- **Per-head attention by slicing.** Multi-head attention slices the static
  hidden axis per head instead of reshaping.
- **Config-driven.** Dimensions (`hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `layer_norm_eps`, …) are read from the model's
  `config.json`, so any standard BERT-family safetensors model works.

## Benchmarks

```bash
cd Benchmarks
swift package --disable-sandbox benchmark
```

`--disable-sandbox` lets the `setup:` step download the model into the Hub cache
(first run only). Loading/compilation happens in `setup:` and is **not** timed;
the measured work is tokenization + the BNNSGraph forward pass + pooling.

On Apple silicon, single-sentence encode latency (`all-MiniLM-L6-v2`) is roughly
**~1.3 ms** (p50). Compared to an equivalent CPU
[`MLTensor`](https://github.com/jkrukowski/swift-embeddings) implementation
benchmarked on the same machine (same model, same sentence), the BNNSGraph path
is **well over 10× faster** and allocates **~180× less**:

| Metric (p50)           | BNNSGraph | MLTensor (CPU) |
| ---------------------- | --------: | -------------: |
| Encode time (wall)     |   ~1.3 ms |         ~19 ms |
| Throughput             |  ~750 / s |        ~52 / s |
| Malloc (total)         |       387 |        ~71,000 |
| Memory (resident peak) |    143 MB |         236 MB |

Exact multipliers vary by host and OS version; the takeaway is the order of
magnitude, not the precise number.

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift format . -i -r --configuration .swift-format
```

## License

[MIT](LICENSE)
