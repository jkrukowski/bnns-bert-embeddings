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
**~1.3 ms** (p50). For context, here it is next to an equivalent
[`MLTensor`](https://github.com/jkrukowski/swift-embeddings) implementation
benchmarked on the same machine (same model, same sentence). `MLTensor` is shown
in two configurations: `.cpuOnly`, and its default `.cpuAndGPU` (for this tiny
workload the GPU dispatch overhead makes the default markedly slower):

| Metric (p50)           | BNNSGraph | MLTensor (`.cpuOnly`) | MLTensor (`.cpuAndGPU`, default) |
| ---------------------- | --------: | --------------------: | -------------------------------: |
| Encode time (wall)     |   ~1.3 ms |               ~3.7 ms |                           ~18 ms |
| Throughput             |  ~760 / s |              ~270 / s |                          ~54 / s |
| Malloc (total)         |       387 |               ~16,000 |                          ~71,000 |
| Memory (resident peak) |   ~140 MB |               ~144 MB |                          ~239 MB |

Against a true **CPU-to-CPU** baseline (`MLTensor` `.cpuOnly`), BNNSGraph is
roughly **~2.8× faster** and allocates **~40× less**, at comparable resident
memory. The gap widens dramatically against `MLTensor`'s default `.cpuAndGPU`
policy, but that mostly reflects GPU-dispatch overhead on a sub-millisecond
workload rather than a like-for-like CPU comparison.

Exact multipliers vary by host and OS version; the takeaway is the direction,
not the precise number.

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift format . -i -r --configuration .swift-format
```

## License

[MIT](LICENSE)
