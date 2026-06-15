# longcat-avatar-mlx-swift

Swift port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven video diffusion model — using
[mlx-swift](https://github.com/ml-explore/mlx-swift) for inference on
Apple Silicon.

> **Status: components ported and parity-tested; end-to-end inference landing (S3.9).**
> All four model components + the pipeline are implemented and pass parity against the
> Python-MLX reference. The CLI (`run-inference`) currently runs from **pre-processed
> `.npy` inputs** and writes a `.npy` tensor; the high-level image/audio/prompt → MP4
> convenience path is still being wired (see `docs/port-roadmap.md`). Per-file
> `TODO(S3.x)` headers track what remains.

## Parity vs Python-MLX

| Component | Parity max_abs | Threshold | Notes |
|---|---|---|---|
| Wan VAE encode | **3.1e-6** | 1e-4 | fp32 throughout |
| Wan VAE decode | **1.76e-3** | 5e-3 | bf16, .cpu-stream attention |
| umT5-XXL | **0.119** | 0.15 | manual matmul+softmax |
| Whisper-large-v3 | **0.016** | 0.15 | fused SDPA |
| Base DiT (48 blocks) | **0.033** | 0.1 | fused SDPA |
| Avatar DiT (full) | **0.32** | 0.5 | + audio path |
| **End-to-end pipeline** | **0.23** | 5.0 | 8 DMD steps × 3-pass CFG + VAE encode/decode |

Modules using `MLXFast.scaledDotProductAttention` are ~10× tighter against
Python-MLX than modules with manual matmul+softmax chains.

## What this is

A Swift Package providing `LongCatVideoAvatar` — the same model the Python
port runs, packaged so you can call it from a SwiftUI / AppKit app on
macOS, iOS, or visionOS without bridging through Python. The four published HF
variants ([mlx-community collection](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375))
are consumed verbatim — no Swift-specific re-conversion.

## Library API

```swift
import LongCatVideoAvatar

let pipeline = try await LongCatAvatarPipeline.fromPretrained(
    "mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged"
)

let video = try await pipeline(
    image: …, audio: …, prompt: "A western man stands on stage…"
)
```

(`LongCatAvatarPipeline.fromPretrained(_:)` and `callAsFunction` are implemented in
`Sources/LongCatVideoAvatar/Pipeline/LongCatAvatarPipeline.swift`.)

## CLI (`run-inference`)

The CLI mirrors the Python `run_inference.py` numerics path. It loads weights from an HF
repo and runs the pipeline on **pre-processed `.npy` inputs** in a directory, writing the
generated video tensor as `.npy`:

```bash
swift run run-inference \
    --repo mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged \
    --input-dir ./fixtures \
    --frames 5 --height 64 --width 64 --seed 0 \
    --out output.npy
```

Flags: `--repo` (default `…bf16-dmd-merged`), `--input-dir` (required), `--frames`,
`--height`, `--width`, `--seed`, `--out`. The `--input-dir` must contain the pre-processed
`.npy` tensors (image / audio mel / text embeds / initial noise) the pipeline expects.

## Project layout

The Swift `Sources/LongCatVideoAvatar/` tree mirrors the Python package, so a reader can
diff the two trees and see only PT-MLX-Python ↔ Swift-MLX op substitutions.

```
Sources/LongCatVideoAvatar/
├── Models/
│   ├── AutoencoderKLWan.swift          # Wan 2.1 VAE
│   ├── UMT5EncoderModel.swift          # umT5-XXL text encoder
│   ├── WhisperEncoder.swift            # Whisper-Large-v3 encoder
│   ├── LongCatVideoDiT.swift           # base 48-block DiT
│   ├── Attention.swift                 # 3D self-attn + text cross-attn
│   ├── Blocks.swift                    # PatchEmbed3D, TimestepEmbedder, SwiGLU
│   ├── RoPE3D.swift                    # 3D + 1D RoPE
│   └── Avatar/                         # Avatar 1.5 overlay
│       ├── AvatarAttention.swift
│       ├── AvatarBlocks.swift
│       └── LongCatVideoDiTAvatar.swift
├── Pipeline/
│   ├── LongCatAvatarPipeline.swift     # 4-component orchestration
│   ├── Guidance.swift                  # 3-pass disentangled CFG + DMD sigmas
│   └── FlowMatchEulerDiscreteScheduler.swift  # DMD-distilled flow-match scheduler
├── Audio/
│   └── AudioProcess.swift              # Whisper post-process + mel via vDSP
└── Utilities/
    ├── WeightLoader.swift              # HF Hub download + cache + quant detect
    └── LoRA.swift                      # DMD LoRA (only for legacy bf16 base)

Sources/RunInference/RunInference.swift  # CLI (.npy in → .npy out)
Tests/LongCatVideoAvatarTests/{Smoke,Parity}/  (+ Resources/ .npy fixtures)
```

## Platforms

`Package.swift` targets macOS 14, iOS 17, visionOS 1. Dependencies: `mlx-swift`
(≥0.21.0), `swift-transformers` (≥0.1.18), `swift-argument-parser` (≥1.3.0).

| Platform | Min version | Practical use |
|---|---|---|
| macOS | 14 | Library + CLI, full 480p inference on M-series 64 GB+ |
| iOS | 17 | Library only; only q4-merged is realistic on iPad Pro 16 GB |
| visionOS | 1 | Library only; same RAM caveats as iOS |

## Running tests

```bash
# Recommended: xcodebuild bundles MLX's default.metallib correctly.
xcodebuild test -scheme LongCatVideoAvatar-Package -destination "platform=macOS"
```

> **Heads-up:** `swift test` from the CLI does **not** bundle MLX's Metal shader
> library — the test binary crashes on first kernel dispatch with
> "Failed to load the default metallib / library not found". Use `xcodebuild test`.

## Companion Python port

[xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
is the production reference (oracle for behavior; 70 smoke tests + opt-in PT parity).
Both ports consume the same four published HF variants:

| Variant | DiT dtype | Disk | 29-frame @ 256×432 |
|---|---|---|---|
| [`bf16-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged) | bf16 | 43 GB | ~105 s (Python) |
| [`bf16`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) | bf16 + LoRA | 46 GB | ~105 s (Python) |
| [`q4-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged) | 4-bit | 24 GB | ~102 s (Python) |
| [`q8-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged) | 8-bit | 31 GB | ~151 s (Python) |

Swift wall-clock numbers will be filled in once the end-to-end path (S3.9) lands.

## License

MIT. Matches upstream Meituan LongCat-Video and the Python port.
