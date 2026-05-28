# longcat-avatar-mlx-swift

Swift port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven video diffusion model — using
[mlx-swift](https://github.com/ml-explore/mlx-swift) for inference on
Apple Silicon.

> **Status:** scaffold only. Module ports are tracked in
> [`docs/port-roadmap.md`](docs/port-roadmap.md). The companion Python
> port at [xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
> is the production-ready reference — start there if you want to run the
> model today.

## What this is

A Swift Package providing `LongCatVideoAvatar` — the same model the Python
port runs, packaged so you can call it from a SwiftUI / AppKit app on
macOS, iOS, or visionOS without bridging through Python.

Same weights, same outputs, same MLX runtime: the four published HF
variants ([mlx-community collection](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375))
are consumed verbatim — no Swift-specific re-conversion needed.

## Quick start (once ports land)

```swift
import LongCatVideoAvatar

let pipeline = try await LongCatAvatarPipeline.fromPretrained(
    "mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged"
)

let video = try await pipeline(
    image: try await Image.load(url: imageURL),
    audio: try await Audio.load(url: audioURL),
    prompt: "A western man stands on stage…"
)
try video.writeMP4(to: outputURL)
```

CLI smoke (Stage 3.9):

```bash
swift run run-inference \
    --repo mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged \
    --image refs/man.png --audio refs/man.mp3 \
    --prompt "A western man stands on stage…" \
    --frames 29 --out output.mp4
```

## Project layout

The Swift `Sources/LongCatVideoAvatar/` tree mirrors the Python package
1:1 — that's a hard rule from the `mlx-porting` skill, not an aesthetic
choice. A reader should be able to diff the two trees and see only
PT-MLX-Python ↔ Swift-MLX op substitutions.

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
│   └── Guidance.swift                  # 3-pass disentangled CFG + DMD sigmas
├── Audio/
│   └── AudioProcess.swift              # Whisper post-process + mel via vDSP
└── Utilities/
    ├── WeightLoader.swift              # HF Hub download + cache + quant detect
    └── LoRA.swift                      # DMD LoRA (only for legacy bf16 base)

Sources/RunInference/                    # CLI mirror of Python run_inference.py
Tests/LongCatVideoAvatarTests/{Smoke,Parity}/
```

## Platforms

`Package.swift` targets:

| Platform | Min version | Practical use |
|---|---|---|
| macOS | 14 | Library + CLI, full 480p inference on M-series 64 GB+ |
| iOS | 17 | Library only; only q4-merged is realistic on iPad Pro 16 GB |
| visionOS | 1 | Library only; same RAM caveats as iOS |

## Porting roadmap

See [`docs/port-roadmap.md`](docs/port-roadmap.md) for the staged plan
(S3.0 scaffold → S3.9 end-to-end inference). Each Swift source file has
a header comment with:
- The Python file it ports
- A summary of the module's architecture
- An explicit `TODO(S3.x)` checklist of what's left

## Companion Python port

[xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
is the production reference. When in doubt about behavior, the Python
port is the oracle — it has 71 smoke tests + opt-in PT parity tests and
~20 captured skill lessons covering the trap-traps we've already hit
(`docs/development/skill-lessons.md` in that repo).

Both ports consume the same four published variants on Hugging Face:

| Variant | DiT dtype | Disk | 29-frame @ 256×432 |
|---|---|---|---|
| [`bf16-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged) | bf16 | 43 GB | ~105 s (Python) |
| [`bf16`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) | bf16 + LoRA | 46 GB | ~105 s (Python) |
| [`q4-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged) | 4-bit | 24 GB | ~102 s (Python) |
| [`q8-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged) | 8-bit | 31 GB | ~151 s (Python) |

Swift wall-clock numbers will be filled in once S3.9 lands.

## License

MIT. Matches upstream Meituan LongCat-Video and the Python port.
