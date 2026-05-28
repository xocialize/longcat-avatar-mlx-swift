# Swift port roadmap

Staged plan for porting [longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)
(Python MLX) to Swift via mlx-swift. Stages mirror Stage 3 of the Python
port's overall plan (the Python port covered Stage 0 = recon, Stage 1 =
core port, Stage 2 = quantization; Swift work is Stage 3).

Each stage corresponds to a tracked task in the project's Claude task
list. The TODO checklists inside each Swift source file are the
authoritative breakdown of what's left for the corresponding stage.

## S3.0 — Scaffold Swift Package + mlx-swift dep ✅

- [x] `Package.swift` targeting macOS 14 / iOS 17 / visionOS 1
- [x] Dependencies: `mlx-swift`, `swift-transformers`, `swift-argument-parser`
- [x] Directory layout mirroring Python `longcat_video_avatar/`
- [x] Stub files for every module with port plan in header
- [x] Trivial CI smoke test so `swift test` stays green during port

## S3.1 — Module stubs (mirror Python) ✅

Done in tandem with S3.0. Each stub file documents the Python source it
ports, its architectural summary, and an explicit `TODO(S3.x)` block.

## S3.2 — Runtime HF weight download + cache

- [ ] `WeightLoader.snapshotDownload(repoId)` via swift-transformers' `Hub`
- [ ] Parse `safetensors.index.json` for sharded components (umT5, DiT)
- [ ] `mx.load`-equivalent for sharded weights through mlx-swift
- [ ] Env override `LONGCAT_AVATAR_WEIGHTS_DIR` for local dev
- [ ] Detect `quantization` block in `dit/config.json`; surface to caller
      so they can apply `MLXNN.quantize` *before* loading bit-packed shards
      (mirror Python L19/L20 lessons)

## S3.3 — Port Wan VAE

First substantive port. Pick this first because:
- Self-contained (no audio / text deps)
- Small (~0.25 GB)
- Has a clear PT-vs-MLX parity oracle (Python port hit 7e-6 encode / 1.2e-2 decode)
- Tests the `WeightLoader` end-to-end

Critical: route AttentionBlock to CPU stream for fp32 precision (Python
CLAUDE.md L10 — Metal GPU is TF32-like for fp32 attention).

## S3.4 — Port umT5-XXL text encoder

- Tokenizer: `swift-transformers/Tokenizers` (T5 sentencepiece)
- Three sharded safetensors via `WeightLoader`
- Per-block relative position bias; gated GeLU FFN
- Parity vs `transformers` PT umT5 on the demo prompt

## S3.5 — Port Whisper encoder

- Mel preprocessing via Accelerate / vDSP (no librosa in Swift)
- Audio I/O via AVFoundation
- All 33 hidden states, group-pool in `AudioProcess.groupPoolHiddenStates`
- Parity vs `transformers` PT Whisper on demo audio

## S3.6 — Port base 48-block DiT

- Self-attn + text cross-attn + SwiGLU FFN per block
- AdaLN-Zero, 6 modulations per block
- 3D RoPE for spatial-temporal positions
- Output `-v` (negative velocity)
- Parity at single-block + full-stack levels

## S3.7 — Avatar overlay

- `SingleStreamAttention` with split-fused Q+KV
- 1D RoPE on audio token positions
- Reference Skip Q-slicing per block
- `AudioProjModel` (3-layer MLP, 33→5 group + windowing already in AudioProcess)
- Separate `audio_adaLN_modulation` linears

## S3.8 — Pipeline + scheduler + 3-pass CFG

- `FlowMatchEulerDiscreteScheduler` (port or use mlx-swift if available)
- Trailing sigma sentinel fix (Python L17)
- 3-pass disentangled CFG combiner
- Audio embed length trim
- Negative velocity flip before `scheduler.step`
- Quant detection: apply `MLXNN.quantize` before load when present

## S3.9 — End-to-end smoke

- Run full pipeline on shipped demo (`man.png` + `man.mp3`)
- Frame-by-frame comparison vs Python output (`/tmp/q4_smoke_output.npy`)
- Visual + temporal-variation sanity check
- Fill in the wall-clock row in README

## Out of scope (for now)

- Long-video continuation / KV cache (Python S1.9 — works there, port later)
- Training / LoRA fine-tuning (Python port is inference-only too)
- Custom Metal kernels — only consider if mlx-swift lacks a needed op and
  swift-mlx-arsenal doesn't have it either (LIKELY: re-port Python's
  `guidance.py` distilled sigma builder into Swift first; lift to a Swift
  equivalent of mlx-arsenal only if a second project needs it)

## How to pick up

Each TODO in the source files has the form `TODO(S3.x): ...` matching
the stage above. Find the next pending stage in the task list, open the
relevant file's TODO checklist, port one module, write its parity test,
move on. The Python port's `docs/development/skill-lessons.md`
(lessons L1-L20) is required reading for the recurring traps.
