//
//  LoRA.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/lora.py
//
//  DMD LoRA loader (only needed for the legacy `bf16` variant; the
//  bf16-dmd-merged / q4-dmd-merged / q8-dmd-merged variants ship with the
//  LoRA pre-merged into the DiT weights).
//
//  Why this is non-trivial:
//  - Meituan encodes LoRA target module names in a custom scheme that has
//    to be decoded before matching against DiT module paths
//    (`decodeModuleName` in Python lora.py).
//  - Two attention pieces are SPLIT-FUSED in the DiT but the LoRA stores
//    them separately: (a) QKV self-attn has a fused qkv linear with a
//    LoRA on the individual Q / K / V; (b) the avatar single-stream
//    attention has a fused KV linear with separate K / V LoRAs.
//  - Merge math has to allocate the fused delta correctly across the
//    fused linear's slices. See Python `compute_merged_delta` for the
//    exact accumulator.
//
//  336 target modules = 7 LoRA targets × 48 blocks.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.8 optional / bf16-base only): port LoRA loader + merger
//   ☐ port decodeModuleName for Meituan's encoded names
//   ☐ port groupLoRATensors (down/up/alpha per module)
//   ☐ port computeMergedDelta with split-fused QKV + KV handling
//   ☐ smoke test on the 336-module count
//
// Note: priority is LOW since the recommended variants ship LoRA pre-merged.
// Only port if a user wants runtime multi-strength LoRA experimentation.
