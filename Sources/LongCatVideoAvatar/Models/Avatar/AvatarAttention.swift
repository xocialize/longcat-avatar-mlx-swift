//
//  AvatarAttention.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/attention.py
//
//  Avatar 1.5 overlay attention:
//
//  - SingleStreamAttention: audio cross-attention with 1D RoPE on token
//    positions. Audio K/V come from AudioProjModel output.
//  - Reference Skip mechanism: at certain blocks the Q is sliced to skip
//    the reference-image tokens (the first frame embedding) so the audio
//    only modulates the generated frames, not the conditioning image.
//
//  Critical:
//  - Reference Skip is Q-slicing only — K/V keep full length. The sliced Q
//    indices come from the block's `reference_skip` config field.
//  - SingleStreamAttention uses split-fused Q + KV (separate qkv linears for
//    q and for fused kv) — the DMD LoRA loader has special handling for
//    this. See Python `lora.py:compute_merged_delta` for the split-fused
//    QKV math when porting the LoRA path.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// TODO(S3.7): port SingleStreamAttention + Reference Skip
//   ☐ port SingleStreamAttention with split-fused Q / KV linears
//   ☐ port 1D RoPE injection on audio token positions
//   ☐ port Reference Skip Q-slicing helper
//   ☐ parity test vs Python avatar attention at small config
