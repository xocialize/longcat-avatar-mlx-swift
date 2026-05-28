//
//  Attention.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/attention.py
//
//  3D self-attention + text cross-attention primitives shared by the base
//  DiT. The Avatar overlay adds SingleStreamAttention (audio cross-attn) in
//  Avatar/Attention.swift.
//
//  Architecture:
//  - SelfAttention3D: 3D RoPE-augmented MHA over (T, H, W) video tokens
//  - MultiHeadCrossAttention: text-conditioning cross-attention with KV from
//    umT5 encoder hidden states. Uses block-diagonal mask for variable text
//    lengths within a batch.
//
//  Critical (port these explicitly — they bit the Python port):
//  - L18: `mx.fast.scaled_dot_product_attention` rejects fp32 mask with bf16
//    Q/K/V. Cast mask to q.dtype before SDPA. (mlx-swift likely has the same
//    constraint — verify when porting.)
//  - Cross-attention mask construction: build per-batch block-diagonal in
//    fp32 with -3.389e38 fill (smaller than bf16 -inf so SDPA's internal
//    softmax doesn't NaN), then cast to q.dtype.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// TODO(S3.6 prereq): port SelfAttention3D + MultiHeadCrossAttention
//   ☐ port SelfAttention3D with 3D RoPE injection (via MLXFast.rope or hand-rolled)
//   ☐ port MultiHeadCrossAttention with block-diagonal mask builder
//   ☐ mask dtype cast (port lesson L18 explicitly)
//   ☐ parity test: each at hidden_size=512 / num_heads=8 / seq=32, max_abs < 1e-3
