//
//  RoPE3D.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/rope_3d.py
//
//  3D RoPE (temporal + height + width) + 1D RoPE for audio cross-attention.
//
//  Architecture:
//  - 3D RoPE splits head_dim into thirds (or per-axis configurable). For
//    LongCat the split is head_dim/3 across (T, H, W).
//  - Precomputed cos/sin tables cached per (T, H, W) shape.
//  - 1D RoPE used by SingleStreamAttention on audio token positions.
//
//  Critical:
//  - mlx-swift exposes `MLXFast.rope` similar to Python's `mx.fast.rope`.
//    Verify it accepts a precomputed (cos, sin) pair OR per-position freqs;
//    different signature than Python's may require an adapter.
//

import Foundation
import MLX
import MLXFast

// TODO(S3.6 prereq): port 3D + 1D RoPE
//   ☐ port build_3d_rope_freqs (T, H, W, head_dim_per_axis) → (cos, sin)
//   ☐ apply_rotary_3d (q, k, cos, sin) → (q_rot, k_rot)
//   ☐ port 1D RoPE for SingleStreamAttention
//   ☐ parity test: numerical match vs Python apply_rotary_3d at config shape
