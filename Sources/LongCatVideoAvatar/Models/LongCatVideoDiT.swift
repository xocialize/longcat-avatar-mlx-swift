//
//  LongCatVideoDiT.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/longcat_video_dit.py
//
//  Base 48-block LongCat-Video DiT (text-only path; the Avatar overlay
//  subclasses in Models/Avatar/LongCatVideoDiTAvatar.swift).
//
//  Architecture:
//  - 48 blocks × {self-attn (3D RoPE), text cross-attn, SwiGLU FFN}
//  - hidden_size=4096, num_heads=32, head_dim=128
//  - Per-block AdaLN gating: 6 modulations (msa_x, msa_y, msa_z, mlp_x, mlp_y, mlp_z)
//  - intermediate_dim=512 for the t_embedder MLP (NOT 4096); confusing name
//  - PatchEmbed3D with patch_size (1, 2, 2): (T, H, W) → (T, H/2, W/2)
//  - final_layer: LayerNorm + Linear back to in_channels * prod(patch_size)
//
//  Forward signature (mirror Python):
//    forward(x, t, text_embeds, text_mask) → noise_pred
//    where x is [B, C, T, H, W] (PT-style; we may want NHWC + permute on entry).
//
//  Critical:
//  - AdaLN-Zero: init the last linear of each modulation MLP to zero (Meituan does)
//  - Apply text mask via the cross-attention path's block-diagonal mask
//  - Output is `-v` (negative velocity); the pipeline flips before scheduler.step.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.6): port LongCatVideoDiT
//   ☐ port AdaLN-Zero modulation helpers
//   ☐ port DiTBlock (self-attn + text cross-attn + SwiGLU FFN with adaLN gates)
//   ☐ port LongCatVideoTransformer3DModel (48 blocks + patch embed + final layer)
//   ☐ from_pretrained: sharded safetensors (3 shards in published variants)
//   ☐ parity test: single-block forward at small shapes vs PT, max_abs < 5e-3
//   ☐ parity test: full 48-block forward at production shapes, max_abs < 1e-2
