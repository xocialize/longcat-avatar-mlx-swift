//
//  Blocks.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/blocks.py
//
//  DiT building blocks. The Avatar overlay adds AudioProjModel in
//  Avatar/Blocks.swift; this file holds the base DiT building blocks.
//
//  Modules to port (preserve Python names):
//  - PatchEmbed3D: Conv3d patchify with patch_size (1, 2, 2)
//  - TimestepEmbedder: fp32 sinusoidal + 2-layer MLP, frequency_embedding_size=256
//  - CaptionEmbedder (y_embedder in checkpoints): 2-layer MLP
//  - SwiGLU FFN: inner = (intermediate_size * 2 / 3) rounded up to multiple_of=256
//
//  Critical:
//  - TimestepEmbedder runs fp32 internally (Meituan's _FP32 convention).
//    AdaLN modulation linears stay fp32 (CLAUDE.md L11 from Python port).
//  - SwiGLU FFN inner-dim formula: round_up(2/3 * intermediate_size, 256).
//    Do NOT use a different rounding; weights won't fit.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.6 prereq): port DiT blocks
//   ☐ port PatchEmbed3D (Conv3d with patch_size triple)
//   ☐ port TimestepEmbedder (fp32 sinusoidal + MLP)
//   ☐ port CaptionEmbedder (2-layer MLP)
//   ☐ port SwiGLU FFN with the correct multiple_of=256 inner-dim rounding
//   ☐ smoke test: shape sanity for each at production sizes
