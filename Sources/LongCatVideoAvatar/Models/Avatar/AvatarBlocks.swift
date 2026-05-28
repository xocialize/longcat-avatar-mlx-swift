//
//  AvatarBlocks.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/blocks.py
//
//  Avatar 1.5 overlay blocks. The main module here is `AudioProjModel`.
//
//  AudioProjModel:
//  - 3-layer MLP that projects Whisper encoder outputs (5-group mean-pooled,
//    25 Hz linear-interp) into the DiT's hidden space for audio cross-attn.
//  - Input shape: [B, T_audio, 5, whisper_hidden]
//  - Output shape: [B, T_audio, context_tokens=32, output_dim=768]
//  - Per-frame "context tokens" pattern — analogous to perceiver / Q-Former.
//
//  Critical:
//  - The 5-group windowing happens in audio_process.py BEFORE entering
//    AudioProjModel. Port both consistently.
//  - The audio adaLN modulation linears are SEPARATE from the base DiT's
//    adaLN (they're `audio_adaLN_modulation` in the checkpoint). Both kept
//    at fp32 per CLAUDE.md L11.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.7): port AudioProjModel
//   ☐ port 3-layer MLP with the right context-token reshape
//   ☐ verify dimensions match config: intermediate_dim=512, output_dim=768,
//     context_tokens=32, audio_window=5, audio_block=5
//   ☐ smoke test: shape sanity with a synthetic Whisper output
