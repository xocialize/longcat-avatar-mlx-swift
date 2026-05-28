//
//  LongCatVideoDiTAvatar.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/longcat_video_dit_avatar.py
//
//  Full Avatar DiT — subclass of LongCatVideoDiT that adds:
//  - SingleStreamAttention (audio cross-attention) per block
//  - AudioProjModel attached as `audio_proj`
//  - Reference Skip block indices (per-block reference_skip config)
//  - Separate audio_adaLN_modulation linears
//
//  Forward signature additions:
//    forward(x, t, text_embeds, text_mask, audio_embeds) → noise_pred
//    where audio_embeds is [B, T_audio, context_tokens=32, output_dim=768]
//    (the AudioProjModel output).
//
//  Critical:
//  - The pipeline applies audio cross-attention AFTER text cross-attention in
//    each block. Order matters because of the residual additions.
//  - The 336 LoRA target modules (7 per block × 48 blocks) are now
//    pre-merged into the published bf16 / q4 / q8 variants — no runtime
//    LoRA loading needed unless using the legacy bf16 (base) variant.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.7): port LongCatVideoAvatarTransformer3DModel
//   ☐ inherit / compose from LongCatVideoDiT (preserve Python's subclass relation)
//   ☐ add per-block SingleStreamAttention with reference_skip routing
//   ☐ attach AudioProjModel
//   ☐ from_pretrained: sharded safetensors (3-7 shards depending on variant)
//   ☐ detect `quantization` block in config.json and apply MLXNN.quantize
//     BEFORE load (mirrors Python L19/L20 loader; see Models/Avatar README)
//   ☐ parity test: full forward at production shape vs PT, max_abs < 1e-2
