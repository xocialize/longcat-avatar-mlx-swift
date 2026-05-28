//
//  AutoencoderKLWan.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/autoencoder_kl_wan.py
//
//  Wan 2.1 VAE — diffusers 0.38 canonical schema. encode 7e-6 / decode 1.2e-2
//  PT-vs-MLX parity in the Python port (Metal-GPU fp32 quirks per the
//  reference's CLAUDE.md L10).
//
//  Architecture summary (from Python port):
//  - conv_in (Conv3d) → downblocks → mid (residual + attention) → conv_out
//  - Inverse for decoder. quant_conv / post_quant_conv around the latent.
//  - AttentionBlock routed to CPU stream in Python via `mx.stream(mx.cpu)` to
//    recover strict fp32 precision; check mlx-swift equivalent (likely
//    `MLX.stream(.cpu) { ... }` or per-op device hints).
//  - WanResample.upsample3d uses a "Rep" string sentinel for first-call
//    temporal-skip — port verbatim (sentinel pattern documented as L8 in
//    the Python port's docs/development/skill-lessons.md).
//
//  Critical:
//  - Conv3d weight layout PT (O,I,T,H,W) → MLX (O,T,H,W,I). MLX is `(O, *K, I)`
//    same as Python MLX; mlx-swift should match. Conversion already handled
//    by the published bf16 safetensors — no re-transpose at load time.
//  - normalize / denormalize helpers live OUTSIDE the encode/decode (raw-z
//    convention from upstream PT). Don't fuse them in.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.3): port AutoencoderKLWan
//   ☐ port `WanResidualBlock`, `WanMidBlock`, `WanUpBlock`, `WanDownBlock`, `WanResample`
//   ☐ port encode() + decode() with normalize/denormalize as separate calls
//   ☐ route AttentionBlock to CPU stream for fp32 precision
//   ☐ from_pretrained(repoId): single safetensors file under vae/
//   ☐ parity test: encode(demo image) max_abs < 1e-3 vs PT reference
//   ☐ parity test: decode(encoded latent) max_abs < 5e-2 vs PT reference
