//
//  LongCatAvatarPipeline.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/pipeline_mlx.py
//
//  High-level orchestration:
//      LongCatAvatarPipeline(vae, textEncoder, audioEncoder, dit, config)
//      pipeline.callAsFunction(image: ..., audioMel: ...,
//                              textEmbeds: ..., textMask: ...,
//                              uncondEmbeds: ..., uncondMask: ...) -> MLXArray
//
//  Stages (mirror Python):
//   1. VAE.encode(ref image) → first-frame latent
//   2. Whisper(audioMel) → 33 hidden states → group-pool (33→5) → 25 Hz
//      linear-interp → AudioProjModel → per-frame audio embeds
//   3. FlowMatchEulerDiscreteScheduler with DMD distilled sigmas
//   4. 8-step denoising loop with 3-pass disentangled CFG (see Guidance.swift)
//   5. VAE.decode(final latent) → video [B, C, T, H, W] in [-1, 1]
//
//  Critical (port these explicitly — they cost the Python port real time):
//  - L14 (Python lessons): mlx-arsenal scheduler.step returns mx.array NOT
//    tuple; check if swift-equivalent has same convention.
//  - L17: scheduler appends 1.0 sentinel; we MUST overwrite trailing sigma
//    to 0.0 after set_timesteps. Otherwise final step re-adds noise.
//  - Negative velocity flip: `noisePred = -noisePred` before scheduler step.
//  - Audio embeds trim: `audioEmbs[:1 + vaeScale * (TLatFull - 1)]` length match.
//

import Foundation
import MLX
import MLXNN
import Tokenizers

public struct PipelineConfig {
    public var numSamplingSteps: Int = 8
    public var guidanceScaleText: Float = 4.0
    public var guidanceScaleAudio: Float = 4.0
    public var schedulerShift: Float = 7.0
    public init() {}
}

// TODO(S3.8): port LongCatAvatarPipeline
//   ☐ port FlowMatchEulerDiscreteScheduler from Python guidance / mlx-arsenal
//     (sigma schedule + step). Reuse mlx-swift if it ships one; else port.
//   ☐ override trailing sigma sentinel (L17 fix)
//   ☐ port disentangled CFG combiner (Guidance.swift)
//   ☐ wire VAE encode + audio path + DiT loop + VAE decode
//   ☐ from_pretrained(repoId): orchestrate all 4 component loads + quant detection
