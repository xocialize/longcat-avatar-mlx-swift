//
//  Guidance.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/guidance.py
//
//  3-pass disentangled CFG combiner + DMD distilled sigma schedule.
//
//  Formula (defaults s_t = s_a = 4.0, matches PT DMD distillation):
//      noise_pred = uncond + s_t * (cond - uncond_text)
//                         + s_a * (uncond_text - uncond)
//
//  Three forward passes per denoising step:
//   1. cond: text + audio cross-attn active
//   2. uncond_text: audio cross-attn active, text masked to empty
//   3. uncond: both text and audio masked
//
//  DMD distilled sigmas:
//      8 values [1.0, 0.876, 0.751, 0.626, 0.500, 0.374, 0.249, 0.124]
//      Pipeline overwrites the trailing sentinel sigma to 0.0 (L17 fix).
//

import Foundation
import MLX

public enum Guidance {
    // TODO(S3.8): port disentangledCFGCombine, flipVelocityForScheduler,
    //             getDMDDistilledSigmas, cfgSplitOutputs
}
