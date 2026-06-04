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
//   1. cond:        text + audio cross-attn active
//   2. uncond_text: audio cross-attn active, text masked to empty
//   3. uncond:      both text and audio masked
//
//  DMD distilled sigmas (8 values descending). Pipeline must override the
//  trailing sentinel sigma to 0.0 (L17) — see PipelineMLX wiring.
//

import Foundation
import MLX

public enum Guidance {

    /// 3-pass disentangled CFG combine. See file header for the formula.
    public static func disentangledCFGCombine(
        noisePredCond: MLXArray,
        noisePredUncondText: MLXArray,
        noisePredUncond: MLXArray,
        textGuidanceScale: Float = 4.0,
        audioGuidanceScale: Float = 4.0
    ) -> MLXArray {
        noisePredUncond
            + MLXArray(textGuidanceScale) * (noisePredCond - noisePredUncondText)
            + MLXArray(audioGuidanceScale) * (noisePredUncondText - noisePredUncond)
    }

    /// LongCat DiT predicts negative velocity (`ε − x_0`). Flip the sign before
    /// `FlowMatchEulerDiscreteScheduler.step()`. Don't forget — easy to miss.
    public static func flipVelocityForScheduler(_ noisePred: MLXArray) -> MLXArray {
        -noisePred
    }

    /// DMD distilled sigma schedule (Avatar v1.5 path).
    ///
    /// Algorithm (mirrors `pipeline_longcat_video_avatar.py:get_timesteps_sigmas`
    /// with `use_distill=True`):
    ///     step_size = num_train_timesteps / num_distill_sample_steps
    ///     idx = round([1..N] * step_size)
    ///     idx = num_train_timesteps - idx
    ///     full = reversed_linspace(0, 1, num_train_timesteps)
    ///     sigmas = reversed([full[i] for i in idx])
    public static func getDMDDistilledSigmas(
        samplingSteps: Int = 8,
        numTrainTimesteps: Int = 1000,
        numDistillSampleSteps: Int = 8,
        modelType: String = "avatar-v1.5"
    ) -> MLXArray {
        precondition(
            modelType == "avatar-v1.5",
            "Only modelType='avatar-v1.5' is implemented; got \(modelType)."
        )
        let stepSize = numTrainTimesteps / numDistillSampleSteps
        var distillIdx: [Int] = []
        for i in 0..<numDistillSampleSteps {
            // Match Python's `round((i + 1) * step_size)` (integer arithmetic since
            // both factors are ints here — step_size is float-typed in Python but
            // numerically equivalent to integer division for these values).
            distillIdx.append((i + 1) * stepSize)
        }
        distillIdx = distillIdx.map { numTrainTimesteps - $0 }

        // Build reversed linspace(0, 1, num_train_timesteps): full[i] = (N-1-i)/(N-1)
        let n = numTrainTimesteps
        let sigmaValues: [Float] = distillIdx.map { idx in
            Float(n - 1 - idx) / Float(n - 1)
        }
        // Reverse so the schedule is descending (high → low sigma)
        return MLXArray(sigmaValues.reversed())
    }

    /// Split a doubled-batch CFG forward output into `(uncondText, cond)`.
    ///
    /// The pipeline stacks `[latents, latents]` paired with text
    /// `[negative_prompt, positive_prompt]` — negative FIRST. So the first
    /// half of the DiT output is `noise_pred_uncond_text` and the second
    /// half is `noise_pred_cond` (matches PT `noise_pred.chunk(2)` ordering).
    public static func cfgSplitOutputs(_ noisePred2batch: MLXArray) -> (MLXArray, MLXArray) {
        let parts = MLX.split(noisePred2batch, parts: 2, axis: 0)
        return (parts[0], parts[1])
    }
}
