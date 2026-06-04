//
//  FlowMatchEulerDiscreteScheduler.swift
//
//  Minimal Flow Matching Euler discrete scheduler — what the Avatar
//  pipeline drives the DiT with. Python imports the same scheduler from
//  `mlx_arsenal.diffusion`. We inline it here rather than depend on a
//  separate arsenal package, since the Swift port is otherwise zero-dep
//  beyond mlx-swift + swift-transformers.
//
//  Behavior matches diffusers' `FlowMatchEulerDiscreteScheduler`:
//
//   - Configurable `shift` (Avatar uses 7.0): rescales sigmas via
//     `shift * sigma / (1 + (shift - 1) * sigma)`. Inverts the standard
//     timestep-by-shift Karras rescaling for flow matching.
//   - `setTimesteps(numSteps:sigmas:)` accepts a custom sigma list.
//     We APPEND 1.0 as the trailing sentinel (matching mlx-arsenal's
//     behavior). The pipeline overwrites that to 0.0 after set (L17 fix).
//   - `step(modelOutput:timestep:sample:)` does the Euler update:
//         x_next = x + (sigma_next - sigma_current) * v
//     where `v` is the velocity prediction (LongCat outputs `-v`, so
//     the pipeline flips before calling step).
//

import Foundation
import MLX

public final class FlowMatchEulerDiscreteScheduler {

    public let numTrainTimesteps: Int
    public let shift: Float
    public let useDynamicShifting: Bool

    /// Sigmas array after `setTimesteps`. Mutable so the pipeline can
    /// patch the trailing sentinel (L17).
    public var sigmas: MLXArray = MLXArray.zeros([1])
    /// Timesteps corresponding to each sigma (excluding the trailing sentinel).
    public var timesteps: MLXArray = MLXArray.zeros([1])

    private var stepIndex: Int = 0

    public init(
        numTrainTimesteps: Int = 1000,
        shift: Float = 7.0,
        useDynamicShifting: Bool = false
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
        self.useDynamicShifting = useDynamicShifting
    }

    /// Set up the schedule. `sigmas` is the user-provided sigma list
    /// (descending). We apply the shift rescaling, then append a trailing
    /// sentinel 1.0 (high noise) — the pipeline overrides to 0.0 for the
    /// final-step clean-denoising fix.
    public func setTimesteps(_ numSteps: Int, sigmas userSigmas: [Float]) {
        precondition(userSigmas.count == numSteps, "sigmas count must match numSteps")

        // Apply shift rescaling: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        let rescaled: [Float] = userSigmas.map { s in
            shift * s / (1 + (shift - 1) * s)
        }

        // Timesteps = sigma * num_train_timesteps
        let timestepValues = rescaled.map { $0 * Float(numTrainTimesteps) }

        // Append the trailing sentinel
        let withSentinel = rescaled + [Float(1.0)]
        self.sigmas = MLXArray(withSentinel.map { $0 }).asType(.float32)
        self.timesteps = MLXArray(timestepValues).asType(.float32)
        self.stepIndex = 0
    }

    /// Euler step. Pipeline calls this in order matching `timesteps`; we
    /// look up the current and next sigma by `stepIndex` rather than by
    /// `timestep` value to avoid float-equality pitfalls.
    public func step(
        modelOutput: MLXArray,
        timestep: Float,
        sample: MLXArray
    ) -> MLXArray {
        _ = timestep   // unused — we track via stepIndex internally
        let sigma = sigmas[stepIndex].item(Float.self)
        let sigmaNext = sigmas[stepIndex + 1].item(Float.self)
        let dt = sigmaNext - sigma
        let result = sample + MLXArray(dt) * modelOutput
        stepIndex += 1
        return result
    }

    /// Reset the step index — call before starting a fresh denoising loop
    /// on the same scheduler instance.
    public func reset() {
        stepIndex = 0
    }
}
