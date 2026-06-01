//
//  PipelineSmoke.swift
//
//  Smoke tests for Pipeline/{Guidance, FlowMatchEulerDiscreteScheduler,
//  LongCatAvatarPipeline} and Audio/AudioProcess. No weights, no end-to-end
//  run (the latter is S3.9 — runs the whole DiT loop).
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class PipelineSmoke: XCTestCase {

    // MARK: - Guidance

    func testDisentangledCFGCombineMatchesFormula() {
        // For cond=(1,1,1), uncond_text=(0,0,0), uncond=(2,2,2),
        // s_t=4.0, s_a=4.0:
        //   noise = uncond + s_t*(cond - uncond_text) + s_a*(uncond_text - uncond)
        //         = 2 + 4*(1-0) + 4*(0-2) = 2 + 4 - 8 = -2
        let cond = MLXArray([Float(1), 1, 1])
        let ut = MLXArray([Float(0), 0, 0])
        let unc = MLXArray([Float(2), 2, 2])
        let out = Guidance.disentangledCFGCombine(
            noisePredCond: cond,
            noisePredUncondText: ut,
            noisePredUncond: unc,
            textGuidanceScale: 4.0,
            audioGuidanceScale: 4.0
        )
        XCTAssertEqual(out.asArray(Float.self), [-2, -2, -2])
    }

    func testFlipVelocityNegates() {
        let x = MLXArray([Float(1), -2, 3])
        let flipped = Guidance.flipVelocityForScheduler(x)
        XCTAssertEqual(flipped.asArray(Float.self), [-1, 2, -3])
    }

    func testGetDMDDistilledSigmasShape() {
        let sigmas = Guidance.getDMDDistilledSigmas(samplingSteps: 8)
        XCTAssertEqual(sigmas.shape, [8])
        // Descending
        let vals = sigmas.asArray(Float.self)
        for i in 1..<vals.count {
            XCTAssertLessThan(vals[i], vals[i - 1])
        }
    }

    func testGetDMDDistilledSigmasMatchesPythonValues() {
        // Bit-exact comparison against the Python port's values
        // (longcat_video_avatar/guidance.py::get_dmd_distilled_sigmas).
        let sigmas = Guidance.getDMDDistilledSigmas(samplingSteps: 8)
        let vals = sigmas.asArray(Float.self)
        let pythonValues: [Float] = [
            1.0,
            0.8748748898506165,
            0.7497497200965881,
            0.6246246099472046,
            0.49949949979782104,
            0.3743743598461151,
            0.24924924969673157,
            0.12412412464618683,
        ]
        XCTAssertEqual(vals.count, pythonValues.count)
        for i in 0..<vals.count {
            XCTAssertEqual(vals[i], pythonValues[i], accuracy: 1e-6, "sigma[\(i)] mismatch")
        }
    }

    func testCFGSplitOutputsRoundTrip() {
        let uncond = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 4)
        let cond = MLXArray([Float(5), 6, 7, 8]).reshaped(1, 4)
        let stacked = MLX.concatenated([uncond, cond], axis: 0)
        let (u, c) = Guidance.cfgSplitOutputs(stacked)
        XCTAssertEqual(u.asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(c.asArray(Float.self), [5, 6, 7, 8])
    }

    // MARK: - FlowMatchEulerDiscreteScheduler

    func testSchedulerSetTimestepsBuildsSigmasAndSentinel() {
        let s = FlowMatchEulerDiscreteScheduler(numTrainTimesteps: 1000, shift: 7.0)
        s.setTimesteps(3, sigmas: [1.0, 0.5, 0.25])
        // After shift=7.0 rescale + append sentinel 1.0
        XCTAssertEqual(s.sigmas.dim(0), 4)   // 3 sigmas + 1 sentinel
        XCTAssertEqual(s.timesteps.dim(0), 3)
        // Last sigma is the trailing sentinel = 1.0
        XCTAssertEqual(s.sigmas[3].item(Float.self), 1.0, accuracy: 1e-5)
    }

    func testSchedulerStepEulerUpdate() {
        let s = FlowMatchEulerDiscreteScheduler(numTrainTimesteps: 1000, shift: 1.0)
        // shift=1.0 → sigmas = userSigmas verbatim
        s.setTimesteps(2, sigmas: [1.0, 0.5])
        // step 0: dt = sigmas[1] - sigmas[0] = 0.5 - 1.0 = -0.5
        let x = MLXArray([Float(1), 2, 3])
        let v = MLXArray([Float(0.1), 0.2, 0.3])
        let out = s.step(modelOutput: v, timestep: 0, sample: x)
        // x + (-0.5)*v = [1 - 0.05, 2 - 0.1, 3 - 0.15] = [0.95, 1.9, 2.85]
        let expected: [Float] = [0.95, 1.9, 2.85]
        let actual = out.asArray(Float.self)
        for i in 0..<3 {
            XCTAssertEqual(actual[i], expected[i], accuracy: 1e-5)
        }
    }

    // MARK: - AudioProcess

    func testGroupPoolWhisperHiddenStatesShape() {
        // 33 layers, each [B=1, T=4, D=8]
        var layers: [MLXArray] = []
        for _ in 0..<33 {
            layers.append(MLXRandom.normal([1, 4, 8]))
        }
        let pooled = AudioProcess.groupPoolWhisperHiddenStates(layers)
        XCTAssertEqual(pooled.shape, [1, 4, 5, 8])
    }

    func testGroupPoolFifthChannelEqualsLastLayer() {
        // The 5th group is the singleton last layer — verify identity.
        var layers: [MLXArray] = []
        for i in 0..<33 {
            layers.append(MLXArray.full([1, 2, 4], values: MLXArray(Float(i))))
        }
        let pooled = AudioProcess.groupPoolWhisperHiddenStates(layers)
        // pooled[0, 0, 4, 0] should equal layer 32's value
        XCTAssertEqual(pooled[0, 0, 4, 0].item(Float.self), 32.0)
    }

    func testGroupPoolFirstChannelIsAverageOfLayers0Through7() {
        var layers: [MLXArray] = []
        for i in 0..<33 {
            layers.append(MLXArray.full([1, 1, 1], values: MLXArray(Float(i))))
        }
        let pooled = AudioProcess.groupPoolWhisperHiddenStates(layers)
        // Group 0 = mean([0..8]) = (0+1+2+3+4+5+6+7)/8 = 3.5
        XCTAssertEqual(pooled[0, 0, 0, 0].item(Float.self), 3.5, accuracy: 1e-5)
    }

    func testLinearInterpolateFeaturesEdgeBehavior() {
        // For T=2, T_out=4: src_idx = linspace(0, 1, 4) = [0, 0.333, 0.667, 1.0]
        // → frac = [0, 0.333, 0.667, 1.0]
        // For features [B=1, T=2, D=1] = [[[0], [10]]]:
        //   out[0] = 0*(1-0) + 10*0 = 0
        //   out[1] = 0*(1-0.333) + 10*0.333 = 3.333
        //   out[2] = 0*(1-0.667) + 10*0.667 = 6.667
        //   out[3] = 0*(1-1) + 10*1 = ... wait, both lo and hi cap at T-1=1
        //   so out[3] = features[1]*(1-1) + features[1]*1 = 10
        let f = MLXArray([Float(0), 10]).reshaped(1, 2, 1)
        let out = AudioProcess.linearInterpolateFeatures(f, inputFps: 1, outputFps: 2, outputLen: 4)
        XCTAssertEqual(out.shape, [1, 4, 1])
        let vals = out.flattened().asArray(Float.self)
        XCTAssertEqual(vals[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(vals[3], 10.0, accuracy: 1e-4)
    }

    func testBuildAvatarAudioEmbeddingsShape() {
        // B=1, T_enc=4, G=5, D=8 → at fps=2, encFps=4 → T_video = 4 * 2/4 = 2
        let g = MLXRandom.normal([1, 4, 5, 8])
        let embs = AudioProcess.buildAvatarAudioEmbeddings(g, fps: 2, encFps: 4, audioWindow: 5)
        XCTAssertEqual(embs.shape, [1, 2, 5, 5, 8])   // [B, T_video, W, G, D]
    }
}
