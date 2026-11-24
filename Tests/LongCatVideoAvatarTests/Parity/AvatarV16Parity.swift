//
//  AvatarV16Parity.swift
//
//  Avatar 1.6 against the Python-MLX reference, on real weights. Opt-in: set
//  LONGCAT_AVATAR_V16_DIR to a converted `LongCat-Video-Avatar-1.6-bf16`
//  directory (recipes/convert_longcat_avatar.py --variant v1.6-bf16).
//
//  Two gates, because one cannot do both jobs:
//
//  - LoRA terms, fp32 (`Resources/lora-parity-v16`, from
//    scripts/dump_lora_swift_fixtures.py): `up(down(x)) * alphaScale` for the
//    real DMD and OPSD LoRAs on six target modules, both split-fused kinds
//    included. fp32 agrees across bindings, so this pins the key decoding, the
//    split order and `alpha_scale` down to a fraction of a percent.
//  - The whole pipeline (`Resources/pipeline-parity-v16`, from
//    scripts/dump_pipeline_swift_fixtures.py): bf16 kernels drift between the
//    Python and Swift bindings by about as much as the DMD → OPSD switch
//    changes this tiny clip (max abs 0.14 either way), so the pass condition
//    is relative: the default schedule must sit closer to Python's
//    default-schedule output than to its DMD-on-all-steps output, and the
//    all-DMD schedule the other way round.
//

import Foundation
import MLX
import MLXNN
import XCTest

@testable import LongCatVideoAvatar

final class AvatarV16Parity: XCTestCase {

    private func variantDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["LONGCAT_AVATAR_V16_DIR"] else {
            throw XCTSkip("Set LONGCAT_AVATAR_V16_DIR to a converted LongCat-Video-Avatar-1.6-bf16 directory.")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func fixture(_ name: String, _ directory: String) throws -> MLXArray {
        guard let url = Bundle.module.url(forResource: name, withExtension: "npy", subdirectory: "Resources/\(directory)") else {
            XCTFail("no \(directory)/\(name).npy in the test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func meanAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self)
    }

    // MARK: - LoRA terms

    static let modules = [
        "blocks.0.attn.qkv", "blocks.17.attn.proj", "blocks.23.cross_attn.q_linear",
        "blocks.31.cross_attn.kv_linear", "blocks.40.ffn.w1", "blocks.47.ffn.w2",
    ]

    func testLoRATermsMatchPythonFP32() throws {
        let root = try variantDirectory()
        let loras = [
            ("dmd", try LoRA.groupLoRATensors(try WeightLoader.loadSafetensors(url: root.appendingPathComponent("lora/dmd_lora.safetensors")))),
            ("opsd", try LoRA.groupLoRATensors(try WeightLoader.loadSafetensors(url: root.appendingPathComponent("lora/opsd_lora.safetensors")))),
        ]
        for (_, groups) in loras { XCTAssertEqual(groups.count, 336, "7 targets × 48 blocks") }

        for (index, path) in Self.modules.enumerated() {
            let x = try fixture("m\(index)_x", "lora-parity-v16")
            for (name, groups) in loras {
                let want = try fixture("m\(index)_\(name)", "lora-parity-v16")
                let other = try fixture("m\(index)_\(name == "dmd" ? "opsd" : "dmd")", "lora-parity-v16")
                let scale = MLX.abs(want).max().item(Float.self)
                let group = try XCTUnwrap(groups[path], "\(name) has no \(path)")
                let fp32 = LoRAGroup(
                    down: group.down.asType(.float32), ups: group.ups.map { $0.asType(.float32) },
                    alphaScale: group.alphaScale)

                // The merge math, on the CPU stream (exact fp32).
                let merged: MLXArray = Stream.withNewDefaultStream(device: .cpu) {
                    let y = MLX.matmul(x, LoRA.computeMergedDelta(fp32).transposed())
                    MLX.eval(y)
                    return y
                }
                XCTAssertLessThan(maxAbs(merged, want), 1e-3 * scale, "\(name) \(path): merged delta")

                // The runtime side-path, over a zero base so only the LoRA term is left.
                let base = Linear(weight: MLXArray.zeros([want.dim(-1), x.dim(-1)]), bias: nil)
                let wrapped = LoRALinear(base)
                wrapped.lora.add(key: name, group: fp32, dtype: .float32)
                XCTAssertEqual(MLX.abs(wrapped(x)).max().item(Float.self), 0, "multiplier 0 is the base")
                wrapped.lora.setMultiplier(1, for: name)
                let sidePath = wrapped(x)
                XCTAssertLessThan(maxAbs(sidePath, want), 2e-2 * scale, "\(name) \(path): runtime side-path")

                // The gate discriminates: the other LoRA's term is far outside either tolerance.
                XCTAssertGreaterThan(maxAbs(sidePath, other), 0.2 * scale, "\(name) \(path) is not distinguishable from the other LoRA")
                print("✓ \(name) \(path): merged \(maxAbs(merged, want) / scale), side-path \(maxAbs(sidePath, want) / scale) of |y|; other LoRA \(maxAbs(sidePath, other) / scale)")
            }
        }
    }

    // MARK: - Pipeline

    private func run(mixSteps: Int?) throws -> MLXArray {
        var config = PipelineConfig()
        config.mixSteps = mixSteps
        let pipeline = try LongCatAvatarPipeline.fromDirectory(try variantDirectory(), config: config)
        XCTAssertEqual(pipeline.loraMix?.modules.count, 336)
        XCTAssertEqual(pipeline.loraMix?.mixSteps, mixSteps ?? 4)
        let f = "pipeline-parity-v16"
        let video = try pipeline(
            image: try fixture("image", f),
            audioMel: try fixture("audio_mel", f),
            textEmbeds: try fixture("text_embeds", f),
            textMask: try fixture("text_mask", f),
            uncondEmbeds: try fixture("uncond_embeds", f),
            uncondMask: try fixture("uncond_mask", f),
            numFrames: 5, height: 64, width: 64,
            initialNoise: [try fixture("initial_noise", f)])
        MLX.eval(video)
        return video
    }

    func testPipelineFollowsTheLoRASchedule() throws {
        let pythonDefault = try fixture("output", "pipeline-parity-v16")
        let pythonAllDMD = try fixture("output_dmd_only", "pipeline-parity-v16")
        XCTAssertEqual(pythonDefault.shape, [1, 3, 5, 64, 64])

        let swiftDefault = try run(mixSteps: nil)
        XCTAssertEqual(swiftDefault.shape, pythonDefault.shape)
        let (own, wrong) = (meanAbs(swiftDefault, pythonDefault), meanAbs(swiftDefault, pythonAllDMD))
        print("default schedule: max_abs \(maxAbs(swiftDefault, pythonDefault)), mean_abs \(own) to Python's default, \(wrong) to Python's all-DMD")
        XCTAssertLessThan(maxAbs(swiftDefault, pythonDefault), 0.4, "structural divergence (bf16 drift measured 0.14)")
        XCTAssertLessThan(own, 0.85 * wrong, "the default schedule should match Python's default-schedule output best")

        let swiftAllDMD = try run(mixSteps: 8)
        let (ownDMD, wrongDMD) = (meanAbs(swiftAllDMD, pythonAllDMD), meanAbs(swiftAllDMD, pythonDefault))
        print("all-DMD schedule: max_abs \(maxAbs(swiftAllDMD, pythonAllDMD)), mean_abs \(ownDMD) to Python's all-DMD, \(wrongDMD) to Python's default")
        XCTAssertLessThan(ownDMD, 0.85 * wrongDMD, "DMD on all steps should match Python's all-DMD output best")
    }
}
