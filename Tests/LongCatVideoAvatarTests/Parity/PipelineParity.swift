//
//  PipelineParity.swift
//
//  End-to-end pipeline parity vs Python-MLX. Runs the full
//  LongCatAvatarPipeline (VAE encode + Whisper + 8-step DMD denoising
//  loop with 3-pass disentangled CFG + VAE decode) on synthetic small
//  inputs with a shared initial-noise tensor, then compares against the
//  Python reference output.
//
//  Opt-in: requires the full bf16-dmd-merged weights (~46 GB). Gated by
//  LONGCAT_PIPELINE_AUTO_DOWNLOAD=1 or LONGCAT_AVATAR_WEIGHTS_DIR.
//

import Foundation
import XCTest
import MLX
import MLXNN
@testable import LongCatVideoAvatar

final class PipelineParity: XCTestCase {

    private static let repoID = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    private func loadFixture(_ name: String) throws -> MLXArray {
        let url = Bundle.module.url(
            forResource: name,
            withExtension: "npy",
            subdirectory: "Resources/pipeline-parity"
        )
        guard let url else {
            XCTFail("Could not locate pipeline-parity fixture \(name).npy in test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    private func skipUnlessOptedIn() throws {
        let env = ProcessInfo.processInfo.environment
        let optIn = env["LONGCAT_PIPELINE_AUTO_DOWNLOAD"] == "1"
        let localDir = env["LONGCAT_AVATAR_WEIGHTS_DIR"]
        if !optIn && localDir == nil {
            throw XCTSkip("""
                Pipeline parity test is opt-in. Set LONGCAT_PIPELINE_AUTO_DOWNLOAD=1 \
                to download the bf16 weights (~46 GB), or set \
                LONGCAT_AVATAR_WEIGHTS_DIR to an unpacked weights dir.
                """)
        }
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = (a - b).asType(.float32)
        return MLX.abs(diff).max().item(Float.self)
    }

    /// Load Avatar DiT into the base class (filtering audio-overlay keys
    /// would lose audio behavior — Avatar wants the FULL key set). Used
    /// here instead of LongCatVideoAvatarTransformer3DModel.fromPretrained
    /// only because the pipeline call site needs to read `xEmbedder.proj.weight`
    /// which is the same in both classes.
    private func loadFullPipeline() async throws -> LongCatAvatarPipeline {
        try await LongCatAvatarPipeline.fromPretrained(Self.repoID)
    }

    func testEndToEndMatchesPythonMLX() async throws {
        try skipUnlessOptedIn()

        print("  Loading pipeline (this can take 30-60s on first run)...")
        let pipeline = try await loadFullPipeline()

        print("  Loading fixtures...")
        let image = try loadFixture("image")
        let audioMel = try loadFixture("audio_mel")
        let textEmbeds = try loadFixture("text_embeds")
        let textMask = try loadFixture("text_mask")
        let uncondEmbeds = try loadFixture("uncond_embeds")
        let uncondMask = try loadFixture("uncond_mask")
        let initialNoise = try loadFixture("initial_noise")
        let pythonOutput = try loadFixture("output")

        XCTAssertEqual(image.shape, [1, 3, 1, 64, 64])
        XCTAssertEqual(audioMel.shape, [1, 128, 200])
        XCTAssertEqual(initialNoise.shape, [1, 16, 2, 8, 8])
        XCTAssertEqual(pythonOutput.shape, [1, 3, 5, 64, 64])

        print("  Running Swift pipeline (this can take 30-90s)...")
        let swiftOutput = pipeline(
            image: image,
            audioMel: audioMel,
            textEmbeds: textEmbeds,
            textMask: textMask,
            uncondEmbeds: uncondEmbeds,
            uncondMask: uncondMask,
            numFrames: 5,
            height: 64,
            width: 64,
            seed: 0,
            initialNoise: initialNoise
        )

        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Pipeline output shape diverged")

        // End-to-end runs the Avatar DiT 3 times per step for 8 steps = 24 DiT
        // passes (each 48 layers), plus VAE encode + decode. Per-step Avatar
        // DiT drift was 0.32; after 24 passes + VAE the compounded divergence
        // depends on how much each step amplifies vs clips the noise (DMD
        // distilled schedule trajectories converge somewhat). Threshold 5.0
        // is the dump-script's conservative estimate; we report the
        // actual measured value.
        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 5.0, """
            Pipeline e2e parity failed: max_abs = \(err) > 5.0.
            A divergence at this magnitude implies a structural port bug
            (vs documented bf16 GPU kernel drift L22 compounded through
            8 denoising steps + VAE encode/decode).
            """)
        print("✓ pipeline.e2e parity max_abs = \(err) (threshold 5.0)")
    }
}
