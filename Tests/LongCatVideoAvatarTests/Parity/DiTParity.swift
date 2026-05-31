//
//  DiTParity.swift
//
//  PT-vs-Swift base DiT parity via Python-MLX-as-intermediate. Mirrors
//  VAE/UMT5/Whisper parity tests.
//
//  Opt-in: requires the bf16 DiT weights (~33 GB sharded). Gated by
//  LONGCAT_DIT_AUTO_DOWNLOAD=1, or LONGCAT_AVATAR_WEIGHTS_DIR pointing
//  at an unpacked weights dir.
//
//  Note: we load the BASE DiT class from the Avatar checkpoint. The
//  audio-overlay weights (audio_proj, audio_adaLN, audio_cross_attn)
//  are filtered out before update(parameters:) since the base class
//  has no slots for them — mirrors Python's strict=False load behavior.
//

import Foundation
import XCTest
import MLX
import MLXNN
@testable import LongCatVideoAvatar

final class DiTParity: XCTestCase {

    private static let repoID = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    private func loadFixture(_ name: String) throws -> MLXArray {
        let url = Bundle.module.url(
            forResource: name,
            withExtension: "npy",
            subdirectory: "Resources/dit-parity"
        )
        guard let url else {
            XCTFail("Could not locate dit-parity fixture \(name).npy in test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    private func skipUnlessOptedIn() throws {
        let env = ProcessInfo.processInfo.environment
        let optIn = env["LONGCAT_DIT_AUTO_DOWNLOAD"] == "1"
        let localDir = env["LONGCAT_AVATAR_WEIGHTS_DIR"]
        if !optIn && localDir == nil {
            throw XCTSkip("""
                DiT parity test is opt-in. Set LONGCAT_DIT_AUTO_DOWNLOAD=1 \
                to download the bf16 DiT weights (~33 GB sharded), or set \
                LONGCAT_AVATAR_WEIGHTS_DIR to an unpacked weights dir.
                """)
        }
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = (a - b).asType(.float32)
        return MLX.abs(diff).max().item(Float.self)
    }

    /// Load base-DiT weights from the avatar checkpoint, filtering out
    /// audio-overlay keys (audio_proj.*, blocks.X.audio_*) — the base
    /// LongCatVideoTransformer3DModel doesn't have slots for those.
    private func loadBaseFiltered(repoID: String) async throws -> LongCatVideoTransformer3DModel {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID)
        let dir = try WeightLoader.componentDirectory("dit", under: root)
        let config: LongCatVideoConfig = try WeightLoader.loadConfig(
            LongCatVideoConfig.self,
            from: dir.appendingPathComponent("config.json")
        )
        let model = LongCatVideoTransformer3DModel.fromConfig(config)
        var weights = try WeightLoader.loadShardedSafetensors(
            indexURL: dir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )
        // Drop avatar-only keys; the base class has no slots for them.
        // The Avatar overlay adds:
        //  - audio_proj.* (AudioProjModel top-level)
        //  - blocks.X.audio_adaLN_modulation.* / audio_cross_attn.*
        //  - blocks.X.pre_video_crs_attn_norm.* (pre-norm for audio path's video input)
        //  - blocks.X.pre_audio_crs_attn_norm.* (only when audio_prenorm=true)
        weights = weights.filter { key, _ in
            if key.contains("audio_proj") { return false }
            if key.contains(".audio_") { return false }
            if key.contains(".pre_video_crs_attn_norm") { return false }
            if key.contains(".pre_audio_crs_attn_norm") { return false }
            return true
        }
        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }

    // MARK: - forward

    func testForwardMatchesPythonMLX() async throws {
        try skipUnlessOptedIn()

        let model = try await loadBaseFiltered(repoID: Self.repoID)

        let hs = try loadFixture("hidden_states")
        let timestep = try loadFixture("timestep")
        let ehs = try loadFixture("encoder_hidden_states")
        let mask = try loadFixture("encoder_attention_mask")
        let pythonOutput = try loadFixture("output")

        XCTAssertEqual(hs.shape, [1, 16, 1, 8, 8])
        XCTAssertEqual(pythonOutput.shape, hs.shape)

        let swiftOutput = model(
            hiddenStates: hs,
            timestep: timestep,
            encoderHiddenStates: ehs,
            encoderAttentionMask: mask,
            numCondLatents: 0
        )
        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Base DiT output shape diverged")

        // Per L22 + S3.5 finding: fused MLXFast.scaledDotProductAttention is
        // far more deterministic across Python-MLX and Swift-MLX than manual
        // matmul+softmax chains. Whisper (32 layers, fused SDPA) measured 0.016
        // vs umT5 (24 layers, manual chain) at 0.119. Base DiT (48 layers,
        // fused SDPA) measured 0.033 — fits the fused-SDPA-dominant model.
        // Threshold 0.1 leaves headroom while catching structural regressions.
        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 0.1, """
            Base DiT forward parity failed: max_abs = \(err) > 0.1.
            A divergence at this magnitude implies a structural port bug
            (vs ~0.033 documented baseline; bf16 GPU kernel drift L22).
            """)
        print("✓ dit.forward parity max_abs = \(err) (threshold 0.1, baseline ~0.033)")
    }
}
