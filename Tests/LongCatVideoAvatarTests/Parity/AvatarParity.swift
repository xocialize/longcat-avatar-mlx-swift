//
//  AvatarParity.swift
//
//  PT-vs-Swift Avatar DiT parity via Python-MLX-as-intermediate. Same pattern
//  as VAE/UMT5/Whisper/baseDiT parity tests. Loads the FULL Avatar checkpoint
//  (no key filtering needed — every published key maps to a class slot).
//
//  Opt-in: requires the bf16 DiT weights (~33 GB sharded). Gated by
//  LONGCAT_AVATAR_AUTO_DOWNLOAD=1, or LONGCAT_AVATAR_WEIGHTS_DIR pointing
//  at an unpacked weights dir.
//

import Foundation
import XCTest
import MLX
import MLXNN
@testable import LongCatVideoAvatar

final class AvatarParity: XCTestCase {

    private static let repoID = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    private func loadFixture(_ name: String) throws -> MLXArray {
        let url = Bundle.module.url(
            forResource: name,
            withExtension: "npy",
            subdirectory: "Resources/avatar-parity"
        )
        guard let url else {
            XCTFail("Could not locate avatar-parity fixture \(name).npy in test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    private func skipUnlessOptedIn() throws {
        let env = ProcessInfo.processInfo.environment
        let optIn = env["LONGCAT_AVATAR_AUTO_DOWNLOAD"] == "1"
        let localDir = env["LONGCAT_AVATAR_WEIGHTS_DIR"]
        if !optIn && localDir == nil {
            throw XCTSkip("""
                Avatar parity test is opt-in. Set LONGCAT_AVATAR_AUTO_DOWNLOAD=1 \
                to download the bf16 DiT weights (~33 GB sharded), or set \
                LONGCAT_AVATAR_WEIGHTS_DIR to an unpacked weights dir.
                """)
        }
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = (a - b).asType(.float32)
        return MLX.abs(diff).max().item(Float.self)
    }

    // MARK: - forward

    func testForwardMatchesPythonMLX() async throws {
        try skipUnlessOptedIn()

        let model = try await LongCatVideoAvatarTransformer3DModel.fromPretrained(Self.repoID)

        let hs = try loadFixture("hidden_states")
        let timestep = try loadFixture("timestep")
        let ehs = try loadFixture("encoder_hidden_states")
        let mask = try loadFixture("encoder_attention_mask")
        let audio = try loadFixture("audio_embs")
        let pythonOutput = try loadFixture("output")

        XCTAssertEqual(hs.shape, [1, 16, 3, 8, 8])
        XCTAssertEqual(audio.shape, [1, 9, 5, 5, 1280])
        XCTAssertEqual(pythonOutput.shape, hs.shape)

        let swiftOutput = model(
            hiddenStates: hs,
            timestep: timestep,
            encoderHiddenStates: ehs,
            audioEmbs: audio,
            encoderAttentionMask: mask,
            numCondLatents: 0
        )
        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Avatar DiT output shape diverged")

        // Avatar DiT measured 0.32 vs base DiT's 0.033 — ~10x looser. The
        // increase tracks the structural additions Avatar makes over base:
        //
        //   1. AudioProjModel: 4-layer MLP at large dims (32000 → 512 → 512
        //      → 24576). Each Linear is a bf16 matmul + per-output bf16
        //      kernel drift between Python-MLX and Swift-MLX.
        //   2. Per-block audio_cross_attn (fused SDPA, OK) +
        //      audio_adaLN_modulation Linear + pre_video_crs_attn_norm —
        //      adds another full attention path through 48 layers.
        //
        // The 5.5% relative error (0.32 / abs.max=5.78) matches umT5's 6.8%
        // pattern and is well above base DiT's 0.8%, but the structural
        // port is correct — every per-layer parity is in expected bf16
        // drift territory; it just compounds more in Avatar.
        //
        // Threshold 0.5 leaves headroom; baseline measured ~0.32.
        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 0.5, """
            Avatar DiT forward parity failed: max_abs = \(err) > 0.5.
            A divergence at this magnitude implies a structural port bug
            (vs ~0.32 documented baseline; bf16 GPU kernel drift L22).
            """)
        print("✓ avatar.forward parity max_abs = \(err) (threshold 0.5, baseline ~0.32)")
    }
}
