//
//  WhisperParity.swift
//
//  PT-vs-Swift Whisper-encoder parity via Python-MLX-as-intermediate.
//  Same pattern as VAEParity and UMT5Parity.
//
//  Opt-in: requires the bf16 Whisper encoder weights (~1.3 GB). Gated by
//  LONGCAT_WHISPER_AUTO_DOWNLOAD=1 (mirrors the Python port's pattern),
//  or LONGCAT_AVATAR_WEIGHTS_DIR pointing at an unpacked weights dir.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class WhisperParity: XCTestCase {

    private static let repoID = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    private func loadFixture(_ name: String) throws -> MLXArray {
        let url = Bundle.module.url(
            forResource: name,
            withExtension: "npy",
            subdirectory: "Resources/whisper-parity"
        )
        guard let url else {
            XCTFail("Could not locate whisper-parity fixture \(name).npy in test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    private func skipUnlessOptedIn() throws {
        let env = ProcessInfo.processInfo.environment
        let optIn = env["LONGCAT_WHISPER_AUTO_DOWNLOAD"] == "1"
        let localDir = env["LONGCAT_AVATAR_WEIGHTS_DIR"]
        if !optIn && localDir == nil {
            throw XCTSkip("""
                Whisper parity test is opt-in. Set LONGCAT_WHISPER_AUTO_DOWNLOAD=1 \
                to download the bf16 Whisper encoder weights (~1.3 GB), or set \
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

        let model = try await WhisperEncoder.fromPretrained(Self.repoID)

        let mel = try loadFixture("input_mel")
        let pythonOutput = try loadFixture("output")
        XCTAssertEqual(mel.shape, [1, 128, 64])
        XCTAssertEqual(pythonOutput.shape, [1, 32, 1280])

        let swiftOutput = model(mel)
        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Whisper output shape diverged")

        // Per L22, expect bf16 kernel drift compounded across 32 layers.
        // Whisper's smaller d_model=1280 (vs umT5's 4096) keeps per-layer
        // noise smaller, but the extra 8 layers + final LayerNorm magnitude
        // dictate the actual threshold. 0.15 matches umT5's empirically-
        // chosen ceiling; tighten later if measured drift is lower.
        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 0.15, """
            Whisper forward parity failed: max_abs = \(err) > 0.15.
            A divergence at this magnitude implies a structural port bug
            (vs documented bf16 GPU matmul kernel drift, L22).
            """)
        print("✓ whisper.forward parity max_abs = \(err) (threshold 0.15, see L22)")
    }
}
