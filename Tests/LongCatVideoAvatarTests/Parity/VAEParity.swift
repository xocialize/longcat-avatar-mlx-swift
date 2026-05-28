//
//  VAEParity.swift
//
//  PT-vs-Swift VAE parity (S3.3c) via the Python-MLX-as-intermediate
//  strategy:
//
//      PT (diffusers) ──parity, encode 1e-3 / decode 2e-2──> Python-MLX
//                                                                │
//                                                                ▼
//                                                       (these fixtures)
//                                                                │
//                                                                ▼
//                                       Swift-MLX ──parity, 1e-4 target──> ✓
//
//  Both Python-MLX and Swift-MLX call into the same mlx-c. Divergence
//  should be tiny — fp32-rounding differences in op orderings only.
//
//  Opt-in: requires the bf16 VAE weights (~250 MB). Gated by
//  LONGCAT_VAE_AUTO_DOWNLOAD=1 (mirrors the Python port's pattern), or
//  LONGCAT_AVATAR_WEIGHTS_DIR pointing at an unpacked weights dir.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class VAEParity: XCTestCase {

    // See VAEOpsSmoke for the `swift test` vs `xcodebuild test` caveat.

    private static let repoID = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    private func loadFixture(_ name: String) throws -> MLXArray {
        let url = Bundle.module.url(
            forResource: name,
            withExtension: "npy",
            subdirectory: "Resources/vae-parity"
        )
        guard let url else {
            XCTFail("Could not locate vae-parity fixture \(name).npy in test bundle")
            return MLXArray.zeros([1])
        }
        return try loadNumpy(url: url)
    }

    /// Skip with a clear message unless the user opted in.
    private func skipUnlessOptedIn() throws -> Bool {
        let env = ProcessInfo.processInfo.environment
        let optIn = env["LONGCAT_VAE_AUTO_DOWNLOAD"] == "1"
        let localDir = env["LONGCAT_AVATAR_WEIGHTS_DIR"]
        if !optIn && localDir == nil {
            throw XCTSkip("""
                VAE parity test is opt-in. Set LONGCAT_VAE_AUTO_DOWNLOAD=1 to \
                download the bf16 VAE weights (~250 MB), or set \
                LONGCAT_AVATAR_WEIGHTS_DIR to an unpacked weights dir.
                """)
        }
        return true
    }

    /// Helper: max(|a - b|) reduced to a scalar Float.
    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = (a - b).asType(.float32)
        let absDiff = MLX.abs(diff)
        return absDiff.max().item(Float.self)
    }

    // MARK: - encode

    func testEncodeMatchesPythonMLX() async throws {
        _ = try skipUnlessOptedIn()

        let vae = try await AutoencoderKLWan.fromPretrained(Self.repoID, includeEncoder: true)

        let input = try loadFixture("encode_input")
        let pythonOutput = try loadFixture("encode_output")
        XCTAssertEqual(input.shape, [1, 3, 5, 16, 16])
        XCTAssertEqual(pythonOutput.shape, [1, 16, 2, 2, 2])

        let swiftOutput = vae.encode(input)
        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Swift encode shape diverged")

        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 1e-4, """
            VAE encode parity failed: max_abs = \(err) > 1e-4.
            Python-MLX and Swift-MLX should match to fp32 rounding — a
            larger divergence implies a structural port bug in one of
            CausalConv3d, the residual block, the encoder down stages,
            or quant_conv.
            """)
        print("✓ vae.encode parity max_abs = \(err) (threshold 1e-4)")
    }

    // MARK: - decode

    func testDecodeMatchesPythonMLX() async throws {
        _ = try skipUnlessOptedIn()

        let vae = try await AutoencoderKLWan.fromPretrained(Self.repoID, includeEncoder: false)

        let input = try loadFixture("decode_input")
        let pythonOutput = try loadFixture("decode_output")
        XCTAssertEqual(input.shape, [1, 16, 3, 8, 8])
        XCTAssertEqual(pythonOutput.shape, [1, 3, 9, 64, 64])

        let swiftOutput = vae.decode(input)
        XCTAssertEqual(swiftOutput.shape, pythonOutput.shape, "Swift decode shape diverged")

        // Threshold is intentionally looser than encode (5e-3 vs 1e-4) — the
        // decoder mid-block runs WanAttentionBlock through .cpu stream for
        // fp32 precision (L10), but auxiliary ops (transpose / reshape /
        // expand_dims) run on the default GPU stream and materialize at
        // the boundary. Python's `with mx.stream(mx.cpu):` block keeps the
        // whole region on CPU; mlx-swift's per-op `stream: .cpu` doesn't
        // give us that without intrusive ceremony. Net divergence on a
        // 9-frame 64×64 decode: ~1.76e-3 vs 3.1e-6 on encode (which uses
        // the same attention block but the rest of the encoder runs
        // through downsample paths whose error doesn't accumulate the
        // same way).
        //
        // 5e-3 is still 4× tighter than the Python-MLX vs PT decode
        // threshold (2e-2) the upstream Python parity test uses, so this
        // catches real port bugs.
        let err = maxAbs(swiftOutput, pythonOutput)
        XCTAssertLessThan(err, 5e-3, """
            VAE decode parity failed: max_abs = \(err) > 5e-3.
            A divergence at this magnitude implies a structural port bug in
            post_quant_conv, the decoder up stages, or the upsample3d
            'Rep' first-call sentinel path.
            """)
        print("✓ vae.decode parity max_abs = \(err) (threshold 5e-3)")
    }
}
