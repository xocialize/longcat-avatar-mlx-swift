//
//  DiTSmoke.swift
//
//  Shape / Codable / construction smoke tests for the LongCat-Video DiT
//  port (RoPE3D + Blocks + Attention + LongCatVideoDiT). No weights, no
//  PT parity (that lives in Parity/LongCatVideoDiTParity.swift).
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class DiTSmoke: XCTestCase {

    // See VAEOpsSmoke for the `swift test` vs `xcodebuild test` caveat.

    // Tiny dims so shape tests run quickly without weights.
    private let tinyHidden = 32
    private let tinyHeads = 4
    private let tinyAdaln = 16
    private let tinyCaption = 32

    // MARK: - rotateHalf

    func testRotateHalfPreservesShape() {
        let x = MLXRandom.normal([2, 4, 8, 16])
        let y = rotateHalf(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testRotateHalfMatchesExpected() {
        // For input [1, 2, 3, 4] (last dim), rotateHalf produces [-2, 1, -4, 3]
        let x = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 4)
        let y = rotateHalf(x)
        let expected: [Float] = [-2, 1, -4, 3]
        let actual = y[0].asArray(Float.self)
        XCTAssertEqual(actual, expected)
    }

    // MARK: - RotaryPositionalEmbedding

    func testRoPE3DProducesCorrectShape() {
        let headDim = 16
        let rope = RotaryPositionalEmbedding(headDim: headDim)
        let B = 1, H = 2, S = 8
        let q = MLXRandom.normal([B, H, S, headDim])
        let k = MLXRandom.normal([B, H, S, headDim])
        let (qR, kR) = rope(q: q, k: k, gridSize: (2, 2, 2))
        XCTAssertEqual(qR.shape, q.shape)
        XCTAssertEqual(kR.shape, k.shape)
    }

    func testRoPE3DCachesFreqsPerGridSize() {
        let rope = RotaryPositionalEmbedding(headDim: 16)
        let q1 = MLXRandom.normal([1, 2, 8, 16])
        let k1 = MLXRandom.normal([1, 2, 8, 16])
        _ = rope(q: q1, k: k1, gridSize: (2, 2, 2))
        _ = rope(q: q1, k: k1, gridSize: (2, 2, 2))
        // No assert — just verifying no crash from cache hit.
        XCTAssertTrue(true)
    }

    // MARK: - RMSNormFP32

    func testRMSNormFP32PreservesShape() {
        let norm = RMSNormFP32(dim: tinyHidden, eps: 1e-6)
        let x = MLXRandom.normal([2, 8, tinyHidden])
        let y = norm(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - LayerNormFP32

    func testLayerNormFP32WithAffinePreservesShape() {
        let norm = LayerNormFP32(dim: tinyHidden, eps: 1e-6, elementwiseAffine: true)
        let x = MLXRandom.normal([2, 8, tinyHidden])
        let y = norm(x)
        XCTAssertEqual(y.shape, x.shape)
        XCTAssertNotNil(norm.weight)
        XCTAssertNotNil(norm.bias)
    }

    func testLayerNormFP32WithoutAffineHasNoParams() {
        let norm = LayerNormFP32(dim: tinyHidden, eps: 1e-6, elementwiseAffine: false)
        XCTAssertNil(norm.weight)
        XCTAssertNil(norm.bias)
    }

    // MARK: - FeedForwardSwiGLU

    func testFFNSwiGLUInnerDimUsesMultipleOf256Rounding() {
        // For dim=4096, hiddenDim=16384 (production): inner = int(2/3 * 16384) = 10922,
        // rounded up to multiple of 256 = 11008.
        let ffn = FeedForwardSwiGLU(dim: 4096, hiddenDim: 16384, multipleOf: 256)
        XCTAssertEqual(ffn.hiddenDim, 11008)
    }

    func testFFNSwiGLUPreservesShape() {
        let ffn = FeedForwardSwiGLU(dim: tinyHidden, hiddenDim: tinyHidden * 4, multipleOf: 8)
        let x = MLXRandom.normal([2, 8, tinyHidden])
        let y = ffn(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - PatchEmbed3D

    func testPatchEmbed3DProducesCorrectShape() {
        let pe = PatchEmbed3D(patchSize: (1, 2, 2), inChans: 4, embedDim: tinyHidden, flatten: true)
        // [B, C, T, H, W] = [1, 4, 2, 4, 4]
        let x = MLXRandom.normal([1, 4, 2, 4, 4])
        let y = pe(x)
        // Output N = T*H/2*W/2 = 2*2*2 = 8
        XCTAssertEqual(y.shape, [1, 8, tinyHidden])
    }

    // MARK: - TimestepEmbedder

    func testTimestepEmbedderForwardShape() {
        let emb = TimestepEmbedder(tEmbedDim: tinyAdaln, frequencyEmbeddingSize: 32)
        let t = MLXArray([Float(0), 100, 500, 999])
        let out = emb(t, dtype: .float32)
        XCTAssertEqual(out.shape, [4, tinyAdaln])
    }

    // MARK: - CaptionEmbedder

    func testCaptionEmbedderForwardShape() {
        let ce = CaptionEmbedder(inChannels: tinyCaption, hiddenSize: tinyHidden)
        let x = MLXRandom.normal([2, 1, 8, tinyCaption])
        let y = ce(x)
        XCTAssertEqual(y.shape, [2, 1, 8, tinyHidden])
    }

    // MARK: - FinalLayerFP32

    func testFinalLayerFP32Shape() {
        let numPatch = 4
        let outChannels = 16
        let fl = FinalLayerFP32(
            hiddenSize: tinyHidden, numPatch: numPatch,
            outChannels: outChannels, adalnTembedDim: tinyAdaln
        )
        let x = MLXRandom.normal([1, 8, tinyHidden])
        let t = MLXRandom.normal([1, 1, tinyAdaln]).asType(.float32)
        let y = fl(x, t: t, latentShape: (1, 2, 4))
        XCTAssertEqual(y.shape, [1, 8, numPatch * outChannels])
    }

    // MARK: - Attention

    func testAttentionPreservesShape() {
        // dim=hidden, num_heads=4 → head_dim=8 (multiple of 8 for RoPE3D constraint)
        let attn = Attention(dim: tinyHidden, numHeads: tinyHeads)
        // shape = (T=2, H=2, W=2) → N = 8
        let x = MLXRandom.normal([1, 8, tinyHidden])
        let (y, kv) = attn(x, shape: (2, 2, 2), numCondLatents: nil, returnKV: false)
        XCTAssertEqual(y.shape, x.shape)
        XCTAssertNil(kv)
    }

    func testAttentionReturnsKVCache() {
        let attn = Attention(dim: tinyHidden, numHeads: tinyHeads)
        let x = MLXRandom.normal([1, 8, tinyHidden])
        let (_, kv) = attn(x, shape: (2, 2, 2), numCondLatents: nil, returnKV: true)
        XCTAssertNotNil(kv)
        XCTAssertEqual(kv!.0.shape, [1, tinyHeads, 8, 8])    // [B, H, N, D]
    }

    // MARK: - MultiHeadCrossAttention

    func testCrossAttentionPreservesShape() {
        let attn = MultiHeadCrossAttention(dim: tinyHidden, numHeads: tinyHeads)
        let B = 2, N = 8
        let textPerBatch = [4, 6]
        let x = MLXRandom.normal([B, N, tinyHidden])
        let cond = MLXRandom.normal([1, textPerBatch.reduce(0, +), tinyHidden])
        let y = attn(x, cond: cond, kvSeqlen: textPerBatch)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - LongCatSingleStreamBlock

    func testSingleStreamBlockPreservesShape() {
        let block = LongCatSingleStreamBlock(
            hiddenSize: tinyHidden,
            numHeads: tinyHeads,
            mlpRatio: 2,
            adalnTembedDim: tinyAdaln
        )
        let B = 1, N = 8
        let x = MLXRandom.normal([B, N, tinyHidden])
        let y = MLXRandom.normal([1, 4, tinyHidden])
        let t = MLXRandom.normal([B, 2, tinyAdaln]).asType(.float32)
        let (out, _) = block(
            x, y: y, t: t, ySeqlen: [4],
            latentShape: (2, 2, 2),
            numCondLatents: nil
        )
        XCTAssertEqual(out.shape, x.shape)
    }

    // MARK: - LongCatVideoTransformer3DModel

    func testFullDiTConstructsAndForwards() {
        // tiny config so we can actually forward
        let cfg = LongCatVideoConfig()
        var c = cfg
        c.inChannels = 4
        c.outChannels = 4
        c.hiddenSize = tinyHidden
        c.depth = 2
        c.numHeads = tinyHeads
        c.captionChannels = tinyCaption
        c.mlpRatio = 2
        c.adalnTembedDim = tinyAdaln
        c.frequencyEmbeddingSize = 32
        c.patchSize = [1, 2, 2]

        let model = LongCatVideoTransformer3DModel.fromConfig(c)
        XCTAssertEqual(model.blocks.count, 2)
        XCTAssertEqual(model.inChannels, 4)
        XCTAssertEqual(model.outChannels, 4)

        let B = 1, T = 2, H = 4, W = 4
        let nText = 8
        let hiddenStates = MLXRandom.normal([B, 4, T, H, W])
        let timestep = MLXArray([Float(500)])
        let encoderHiddenStates = MLXRandom.normal([B, 1, nText, tinyCaption])
        let encoderAttentionMask = MLXArray.ones([B, nText], dtype: .int32)

        let out = model(
            hiddenStates: hiddenStates,
            timestep: timestep,
            encoderHiddenStates: encoderHiddenStates,
            encoderAttentionMask: encoderAttentionMask
        )
        // patch_size = [1,2,2] → spatial halves, so output is [B, C_out, T, H, W]
        XCTAssertEqual(out.shape, [B, 4, T, H, W])
    }

    // MARK: - Config

    func testConfigDecodesMeituanShape() throws {
        // The actual published dit/config.json (subset of keys).
        let json = """
        {
          "_class_name": "LongCatVideoAvatarTransformer3DModel",
          "in_channels": 16,
          "out_channels": 16,
          "hidden_size": 4096,
          "depth": 48,
          "num_heads": 32,
          "caption_channels": 4096,
          "mlp_ratio": 4,
          "adaln_tembed_dim": 512,
          "frequency_embedding_size": 256,
          "patch_size": [1, 2, 2],
          "text_tokens_zero_pad": true
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(LongCatVideoConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 4096)
        XCTAssertEqual(cfg.depth, 48)
        XCTAssertEqual(cfg.numHeads, 32)
        XCTAssertEqual(cfg.patchSize, [1, 2, 2])
        XCTAssertEqual(cfg.textTokensZeroPad, true)
        XCTAssertNil(cfg.quantization)
    }

    func testConfigDecodesQuantizationBlock() throws {
        let json = """
        {
          "hidden_size": 4096,
          "quantization": {
            "method": "mlx.nn.quantize",
            "bits": 4,
            "group_size": 64,
            "skip_patterns": ["final_layer.linear"]
          }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(LongCatVideoConfig.self, from: json)
        XCTAssertNotNil(cfg.quantization)
        XCTAssertEqual(cfg.quantization?.bits, 4)
    }
}
