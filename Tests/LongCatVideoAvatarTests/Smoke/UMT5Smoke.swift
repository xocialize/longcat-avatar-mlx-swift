//
//  UMT5Smoke.swift
//
//  Shape + Codable smoke tests for the UMT5 port. No weights, no PT
//  parity (that lives in Parity/UMT5Parity.swift once fixtures land).
//
//  See VAEOpsSmoke for the `swift test` vs `xcodebuild test` caveat.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class UMT5Smoke: XCTestCase {

    // Tiny dims so shape tests run quickly without weights.
    private let tinyVocab = 256
    private let tinyDim = 64
    private let tinyHeads = 4
    private let tinyDimAttn = 64
    private let tinyDimFFN = 128
    private let tinyLayers = 2

    private func tinyModel() -> UMT5EncoderModel {
        UMT5EncoderModel(
            vocabSize: tinyVocab,
            dim: tinyDim,
            dimAttn: tinyDimAttn,
            dimFFN: tinyDimFFN,
            numHeads: tinyHeads,
            numLayers: tinyLayers,
            numBuckets: 16,
            sharedPos: false
        )
    }

    // MARK: - T5LayerNorm

    func testT5LayerNormPreservesShape() {
        let norm = T5LayerNorm(dim: tinyDim)
        let x = MLXRandom.normal([2, 8, tinyDim])
        let y = norm(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - T5RelativeEmbedding

    func testRelativeEmbeddingShape() {
        let emb = T5RelativeEmbedding(numBuckets: 16, numHeads: tinyHeads, bidirectional: true)
        let bias = emb(lq: 8, lk: 8)
        XCTAssertEqual(bias.shape, [1, tinyHeads, 8, 8])
    }

    func testRelativeEmbeddingAsymmetricSequenceShape() {
        let emb = T5RelativeEmbedding(numBuckets: 16, numHeads: tinyHeads, bidirectional: true)
        let bias = emb(lq: 4, lk: 12)
        XCTAssertEqual(bias.shape, [1, tinyHeads, 4, 12])
    }

    // MARK: - T5Attention

    func testT5AttentionPreservesShape() {
        let attn = T5Attention(dim: tinyDim, dimAttn: tinyDimAttn, numHeads: tinyHeads)
        let x = MLXRandom.normal([2, 8, tinyDim])
        let y = attn(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testT5AttentionWithPosBias() {
        let attn = T5Attention(dim: tinyDim, dimAttn: tinyDimAttn, numHeads: tinyHeads)
        let x = MLXRandom.normal([2, 8, tinyDim])
        let bias = MLXRandom.normal([1, tinyHeads, 8, 8])
        let y = attn(x, posBias: bias)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testT5AttentionWith2DMask() {
        let attn = T5Attention(dim: tinyDim, dimAttn: tinyDimAttn, numHeads: tinyHeads)
        let x = MLXRandom.normal([2, 8, tinyDim])
        let mask = MLXArray.ones([2, 8])  // [B, L]
        let y = attn(x, mask: mask)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - T5FeedForward

    func testFeedForwardShape() {
        let ffn = T5FeedForward(dim: tinyDim, dimFFN: tinyDimFFN)
        let x = MLXRandom.normal([2, 8, tinyDim])
        let y = ffn(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - T5SelfAttentionBlock

    func testBlockSharedPosFalseHasOwnEmbedding() {
        let block = T5SelfAttentionBlock(
            dim: tinyDim, dimAttn: tinyDimAttn, dimFFN: tinyDimFFN,
            numHeads: tinyHeads, numBuckets: 16, sharedPos: false
        )
        XCTAssertNotNil(block.posEmbedding)
    }

    func testBlockSharedPosTrueHasNoEmbedding() {
        let block = T5SelfAttentionBlock(
            dim: tinyDim, dimAttn: tinyDimAttn, dimFFN: tinyDimFFN,
            numHeads: tinyHeads, numBuckets: 16, sharedPos: true
        )
        XCTAssertNil(block.posEmbedding)
    }

    func testBlockForwardPreservesShape() {
        let block = T5SelfAttentionBlock(
            dim: tinyDim, dimAttn: tinyDimAttn, dimFFN: tinyDimFFN,
            numHeads: tinyHeads, numBuckets: 16, sharedPos: false
        )
        let x = MLXRandom.normal([2, 8, tinyDim])
        let y = block(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - UMT5EncoderModel

    func testEncoderConstructsAtTinyConfig() {
        let model = tinyModel()
        XCTAssertEqual(model.blocks.count, tinyLayers)
        XCTAssertEqual(model.dim, tinyDim)
        XCTAssertFalse(model.sharedPos)
        XCTAssertNil(model.posEmbedding)
    }

    func testEncoderForwardShape() {
        let model = tinyModel()
        let ids = MLXArray.zeros([2, 8], dtype: .int32)
        let out = model(ids)
        XCTAssertEqual(out.shape, [2, 8, tinyDim])
    }

    func testEncoderForwardWithMask() {
        let model = tinyModel()
        let ids = MLXArray.zeros([2, 8], dtype: .int32)
        let mask = MLXArray.ones([2, 8], dtype: .int32)
        let out = model(ids, mask: mask)
        XCTAssertEqual(out.shape, [2, 8, tinyDim])
    }

    // MARK: - UMT5Config Codable

    func testConfigDecodesMeituanShape() throws {
        // Mirrors the actual published text_encoder/config.json.
        let json = """
        {
          "_class_name": "UMT5EncoderModel",
          "vocab_size": 256384,
          "d_model": 4096,
          "d_ff": 10240,
          "d_kv": 64,
          "num_heads": 64,
          "num_layers": 24,
          "relative_attention_num_buckets": 32
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(UMT5Config.self, from: json)
        XCTAssertEqual(cfg.vocabSize, 256384)
        XCTAssertEqual(cfg.dim, 4096)
        XCTAssertEqual(cfg.dimFFN, 10240)
        XCTAssertEqual(cfg.numHeads, 64)
        XCTAssertEqual(cfg.dKV, 64)
        XCTAssertEqual(cfg.numLayers, 24)
        XCTAssertEqual(cfg.numBuckets, 32)
    }

    func testConfigFallsBackToDefaults() throws {
        // Empty JSON falls back to UMT5 defaults.
        let json = "{}".data(using: .utf8)!
        let cfg = try JSONDecoder().decode(UMT5Config.self, from: json)
        XCTAssertEqual(cfg.vocabSize, 256384)
        XCTAssertEqual(cfg.dim, 4096)
    }

    // MARK: - fromConfig

    func testFromConfigMatchesExplicitInit() {
        var cfg = UMT5Config()
        cfg.numLayers = tinyLayers
        cfg.dim = tinyDim
        cfg.numHeads = tinyHeads
        cfg.dKV = tinyDimAttn / tinyHeads
        cfg.dimFFN = tinyDimFFN
        cfg.vocabSize = tinyVocab
        cfg.numBuckets = 16

        let model = UMT5EncoderModel.fromConfig(cfg)
        XCTAssertEqual(model.blocks.count, tinyLayers)
        XCTAssertEqual(model.dim, tinyDim)
        XCTAssertFalse(model.sharedPos)  // umT5 always per-block
    }
}
