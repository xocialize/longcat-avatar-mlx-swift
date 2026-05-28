//
//  WhisperSmoke.swift
//
//  Shape + Codable smoke tests for the Whisper encoder port. No weights,
//  no PT parity (that lives in Parity/WhisperParity.swift once fixtures
//  land).
//
//  See VAEOpsSmoke for the `swift test` vs `xcodebuild test` caveat.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class WhisperSmoke: XCTestCase {

    // Tiny dims so shape tests run quickly without weights.
    private let tinyMelBins = 16
    private let tinyDim = 32
    private let tinyHeads = 4
    private let tinyFFN = 64
    private let tinyLayers = 2
    private let tinyMaxPos = 32

    private func tinyModel() -> WhisperEncoder {
        WhisperEncoder(
            dModel: tinyDim,
            numLayers: tinyLayers,
            numHeads: tinyHeads,
            ffnDim: tinyFFN,
            numMelBins: tinyMelBins,
            maxSourcePositions: tinyMaxPos
        )
    }

    // MARK: - WhisperAttention

    func testAttentionPreservesShape() {
        let attn = WhisperAttention(dModel: tinyDim, numHeads: tinyHeads)
        let x = MLXRandom.normal([2, 16, tinyDim])
        let y = attn(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testAttentionKProjHasNoBias() {
        let attn = WhisperAttention(dModel: tinyDim, numHeads: tinyHeads)
        // HF quirk: k_proj has no bias; q_proj/v_proj/out_proj do.
        XCTAssertNil(attn.kProj.bias)
        XCTAssertNotNil(attn.qProj.bias)
        XCTAssertNotNil(attn.vProj.bias)
        XCTAssertNotNil(attn.outProj.bias)
    }

    // MARK: - WhisperEncoderLayer

    func testEncoderLayerPreservesShape() {
        let layer = WhisperEncoderLayer(dModel: tinyDim, numHeads: tinyHeads, ffnDim: tinyFFN)
        let x = MLXRandom.normal([2, 16, tinyDim])
        let y = layer(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - WhisperEncoder

    func testEncoderConstructsAtTinyConfig() {
        let model = tinyModel()
        XCTAssertEqual(model.layers.count, tinyLayers)
        XCTAssertEqual(model.dModel, tinyDim)
        XCTAssertEqual(model.maxSourcePositions, tinyMaxPos)
    }

    func testEncoderForwardHalvesTimeViaConv2Stride2() {
        let model = tinyModel()
        // [B, mel, T_mel] = [1, 16, 16]
        let mel = MLXRandom.normal([1, tinyMelBins, 16])
        let y = model(mel)
        // After conv2 stride=2: T_enc = T_mel / 2 = 8
        XCTAssertEqual(y.shape, [1, 8, tinyDim])
    }

    func testEncoderAllHiddenStatesReturnsNumLayersPlusOne() {
        let model = tinyModel()
        let mel = MLXRandom.normal([1, tinyMelBins, 8])
        let allHidden = model.allHiddenStates(mel)
        // 1 (post-conv frontend, before any layer) + numLayers
        XCTAssertEqual(allHidden.count, tinyLayers + 1)
        // Every hidden state is [B, T_enc=4, dModel]
        for h in allHidden {
            XCTAssertEqual(h.shape, [1, 4, tinyDim])
        }
    }

    func testEncoderHonorsMaxSourcePositions() {
        // T_enc after conv2 stride=2 must be <= max_source_positions.
        // The positional embedding gets sliced to T_enc length.
        let model = tinyModel()
        let melLen = tinyMaxPos * 2  // → T_enc = tinyMaxPos (the max)
        let mel = MLXRandom.normal([1, tinyMelBins, melLen])
        let y = model(mel)
        XCTAssertEqual(y.dim(1), tinyMaxPos)
    }

    // MARK: - WhisperConfig Codable

    func testConfigDecodesMeituanShape() throws {
        // Exact shape of the published audio_encoder/config.json (subset of keys).
        let json = """
        {
          "d_model": 1280,
          "encoder_layers": 32,
          "encoder_attention_heads": 20,
          "encoder_ffn_dim": 5120,
          "num_mel_bins": 128,
          "max_source_positions": 1500
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(WhisperConfig.self, from: json)
        XCTAssertEqual(cfg.dModel, 1280)
        XCTAssertEqual(cfg.encoderLayers, 32)
        XCTAssertEqual(cfg.encoderAttentionHeads, 20)
        XCTAssertEqual(cfg.encoderFfnDim, 5120)
        XCTAssertEqual(cfg.numMelBins, 128)
        XCTAssertEqual(cfg.maxSourcePositions, 1500)
    }

    func testConfigFallsBackToDefaults() throws {
        let cfg = try JSONDecoder().decode(WhisperConfig.self, from: "{}".data(using: .utf8)!)
        XCTAssertEqual(cfg.dModel, 1280)
        XCTAssertEqual(cfg.encoderLayers, 32)
    }

    func testConfigIgnoresIrrelevantHFKeys() throws {
        // Real HF config has dozens of decoder-only / generation keys; the
        // encoder-only loader must not choke.
        let json = """
        {
          "d_model": 1280,
          "encoder_layers": 32,
          "decoder_layers": 32,
          "vocab_size": 51866,
          "max_target_positions": 448,
          "is_encoder_decoder": true
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(WhisperConfig.self, from: json)
        XCTAssertEqual(cfg.dModel, 1280)
        XCTAssertEqual(cfg.encoderLayers, 32)
    }

    // MARK: - fromConfig

    func testFromConfigMatchesExplicitInit() {
        var cfg = WhisperConfig()
        cfg.dModel = tinyDim
        cfg.encoderLayers = tinyLayers
        cfg.encoderAttentionHeads = tinyHeads
        cfg.encoderFfnDim = tinyFFN
        cfg.numMelBins = tinyMelBins
        cfg.maxSourcePositions = tinyMaxPos

        let model = WhisperEncoder.fromConfig(cfg)
        XCTAssertEqual(model.layers.count, tinyLayers)
        XCTAssertEqual(model.dModel, tinyDim)
    }
}
