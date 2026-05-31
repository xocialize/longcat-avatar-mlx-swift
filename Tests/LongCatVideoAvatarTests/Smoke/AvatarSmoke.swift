//
//  AvatarSmoke.swift
//
//  Shape + Codable smoke tests for the Avatar overlay port — AudioProjModel,
//  AvatarAttention (Reference Skip), SingleStreamAttention (audio cross-attn),
//  LongCatAvatarSingleStreamBlock, LongCatVideoAvatarTransformer3DModel.
//  No weights, no parity (that's in AvatarParity.swift).
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class AvatarSmoke: XCTestCase {

    // Tiny config — keep tests fast.
    private let tinyHidden = 32
    private let tinyHeads = 4
    private let tinyAdaln = 16
    private let tinyOutputDim = 16
    private let tinyAudioWindow = 5
    private let tinyAudioBlocks = 5
    private let tinyAudioChannels = 8
    private let tinyIntermediate = 16
    private let tinyContextTokens = 4
    private let tinyVaeScale = 4

    // MARK: - AudioProjModel

    func testAudioProjModelProducesCorrectShape() {
        let m = AudioProjModel(
            seqLen: tinyAudioWindow,
            seqLenVF: tinyAudioWindow + tinyVaeScale - 1,
            blocksCount: tinyAudioBlocks,
            channels: tinyAudioChannels,
            intermediateDim: tinyIntermediate,
            outputDim: tinyOutputDim,
            contextTokens: tinyContextTokens
        )
        // First-frame input: [B, 1, W, S, C]
        let af = MLXRandom.normal([1, 1, tinyAudioWindow, tinyAudioBlocks, tinyAudioChannels])
        // Latter-frame input: [B, T-1, W', S, C]
        let avf = MLXRandom.normal([1, 2, tinyAudioWindow + tinyVaeScale - 1, tinyAudioBlocks, tinyAudioChannels])
        let out = m(audioEmbeds: af, audioEmbedsVF: avf)
        XCTAssertEqual(out.shape, [1, 3, tinyContextTokens, tinyOutputDim])
    }

    // MARK: - AvatarAttention

    func testAvatarAttentionStandardPathPreservesShape() {
        let attn = AvatarAttention(dim: tinyHidden, numHeads: tinyHeads)
        let B = 1, T = 2, H = 2, W = 2, N = T * H * W
        let x = MLXRandom.normal([B, N, tinyHidden])
        let (out, kv, refMap) = attn(x, shape: (T, H, W))
        XCTAssertEqual(out.shape, x.shape)
        XCTAssertNil(kv)
        XCTAssertNil(refMap)
    }

    func testAvatarAttentionReturnsKVCache() {
        let attn = AvatarAttention(dim: tinyHidden, numHeads: tinyHeads)
        let x = MLXRandom.normal([1, 8, tinyHidden])
        let (_, kv, _) = attn(x, shape: (2, 2, 2), returnKV: true)
        XCTAssertNotNil(kv)
        XCTAssertEqual(kv!.0.shape, [1, tinyHeads, 8, 8])
    }

    func testAvatarAttentionWithCondLatents() {
        let attn = AvatarAttention(dim: tinyHidden, numHeads: tinyHeads)
        // T=4 frames, 1 cond latent at the head
        let B = 1, T = 4, H = 2, W = 2, N = T * H * W
        let x = MLXRandom.normal([B, N, tinyHidden])
        let (out, _, _) = attn(x, shape: (T, H, W), numCondLatents: 1)
        XCTAssertEqual(out.shape, x.shape)
    }

    // MARK: - SingleStreamAttention

    func testSingleStreamAttentionPreservesShape() {
        let attn = SingleStreamAttention(
            dim: tinyHidden,
            encoderHiddenStatesDim: tinyOutputDim,
            numHeads: tinyHeads
        )
        let B = 1, T = 2, H = 2, W = 2, N = T * H * W
        let x = MLXRandom.normal([B, N, tinyHidden])
        let cond = MLXRandom.normal([B * T, tinyContextTokens, tinyOutputDim])
        let (audioOutCond, audioOutNoise) = attn(x, cond: cond, shape: (T, H, W))
        XCTAssertNil(audioOutCond)
        XCTAssertEqual(audioOutNoise.shape, x.shape)
    }

    func testSingleStreamAttentionWithCondLatents() {
        let attn = SingleStreamAttention(
            dim: tinyHidden,
            encoderHiddenStatesDim: tinyOutputDim,
            numHeads: tinyHeads
        )
        // T=3 frames, ncl=1 cond at head
        let B = 1, T = 3, H = 2, W = 2, N = T * H * W
        let nCondTokens = (N / T) * 1
        let x = MLXRandom.normal([B, N, tinyHidden])
        let cond = MLXRandom.normal([B * T, tinyContextTokens, tinyOutputDim])
        let (audioOutCond, audioOutNoise) = attn(x, cond: cond, shape: (T, H, W), numCondLatents: 1)
        XCTAssertNotNil(audioOutCond)
        XCTAssertEqual(audioOutCond?.shape, [B, nCondTokens, tinyHidden])
        XCTAssertEqual(audioOutNoise.shape, [B, N - nCondTokens, tinyHidden])
    }

    // MARK: - LongCatAvatarSingleStreamBlock

    func testAvatarBlockPreservesShape() {
        let block = LongCatAvatarSingleStreamBlock(
            hiddenSize: tinyHidden,
            numHeads: tinyHeads,
            mlpRatio: 2,
            adalnTembedDim: tinyAdaln,
            outputDim: tinyOutputDim
        )
        let B = 1, T = 2, H = 2, W = 2, N = T * H * W
        let x = MLXRandom.normal([B, N, tinyHidden])
        let y = MLXRandom.normal([1, 4, tinyHidden])
        let t = MLXRandom.normal([B, T, tinyAdaln]).asType(.float32)
        let audioHidden = MLXRandom.normal([B * T, tinyContextTokens, tinyOutputDim])
        let out = block(
            x, y: y, t: t, ySeqlen: [4],
            latentShape: (T, H, W),
            audioHiddenStates: audioHidden
        )
        XCTAssertEqual(out.shape, x.shape)
    }

    // MARK: - LongCatVideoAvatarConfig

    func testAvatarConfigDecodesMeituanShape() throws {
        // Mirrors the actual published dit/config.json (subset).
        let json = """
        {
          "_class_name": "LongCatVideoAvatarTransformer3DModel",
          "hidden_size": 4096,
          "depth": 48,
          "num_heads": 32,
          "patch_size": [1, 2, 2],
          "audio_window": 5,
          "audio_block": 5,
          "audio_channel": 1280,
          "intermediate_dim": 512,
          "output_dim": 768,
          "context_tokens": 32,
          "vae_scale": 4,
          "audio_prenorm": false,
          "class_range": 24,
          "class_interval": 4,
          "text_tokens_zero_pad": true
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(LongCatVideoAvatarConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 4096)
        XCTAssertEqual(cfg.depth, 48)
        XCTAssertEqual(cfg.audioWindow, 5)
        XCTAssertEqual(cfg.audioBlock, 5)
        XCTAssertEqual(cfg.audioChannel, 1280)
        XCTAssertEqual(cfg.outputDim, 768)
        XCTAssertEqual(cfg.contextTokens, 32)
        XCTAssertEqual(cfg.vaeScale, 4)
        XCTAssertFalse(cfg.audioPrenorm)
        XCTAssertEqual(cfg.classRange, 24)
        XCTAssertEqual(cfg.classInterval, 4)
        XCTAssertTrue(cfg.textTokensZeroPad)
    }

    // MARK: - Full tiny Avatar DiT

    func testAvatarDiTConstructsAndForwards() {
        var cfg = LongCatVideoAvatarConfig()
        cfg.inChannels = 4
        cfg.outChannels = 4
        cfg.hiddenSize = tinyHidden
        cfg.depth = 2
        cfg.numHeads = tinyHeads
        cfg.captionChannels = tinyHidden
        cfg.mlpRatio = 2
        cfg.adalnTembedDim = tinyAdaln
        cfg.frequencyEmbeddingSize = 32
        cfg.patchSize = [1, 2, 2]
        cfg.audioWindow = tinyAudioWindow
        cfg.audioBlock = tinyAudioBlocks
        cfg.audioChannel = tinyAudioChannels
        cfg.intermediateDim = tinyIntermediate
        cfg.outputDim = tinyOutputDim
        cfg.contextTokens = tinyContextTokens
        cfg.vaeScale = tinyVaeScale

        let model = LongCatVideoAvatarTransformer3DModel.fromConfig(cfg)
        XCTAssertEqual(model.blocks.count, 2)

        // Tiny input: B=1, C=4, T=3, H=4, W=4 (after patchify: 3 latent frames, 8 spatial tokens each = 24 visual tokens)
        let B = 1
        let T = 3
        let H = 4, W = 4
        let nText = 8
        let hiddenStates = MLXRandom.normal([B, 4, T, H, W])
        let timestep = MLXArray([Float(500)])
        let encoderHiddenStates = MLXRandom.normal([B, 1, nText, tinyHidden])
        let mask = MLXArray.ones([B, nText], dtype: .int32)
        // audio_embs shape: [B, T_audio, W=5, S=5, C=audioChannels]
        // T_audio = 1 (first frame) + (T-1) * vae_scale = 1 + 2 * 4 = 9
        let audioEmbs = MLXRandom.normal([B, 9, tinyAudioWindow, tinyAudioBlocks, tinyAudioChannels])

        let out = model(
            hiddenStates: hiddenStates,
            timestep: timestep,
            encoderHiddenStates: encoderHiddenStates,
            audioEmbs: audioEmbs,
            encoderAttentionMask: mask,
            numCondLatents: 0
        )
        XCTAssertEqual(out.shape, [B, 4, T, H, W])
    }
}
