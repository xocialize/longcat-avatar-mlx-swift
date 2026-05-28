//
//  VAECompositesSmoke.swift
//
//  Shape + construction smoke tests for the composite Wan VAE blocks
//  (WanResidualBlock, WanMidBlock, WanUpBlock), encoder/decoder, and
//  AutoencoderKLWan top-level. No weights, no PT parity — parity lives
//  in the parity test target and requires the downloaded HF weights.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class VAECompositesSmoke: XCTestCase {

    // See VAEOpsSmoke for the `swift test` vs `xcodebuild test` caveat.

    // MARK: - WanResidualBlock

    func testResidualBlockEqualDimsHasNoShortcut() {
        let block = WanResidualBlock(inDim: 16, outDim: 16)
        XCTAssertNil(block.convShortcut)
    }

    func testResidualBlockUnequalDimsHasShortcut() {
        let block = WanResidualBlock(inDim: 8, outDim: 16)
        XCTAssertNotNil(block.convShortcut)
    }

    func testResidualBlockForwardShape() {
        let block = WanResidualBlock(inDim: 8, outDim: 16)
        let x = MLXRandom.normal([1, 8, 2, 8, 8])
        let y = block(x)
        XCTAssertEqual(y.shape, [1, 16, 2, 8, 8])
    }

    // MARK: - WanMidBlock

    func testMidBlockHasOneMoreResnetThanAttention() {
        let mid = WanMidBlock(dim: 16, numLayers: 1)
        XCTAssertEqual(mid.resnets.count, 2)        // [resnet, attn, resnet]
        XCTAssertEqual(mid.attentions.count, 1)
    }

    func testMidBlockPreservesShape() {
        let mid = WanMidBlock(dim: 16, numLayers: 1)
        let x = MLXRandom.normal([1, 16, 2, 4, 4])
        let y = mid(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - WanUpBlock

    func testUpBlockHasNumResBlocksPlusOneResnets() {
        let up = WanUpBlock(inDim: 16, outDim: 8, numResBlocks: 2, upsampleMode: "upsample3d")
        // numResBlocks+1 resnets, plus optional upsamplers
        XCTAssertEqual(up.resnets.count, 3)
        XCTAssertNotNil(up.upsamplers)
        XCTAssertEqual(up.upsamplers?.count, 1)
    }

    func testUpBlockWithoutUpsamplers() {
        let up = WanUpBlock(inDim: 16, outDim: 8, numResBlocks: 2, upsampleMode: nil)
        XCTAssertNil(up.upsamplers)
    }

    // MARK: - WanEncoder3d / WanDecoder3d structure

    func testEncoderConstructsWithDefaults() {
        let enc = WanEncoder3d()
        // Default dim_mult = [1,2,4,4], default temperal_downsample = [false, true, true]:
        // - 3 down stages × 2 res blocks = 6 residual layers
        // - 3 downsample stages (between 4 dim levels: only 3 transitions)
        // - 0 attention layers (attn_scales=[])
        // total = 6 residuals + 3 resamples = 9
        var residuals = 0, attentions = 0, resamples = 0
        for layer in enc.downBlocks {
            if layer is WanResidualBlock { residuals += 1 }
            else if layer is WanAttentionBlock { attentions += 1 }
            else if layer is WanResample { resamples += 1 }
            else { XCTFail("unexpected down-block layer type: \(type(of: layer))") }
        }
        XCTAssertEqual(residuals, 8)   // 4 channel transitions × 2 res blocks
        XCTAssertEqual(attentions, 0)
        XCTAssertEqual(resamples, 3)   // 3 between-stage downsamples
    }

    func testDecoderConstructsWithDefaults() {
        let dec = WanDecoder3d()
        // Default dim_mult = [1,2,4,4] → 4 up-blocks (one per dim level)
        XCTAssertEqual(dec.upBlocks.count, 4)
        // First three carry an upsampler; the last does not (mults.count - 1 transitions)
        XCTAssertNotNil(dec.upBlocks[0].upsamplers)
        XCTAssertNotNil(dec.upBlocks[1].upsamplers)
        XCTAssertNotNil(dec.upBlocks[2].upsamplers)
        XCTAssertNil(dec.upBlocks[3].upsamplers)
    }

    // MARK: - AutoencoderKLWan top-level

    func testAutoencoderConstructsAtDefaultConfig() {
        let vae = AutoencoderKLWan()
        XCTAssertEqual(vae.zDim, 16)
        XCTAssertNotNil(vae.encoder)
        XCTAssertNotNil(vae.quantConv)
        XCTAssertEqual(vae.mean.shape, [16])
        XCTAssertEqual(vae.std.shape, [16])
    }

    func testAutoencoderDecoderOnlyOmitsEncoder() {
        let vae = AutoencoderKLWan(includeEncoder: false)
        XCTAssertNil(vae.encoder)
        XCTAssertNil(vae.quantConv)
        // Decoder side is always present
        XCTAssertNotNil(vae.postQuantConv)
    }

    func testNormalizeDenormalizeRoundTripsCloseToIdentity() {
        let vae = AutoencoderKLWan()
        let mu = MLXRandom.normal([1, 16, 2, 4, 4])
        let normalized = vae.normalizeLatents(mu)
        let recovered = vae.denormalizeLatents(normalized)
        let err = (recovered - mu).square().mean().item(Float.self)
        // Within fp32 rounding — should be effectively zero
        XCTAssertLessThan(err, 1e-5)
    }

    func testEncoderCacheSlotCountMatchesPython() {
        // Python autoencoder_kl_wan.py:_count_encoder_cache_slots at the
        // default config returns: 1 (conv_in) + 8*1 (4 stages × 2 residuals)
        //                       + 2 (last-stage residuals reuse same code)
        //                       + 3 (downsamples — but only downsample3d counts)
        // For default temperal_downsample = [false, true, true] →
        // 2 downsample3d slots (not 3).
        // Recount: residuals contribute 2 slots each → 16, conv_in=1,
        // downsample3d=2 (stages 1,2), mid_block resnets = 2 × 2 = 4,
        // conv_out=1. Total = 1 + 16 + 2 + 4 + 1 = 24.
        let vae = AutoencoderKLWan()
        let n = vae.countEncoderCacheSlots()
        XCTAssertEqual(n, 24, "encoder cache slot count drifted from Python — likely structural bug")
    }

    func testDecoderCacheSlotCountMatchesPython() {
        // Default temperal_upsample = reversed([false, true, true]) = [true, true, false]
        // conv_in=1, mid_block resnets = 2×2 = 4,
        // 4 up_blocks × (numResBlocks+1)=3 residuals × 2 slots/residual = 24,
        // 2 upsample3d slots (first two up blocks),
        // conv_out=1. Total = 1 + 4 + 24 + 2 + 1 = 32.
        let vae = AutoencoderKLWan()
        let n = vae.countDecoderCacheSlots()
        XCTAssertEqual(n, 32, "decoder cache slot count drifted from Python — likely structural bug")
    }

    // MARK: - WanVAEConfig Codable

    func testVAEConfigDecodesMeituanShape() throws {
        // Mirrors the actual published vae/config.json content.
        let json = """
        {
          "_class_name": "AutoencoderKLWan",
          "z_dim": 16,
          "base_dim": 96,
          "dim_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_scales": [],
          "temperal_downsample": [false, true, true],
          "latents_mean": \(DefaultVAEMean),
          "latents_std": \(DefaultVAEStd)
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(WanVAEConfig.self, from: json)
        XCTAssertEqual(config.zDim, 16)
        XCTAssertEqual(config.baseDim, 96)
        XCTAssertEqual(config.dimMult, [1, 2, 4, 4])
        XCTAssertEqual(config.temperalDownsample, [false, true, true])
        XCTAssertEqual(config.latentsMean?.count, 16)
    }
}
