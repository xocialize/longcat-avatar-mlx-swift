//
//  VAEOpsSmoke.swift
//
//  Shape + basic-forward smoke tests for the Wan VAE op primitives ported
//  in S3.3a (no weights, no PT parity yet — that lands with S3.3b alongside
//  the composite blocks and encoder/decoder).
//
//  Each test mirrors the assumptions the Python composite blocks rely on:
//  if these shapes are wrong, every downstream block forward will be too.
//

import Foundation
import XCTest
import MLX
@testable import LongCatVideoAvatar

final class VAEOpsSmoke: XCTestCase {

    // NOTE: `swift test` from the CLI does not bundle the Metal shader
    // library (`default.metallib`) into the test binary on macOS — MLX
    // operations crash on first dispatch with "Failed to load the default
    // metallib". Run via `xcodebuild test -scheme LongCatVideoAvatar-Package
    // -destination "platform=macOS"` instead (Xcode's build system handles
    // the resource compile pass that SwiftPM CLI skips). The README's
    // contributor section calls this out.

    // MARK: - CausalConv3d

    func testCausalConv3dPreservesShapeWithKernel3Padding1() {
        // Configuration used by ResidualBlock.conv1 / conv2 in the encoder.
        let conv = CausalConv3d(inputChannels: 4, outputChannels: 4, kernelSize: 3, padding: 1)
        let x = MLXRandom.normal([1, 4, 2, 8, 8])  // [B, C, T, H, W]
        let y = conv(x)
        XCTAssertEqual(y.shape, [1, 4, 2, 8, 8])
    }

    func testCausalConv3dChangesChannelsWithKernel1() {
        // conv_shortcut / quant_conv / post_quant_conv configuration.
        let conv = CausalConv3d(inputChannels: 8, outputChannels: 16, kernelSize: 1)
        let x = MLXRandom.normal([1, 8, 3, 4, 4])
        let y = conv(x)
        XCTAssertEqual(y.shape, [1, 16, 3, 4, 4])
    }

    func testCausalConv3dHonorsTemporalKernel() {
        // upsample3d / downsample3d time_conv configuration: (3,1,1) kernel.
        let conv = CausalConv3d(
            inputChannels: 6, outputChannels: 12,
            kernelSize: (3, 1, 1), stride: (1, 1, 1), padding: (1, 0, 0)
        )
        let x = MLXRandom.normal([1, 6, 4, 5, 5])
        let y = conv(x)
        // Causal pad k-stride=2 at front; padding doesn't contribute extra (we
        // pad temporally before conv). T_out = (T + causalPadT - kT)/stride + 1.
        XCTAssertEqual(y.dim(0), 1)
        XCTAssertEqual(y.dim(1), 12)   // out channels
        XCTAssertEqual(y.dim(3), 5)    // H unchanged
        XCTAssertEqual(y.dim(4), 5)    // W unchanged
        // T preserved by causal padding equal to (k-1).
        XCTAssertEqual(y.dim(2), 4)
    }

    func testCausalConv3dAcceptsExternalCache() {
        // cacheX shape: [B, C, CACHE_T, H, W]
        let conv = CausalConv3d(inputChannels: 4, outputChannels: 4, kernelSize: 3, padding: 1)
        let x = MLXRandom.normal([1, 4, 2, 4, 4])
        let cacheX = MLXRandom.normal([1, 4, WanVAECacheT, 4, 4])
        let y = conv(x, cacheX: cacheX)
        // The cache substitutes for the causal pad → output T == input T.
        XCTAssertEqual(y.shape, [1, 4, 2, 4, 4])
    }

    // MARK: - WanRMSNorm

    func testWanRMSNormShape3DChannelFirst() {
        // images=true: gamma has shape (dim, 1, 1) — used per-frame by attention norm
        let norm = WanRMSNorm(dim: 8, channelFirst: true, images: true)
        let x = MLXRandom.normal([2, 8, 16, 16])  // [BT, C, H, W]
        let y = norm(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testWanRMSNormShape4DChannelFirst() {
        // images=false: gamma has shape (dim, 1, 1, 1) — used by ResidualBlock
        // norms over [B, C, T, H, W]
        let norm = WanRMSNorm(dim: 4, channelFirst: true, images: false)
        let x = MLXRandom.normal([1, 4, 3, 8, 8])
        let y = norm(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    func testWanRMSNormPreservesMagnitude() {
        // With gamma=ones and scale=sqrt(dim), output magnitude per channel
        // should be roughly unit when input is unit-Gaussian.
        let norm = WanRMSNorm(dim: 8, channelFirst: true, images: true)
        let x = MLXRandom.normal([1, 8, 4, 4])
        let y = norm(x)
        let mag = y.square().mean().item(Float.self)
        // Should be ~1 (the scale factor of sqrt(dim) cancels the 1/sqrt(dim) divisor).
        XCTAssertGreaterThan(mag, 0.5)
        XCTAssertLessThan(mag, 2.0)
    }

    // MARK: - WanAttentionBlock

    func testWanAttentionBlockPreservesShape() {
        // Mid-block attention is over [B, C, T, H, W] → same shape out.
        let attn = WanAttentionBlock(dim: 16)
        let x = MLXRandom.normal([1, 16, 2, 8, 8])
        let y = attn(x)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - WanResample

    func testWanResampleUpsample2dShape() {
        // upsample2d halves channels, doubles H + W, keeps T.
        let up = WanResample(dim: 8, mode: "upsample2d")
        let x = MLXRandom.normal([1, 8, 2, 4, 4])  // [B, 8, T, 4, 4]
        let y = up(x)
        XCTAssertEqual(y.shape, [1, 4, 2, 8, 8])    // C/2, 2H, 2W
    }

    func testWanResampleDownsample2dShape() {
        // downsample2d keeps channels, halves H + W (via stride-2 conv).
        let down = WanResample(dim: 8, mode: "downsample2d")
        let x = MLXRandom.normal([1, 8, 2, 8, 8])
        let y = down(x)
        XCTAssertEqual(y.dim(0), 1)
        XCTAssertEqual(y.dim(1), 8)
        XCTAssertEqual(y.dim(2), 2)
        XCTAssertEqual(y.dim(3), 4)
        XCTAssertEqual(y.dim(4), 4)
    }

    func testWanResampleUpsample3dFirstCallSentinel() {
        // First call with empty feat_cache plants the "Rep" sentinel and
        // skips time_conv → output T unchanged from spatial-only upsample.
        let up = WanResample(dim: 8, mode: "upsample3d")
        let cache = WanFeatCacheRef(slotCount: 1)
        let idx = WanFeatIdxRef()
        let x = MLXRandom.normal([1, 8, 1, 4, 4])
        let y = up(x, featCache: cache, featIdx: idx)
        // Channels halve (spatial conv), H+W double, T unchanged on first call.
        XCTAssertEqual(y.shape, [1, 4, 1, 8, 8])
        // Sentinel planted, idx advanced.
        XCTAssertEqual(idx.value, 1)
        if case .rep = cache.slot(at: 0) { /* ok */ } else {
            XCTFail("Expected .rep sentinel after first call")
        }
    }

    func testWanResampleUpsample3dSecondCallDoublesT() {
        // After first-call sentinel, second call runs time_conv and doubles T.
        let up = WanResample(dim: 8, mode: "upsample3d")
        let cache = WanFeatCacheRef(slotCount: 1)
        let idx1 = WanFeatIdxRef()
        let x1 = MLXRandom.normal([1, 8, 1, 4, 4])
        _ = up(x1, featCache: cache, featIdx: idx1)
        // Now slot 0 is .rep — second call goes through the time_conv path.
        let idx2 = WanFeatIdxRef()
        let x2 = MLXRandom.normal([1, 8, 1, 4, 4])
        let y = up(x2, featCache: cache, featIdx: idx2)
        // T should double (1 → 2) plus spatial upsample.
        XCTAssertEqual(y.dim(2), 2)
        XCTAssertEqual(y.dim(3), 8)
        XCTAssertEqual(y.dim(4), 8)
    }

    // MARK: - feat_cache state helpers

    func testFeatCacheRefStartsEmpty() {
        let cache = WanFeatCacheRef(slotCount: 5)
        for i in 0..<5 {
            if case .empty = cache.slot(at: i) { /* ok */ } else {
                XCTFail("Slot \(i) should be .empty initially")
            }
        }
    }

    func testFeatIdxRefAdvancesAndResets() {
        let idx = WanFeatIdxRef()
        XCTAssertEqual(idx.value, 0)
        idx.advance(); idx.advance(); idx.advance()
        XCTAssertEqual(idx.value, 3)
        idx.reset()
        XCTAssertEqual(idx.value, 0)
    }
}
