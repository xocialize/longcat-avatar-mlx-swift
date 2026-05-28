//
//  AutoencoderKLWan.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/autoencoder_kl_wan.py
//
//  Wan 2.1 VAE — diffusers 0.38 canonical schema (`is_residual=False`
//  variant, as shipped by Meituan). The Python port hit PT-vs-MLX parity
//  of 7e-6 (encode) / 1.2e-2 (decode); we target the same.
//
//  Architectural notes (all carried over from the Python port — do NOT
//  refactor away from these without porting the corresponding skill
//  lesson):
//
//  - **L1** — diffusers Wan 2.1 has a DIFFERENT channel pattern than the
//    older `Blaizzy/mlx-video` Wan port. We mirror diffusers 0.38 here,
//    not mlx-video.
//
//  - **L10** — MLX's Metal GPU loses ~3-4 bits of fp32 precision on the
//    long QK^T / softmax-V matmuls. The VAE has only two AttentionBlocks
//    (one per WanMidBlock). We route their SDPA chain through `.cpu`
//    stream to recover strict fp32; perf cost is negligible.
//
//  - **L8** — `WanResample.upsample3d` uses a "Rep" string sentinel in
//    `feat_cache` to make the FIRST call skip `time_conv` entirely. This
//    is what makes the first decoded latent produce 1 video frame and
//    subsequent latents produce 2. Port verbatim — re-deriving this
//    branch from the diffusers 0.38 source is non-trivial and dropping
//    it produces noise output.
//
//  STAGE STATUS:
//  - S3.3a (this PR): op primitives (CausalConv3d, WanRMSNorm,
//    WanAttentionBlock, WanResample). Shape smoke tests.
//  - S3.3b (next PR): composite blocks + encoder/decoder + AutoencoderKLWan
//    + parity test against the Python port's reference outputs.
//

import Foundation
import MLX
import MLXNN

// MARK: - Constants

/// Number of trailing temporal frames each `CausalConv3d` retains in
/// `feat_cache` to provide causal context to the next chunk.
/// Matches Python `CACHE_T`.
public let WanVAECacheT: Int = 2

/// Default per-channel normalization stats for `z_dim=16` (Wan 2.1).
/// Overridden when present in `vae/config.json`.
public let DefaultVAEMean: [Float] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
public let DefaultVAEStd: [Float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

// MARK: - feat_cache sentinel

/// `WanResample.upsample3d` uses a one-shot sentinel in `feat_cache` to
/// signal "first call, skip time_conv". Python encodes this as the string
/// `"Rep"`; Swift uses an explicit enum for type safety.
///
/// (Not `Sendable` because `MLXArray` isn't — and we don't cross actor
/// boundaries with these. The Python `feat_cache` is also single-threaded.)
public enum WanFeatCacheSlot {
    case empty                    // nil in Python — no cache yet
    case rep                      // "Rep" in Python — first call sentinel
    case tensor(MLXArray)         // a real cached tensor
}

// MARK: - CausalConv3d

/// 3D convolution with causal temporal padding. Matches diffusers'
/// `WanCausalConv3d`.
///
/// Unlike the Python port (which had to emulate Conv3d via per-frame
/// Conv2d because Python MLX historically lacked Conv3d), the Swift port
/// uses `MLXNN.Conv3d` directly. Weight layout `(O, kT, kH, kW, I)` is
/// identical in both runtimes — weights load with no transpose.
///
/// Layout convention: this layer accepts/returns **PyTorch-style**
/// `[B, C, T, H, W]` for consistency with the Python port's call sites,
/// then transposes to/from MLX's preferred `[B, T, H, W, C]` internally.
public final class CausalConv3d: Module, @unchecked Sendable {
    public let kernelSize: (Int, Int, Int)
    public let stride: (Int, Int, Int)
    public let padH: Int
    public let padW: Int
    public let causalPadT: Int

    // Weights: `(O, kT, kH, kW, I)` — matches both Python MLX and mlx-swift.
    public let weight: MLXArray
    public let bias: MLXArray

    private let outputChannels: Int
    private let inputChannels: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (0, 0, 0)
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padH = padding.1
        self.padW = padding.2
        // Causal temporal padding: dilation*(k-1) + (1-stride). dilation=1 → k-stride.
        self.causalPadT = kernelSize.0 - stride.0
        self.outputChannels = outputChannels
        self.inputChannels = inputChannels

        self.weight = MLXArray.zeros([
            outputChannels,
            kernelSize.0, kernelSize.1, kernelSize.2,
            inputChannels,
        ])
        self.bias = MLXArray.zeros([outputChannels])
        super.init()
    }

    /// Convenience for the common scalar-kernel case.
    public convenience init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize k: Int,
        padding: Int = 0
    ) {
        self.init(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: (k, k, k),
            stride: (1, 1, 1),
            padding: (padding, padding, padding)
        )
    }

    /// Apply the conv. `x` is `[B, C, T, H, W]`. Optional `cacheX`
    /// provides causal context from the previous chunk (see WanVAECacheT).
    public func callAsFunction(_ x: MLXArray, cacheX: MLXArray? = nil) -> MLXArray {
        var x = x
        let b = x.dim(0)
        let c = x.dim(1)
        let h = x.dim(3)
        let w = x.dim(4)

        var causalPad = causalPadT
        if let cacheX, causalPad > 0 {
            x = concatenated([cacheX, x], axis: 2)
            causalPad = Swift.max(0, causalPad - cacheX.dim(2))
        }

        if causalPad > 0 {
            let padT = MLXArray.zeros([b, c, causalPad, h, w], dtype: x.dtype)
            x = concatenated([padT, x], axis: 2)
        }

        if padH > 0 || padW > 0 {
            x = padded(
                x,
                widths: [
                    .init((0, 0)), .init((0, 0)), .init((0, 0)),
                    .init((padH, padH)), .init((padW, padW)),
                ]
            )
        }

        // [B, C, T, H, W] → [B, T, H, W, C] for mlx-swift Conv3d (NDHWC)
        x = x.transposed(0, 2, 3, 4, 1)
        let out = conv3d(
            x, weight,
            stride: IntOrTriple((stride.0, stride.1, stride.2)),
            padding: 0
        ) + bias
        // [B, T_out, H_out, W_out, O] → [B, O, T_out, H_out, W_out]
        return out.transposed(0, 4, 1, 2, 3)
    }
}

// MARK: - WanRMSNorm

/// Channel-first L2-normalize + learned scale. Matches diffusers'
/// `WanRMS_norm`.
///
/// Two storage shapes for `gamma`:
/// - `images: true` (used by AttentionBlock per-frame): `(C, 1, 1)`
/// - `images: false` (used by ResidualBlock 3D): `(C, 1, 1, 1)`
public final class WanRMSNorm: Module, @unchecked Sendable {
    public let scale: Float
    public let gamma: MLXArray
    public let channelFirst: Bool

    public init(dim: Int, channelFirst: Bool = true, images: Bool = true) {
        self.scale = Foundation.sqrt(Float(dim))
        self.channelFirst = channelFirst
        if channelFirst {
            if images {
                self.gamma = MLXArray.ones([dim, 1, 1])
            } else {
                self.gamma = MLXArray.ones([dim, 1, 1, 1])
            }
        } else {
            self.gamma = MLXArray.ones([dim])
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normDim = channelFirst ? 1 : -1
        let squared = x * x
        let sumSq = squared.sum(axis: normDim, keepDims: true)
        let norm = sqrt(MLX.clip(sumSq, min: MLXArray(Float(1e-12))))
        return (x / norm) * scale * gamma
    }
}

// MARK: - WanAttentionBlock

/// Single-head spatial self-attention (per-frame). Matches diffusers'
/// `WanAttentionBlock`.
///
/// **L10 (skill lesson):** the QK^T → softmax → ·V chain runs on the
/// `.cpu` stream to keep strict fp32 precision. The Python port routes
/// this via `with mx.stream(mx.cpu):`; mlx-swift takes per-op
/// `stream: .cpu` arguments — equivalent semantics.
public final class WanAttentionBlock: Module, @unchecked Sendable {
    public let norm: WanRMSNorm
    public let toQKV: Conv2d
    public let proj: Conv2d
    public let dim: Int

    public init(dim: Int) {
        self.dim = dim
        self.norm = WanRMSNorm(dim: dim, images: true)
        self.toQKV = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        self.proj = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4)

        // [B, C, T, H, W] → [BT, C, H, W] (fold time into batch for per-frame attn)
        var xf = x.transposed(0, 2, 1, 3, 4).reshaped(b * t, c, h, w)
        xf = norm(xf)
        // → NHWC for Conv2d
        xf = xf.transposed(0, 2, 3, 1)

        let qkv = toQKV(xf)
        let qkvReshaped = qkv.reshaped(b * t, h * w, 3, c).transposed(2, 0, 1, 3)
        let q = qkvReshaped[0].expandedDimensions(axis: 1)
        let k = qkvReshaped[1].expandedDimensions(axis: 1)
        let v = qkvReshaped[2].expandedDimensions(axis: 1)

        // L10: route the SDPA chain through .cpu for strict fp32.
        let invSqrtD = Float(1.0) / Foundation.sqrt(Float(c))
        let logits = MLX.matmul(q, k.transposed(0, 1, 3, 2), stream: .cpu) * invSqrtD
        let attn = softmax(logits, axis: -1, stream: .cpu)
        var out = MLX.matmul(attn, v, stream: .cpu)
        // force materialization before leaving the cpu stream (mirrors Python mx.eval)
        out.eval()

        out = out.squeezed(axis: 1).reshaped(b * t, h, w, c)
        out = proj(out)
        // [BT, H, W, C] → [B, T, H, W, C] → [B, C, T, H, W]
        out = out.reshaped(b, t, h, w, c).transposed(0, 4, 1, 2, 3)
        return out + identity
    }
}

// MARK: - WanResample

/// Up/down sample stage. Matches diffusers' `WanResample`.
///
/// Modes (string parity with diffusers):
/// - `"upsample2d"`: per-frame nearest-2x + Conv2d that halves channels
/// - `"upsample3d"`: same + a temporal `time_conv` that doubles T (with
///   the "Rep" first-call sentinel)
/// - `"downsample2d"`: stride-2 Conv2d, channels unchanged
/// - `"downsample3d"`: same + stride-2 temporal `time_conv`
public final class WanResample: Module, @unchecked Sendable {
    public let mode: String
    public let dim: Int
    public let resample: Conv2d
    public let timeConv: CausalConv3d?

    public init(dim: Int, mode: String) {
        precondition(
            ["upsample2d", "upsample3d", "downsample2d", "downsample3d"].contains(mode),
            "Unknown WanResample mode: \(mode)"
        )
        self.mode = mode
        self.dim = dim

        if mode.hasPrefix("upsample") {
            self.resample = Conv2d(
                inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1
            )
            self.timeConv = (mode == "upsample3d")
                ? CausalConv3d(
                    inputChannels: dim, outputChannels: dim * 2,
                    kernelSize: (3, 1, 1), stride: (1, 1, 1), padding: (1, 0, 0)
                )
                : nil
        } else {
            self.resample = Conv2d(
                inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2
            )
            self.timeConv = (mode == "downsample3d")
                ? CausalConv3d(
                    inputChannels: dim, outputChannels: dim,
                    kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)
                )
                : nil
        }
        super.init()
    }

    /// Apply the resample. `featCache`/`featIdx` are passed through chunked
    /// encode/decode; for one-shot inference they can be omitted.
    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        var x = x
        let b = x.dim(0)
        var c = x.dim(1)
        var t = x.dim(2)
        let h0 = x.dim(3)
        let w0 = x.dim(4)

        // ----- temporal step (upsample3d only) -----
        if mode == "upsample3d", let timeConv {
            if let featCache, let featIdx {
                let idx = featIdx.value
                switch featCache.slot(at: idx) {
                case .empty:
                    // First call: skip time_conv, plant "Rep" sentinel.
                    featCache.set(.rep, at: idx)
                    featIdx.advance()
                case .rep, .tensor:
                    var cacheX = x[0..., 0..., (max(0, t - WanVAECacheT))...]
                    if cacheX.dim(2) < 2 {
                        switch featCache.slot(at: idx) {
                        case .tensor(let cached):
                            cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
                        case .rep:
                            cacheX = concatenated([MLXArray.zeros(like: cacheX), cacheX], axis: 2)
                        case .empty:
                            break  // unreachable here
                        }
                    }
                    switch featCache.slot(at: idx) {
                    case .rep:
                        x = timeConv(x)
                    case .tensor(let cached):
                        x = timeConv(x, cacheX: cached)
                    case .empty:
                        break  // unreachable
                    }
                    featCache.set(.tensor(cacheX), at: idx)
                    featIdx.advance()
                    // Double T via reshape-then-stack: [B, 2*C/2, T, H, W] → [B, C/2, 2T, H, W]
                    t = x.dim(2)
                    c = x.dim(1)
                    let reshaped = x.reshaped(b, 2, c / 2, t, h0, w0)
                    let stacked = MLX.stacked(
                        [reshaped[0..., 0], reshaped[0..., 1]],
                        axis: 3
                    )
                    x = stacked.reshaped(b, c / 2, t * 2, h0, w0)
                    t = t * 2
                    c = c / 2
                }
            }
        }

        // ----- spatial step -----
        if mode.hasPrefix("upsample") {
            // [B, C, T, H, W] → [B*T, H, W, C]
            var xs = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h0, w0, c)
            xs = MLX.repeated(xs, count: 2, axis: 1)
            xs = MLX.repeated(xs, count: 2, axis: 2)
            xs = resample(xs)
            let cOut = xs.dim(-1)
            return xs.reshaped(b, t, h0 * 2, w0 * 2, cOut).transposed(0, 4, 1, 2, 3)
        } else {
            var xs = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h0, w0, c)
            xs = padded(
                xs,
                widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]
            )
            xs = resample(xs)
            let cOut = xs.dim(-1)
            let hOut = xs.dim(1), wOut = xs.dim(2)
            var x2 = xs.reshaped(b, t, hOut, wOut, cOut).transposed(0, 4, 1, 2, 3)

            if mode == "downsample3d", let timeConv {
                if let featCache, let featIdx {
                    let idx = featIdx.value
                    switch featCache.slot(at: idx) {
                    case .empty:
                        featCache.set(.tensor(x2), at: idx)
                        featIdx.advance()
                    case .tensor(let cached):
                        let cacheX = x2[0..., 0..., (x2.dim(2) - 1)...]
                        x2 = timeConv(x2, cacheX: cached[0..., 0..., (cached.dim(2) - 1)...])
                        featCache.set(.tensor(cacheX), at: idx)
                        featIdx.advance()
                    case .rep:
                        // Should not occur in downsample path
                        break
                    }
                } else {
                    x2 = timeConv(x2)
                }
            }
            return x2
        }
    }
}

// MARK: - chunked-encode/decode state helpers

/// Reference wrapper so the chunked encode/decode loop can mutate
/// `feat_cache` slots from inside a recursive call (matching Python's
/// list-of-Optional pattern).
public final class WanFeatCacheRef: @unchecked Sendable {
    private var slots: [WanFeatCacheSlot]

    public init(slotCount: Int) {
        slots = Array(repeating: .empty, count: slotCount)
    }

    public func slot(at idx: Int) -> WanFeatCacheSlot { slots[idx] }

    public func set(_ slot: WanFeatCacheSlot, at idx: Int) { slots[idx] = slot }
}

/// Reference wrapper for the single integer `feat_idx[0]` Python uses.
public final class WanFeatIdxRef: @unchecked Sendable {
    public private(set) var value: Int = 0

    public init() {}

    public func advance() { value += 1 }
    public func reset() { value = 0 }
}

// =============================================================================
// TODO(S3.3b): composite blocks + encoder/decoder + top-level AutoencoderKLWan
// =============================================================================
//
// The remaining work (lands as a follow-up PR per the staged-port discipline):
//
//   ☐ WanResidualBlock        (norm1, conv1, norm2, conv2, conv_shortcut?)
//   ☐ WanMidBlock             ([resnet, attn, resnet])
//   ☐ WanUpBlock              (resnets[..R] + optional upsamplers[0])
//   ☐ WanEncoder3d            (conv_in + flat down_blocks + mid_block + norm/conv_out)
//   ☐ WanDecoder3d            (conv_in + mid_block + nested up_blocks + norm/conv_out)
//   ☐ AutoencoderKLWan        (encoder + quant_conv + post_quant_conv + decoder,
//                              encode(), decode(), normalizeLatents(), denormalizeLatents())
//   ☐ fromPretrained(repoID:) using WeightLoader (S3.2 land)
//   ☐ Parity test against the Python port's reference .npy outputs:
//     - encode max_abs < 1e-3
//     - decode max_abs < 5e-2
//
