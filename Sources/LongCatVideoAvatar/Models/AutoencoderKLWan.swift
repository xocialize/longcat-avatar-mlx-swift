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
    public var norm: WanRMSNorm
    @ModuleInfo(key: "to_qkv") public var toQKV: Conv2d
    public var proj: Conv2d
    public let dim: Int

    public init(dim: Int) {
        self.dim = dim
        self.norm = WanRMSNorm(dim: dim, images: true)
        self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
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

    /// Mirrors Python `self.resample = [None, nn.Conv2d(...)]`. The
    /// safetensors keys are `resample.1.weight` / `resample.1.bias` —
    /// no key for index 0. Module reflection iterates `[Conv2d?]` as
    /// `resample.0` (nil → omitted) and `resample.1` (the Conv2d).
    public var resample: [Conv2d?]

    @ModuleInfo(key: "time_conv") public var timeConv: CausalConv3d?

    public init(dim: Int, mode: String) {
        precondition(
            ["upsample2d", "upsample3d", "downsample2d", "downsample3d"].contains(mode),
            "Unknown WanResample mode: \(mode)"
        )
        self.mode = mode
        self.dim = dim

        if mode.hasPrefix("upsample") {
            self.resample = [
                nil,
                Conv2d(inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1),
            ]
            self._timeConv.wrappedValue = (mode == "upsample3d")
                ? CausalConv3d(
                    inputChannels: dim, outputChannels: dim * 2,
                    kernelSize: (3, 1, 1), stride: (1, 1, 1), padding: (1, 0, 0)
                )
                : nil
        } else {
            self.resample = [
                nil,
                Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2),
            ]
            self._timeConv.wrappedValue = (mode == "downsample3d")
                ? CausalConv3d(
                    inputChannels: dim, outputChannels: dim,
                    kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)
                )
                : nil
        }
        super.init()
    }

    /// Convenience for the resample conv (Python's `self.resample[1]`).
    private var resampleConv: Conv2d {
        guard let c = resample.last, let conv = c else {
            fatalError("resample[1] missing — WanResample structural drift")
        }
        return conv
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
            xs = resampleConv(xs)
            let cOut = xs.dim(-1)
            return xs.reshaped(b, t, h0 * 2, w0 * 2, cOut).transposed(0, 4, 1, 2, 3)
        } else {
            var xs = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h0, w0, c)
            xs = padded(
                xs,
                widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]
            )
            xs = resampleConv(xs)
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
// Composite blocks (S3.3b) — match diffusers' WanResidualBlock /
// WanMidBlock / WanUpBlock. The "down stage" in the encoder is a FLAT
// list mixing residual blocks, optional attention, and resamples
// (per the Wan 2.1 `is_residual=False` schema) — kept as `[any Module]`
// rather than wrapped in a "DownBlock" class to mirror the Python port.
// =============================================================================

/// ResNet block with named `norm1` / `conv1` / `norm2` / `conv2` /
/// `convShortcut?` fields. Matches diffusers' `WanResidualBlock`.
public final class WanResidualBlock: Module, @unchecked Sendable {
    public let inDim: Int
    public let outDim: Int

    public var norm1: WanRMSNorm
    public var conv1: CausalConv3d
    public var norm2: WanRMSNorm
    public var conv2: CausalConv3d
    /// `nil` when `inDim == outDim` (matches PT's `nn.Identity()` slot).
    @ModuleInfo(key: "conv_shortcut") public var convShortcut: CausalConv3d?

    public init(inDim: Int, outDim: Int) {
        self.inDim = inDim
        self.outDim = outDim

        self.norm1 = WanRMSNorm(dim: inDim, images: false)
        self.conv1 = CausalConv3d(inputChannels: inDim, outputChannels: outDim, kernelSize: 3, padding: 1)
        self.norm2 = WanRMSNorm(dim: outDim, images: false)
        self.conv2 = CausalConv3d(inputChannels: outDim, outputChannels: outDim, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = (inDim != outDim)
            ? CausalConv3d(inputChannels: inDim, outputChannels: outDim, kernelSize: 1)
            : nil
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        let h = (convShortcut == nil) ? x : convShortcut!(x)

        var y = silu(norm1(x))
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached):
                y = conv1(y, cacheX: cached)
            case .empty, .rep:
                y = conv1(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = conv1(y)
        }

        y = silu(norm2(y))
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached):
                y = conv2(y, cacheX: cached)
            case .empty, .rep:
                y = conv2(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = conv2(y)
        }

        return y + h
    }
}

/// Middle block: `[resnet, attn, resnet, attn, resnet, …]` — alternating,
/// always starting and ending with a resnet. `num_layers=N` adds N
/// attention/resnet pairs after the leading resnet (so total: 1 + 2N
/// modules). Matches diffusers' `WanMidBlock`.
public final class WanMidBlock: Module, @unchecked Sendable {
    public let resnets: [WanResidualBlock]
    public let attentions: [WanAttentionBlock]

    public init(dim: Int, numLayers: Int = 1) {
        var rs = [WanResidualBlock(inDim: dim, outDim: dim)]
        var atts: [WanAttentionBlock] = []
        for _ in 0..<numLayers {
            atts.append(WanAttentionBlock(dim: dim))
            rs.append(WanResidualBlock(inDim: dim, outDim: dim))
        }
        self.resnets = rs
        self.attentions = atts
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        var y = resnets[0](x, featCache: featCache, featIdx: featIdx)
        for i in 0..<attentions.count {
            y = attentions[i](y)
            y = resnets[i + 1](y, featCache: featCache, featIdx: featIdx)
        }
        return y
    }
}

/// Decoder up-stage. `resnets[..R+1]` followed by an optional
/// `upsamplers[0]` (matches Python — Python uses `resnets` as the
/// public attribute name + `upsamplers` as a 1-element list).
public final class WanUpBlock: Module, @unchecked Sendable {
    public let resnets: [WanResidualBlock]
    public let upsamplers: [WanResample]?

    public init(inDim: Int, outDim: Int, numResBlocks: Int, upsampleMode: String?) {
        var rs: [WanResidualBlock] = []
        var current = inDim
        for _ in 0..<(numResBlocks + 1) {
            rs.append(WanResidualBlock(inDim: current, outDim: outDim))
            current = outDim
        }
        self.resnets = rs
        self.upsamplers = upsampleMode.map { [WanResample(dim: outDim, mode: $0)] }
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        var y = x
        for r in resnets {
            y = r(y, featCache: featCache, featIdx: featIdx)
        }
        if let ups = upsamplers, let head = ups.first {
            y = head(y, featCache: featCache, featIdx: featIdx)
        }
        return y
    }
}

// =============================================================================
// Encoder / Decoder
// =============================================================================

/// 3D VAE Encoder. Matches diffusers' `WanEncoder3d` with `is_residual=false`.
///
/// `downBlocks` mirrors Python's flat heterogeneous list mixing
/// `WanResidualBlock` / `WanAttentionBlock` / `WanResample`. Declared as
/// `[Module]` so mlx-swift's Mirror reflection iterates it as a list of
/// modules and produces safetensors-compatible paths like
/// `down_blocks.0.weight`, `down_blocks.5.resample.1.weight`, etc.
/// Runtime forward() switch-casts on concrete type.
public final class WanEncoder3d: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_in") public var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") public var downBlocks: [Module]
    @ModuleInfo(key: "mid_block") public var midBlock: WanMidBlock
    @ModuleInfo(key: "norm_out") public var normOut: WanRMSNorm
    @ModuleInfo(key: "conv_out") public var convOut: CausalConv3d

    public init(
        inChannels: Int = 3,
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int]? = nil,
        numResBlocks: Int = 2,
        attnScales: [Float] = [],
        temperalDownsample: [Bool]? = nil
    ) {
        let mults = dimMult ?? [1, 2, 4, 4]
        let tempDown = temperalDownsample ?? [false, true, true]
        let dims = [dim] + mults.map { dim * $0 }
        var scale: Float = 1.0

        self._convIn.wrappedValue = CausalConv3d(
            inputChannels: inChannels, outputChannels: dims[0], kernelSize: 3, padding: 1
        )

        var blocks: [Module] = []
        var outDim = dims[0]
        for i in 0..<(dims.count - 1) {
            var inD = dims[i]
            let outD = dims[i + 1]
            for _ in 0..<numResBlocks {
                blocks.append(WanResidualBlock(inDim: inD, outDim: outD))
                if attnScales.contains(scale) {
                    blocks.append(WanAttentionBlock(dim: outD))
                }
                inD = outD
            }
            if i != mults.count - 1 {
                let mode = tempDown[i] ? "downsample3d" : "downsample2d"
                blocks.append(WanResample(dim: outD, mode: mode))
                scale /= 2.0
            }
            outDim = outD
        }
        self._downBlocks.wrappedValue = blocks

        self._midBlock.wrappedValue = WanMidBlock(dim: outDim, numLayers: 1)
        self._normOut.wrappedValue = WanRMSNorm(dim: outDim, images: false)
        self._convOut.wrappedValue = CausalConv3d(
            inputChannels: outDim, outputChannels: zDim, kernelSize: 3, padding: 1
        )
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        var y = x

        // ----- conv_in (cache-aware) -----
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached): y = convIn(y, cacheX: cached)
            case .empty, .rep:        y = convIn(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = convIn(y)
        }

        // ----- down blocks (runtime cast on the heterogeneous list) -----
        for layer in downBlocks {
            if let r = layer as? WanResidualBlock {
                y = r(y, featCache: featCache, featIdx: featIdx)
            } else if let a = layer as? WanAttentionBlock {
                y = a(y)   // stateless per-frame
            } else if let s = layer as? WanResample {
                y = s(y, featCache: featCache, featIdx: featIdx)
            } else {
                fatalError("unexpected down-block layer: \(type(of: layer))")
            }
        }

        y = midBlock(y, featCache: featCache, featIdx: featIdx)

        y = silu(normOut(y))
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached): y = convOut(y, cacheX: cached)
            case .empty, .rep:        y = convOut(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = convOut(y)
        }
        return y
    }
}

/// 3D VAE Decoder. Matches diffusers' `WanDecoder3d` with `is_residual=false`.
public final class WanDecoder3d: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_in") public var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") public var midBlock: WanMidBlock
    @ModuleInfo(key: "up_blocks") public var upBlocks: [WanUpBlock]
    @ModuleInfo(key: "norm_out") public var normOut: WanRMSNorm
    @ModuleInfo(key: "conv_out") public var convOut: CausalConv3d

    public init(
        outChannels: Int = 3,
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int]? = nil,
        numResBlocks: Int = 2,
        temperalUpsample: [Bool]? = nil
    ) {
        let mults = dimMult ?? [1, 2, 4, 4]
        let tempUp = temperalUpsample ?? [true, true, false]

        // dims = [dim*mults.last, dim*mults.reversed()...]
        let reversedMults = Array(mults.reversed())
        let dims = [dim * mults.last!] + reversedMults.map { dim * $0 }

        self._convIn.wrappedValue = CausalConv3d(
            inputChannels: zDim, outputChannels: dims[0], kernelSize: 3, padding: 1
        )
        self._midBlock.wrappedValue = WanMidBlock(dim: dims[0], numLayers: 1)

        var ups: [WanUpBlock] = []
        var outDim = dims[0]
        for i in 0..<(dims.count - 1) {
            var inD = dims[i]
            let outD = dims[i + 1]
            // Wan 2.1: starting from stage 1, the previous upsample halves channels
            if i > 0 { inD = inD / 2 }
            let upFlag = (i != mults.count - 1)
            let mode: String? = upFlag ? (tempUp[i] ? "upsample3d" : "upsample2d") : nil
            ups.append(WanUpBlock(
                inDim: inD, outDim: outD, numResBlocks: numResBlocks, upsampleMode: mode
            ))
            outDim = outD
        }
        self._upBlocks.wrappedValue = ups

        self._normOut.wrappedValue = WanRMSNorm(dim: outDim, images: false)
        self._convOut.wrappedValue = CausalConv3d(
            inputChannels: outDim, outputChannels: outChannels, kernelSize: 3, padding: 1
        )
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        featCache: WanFeatCacheRef? = nil,
        featIdx: WanFeatIdxRef? = nil
    ) -> MLXArray {
        var y = x

        // ----- conv_in -----
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached): y = convIn(y, cacheX: cached)
            case .empty, .rep:        y = convIn(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = convIn(y)
        }

        y = midBlock(y, featCache: featCache, featIdx: featIdx)
        for up in upBlocks {
            y = up(y, featCache: featCache, featIdx: featIdx)
        }

        y = silu(normOut(y))
        if let featCache, let featIdx {
            let idx = featIdx.value
            var cacheX = y[0..., 0..., (max(0, y.dim(2) - WanVAECacheT))...]
            if cacheX.dim(2) < WanVAECacheT, case .tensor(let cached) = featCache.slot(at: idx) {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            switch featCache.slot(at: idx) {
            case .tensor(let cached): y = convOut(y, cacheX: cached)
            case .empty, .rep:        y = convOut(y)
            }
            featCache.set(.tensor(cacheX), at: idx)
            featIdx.advance()
        } else {
            y = convOut(y)
        }
        return y
    }
}

// =============================================================================
// Top-level AutoencoderKLWan
// =============================================================================

/// Meituan-style `vae/config.json`.
public struct WanVAEConfig: Codable, Sendable {
    public var zDim: Int = 16
    public var baseDim: Int = 96
    public var dimMult: [Int]? = nil
    public var numResBlocks: Int = 2
    public var attnScales: [Float]? = nil
    public var temperalDownsample: [Bool]? = nil
    public var latentsMean: [Float]? = nil
    public var latentsStd: [Float]? = nil

    enum CodingKeys: String, CodingKey {
        case zDim = "z_dim"
        case baseDim = "base_dim"
        case dimMult = "dim_mult"
        case numResBlocks = "num_res_blocks"
        case attnScales = "attn_scales"
        case temperalDownsample = "temperal_downsample"  // [sic] — upstream typo preserved
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        zDim = try c.decodeIfPresent(Int.self, forKey: .zDim) ?? 16
        baseDim = try c.decodeIfPresent(Int.self, forKey: .baseDim) ?? 96
        dimMult = try c.decodeIfPresent([Int].self, forKey: .dimMult)
        numResBlocks = try c.decodeIfPresent(Int.self, forKey: .numResBlocks) ?? 2
        attnScales = try c.decodeIfPresent([Float].self, forKey: .attnScales)
        temperalDownsample = try c.decodeIfPresent([Bool].self, forKey: .temperalDownsample)
        latentsMean = try c.decodeIfPresent([Float].self, forKey: .latentsMean)
        latentsStd = try c.decodeIfPresent([Float].self, forKey: .latentsStd)
    }
}

public final class AutoencoderKLWan: Module, @unchecked Sendable {
    public let zDim: Int
    public let mean: MLXArray
    public let std: MLXArray
    public let invStd: MLXArray

    // Decoder side: always present
    public var decoder: WanDecoder3d
    @ModuleInfo(key: "post_quant_conv") public var postQuantConv: CausalConv3d

    // Encoder side: optional (decoder-only init for inference-as-decoder)
    public var encoder: WanEncoder3d?
    @ModuleInfo(key: "quant_conv") public var quantConv: CausalConv3d?

    public init(config: WanVAEConfig = .init(), includeEncoder: Bool = true) {
        let zDim = config.zDim
        let baseDim = config.baseDim
        let dimMult = config.dimMult ?? [1, 2, 4, 4]
        let numResBlocks = config.numResBlocks
        let attnScales = config.attnScales ?? []
        let tempDown = config.temperalDownsample ?? [false, true, true]
        let latentsMean = config.latentsMean ?? DefaultVAEMean
        let latentsStd = config.latentsStd ?? DefaultVAEStd

        self.zDim = zDim
        self.mean = MLXArray(latentsMean)
        let stdArr = MLXArray(latentsStd)
        self.std = stdArr
        self.invStd = MLXArray(Float(1.0)) / stdArr

        let tempUp = Array(tempDown.reversed())
        self.decoder = WanDecoder3d(
            outChannels: 3, dim: baseDim, zDim: zDim, dimMult: dimMult,
            numResBlocks: numResBlocks, temperalUpsample: tempUp
        )
        self._postQuantConv.wrappedValue = CausalConv3d(
            inputChannels: zDim, outputChannels: zDim, kernelSize: 1
        )

        if includeEncoder {
            self.encoder = WanEncoder3d(
                inChannels: 3, dim: baseDim, zDim: zDim * 2, dimMult: dimMult,
                numResBlocks: numResBlocks, attnScales: attnScales,
                temperalDownsample: tempDown
            )
            self._quantConv.wrappedValue = CausalConv3d(
                inputChannels: zDim * 2, outputChannels: zDim * 2, kernelSize: 1
            )
        } else {
            self.encoder = nil
            self._quantConv.wrappedValue = nil
        }
        super.init()
    }

    // MARK: - normalize / denormalize

    public func normalizeLatents(_ mu: MLXArray) -> MLXArray {
        let m = mean.reshaped(1, -1, 1, 1, 1)
        let i = invStd.reshaped(1, -1, 1, 1, 1)
        return (mu - m) * i
    }

    public func denormalizeLatents(_ z: MLXArray) -> MLXArray {
        let m = mean.reshaped(1, -1, 1, 1, 1)
        let i = invStd.reshaped(1, -1, 1, 1, 1)
        return z / i + m
    }

    // MARK: - cache slot counts

    /// Count CausalConv3d slots that participate in the chunked-encode cache.
    public func countEncoderCacheSlots() -> Int {
        guard let encoder else { return 0 }
        var n = 1  // conv_in
        for layer in encoder.downBlocks {
            if layer is WanResidualBlock {
                n += 2
            } else if let r = layer as? WanResample, r.mode == "downsample3d" {
                n += 1
            }
        }
        for _ in encoder.midBlock.resnets { n += 2 }
        n += 1  // conv_out
        return n
    }

    /// Count CausalConv3d slots that participate in the chunked-decode cache.
    public func countDecoderCacheSlots() -> Int {
        var n = 1  // conv_in
        for _ in decoder.midBlock.resnets { n += 2 }
        for up in decoder.upBlocks {
            for _ in up.resnets { n += 2 }
            if let head = up.upsamplers?.first, head.mode == "upsample3d" {
                n += 1
            }
        }
        n += 1  // conv_out
        return n
    }

    // MARK: - encode / decode

    /// Video `[B, 3, T, H, W]` in `[-1, 1]` → raw latent mean
    /// `[B, zDim, T_lat, H_lat, W_lat]`. Call `normalizeLatents` before
    /// feeding the DiT.
    public func encode(_ x: MLXArray) -> MLXArray {
        guard let encoder, let quantConv else {
            fatalError("encode() called on decoder-only AutoencoderKLWan; reconstruct with includeEncoder: true")
        }
        let slots = countEncoderCacheSlots()
        let featCache = WanFeatCacheRef(slotCount: slots)

        let t = x.dim(2)
        let numChunks = 1 + (t - 1) / 4

        var out: MLXArray? = nil
        for i in 0..<numChunks {
            let featIdx = WanFeatIdxRef()
            let chunk: MLXArray
            if i == 0 {
                chunk = x[0..., 0..., 0..<1]
            } else {
                chunk = x[0..., 0..., (1 + 4 * (i - 1))..<(1 + 4 * i)]
            }
            let cOut = encoder(chunk, featCache: featCache, featIdx: featIdx)
            out = (out == nil) ? cOut : concatenated([out!, cOut], axis: 2)
        }
        let pre = quantConv(out!)
        let parts = MLX.split(pre, parts: 2, axis: 1)
        return parts[0]   // mu; discard logvar
    }

    /// Raw (post-denormalization) latent → video `[B, 3, T, H, W]` in `[-1, 1]`.
    public func decode(_ z: MLXArray) -> MLXArray {
        let x = postQuantConv(z)
        let slots = countDecoderCacheSlots()
        let featCache = WanFeatCacheRef(slotCount: slots)

        let numFrame = x.dim(2)
        var out: MLXArray? = nil
        for i in 0..<numFrame {
            let featIdx = WanFeatIdxRef()
            let chunk = x[0..., 0..., i..<(i + 1)]
            let cOut = decoder(chunk, featCache: featCache, featIdx: featIdx)
            out = (out == nil) ? cOut : concatenated([out!, cOut], axis: 2)
        }
        return MLX.clip(out!, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
    }

    // MARK: - fromPretrained

    /// Download (if needed), load `vae/config.json` + `vae/diffusion_pytorch_model.safetensors`,
    /// and construct a fully-initialized `AutoencoderKLWan`. Default repo is
    /// the recommended `bf16-dmd-merged` variant; any of the four published
    /// variants work — the VAE is identical across them.
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        includeEncoder: Bool = true,
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> AutoencoderKLWan {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let vaeDir = try WeightLoader.componentDirectory("vae", under: root)
        let config: WanVAEConfig = try WeightLoader.loadConfig(
            WanVAEConfig.self,
            from: vaeDir.appendingPathComponent("config.json")
        )
        let model = AutoencoderKLWan(config: config, includeEncoder: includeEncoder)
        var weights = try WeightLoader.loadSafetensors(
            url: vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        )
        // Decoder-only construction: drop encoder-side weights from the
        // safetensors before update, otherwise `.noUnusedKeys` rejects them.
        if !includeEncoder {
            weights = weights.filter { key, _ in
                !(key.hasPrefix("encoder.") || key.hasPrefix("quant_conv."))
            }
        }
        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
