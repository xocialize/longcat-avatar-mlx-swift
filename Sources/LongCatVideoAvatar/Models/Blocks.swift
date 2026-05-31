//
//  Blocks.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/blocks.py
//
//  DiT building blocks: FeedForwardSwiGLU, RMSNorm_FP32, LayerNorm_FP32,
//  modulate_fp32, PatchEmbed3D, TimestepEmbedder, CaptionEmbedder,
//  FinalLayer_FP32.
//
//  The `_FP32` suffix on RMSNorm / LayerNorm / FinalLayer is a Meituan
//  convention: their forward casts the input to fp32 for the norm + scale
//  + bias computation, then casts back. We replicate exactly — silent
//  bf16 accumulation in AdaLN modulation breaks parity in their training.
//

import Foundation
import MLX
import MLXNN

// MARK: - FeedForwardSwiGLU

/// SwiGLU-gated FFN. Internal `hiddenDim` follows the SwiGLU 2/3 rule.
/// For `dim=4096, hiddenDim=16384, multipleOf=256` (LongCat defaults):
/// `int(2 * 16384 / 3) = 10922`, rounded up to `256 * 43 = 11008`.
public final class FeedForwardSwiGLU: Module, @unchecked Sendable {
    public let dim: Int
    public let hiddenDim: Int

    public var w1: Linear
    public var w2: Linear
    public var w3: Linear

    public init(dim: Int, hiddenDim: Int, multipleOf: Int = 256, ffnDimMultiplier: Float? = nil) {
        var h = Int(2.0 * Float(hiddenDim) / 3.0)
        if let m = ffnDimMultiplier {
            h = Int(m * Float(h))
        }
        h = multipleOf * ((h + multipleOf - 1) / multipleOf)

        self.dim = dim
        self.hiddenDim = h
        self.w1 = Linear(dim, h, bias: false)
        self.w2 = Linear(h, dim, bias: false)
        self.w3 = Linear(dim, h, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - RMSNormFP32

/// RMS norm with fp32 internal compute. Matches PT `RMSNorm_FP32`.
///
/// Forward: `((x.float() * rsqrt(mean(x.float()**2) + eps)).type_as(x)) * weight`.
/// Norm computation runs in fp32 regardless of input dtype; the learned
/// `weight` is applied in the input dtype.
public final class RMSNormFP32: Module, @unchecked Sendable {
    public let eps: Float
    public let weight: MLXArray

    public init(dim: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let outDtype = x.dtype
        let xf = x.asType(.float32)
        let norm = xf * MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + MLXArray(eps))
        return norm.asType(outDtype) * weight
    }
}

// MARK: - LayerNormFP32

/// LayerNorm with fp32 internal compute. `elementwiseAffine=false` means
/// no weight/bias — used inside `modulateFP32`. `elementwiseAffine=true`
/// adds learned `weight` and `bias`.
public final class LayerNormFP32: Module, @unchecked Sendable {
    public let dim: Int
    public let eps: Float
    public let elementwiseAffine: Bool

    public var weight: MLXArray?
    public var bias: MLXArray?

    public init(dim: Int, eps: Float, elementwiseAffine: Bool) {
        self.dim = dim
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        if elementwiseAffine {
            self.weight = MLXArray.ones([dim])
            self.bias = MLXArray.zeros([dim])
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let outDtype = x.dtype
        let xf = x.asType(.float32)
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let diff = xf - mean
        let variance = MLX.mean(diff * diff, axis: -1, keepDims: true)
        var normalized = diff * MLX.rsqrt(variance + MLXArray(eps))
        if elementwiseAffine, let w = weight, let b = bias {
            normalized = normalized * w.asType(.float32) + b.asType(.float32)
        }
        return normalized.asType(outDtype)
    }
}

// MARK: - modulateFP32

/// AdaLN-Zero style modulation in fp32, then cast back.
///
/// Asserts `shift.dtype == scale.dtype == fp32`. Result is
/// `norm(x.float()) * (scale + 1) + shift`, returned in `x.dtype`.
public func modulateFP32(
    _ normFunc: (MLXArray) -> MLXArray,
    _ x: MLXArray,
    shift: MLXArray,
    scale: MLXArray
) -> MLXArray {
    precondition(
        shift.dtype == .float32 && scale.dtype == .float32,
        "Modulation params must be fp32; AdaLN math diverges in bf16."
    )
    let dtype = x.dtype
    var xf = normFunc(x.asType(.float32))
    xf = xf * (scale + Float(1)) + shift
    return xf.asType(dtype)
}

// MARK: - PatchEmbed3D

/// Conv3d placeholder so checkpoint keys map to `x_embedder.proj.weight`.
/// Weight layout `(O, kT, kH, kW, I)` matches mlx-swift's MLXNN.Conv3d.
public final class Conv3dPlaceholder: Module, @unchecked Sendable {
    public let weight: MLXArray
    public let bias: MLXArray

    public init(outChannels: Int, inChannels: Int, kernelSize: (Int, Int, Int)) {
        self.weight = MLXArray.zeros([outChannels, kernelSize.0, kernelSize.1, kernelSize.2, inChannels])
        self.bias = MLXArray.zeros([outChannels])
        super.init()
    }
}

/// 3D patchify via native MLXNN.Conv3d under the hood. Weight shape and
/// bias shape match the PT checkpoint layout exactly so they load with
/// no transpose. Python emulates Conv3d via per-frame Conv2d because
/// Python MLX historically lacked native Conv3d — mlx-swift has it, so
/// we use the real op for cleaner code + lower latency.
public final class PatchEmbed3D: Module, @unchecked Sendable {
    public let patchSize: (Int, Int, Int)
    public let inChans: Int
    public let embedDim: Int
    public let flatten: Bool

    public var proj: Conv3dPlaceholder

    public init(
        patchSize: (Int, Int, Int) = (2, 4, 4),
        inChans: Int = 3,
        embedDim: Int = 96,
        flatten: Bool = true
    ) {
        self.patchSize = patchSize
        self.inChans = inChans
        self.embedDim = embedDim
        self.flatten = flatten
        self.proj = Conv3dPlaceholder(
            outChannels: embedDim,
            inChannels: inChans,
            kernelSize: patchSize
        )
        super.init()
    }

    /// - x: `[B, C, T, H, W]` (channel-second, PT convention)
    /// - Returns: `[B, N, C]` when `flatten=true`, else `[B, C, T_p, H_p, W_p]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let h = x.dim(3)
        let w = x.dim(4)
        let (kt, kh, kw) = patchSize

        // Pad to multiples of patch_size if needed
        if w % kw != 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, kw - w % kw))])
        }
        if h % kh != 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, kh - h % kh)), .init((0, 0))])
        }
        if t % kt != 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, kt - t % kt)), .init((0, 0)), .init((0, 0))])
        }

        // [B, C, T, H, W] → [B, T, H, W, C] for mlx-swift conv3d
        x = x.transposed(0, 2, 3, 4, 1)

        // Use native conv3d. Weight is (O, kT, kH, kW, I) already.
        let out = conv3d(
            x, proj.weight,
            stride: IntOrTriple((kt, kh, kw)),
            padding: 0
        ) + proj.bias

        let tOut = out.dim(1), hOut = out.dim(2), wOut = out.dim(3)
        if flatten {
            // [B, T_p, H_p, W_p, C] → [B, T_p*H_p*W_p, C]
            return out.reshaped(b, tOut * hOut * wOut, embedDim)
        } else {
            // [B, T_p, H_p, W_p, C] → [B, C, T_p, H_p, W_p]
            return out.transposed(0, 4, 1, 2, 3)
        }
    }
}

// MARK: - Timestep embedding utility

/// Sinusoidal timestep embedding. `t` is 1D of shape `(N,)`.
public func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
    let half = dim / 2
    let scale = -Float(log(Double(maxPeriod)))
    let halfRange = MLXArray((0..<half).map { Float($0) })
    let freqs = MLX.exp(MLXArray(scale) * halfRange / Float(half))
    let args = t[0..., .newAxis].asType(.float32) * freqs[.newAxis, 0...]
    var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    if dim % 2 != 0 {
        let pad = MLXArray.zeros(like: embedding[0..., 0..<1])
        embedding = MLX.concatenated([embedding, pad], axis: -1)
    }
    return embedding
}

// MARK: - TimestepEmbedder

/// Sinusoidal embedding + 2-layer MLP. Output is fp32 by convention.
/// PT key names: `mlp.0.{weight,bias}`, `mlp.2.{weight,bias}` (the middle
/// SiLU has no params). Mirrored with a `[Linear?]` list pattern (same
/// approach as `WanResample.resample`).
public final class TimestepEmbedder: Module, @unchecked Sendable {
    public let tEmbedDim: Int
    public let frequencyEmbeddingSize: Int

    /// Mirrors Python `self.mlp = [Linear, None (SiLU), Linear]`. Safetensors
    /// keys are `mlp.0.weight/bias` and `mlp.2.weight/bias`.
    public var mlp: [Linear?]

    public init(tEmbedDim: Int, frequencyEmbeddingSize: Int = 256) {
        self.tEmbedDim = tEmbedDim
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self.mlp = [
            Linear(frequencyEmbeddingSize, tEmbedDim, bias: true),
            nil,
            Linear(tEmbedDim, tEmbedDim, bias: true),
        ]
        super.init()
    }

    public func callAsFunction(_ t: MLXArray, dtype: DType = .float32) -> MLXArray {
        var tFreq = timestepEmbedding(t, dim: frequencyEmbeddingSize)
        if tFreq.dtype != dtype {
            tFreq = tFreq.asType(dtype)
        }
        var x = mlp[0]!(tFreq)
        x = silu(x)
        x = mlp[2]!(x)
        return x
    }
}

// MARK: - CaptionEmbedder

/// 2-layer MLP with tanh-approximate GELU. PT name: `y_proj`.
public final class CaptionEmbedder: Module, @unchecked Sendable {
    public let inChannels: Int
    public let hiddenSize: Int

    /// PT pattern: y_proj.0 = Linear, y_proj.1 = GELU(tanh), y_proj.2 = Linear.
    @ModuleInfo(key: "y_proj") public var yProj: [Linear?]

    public init(inChannels: Int, hiddenSize: Int) {
        self.inChannels = inChannels
        self.hiddenSize = hiddenSize
        self._yProj.wrappedValue = [
            Linear(inChannels, hiddenSize, bias: true),
            nil,
            Linear(hiddenSize, hiddenSize, bias: true),
        ]
        super.init()
    }

    public func callAsFunction(_ caption: MLXArray) -> MLXArray {
        var x = yProj[0]!(caption)
        x = geluApproximate(x)
        x = yProj[2]!(x)
        return x
    }
}

// MARK: - FinalLayerFP32

/// Final AdaLN-Zero head: LN(no-affine) → Linear, with shift/scale from t.
///
/// PT: `norm_final` (LayerNorm_FP32 no-affine), `linear` (Linear),
/// `adaLN_modulation` (`nn.Sequential` of SiLU + Linear). The `[None, Linear]`
/// pattern is mirrored with `[Linear?]`.
public final class FinalLayerFP32: Module, @unchecked Sendable {
    public let hiddenSize: Int
    public let numPatch: Int
    public let outChannels: Int
    public let adalnTembedDim: Int

    @ModuleInfo(key: "norm_final") public var normFinal: LayerNormFP32
    public var linear: Linear

    /// PT `adaLN_modulation` Sequential: [SiLU, Linear]. Index 0 has no
    /// params; index 1 is the Linear.
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: [Linear?]

    public init(hiddenSize: Int, numPatch: Int, outChannels: Int, adalnTembedDim: Int) {
        self.hiddenSize = hiddenSize
        self.numPatch = numPatch
        self.outChannels = outChannels
        self.adalnTembedDim = adalnTembedDim

        self._normFinal.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: false)
        self.linear = Linear(hiddenSize, numPatch * outChannels, bias: true)
        self._adaLNModulation.wrappedValue = [
            nil,
            Linear(adalnTembedDim, 2 * hiddenSize, bias: true),
        ]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, t: MLXArray, latentShape: (Int, Int, Int)) -> MLXArray {
        precondition(t.dtype == .float32, "FinalLayer expects fp32 timestep embedding")
        let b = x.dim(0), n = x.dim(1), c = x.dim(2)
        let tFrames = latentShape.0

        // SiLU + Linear(t) → split(2)
        let tIn = silu(t)
        var ada = adaLNModulation[1]!(tIn)        // [B, T, 2*C]
        ada = ada[0..., 0..., .newAxis, 0...]      // [B, T, 1, 2*C]
        let halves = MLX.split(ada, parts: 2, axis: -1)
        let shift = halves[0]
        let scale = halves[1]

        let reshaped = x.reshaped(b, tFrames, -1, c)
        let modulated = modulateFP32({ self.normFinal($0) }, reshaped, shift: shift, scale: scale)
        return linear(modulated.reshaped(b, n, c))
    }
}
