//
//  AvatarAttention.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/attention.py
//
//  Two classes:
//  - AvatarAttention: visual self-attention with Reference Skip Q-slicing.
//    Subclasses the base Attention (identical parameters). When invoked with
//    num_cond_latents > 0 AND mask_frame_range > 0, splits the noise-region
//    Q into front / maskref / back chunks; the maskref chunk attends only
//    to non-reference K/V to prevent the reference image from inducing
//    repetitive motion in nearby frames.
//  - SingleStreamAttention: audio cross-attention. Visual tokens are Q
//    (from DiT hidden state, dim=hidden_size). Audio context tokens are K/V
//    (from AudioProjModel, dim=output_dim=768). Each video latent frame
//    attends to its own 32 audio tokens. Optional MultiTalk L-RoPE routing
//    via x_ref_attn_map for 2-person conversations.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - AvatarAttention

/// Avatar self-attention with Reference Skip Q-slicing. Inherits all
/// parameters from the base `Attention` — checkpoint keys for visual
/// self-attn are identical between base and avatar.
public final class AvatarAttention: Module, @unchecked Sendable {
    public let dim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public var qkv: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNormFP32
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNormFP32
    public var proj: Linear

    @ModuleInfo(key: "rope_3d") public var rope3d: RotaryPositionalEmbedding

    public init(dim: Int, numHeads: Int) {
        precondition(dim % numHeads == 0, "dim must be divisible by numHeads")
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = Float(1.0) / Foundation.sqrt(Float(self.headDim))

        self.qkv = Linear(dim, dim * 3, bias: true)
        self._qNorm.wrappedValue = RMSNormFP32(dim: self.headDim, eps: 1e-6)
        self._kNorm.wrappedValue = RMSNormFP32(dim: self.headDim, eps: 1e-6)
        self.proj = Linear(dim, dim)
        self._rope3d.wrappedValue = RotaryPositionalEmbedding(headDim: self.headDim)
        super.init()
    }

    private func processAttn(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
    }

    /// Forward.
    /// - x: `[B, N, C]`
    /// - shape: `(T, H, W)`
    /// - Returns: `(out, kvCache?, xRefAttnMap?)`. `xRefAttnMap` is non-nil
    ///   only when `refTargetMasks` is provided (MultiTalk single-talker
    ///   inference returns `nil`).
    public func callAsFunction(
        _ x: MLXArray,
        shape: (Int, Int, Int),
        numCondLatents: Int? = nil,
        returnKV: Bool = false,
        numRefLatents: Int? = nil,
        refImgIndex: Int? = nil,
        maskFrameRange: Int? = nil,
        refTargetMasks: MLXArray? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, MLXArray?) {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)
        let TLat = shape.0
        let tokensPerFrame = N / TLat

        var qkvOut = qkv(x)
        qkvOut = qkvOut.reshaped(B, N, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        q = qNorm(q)
        k = kNorm(k)

        var kvCacheOut: (MLXArray, MLXArray)? = nil
        if returnKV {
            kvCacheOut = (k, v)
        }

        // 3D RoPE (Avatar variant supports frame_index/num_ref_latents)
        (q, k) = rope3d(q: q, k: k, gridSize: shape, frameIndex: refImgIndex, numRefLatents: numRefLatents)

        let out: MLXArray

        // Reference Skip Q-slicing path
        if let ncl = numCondLatents, ncl > 1,
           let nrl = numRefLatents,
           refImgIndex != nil,
           let mfr = maskFrameRange, mfr > 0 {
            let numRefThw = tokensPerFrame
            let nclThw = ncl * tokensPerFrame

            let qRef = q[0..., 0..., 0..<numRefThw]
            let kRef = k[0..., 0..., 0..<numRefThw]
            let vRef = v[0..., 0..., 0..<numRefThw]
            let xRef = processAttn(qRef, kRef, vRef)

            let qCond = q[0..., 0..., numRefThw..<nclThw]
            let kCond = k[0..., 0..., numRefThw..<nclThw]
            let vCond = v[0..., 0..., numRefThw..<nclThw]
            let xCond = processAttn(qCond, kCond, vCond)

            let numNoisyFrames = TLat - ncl
            if ncl == TLat {
                // No noise queries — short-circuit
                out = MLX.concatenated([xRef, xCond], axis: 2)
            } else {
                let qNoise = q[0..., 0..., nclThw...]
                let startNoise = (refImgIndex ?? 0) - mfr - ncl + nrl
                let endNoise = (refImgIndex ?? 0) + mfr - ncl + nrl + 1

                if startNoise >= 0 && endNoise > startNoise && endNoise <= numNoisyFrames {
                    let startPos = startNoise * tokensPerFrame
                    let endPos = endNoise * tokensPerFrame

                    let qNoiseFront = qNoise[0..., 0..., 0..<startPos]
                    let qNoiseMaskref = qNoise[0..., 0..., startPos..<endPos]
                    let qNoiseBack = qNoise[0..., 0..., endPos...]

                    let kNonRef = k[0..., 0..., numRefThw...]
                    let vNonRef = v[0..., 0..., numRefThw...]

                    let xNoiseFront = processAttn(qNoiseFront, k, v)
                    let xNoiseBack = processAttn(qNoiseBack, k, v)
                    let xNoiseMaskref = processAttn(qNoiseMaskref, kNonRef, vNonRef)
                    let xNoise = MLX.concatenated([xNoiseFront, xNoiseMaskref, xNoiseBack], axis: 2)
                    out = MLX.concatenated([xRef, xCond, xNoise], axis: 2)
                } else {
                    let xNoise = processAttn(qNoise, k, v)
                    out = MLX.concatenated([xRef, xCond, xNoise], axis: 2)
                }
            }
        } else if let ncl = numCondLatents, ncl > 0 {
            // Standard cond branching (matches base)
            let nclThw = ncl * tokensPerFrame
            let qCond = q[0..., 0..., 0..<nclThw]
            let kCond = k[0..., 0..., 0..<nclThw]
            let vCond = v[0..., 0..., 0..<nclThw]
            let xCond = processAttn(qCond, kCond, vCond)
            let qNoise = q[0..., 0..., nclThw...]
            let xNoise = processAttn(qNoise, k, v)
            out = MLX.concatenated([xCond, xNoise], axis: 2)
        } else {
            out = processAttn(q, k, v)
        }

        let folded = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        let result = proj(folded)

        // MultiTalk x_ref_attn_map (NOT implemented yet — single-talker is priority)
        if refTargetMasks != nil {
            fatalError("MultiTalk x_ref_attn_map computation not yet ported — single-talker only")
        }

        return (result, kvCacheOut, nil)
    }
}

// MARK: - SingleStreamAttention

/// Audio cross-attention with optional MultiTalk L-RoPE routing.
///
/// Visual tokens are Q (from DiT hidden state). Audio context tokens are K/V
/// (from `AudioProjModel`, dim=output_dim=768). Each video latent frame
/// attends to its own 32 audio tokens.
public final class SingleStreamAttention: Module, @unchecked Sendable {
    public let dim: Int
    public let encoderHiddenStatesDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    public let classInterval: Int
    public let classRange: Int

    @ModuleInfo(key: "q_linear") public var qLinear: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNormFP32?
    public var proj: Linear

    @ModuleInfo(key: "kv_linear") public var kvLinear: Linear
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNormFP32?

    /// 1D RoPE table for MultiTalk L-RoPE routing.
    @ModuleInfo(key: "rope_1d") public var rope1d: RotaryPositionalEmbedding1D

    // L-RoPE position constants
    public let ropeH1: (Int, Int)
    public let ropeH2: (Int, Int)
    public let ropeBak: Int

    public init(
        dim: Int,
        encoderHiddenStatesDim: Int,
        numHeads: Int,
        qkvBias: Bool = true,
        qkNorm: Bool = true,
        eps: Float = 1e-6,
        classRange: Int = 24,
        classInterval: Int = 4
    ) {
        precondition(dim % numHeads == 0, "dim must be divisible by numHeads")
        self.dim = dim
        self.encoderHiddenStatesDim = encoderHiddenStatesDim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = Float(1.0) / Foundation.sqrt(Float(self.headDim))

        self._qLinear.wrappedValue = Linear(dim, dim, bias: qkvBias)
        self._qNorm.wrappedValue = qkNorm ? RMSNormFP32(dim: self.headDim, eps: eps) : nil
        self.proj = Linear(dim, dim)
        self._kvLinear.wrappedValue = Linear(encoderHiddenStatesDim, dim * 2, bias: qkvBias)
        self._kNorm.wrappedValue = qkNorm ? RMSNormFP32(dim: self.headDim, eps: eps) : nil

        self.classInterval = classInterval
        self.classRange = classRange
        self.ropeH1 = (0, classInterval)
        self.ropeH2 = (classRange - classInterval, classRange)
        self.ropeBak = classRange / 2
        self._rope1d.wrappedValue = RotaryPositionalEmbedding1D(headDim: self.headDim)
        super.init()
    }

    private func processCrossAttn(
        _ x: MLXArray,
        cond: MLXArray,
        framesNum: Int,
        xRefAttnMap: MLXArray? = nil,
        humanNum: Int? = nil
    ) -> MLXArray {
        let nT = framesNum
        let outDtype = x.dtype

        // [B, N_t*S, C] → [B*N_t, S, C]
        let xR = x.reshaped(x.dim(0) * nT, -1, x.dim(-1))
        let B = xR.dim(0), N = xR.dim(1), C = xR.dim(2)

        // Q from visual tokens
        var q = qLinear(xR).reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        if let qn = qNorm {
            q = qn(q)
        }

        // MultiTalk routing path — not implemented for single-talker
        if xRefAttnMap != nil {
            fatalError("MultiTalk x_ref_attn_map routing not yet ported in Swift — single-talker only")
        }

        // K, V from audio context tokens
        let nA = cond.dim(1)
        var encoderKV = kvLinear(cond).reshaped(B, nA, 2, numHeads, headDim)
        encoderKV = encoderKV.transposed(2, 0, 3, 1, 4)   // [2, B, H, N_a, D]
        var encoderK = encoderKV[0]
        let encoderV = encoderKV[1]
        if let kn = kNorm {
            encoderK = kn(encoderK)
        }

        // SDPA over audio K/V
        let xAttn = MLXFast.scaledDotProductAttention(
            queries: q, keys: encoderK, values: encoderV,
            scale: scale, mask: .none
        )
        // [B*N_t, H, S, D] → [B*N_t, S, H, D] → [B*N_t, S, C]
        let folded = xAttn.transposed(0, 2, 1, 3).reshaped(B, N, C)
        var result = proj(folded)
        // [B*N_t, S, C] → [B_orig, N_t*S, C]
        result = result.reshaped(B / nT, nT * N, C)
        return result.asType(outDtype)
    }

    /// Forward.
    /// - x: `[B, N_visual, C]`
    /// - cond: `[B*N_t, audio_tokens, C_audio]` (per-frame audio tokens)
    /// - shape: `(T, H, W)`
    /// - Returns: `(audioOutputCond, audioOutputNoise)`. When
    ///   `numCondLatents == 0`, `audioOutputCond` is `nil` and
    ///   `audioOutputNoise` covers all visual tokens. When
    ///   `numCondLatents > 0`, `audioOutputCond` is zeros for the cond
    ///   region and `audioOutputNoise` covers only the noise region.
    public func callAsFunction(
        _ x: MLXArray,
        cond: MLXArray,
        shape: (Int, Int, Int),
        numCondLatents: Int? = nil,
        xRefAttnMap: MLXArray? = nil,
        humanNum: Int? = nil
    ) -> (MLXArray?, MLXArray) {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)

        if numCondLatents == nil || numCondLatents == 0 {
            let output = processCrossAttn(x, cond: cond, framesNum: shape.0, xRefAttnMap: xRefAttnMap, humanNum: humanNum)
            return (nil, output)
        }

        let ncl = numCondLatents!
        precondition(ncl > 0)
        let nclThw = ncl * (N / shape.0)
        let xNoise = x[0..., nclThw...]

        // Drop cond rows from cond: [B*N_t, M, C] → [B, N_t, M, C] → drop first ncl frames
        let condBlock = cond.reshaped(x.dim(0), shape.0, cond.dim(1), cond.dim(2))
        let condRest = condBlock[0..., ncl...]
        let condR = condRest.reshaped(x.dim(0) * (shape.0 - ncl), cond.dim(1), cond.dim(2))

        let framesNum = shape.0 - ncl
        let outputNoise: MLXArray
        if let hn = humanNum, hn >= 2 {
            outputNoise = processCrossAttn(xNoise, cond: condR, framesNum: framesNum, xRefAttnMap: xRefAttnMap, humanNum: hn)
        } else {
            outputNoise = processCrossAttn(xNoise, cond: condR, framesNum: framesNum)
        }
        let outputCond = MLXArray.zeros([B, nclThw, C], dtype: outputNoise.dtype)
        return (outputCond, outputNoise)
    }
}
