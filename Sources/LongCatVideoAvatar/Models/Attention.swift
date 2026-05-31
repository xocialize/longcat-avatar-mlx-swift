//
//  Attention.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/attention.py
//
//  Two classes:
//  - Attention: visual self-attention with QKNorm + 3D RoPE. Supports
//    `numCondLatents` branching for image-to-video / video-continuation.
//    Returns optional KV cache for long-video continuation.
//  - MultiHeadCrossAttention: text cross-attention. Handles variable-
//    length text packing — text tokens for the whole batch are
//    concatenated into a single sequence `[1, N_valid_total, C]` with
//    per-batch `kvSeqlen`. We build a block-diagonal mask for SDPA.
//
//  Uses MLXFast.scaledDotProductAttention (the fused kernel). Per
//  L22 + Whisper finding, fused SDPA is much more deterministic across
//  Python-MLX and Swift-MLX than the manual matmul+softmax chain.
//
//  L18 (Python lessons): SDPA's mask must promote to Q dtype. We build
//  the additive mask in fp32 for sentinel precision then cast to q.dtype.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Attention (visual self-attention)

/// Visual self-attention with QKNorm + 3D RoPE. Matches base
/// `modules/attention.py:Attention`. Avatar overlay adds Reference Skip
/// Q-slicing in `Models/Avatar/AvatarAttention.swift` (S3.7).
public final class Attention: Module, @unchecked Sendable {
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

    /// Dense SDPA wrapper. `q, k, v: [B, H, S, D]`. Returns `[B, H, S_q, D]`.
    private func processAttn(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
    }

    /// Forward.
    /// - x: `[B, N, C]`
    /// - shape: `(T, H, W)`
    /// - returnKV: if true, returns `(out, (kCache, vCache))`
    public func callAsFunction(
        _ x: MLXArray,
        shape: (Int, Int, Int),
        numCondLatents: Int? = nil,
        returnKV: Bool = false
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)

        var qkvOut = qkv(x)
        // [B, N, 3*C] → [B, N, 3, H, D] → [3, B, H, N, D]
        qkvOut = qkvOut.reshaped(B, N, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        q = qNorm(q)
        k = kNorm(k)

        let kCache = k
        let vCache = v

        (q, k) = rope3d(q: q, k: k, gridSize: shape)

        let out: MLXArray
        if let ncl = numCondLatents, ncl > 0 {
            // Image-to-video / video-continuation: process conditioning
            // tokens separately from noise tokens.
            let tokensPerFrame = N / shape.0
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

        // [B, H, N, D] → [B, N, H, D] → [B, N, C]
        let folded = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        let result = proj(folded)
        return (result, returnKV ? (kCache, vCache) : nil)
    }

    /// Chunked continuation path. Mirrors PT `forward_with_kv_cache`.
    public func forwardWithKVCache(
        _ x: MLXArray,
        shape: (Int, Int, Int),
        numCondLatents: Int,
        kvCache: (MLXArray, MLXArray)
    ) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)
        var qkvOut = qkv(x)
        qkvOut = qkvOut.reshaped(B, N, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        q = qNorm(q)
        k = kNorm(k)

        let (T, H, W) = shape
        var kCache = kvCache.0
        var vCache = kvCache.1
        if kCache.dim(0) == 1 && B > 1 {
            var newShape = kCache.shape
            newShape[0] = B
            kCache = MLX.broadcast(kCache, to: newShape)
            vCache = MLX.broadcast(vCache, to: newShape)
        }

        let kFull: MLXArray
        let vFull: MLXArray
        if numCondLatents > 0 {
            kFull = MLX.concatenated([kCache, k], axis: 2)
            vFull = MLX.concatenated([vCache, v], axis: 2)
            // Pad q with zeros for cache positions so RoPE indices align
            let qPaddingIn = MLX.concatenated([MLXArray.zeros(like: kCache), q], axis: 2)
            let (qPadded, _) = rope3d(
                q: qPaddingIn, k: kFull,
                gridSize: (T + numCondLatents, H, W)
            )
            // Slice noise-part of q (the prefix is the zero-padded cache positions)
            q = qPadded[0..., 0..., (qPadded.dim(2) - N)...]
        } else {
            kFull = MLX.concatenated([kCache, k], axis: 2)
            vFull = MLX.concatenated([vCache, v], axis: 2)
        }

        let out = processAttn(q, kFull, vFull)
        let folded = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        return proj(folded)
    }
}

// MARK: - MultiHeadCrossAttention (text cross-attention)

/// Text cross-attention with variable-length text packing.
///
/// PT uses `flash_attn_varlen_func` with `cu_seqlens` to pack variable-
/// length text across the batch into a single sequence
/// `[1, sum(N_valid_i), C]`. MLX equivalent: build a block-diagonal
/// additive mask of shape `[1, 1, B*N_visual, sum(N_valid_i)]` so each
/// batch item's visual queries attend only to its own text slice.
public final class MultiHeadCrossAttention: Module, @unchecked Sendable {
    public let dim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_linear") public var qLinear: Linear
    @ModuleInfo(key: "kv_linear") public var kvLinear: Linear
    public var proj: Linear

    @ModuleInfo(key: "q_norm") public var qNorm: RMSNormFP32
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNormFP32

    public init(dim: Int, numHeads: Int) {
        precondition(dim % numHeads == 0, "dim must be divisible by numHeads")
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = Float(1.0) / Foundation.sqrt(Float(self.headDim))

        self._qLinear.wrappedValue = Linear(dim, dim)
        self._kvLinear.wrappedValue = Linear(dim, dim * 2)
        self.proj = Linear(dim, dim)
        self._qNorm.wrappedValue = RMSNormFP32(dim: self.headDim, eps: 1e-6)
        self._kNorm.wrappedValue = RMSNormFP32(dim: self.headDim, eps: 1e-6)
        super.init()
    }

    private func processCrossAttn(
        _ x: MLXArray,
        cond: MLXArray,
        kvSeqlen: [Int]
    ) -> MLXArray {
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)
        precondition(C == dim && cond.dim(2) == dim)

        // Pack x across batch the same way: [1, B*N, C]
        let xPacked = x.reshaped(1, B * N, C)
        var q = qLinear(xPacked).reshaped(1, B * N, numHeads, headDim)
        let kvOut = kvLinear(cond).reshaped(1, -1, 2, numHeads, headDim)
        var k = kvOut[0..., 0..., 0]
        let v = kvOut[0..., 0..., 1]

        q = qNorm(q)
        k = kNorm(k)

        // [1, S, H, D] → [1, H, S, D] for SDPA
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        // Block-diagonal mask: queries from batch i attend only to
        // KV tokens from batch i. Build in fp32 then cast to q.dtype (L18).
        let sumKV = kvSeqlen.reduce(0, +)
        // MLXArray is a reference type; subscript-assign mutates the buffer
        // even when bound via `let`.
        let maskFp32 = MLXArray.full([B * N, sumKV], values: MLXArray(Float(-3.389e38)), dtype: .float32)
        var kvOffset = 0
        var qOffset = 0
        for ki in kvSeqlen {
            let block = MLXArray.zeros([N, ki], dtype: .float32)
            maskFp32[qOffset..<(qOffset + N), kvOffset..<(kvOffset + ki)] = block
            qOffset += N
            kvOffset += ki
        }
        let mask = maskFp32[.newAxis, .newAxis, 0..., 0...].asType(q.dtype)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: vT,
            scale: scale, mask: .array(mask)
        )
        // [1, H, B*N, D] → [1, B*N, H, D] → [B, N, C]
        let folded = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        return proj(folded)
    }

    /// Forward.
    /// - x: `[B, N, C]` (visual)
    /// - cond: `[1, sum(kvSeqlen), C]` (packed text)
    public func callAsFunction(
        _ x: MLXArray,
        cond: MLXArray,
        kvSeqlen: [Int],
        numCondLatents: Int? = nil,
        shape: (Int, Int, Int)? = nil
    ) -> MLXArray {
        if let ncl = numCondLatents, ncl > 0 {
            precondition(shape != nil, "shape required when numCondLatents > 0")
            let B = x.dim(0), N = x.dim(1), C = x.dim(2)
            let tokensPerFrame = N / shape!.0
            let nclThw = ncl * tokensPerFrame
            let xNoise = x[0..., nclThw...]
            let outNoise = processCrossAttn(xNoise, cond: cond, kvSeqlen: kvSeqlen)
            let zerosCond = MLXArray.zeros([B, nclThw, C], dtype: outNoise.dtype)
            return MLX.concatenated([zerosCond, outNoise], axis: 1)
        }
        return processCrossAttn(x, cond: cond, kvSeqlen: kvSeqlen)
    }
}
