//
//  RoPE3D.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/rope_3d.py
//
//  3D RoPE for video tokens. Head dim is split into three sub-dimensions:
//
//      dim_t = head_dim - 4 * (head_dim // 6)
//      dim_h = 2 * (head_dim // 6)
//      dim_w = 2 * (head_dim // 6)
//
//  For LongCat (head_dim=128): dim_t=44, dim_h=42, dim_w=42 (sum=128).
//
//  Uses manual rotate_half rather than MLXFast.rope because head_dim is
//  partitioned across (T, H, W) axes — MLXFast.rope only handles
//  contiguous-dim cases. Python uses the same manual path. Math is
//  bit-equivalent.
//

import Foundation
import MLX
import MLXNN

/// Pair-wise rotation: `[..., d*r]` reshaped to `[..., d, r=2]`, returns
/// `[-x2, x1]` per pair, then flatten back. Equivalent to PT's
/// `rotate_half` used in standard RoPE.
public func rotateHalf(_ x: MLXArray) -> MLXArray {
    var s = x.shape
    let last = s.removeLast()
    precondition(last % 2 == 0, "rotateHalf requires last dim to be even")
    let pairs = x.reshaped(s + [last / 2, 2])
    let x1 = pairs[.ellipsis, 0]
    let x2 = pairs[.ellipsis, 1]
    let out = MLX.stacked([-x2, x1], axis: -1)
    return out.reshaped(s + [last])
}

// MARK: - RotaryPositionalEmbedding (3D)

/// 3D RoPE for video tokens. Per-grid-size cos/sin tables are memoized.
public final class RotaryPositionalEmbedding: Module, @unchecked Sendable {
    public let headDim: Int
    public let base: Float = 10000

    // Cache key: "T.H.W-frameIndex-numRefLatents"
    private var freqsCache: [String: MLXArray] = [:]

    public init(headDim: Int) {
        precondition(headDim % 8 == 0, "headDim must be a multiple of 8 for 3D RoPE")
        self.headDim = headDim
        super.init()
    }

    private func cacheKey(
        gridSize: (Int, Int, Int),
        frameIndex: Int?,
        numRefLatents: Int?
    ) -> String {
        "\(gridSize.0).\(gridSize.1).\(gridSize.2)-\(frameIndex.map(String.init) ?? "_")-\(numRefLatents.map(String.init) ?? "_")"
    }

    private func precomputeFreqs(
        gridSize: (Int, Int, Int),
        frameIndex: Int?,
        numRefLatents: Int?
    ) -> MLXArray {
        let (numFrames, height, width) = gridSize
        let dimT = headDim - 4 * (headDim / 6)
        let dimH = 2 * (headDim / 6)
        let dimW = 2 * (headDim / 6)

        // Inverse freqs per sub-axis (half-dim each, repeated below)
        let idxT = MLXArray(stride(from: 0, to: dimT, by: 2).map { Float($0) })[0..<(dimT / 2)]
        let idxH = MLXArray(stride(from: 0, to: dimH, by: 2).map { Float($0) })[0..<(dimH / 2)]
        let idxW = MLXArray(stride(from: 0, to: dimW, by: 2).map { Float($0) })[0..<(dimW / 2)]
        var freqsT = MLXArray(Float(1.0)) / (MLXArray(Float(base)) ** (idxT / Float(dimT)))
        var freqsH = MLXArray(Float(1.0)) / (MLXArray(Float(base)) ** (idxH / Float(dimH)))
        var freqsW = MLXArray(Float(1.0)) / (MLXArray(Float(base)) ** (idxW / Float(dimW)))

        // Grid points
        let gridT: MLXArray
        if let frameIndex, let numRefLatents {
            // Reference image at position `frameIndex`, rest is a contiguous
            // range over [0, numFrames - numRefLatents).
            let ref = MLXArray([Float(frameIndex)])
            let cont = MLXArray((0..<(numFrames - numRefLatents)).map { Float($0) })
            gridT = MLX.concatenated([ref, cont], axis: 0)
        } else {
            gridT = MLXArray((0..<numFrames).map { Float($0) })
        }
        let gridH = MLXArray((0..<height).map { Float($0) })
        let gridW = MLXArray((0..<width).map { Float($0) })

        // Outer products → per-position freqs along each axis
        freqsT = gridT[0..., .newAxis] * freqsT[.newAxis, 0...]   // (T, dim_t/2)
        freqsH = gridH[0..., .newAxis] * freqsH[.newAxis, 0...]   // (H, dim_h/2)
        freqsW = gridW[0..., .newAxis] * freqsW[.newAxis, 0...]   // (W, dim_w/2)

        // Repeat each pair (interleave each freq twice so rotateHalf pairs
        // adjacent rotations correctly)
        freqsT = MLX.repeated(freqsT, count: 2, axis: -1)   // (T, dim_t)
        freqsH = MLX.repeated(freqsH, count: 2, axis: -1)   // (H, dim_h)
        freqsW = MLX.repeated(freqsW, count: 2, axis: -1)   // (W, dim_w)

        // Broadcast to (T, H, W, head_dim)
        let T = numFrames, H = height, W = width
        let freqsTb = MLX.broadcast(freqsT[0..., .newAxis, .newAxis, 0...], to: [T, H, W, dimT])
        let freqsHb = MLX.broadcast(freqsH[.newAxis, 0..., .newAxis, 0...], to: [T, H, W, dimH])
        let freqsWb = MLX.broadcast(freqsW[.newAxis, .newAxis, 0..., 0...], to: [T, H, W, dimW])
        let freqs = MLX.concatenated([freqsTb, freqsHb, freqsWb], axis: -1)

        // Flatten to (T*H*W, head_dim)
        return freqs.reshaped(T * H * W, headDim)
    }

    /// Apply 3D RoPE to q and k.
    /// - q, k: `[B, head, seq, head_dim]`
    /// - gridSize: `(T, H, W)`
    /// - Returns: rotated `(q, k)` with the same shape as input.
    public func callAsFunction(
        q: MLXArray,
        k: MLXArray,
        gridSize: (Int, Int, Int),
        frameIndex: Int? = nil,
        numRefLatents: Int? = nil
    ) -> (MLXArray, MLXArray) {
        let key = cacheKey(gridSize: gridSize, frameIndex: frameIndex, numRefLatents: numRefLatents)
        let freqs: MLXArray
        if let cached = freqsCache[key] {
            freqs = cached
        } else {
            let computed = precomputeFreqs(
                gridSize: gridSize,
                frameIndex: frameIndex,
                numRefLatents: numRefLatents
            )
            freqsCache[key] = computed
            freqs = computed
        }
        let cos = MLX.cos(freqs)[.newAxis, .newAxis, 0..., 0...]
        let sin = MLX.sin(freqs)[.newAxis, .newAxis, 0..., 0...]

        let outDtype = q.dtype
        let qF = q.asType(.float32)
        let kF = k.asType(.float32)
        let qR = qF * cos + rotateHalf(qF) * sin
        let kR = kF * cos + rotateHalf(kF) * sin
        return (qR.asType(outDtype), kR.asType(outDtype))
    }
}

// MARK: - RotaryPositionalEmbedding1D

/// 1D RoPE applied at arbitrary positions. Used for MultiTalk human routing.
public final class RotaryPositionalEmbedding1D: Module, @unchecked Sendable {
    public let headDim: Int
    public let base: Float = 10000

    public init(headDim: Int) {
        self.headDim = headDim
        super.init()
    }

    private func precomputeFreqs(posIndices: MLXArray) -> MLXArray {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })[0..<(headDim / 2)]
        var freqs = MLXArray(Float(1.0)) / (MLXArray(Float(base)) ** (idx / Float(headDim)))
        freqs = posIndices.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
        return MLX.repeated(freqs, count: 2, axis: -1)   // (seq, head_dim)
    }

    /// Apply 1D RoPE to x at the given positions.
    /// - x: `[B, head, seq, head_dim]`
    /// - posIndices: `[seq]` integer positions
    /// - Returns: rotated x with the same shape as input.
    public func callAsFunction(_ x: MLXArray, posIndices: MLXArray) -> MLXArray {
        let freqs = precomputeFreqs(posIndices: posIndices)
        let cos = MLX.cos(freqs)[.newAxis, .newAxis, 0..., 0...]
        let sin = MLX.sin(freqs)[.newAxis, .newAxis, 0..., 0...]
        let outDtype = x.dtype
        let xF = x.asType(.float32)
        let xR = xF * cos + rotateHalf(xF) * sin
        return xR.asType(outDtype)
    }
}
