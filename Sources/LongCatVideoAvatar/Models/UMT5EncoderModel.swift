//
//  UMT5EncoderModel.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/umt5.py
//
//  UMT5-XXL text encoder — 11 B params, sharded as 3 safetensors. Published
//  weights use the compact MLX names (token_embedding / pos_embedding /
//  blocks.X.attn.{q,k,v,o} / blocks.X.ffn.{gate_proj,fc1,fc2} /
//  blocks.X.norm{1,2} / norm) — the HF/transformers verbose names were
//  pre-renamed at conversion time via `rename_pt_to_mx` in the Python port.
//
//  Compared to vanilla T5:
//  - Per-block relative position bias (`shared_pos=false`)
//  - Gated GeLU FFN with `gate_proj` / `fc1` / `fc2` (T5 1.1 style)
//  - RMSNorm (HF still calls it `T5LayerNorm`)
//  - No bias on Linear projections
//  - No 1/sqrt(d) scaling on QK^T; softmax done in fp32
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - T5LayerNorm

/// RMS-based layer normalization (T5/UMT5 style). Routes through
/// `MLXFast.rmsNorm` (analogous to Python's `mx.fast.rms_norm`).
public final class T5LayerNorm: Module, @unchecked Sendable {
    public let eps: Float
    public let weight: MLXArray

    public init(dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - T5RelativeEmbedding

/// T5-style bucketed relative position bias. For UMT5 (`shared_pos=false`)
/// one instance lives per block.
public final class T5RelativeEmbedding: Module, @unchecked Sendable {
    public let numBuckets: Int
    public let numHeads: Int
    public let bidirectional: Bool
    public let maxDist: Int
    public var embedding: Embedding

    public init(
        numBuckets: Int,
        numHeads: Int,
        bidirectional: Bool = true,
        maxDist: Int = 128
    ) {
        self.numBuckets = numBuckets
        self.numHeads = numHeads
        self.bidirectional = bidirectional
        self.maxDist = maxDist
        self.embedding = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
        super.init()
    }

    private func relativePositionBucket(_ relPos: MLXArray) -> MLXArray {
        var relPos = relPos
        var relBuckets: MLXArray
        let halfBuckets: Int

        if bidirectional {
            halfBuckets = numBuckets / 2
            relBuckets = (relPos .> 0).asType(.int32) * halfBuckets
            relPos = MLX.abs(relPos)
        } else {
            halfBuckets = numBuckets
            relBuckets = MLXArray.zeros(like: relPos).asType(.int32)
            relPos = MLX.maximum(-relPos, MLXArray.zeros(like: relPos))
        }

        let maxExact = halfBuckets / 2
        let isSmall = relPos .< maxExact

        let relPosF = relPos.asType(.float32)
        let logRatio = MLX.log(relPosF / Float(maxExact)) / Float(log(Double(maxDist) / Double(maxExact)))
        var relPosLarge = (MLXArray(Float(maxExact)) + logRatio * Float(halfBuckets - maxExact)).asType(.int32)
        relPosLarge = MLX.minimum(
            relPosLarge,
            MLXArray.full(relPosLarge.shape, values: MLXArray(Int32(halfBuckets - 1)))
        )

        relBuckets = relBuckets + MLX.which(isSmall, relPos.asType(.int32), relPosLarge)
        return relBuckets
    }

    /// Returns `[1, num_heads, lq, lk]`.
    public func callAsFunction(lq: Int, lk: Int) -> MLXArray {
        let positionsK = MLXArray(0..<lk).expandedDimensions(axis: 0)
        let positionsQ = MLXArray(0..<lq).expandedDimensions(axis: 1)
        let relPos = positionsK - positionsQ
        let buckets = relativePositionBucket(relPos)
        let embeds = embedding(buckets)
        return embeds.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}

// MARK: - T5Attention

/// T5/UMT5 self-attention. No 1/sqrt(d) scaling (unscaled QK^T). Softmax
/// in fp32 regardless of the input dtype (per the original T5 paper).
public final class T5Attention: Module, @unchecked Sendable {
    public let dim: Int
    public let dimAttn: Int
    public let numHeads: Int
    public let headDim: Int

    public var q: Linear
    public var k: Linear
    public var v: Linear
    public var o: Linear

    public init(dim: Int, dimAttn: Int, numHeads: Int) {
        precondition(dimAttn % numHeads == 0)
        self.dim = dim
        self.dimAttn = dimAttn
        self.numHeads = numHeads
        self.headDim = dimAttn / numHeads

        self.q = Linear(dim, dimAttn, bias: false)
        self.k = Linear(dim, dimAttn, bias: false)
        self.v = Linear(dim, dimAttn, bias: false)
        self.o = Linear(dimAttn, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let n = numHeads
        let c = headDim

        let qx = q(x).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)
        let kx = k(x).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)
        let vx = v(x).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)

        // T5 convention: NO 1/sqrt(d) scaling, softmax in fp32.
        var attn = MLX.matmul(qx.asType(.float32), kx.asType(.float32).transposed(0, 1, 3, 2))
        if let posBias {
            attn = attn + posBias.asType(.float32)
        }
        if let mask {
            var m = mask
            if m.ndim == 2 { m = m[0..., .newAxis, .newAxis, 0...] }
            else if m.ndim == 3 { m = m[0..., .newAxis, 0..., 0...] }
            let additive = MLX.which(m .== 0, MLXArray(Float(-3.389e38)), MLXArray(Float(0)))
                .asType(.float32)
            attn = attn + additive
        }
        let sm = MLX.softmax(attn, axis: -1).asType(qx.dtype)
        let out = MLX.matmul(sm, vx).transposed(0, 2, 1, 3).reshaped(b, -1, n * c)
        return o(out)
    }
}

// MARK: - T5FeedForward

/// Gated GeLU FFN (T5 1.1 / UMT5): `gate_proj` gates `fc1` → `fc2`.
public final class T5FeedForward: Module, @unchecked Sendable {
    public let dim: Int
    public let dimFFN: Int

    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    public var fc1: Linear
    public var fc2: Linear

    public init(dim: Int, dimFFN: Int) {
        self.dim = dim
        self.dimFFN = dimFFN
        self._gateProj.wrappedValue = Linear(dim, dimFFN, bias: false)
        self.fc1 = Linear(dim, dimFFN, bias: false)
        self.fc2 = Linear(dimFFN, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python uses GELU(approx="tanh") on gate_proj output.
        fc2(fc1(x) * geluApproximate(gateProj(x)))
    }
}

// MARK: - T5SelfAttentionBlock

/// One UMT5 encoder block: pre-LN self-attn + pre-LN gated FFN. For UMT5
/// the per-block relative position bias lives here (`posEmbedding`); for
/// vanilla T5 (`sharedPos=true`) it's `nil` and the encoder passes in the
/// shared bias.
public final class T5SelfAttentionBlock: Module, @unchecked Sendable {
    public let sharedPos: Bool

    public var norm1: T5LayerNorm
    public var attn: T5Attention
    public var norm2: T5LayerNorm
    public var ffn: T5FeedForward

    @ModuleInfo(key: "pos_embedding") public var posEmbedding: T5RelativeEmbedding?

    public init(
        dim: Int,
        dimAttn: Int,
        dimFFN: Int,
        numHeads: Int,
        numBuckets: Int,
        sharedPos: Bool = true
    ) {
        self.sharedPos = sharedPos
        self.norm1 = T5LayerNorm(dim: dim)
        self.attn = T5Attention(dim: dim, dimAttn: dimAttn, numHeads: numHeads)
        self.norm2 = T5LayerNorm(dim: dim)
        self.ffn = T5FeedForward(dim: dim, dimFFN: dimFFN)
        self._posEmbedding.wrappedValue = sharedPos ? nil :
            T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let e: MLXArray? = sharedPos ? posBias : posEmbedding!(lq: x.dim(1), lk: x.dim(1))
        var y = x + attn(norm1(x), mask: mask, posBias: e)
        y = y + ffn(norm2(y))
        return y
    }
}

// MARK: - UMT5EncoderModel

/// HF-style umT5 `config.json`.
public struct UMT5Config: Codable, Sendable {
    public var vocabSize: Int = 256384
    public var dim: Int = 4096
    public var dimFFN: Int = 10240
    public var numHeads: Int = 64
    public var dKV: Int = 64
    public var numLayers: Int = 24
    public var numBuckets: Int = 32

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dim = "d_model"
        case dimFFN = "d_ff"
        case numHeads = "num_heads"
        case dKV = "d_kv"
        case numLayers = "num_layers"
        case numBuckets = "relative_attention_num_buckets"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 256384
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 4096
        dimFFN = try c.decodeIfPresent(Int.self, forKey: .dimFFN) ?? 10240
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 64
        dKV = try c.decodeIfPresent(Int.self, forKey: .dKV) ?? 64
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 24
        numBuckets = try c.decodeIfPresent(Int.self, forKey: .numBuckets) ?? 32
    }
}

public final class UMT5EncoderModel: Module, @unchecked Sendable {
    public let dim: Int
    public let sharedPos: Bool

    @ModuleInfo(key: "token_embedding") public var tokenEmbedding: Embedding
    @ModuleInfo(key: "pos_embedding") public var posEmbedding: T5RelativeEmbedding?
    public var blocks: [T5SelfAttentionBlock]
    public var norm: T5LayerNorm

    public init(
        vocabSize: Int = 256384,
        dim: Int = 4096,
        dimAttn: Int = 4096,
        dimFFN: Int = 10240,
        numHeads: Int = 64,
        numLayers: Int = 24,
        numBuckets: Int = 32,
        sharedPos: Bool = false   // UMT5 default — per-block relative bias
    ) {
        self.dim = dim
        self.sharedPos = sharedPos

        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._posEmbedding.wrappedValue = sharedPos
            ? T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
            : nil
        var bs: [T5SelfAttentionBlock] = []
        for _ in 0..<numLayers {
            bs.append(T5SelfAttentionBlock(
                dim: dim, dimAttn: dimAttn, dimFFN: dimFFN,
                numHeads: numHeads, numBuckets: numBuckets,
                sharedPos: sharedPos
            ))
        }
        self.blocks = bs
        self.norm = T5LayerNorm(dim: dim)
        super.init()
    }

    /// Construct from a HF-style umT5 `config.json`.
    public static func fromConfig(_ config: UMT5Config) -> UMT5EncoderModel {
        UMT5EncoderModel(
            vocabSize: config.vocabSize,
            dim: config.dim,
            dimAttn: config.numHeads * config.dKV,
            dimFFN: config.dimFFN,
            numHeads: config.numHeads,
            numLayers: config.numLayers,
            numBuckets: config.numBuckets,
            sharedPos: false
        )
    }

    /// Forward pass.
    /// - ids:  `[B, L]` token ids
    /// - mask: `[B, L]` attention mask (1=keep, 0=pad); optional
    /// - Returns: `[B, L, dim]` hidden states
    public func callAsFunction(_ ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = tokenEmbedding(ids)
        let e = posEmbedding?(lq: x.dim(1), lk: x.dim(1))
        for block in blocks {
            x = block(x, mask: mask, posBias: e)
        }
        x = norm(x)
        return x
    }

    /// Download + load the published 3-shard umT5 weights into a fully-
    /// initialized model.
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> UMT5EncoderModel {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let dir = try WeightLoader.componentDirectory("text_encoder", under: root)
        let config: UMT5Config = try WeightLoader.loadConfig(
            UMT5Config.self,
            from: dir.appendingPathComponent("config.json")
        )
        let model = UMT5EncoderModel.fromConfig(config)
        let weights = try WeightLoader.loadShardedSafetensors(
            indexURL: dir.appendingPathComponent("model.safetensors.index.json")
        )
        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
