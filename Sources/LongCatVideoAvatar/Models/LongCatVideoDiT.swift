//
//  LongCatVideoDiT.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/longcat_video_dit.py
//
//  Base 48-block LongCat-Video DiT. The Avatar overlay (audio cross-attn,
//  Reference Skip, AudioProjModel) subclasses this in
//  Models/Avatar/LongCatVideoDiTAvatar.swift (S3.7).
//
//  Default config: hidden_size=4096, depth=48, num_heads=32 (head_dim=128),
//  patch_size=(1,2,2), adaln_tembed_dim=512, frequency_embedding_size=256.
//

import Foundation
import MLX
import MLXNN

// MARK: - LongCatSingleStreamBlock

/// One DiT block: self-attn + text cross-attn + SwiGLU FFN.
///
/// AdaLN-Zero modulation on self-attn (6-param: shift/scale/gate × msa/mlp).
/// Text cross-attn is NOT AdaLN-modulated (residual w/ pre-norm only).
public final class LongCatSingleStreamBlock: Module, @unchecked Sendable {
    public let hiddenSize: Int

    /// PT `adaLN_modulation` Sequential: [SiLU, Linear]. Index 0 has no
    /// params; index 1 is the Linear.
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: [Linear?]

    @ModuleInfo(key: "mod_norm_attn") public var modNormAttn: LayerNormFP32
    @ModuleInfo(key: "mod_norm_ffn") public var modNormFFN: LayerNormFP32
    @ModuleInfo(key: "pre_crs_attn_norm") public var preCrsAttnNorm: LayerNormFP32

    public var attn: Attention
    @ModuleInfo(key: "cross_attn") public var crossAttn: MultiHeadCrossAttention
    public var ffn: FeedForwardSwiGLU

    public init(hiddenSize: Int, numHeads: Int, mlpRatio: Int, adalnTembedDim: Int) {
        self.hiddenSize = hiddenSize

        self._adaLNModulation.wrappedValue = [
            nil,
            Linear(adalnTembedDim, 6 * hiddenSize, bias: true),
        ]
        self._modNormAttn.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: false)
        self._modNormFFN.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: false)
        self._preCrsAttnNorm.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: true)

        self.attn = Attention(dim: hiddenSize, numHeads: numHeads)
        self._crossAttn.wrappedValue = MultiHeadCrossAttention(dim: hiddenSize, numHeads: numHeads)
        self.ffn = FeedForwardSwiGLU(dim: hiddenSize, hiddenDim: hiddenSize * mlpRatio)
        super.init()
    }

    /// Forward.
    /// - x: visual tokens `[B, N, C]`
    /// - y: packed text `[1, sum(ySeqlen), C]`
    /// - t: timestep embedding `[B, T, C_t]` (fp32)
    public func callAsFunction(
        _ x: MLXArray,
        y: MLXArray,
        t: MLXArray,
        ySeqlen: [Int],
        latentShape: (Int, Int, Int),
        numCondLatents: Int? = nil,
        returnKV: Bool = false,
        kvCache: (MLXArray, MLXArray)? = nil,
        skipCrsAttn: Bool = false
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let xDtype = x.dtype
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)
        let T = latentShape.0

        // adaLN params (fp32). t is already fp32 by convention.
        let tIn = silu(t)
        var ada = adaLNModulation[1]!(tIn)         // [B, T, 6*C]
        ada = ada[0..., 0..., .newAxis, 0...]       // [B, T, 1, 6*C]
        let parts = MLX.split(ada, parts: 6, axis: -1)
        let shiftMSA = parts[0]
        let scaleMSA = parts[1]
        let gateMSA = parts[2]
        let shiftMLP = parts[3]
        let scaleMLP = parts[4]
        let gateMLP = parts[5]

        // Self-attn with AdaLN modulation
        let xM = modulateFP32(
            { self.modNormAttn($0) },
            x.reshaped(B, T, -1, C),
            shift: shiftMSA, scale: scaleMSA
        ).reshaped(B, N, C)

        var xResult: MLXArray
        var newKV: (MLXArray, MLXArray)? = nil

        if let cache = kvCache {
            xResult = self.attn.forwardWithKVCache(
                xM, shape: latentShape,
                numCondLatents: numCondLatents ?? 0,
                kvCache: cache
            )
            newKV = cache   // forwardWithKVCache doesn't return new cache
        } else if returnKV {
            let (out, kv) = self.attn(xM, shape: latentShape, numCondLatents: numCondLatents, returnKV: true)
            xResult = out
            newKV = kv
        } else {
            let (out, _) = self.attn(xM, shape: latentShape, numCondLatents: numCondLatents, returnKV: false)
            xResult = out
        }

        // Residual with gate (fp32 multiply, then back to x dtype)
        let gateMSAf = gateMSA.asType(.float32)
        let xSf = xResult.reshaped(B, T, -1, C).asType(.float32)
        var xOut = (x.asType(.float32) + (gateMSAf * xSf).reshaped(B, N, C)).asType(xDtype)

        // Text cross-attn (no AdaLN modulation; pre-norm + residual)
        if !skipCrsAttn {
            let nclForCross = (kvCache != nil) ? nil : numCondLatents
            xOut = xOut + self.crossAttn(
                self.preCrsAttnNorm(xOut),
                cond: y,
                kvSeqlen: ySeqlen,
                numCondLatents: nclForCross,
                shape: latentShape
            )
        }

        // FFN with AdaLN modulation
        let xMM = modulateFP32(
            { self.modNormFFN($0) },
            xOut.reshaped(B, T, -1, C),
            shift: shiftMLP, scale: scaleMLP
        ).reshaped(B, N, C)
        let xS = self.ffn(xMM)
        let gateMLPf = gateMLP.asType(.float32)
        let xS2f = xS.reshaped(B, T, -1, C).asType(.float32)
        xOut = (xOut.asType(.float32) + (gateMLPf * xS2f).reshaped(B, N, C)).asType(xDtype)

        return (xOut, returnKV ? newKV : nil)
    }
}

// MARK: - LongCatVideoConfig

/// Meituan-style DiT `config.json`.
public struct LongCatVideoConfig: Codable, Sendable {
    public var inChannels: Int = 16
    public var outChannels: Int = 16
    public var hiddenSize: Int = 4096
    public var depth: Int = 48
    public var numHeads: Int = 32
    public var captionChannels: Int = 4096
    public var mlpRatio: Int = 4
    public var adalnTembedDim: Int = 512
    public var frequencyEmbeddingSize: Int = 256
    public var patchSize: [Int] = [1, 2, 2]
    public var textTokensZeroPad: Bool = false
    public var quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case hiddenSize = "hidden_size"
        case depth
        case numHeads = "num_heads"
        case captionChannels = "caption_channels"
        case mlpRatio = "mlp_ratio"
        case adalnTembedDim = "adaln_tembed_dim"
        case frequencyEmbeddingSize = "frequency_embedding_size"
        case patchSize = "patch_size"
        case textTokensZeroPad = "text_tokens_zero_pad"
        case quantization
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 16
        outChannels = try c.decodeIfPresent(Int.self, forKey: .outChannels) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 48
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 32
        captionChannels = try c.decodeIfPresent(Int.self, forKey: .captionChannels) ?? 4096
        mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 4
        adalnTembedDim = try c.decodeIfPresent(Int.self, forKey: .adalnTembedDim) ?? 512
        frequencyEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .frequencyEmbeddingSize) ?? 256
        patchSize = try c.decodeIfPresent([Int].self, forKey: .patchSize) ?? [1, 2, 2]
        textTokensZeroPad = try c.decodeIfPresent(Bool.self, forKey: .textTokensZeroPad) ?? false
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - LongCatVideoTransformer3DModel

public final class LongCatVideoTransformer3DModel: Module, @unchecked Sendable {
    public let patchSize: (Int, Int, Int)
    public let inChannels: Int
    public let outChannels: Int
    public let textTokensZeroPad: Bool
    public let depth: Int

    @ModuleInfo(key: "x_embedder") public var xEmbedder: PatchEmbed3D
    @ModuleInfo(key: "t_embedder") public var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "y_embedder") public var yEmbedder: CaptionEmbedder
    public var blocks: [LongCatSingleStreamBlock]
    @ModuleInfo(key: "final_layer") public var finalLayer: FinalLayerFP32

    public init(
        inChannels: Int = 16,
        outChannels: Int = 16,
        hiddenSize: Int = 4096,
        depth: Int = 48,
        numHeads: Int = 32,
        captionChannels: Int = 4096,
        mlpRatio: Int = 4,
        adalnTembedDim: Int = 512,
        frequencyEmbeddingSize: Int = 256,
        patchSize: (Int, Int, Int) = (1, 2, 2),
        textTokensZeroPad: Bool = false
    ) {
        precondition(patchSize.0 == 1, "Temporal patchify dim must be 1 (matches Meituan)")

        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.textTokensZeroPad = textTokensZeroPad
        self.depth = depth

        self._xEmbedder.wrappedValue = PatchEmbed3D(
            patchSize: patchSize,
            inChans: inChannels,
            embedDim: hiddenSize
        )
        self._tEmbedder.wrappedValue = TimestepEmbedder(
            tEmbedDim: adalnTembedDim,
            frequencyEmbeddingSize: frequencyEmbeddingSize
        )
        self._yEmbedder.wrappedValue = CaptionEmbedder(
            inChannels: captionChannels,
            hiddenSize: hiddenSize
        )
        var bs: [LongCatSingleStreamBlock] = []
        for _ in 0..<depth {
            bs.append(LongCatSingleStreamBlock(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                mlpRatio: mlpRatio,
                adalnTembedDim: adalnTembedDim
            ))
        }
        self.blocks = bs
        let numPatch = patchSize.0 * patchSize.1 * patchSize.2
        self._finalLayer.wrappedValue = FinalLayerFP32(
            hiddenSize: hiddenSize,
            numPatch: numPatch,
            outChannels: outChannels,
            adalnTembedDim: adalnTembedDim
        )
        super.init()
    }

    public static func fromConfig(_ config: LongCatVideoConfig) -> LongCatVideoTransformer3DModel {
        let ps = config.patchSize
        return LongCatVideoTransformer3DModel(
            inChannels: config.inChannels,
            outChannels: config.outChannels,
            hiddenSize: config.hiddenSize,
            depth: config.depth,
            numHeads: config.numHeads,
            captionChannels: config.captionChannels,
            mlpRatio: config.mlpRatio,
            adalnTembedDim: config.adalnTembedDim,
            frequencyEmbeddingSize: config.frequencyEmbeddingSize,
            patchSize: (ps[0], ps[1], ps[2]),
            textTokensZeroPad: config.textTokensZeroPad
        )
    }

    /// Forward.
    /// - hiddenStates: `[B, C_in, T, H, W]` noisy latent
    /// - timestep: `[B]` or `[B, T]` (per-frame timestep)
    /// - encoderHiddenStates: `[B, 1, N_text, C_text]` text embeddings
    /// - encoderAttentionMask: `[B, 1, 1, N_text]` or `[B, N_text]` valid mask
    /// - Returns: `[B, C_out, T, H, W]`
    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        numCondLatents: Int = 0
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(2), H = hiddenStates.dim(3), W = hiddenStates.dim(4)
        let nT = T / patchSize.0
        let nH = H / patchSize.1
        let nW = W / patchSize.2

        // Expand timestep [B] -> [B, N_t] if needed
        var ts = timestep
        if ts.ndim == 1 {
            ts = MLX.broadcast(ts[0..., .newAxis], to: [B, nT])
        }

        // Take embedder weight dtype as inference dtype
        let dtype = xEmbedder.proj.weight.dtype
        var hs = hiddenStates.asType(dtype)
        ts = ts.asType(dtype)
        var ehs = encoderHiddenStates.asType(dtype)

        hs = xEmbedder(hs)   // [B, N, C]

        // t_embedder runs in fp32 (matches PT amp.autocast(fp32))
        let t = tEmbedder(ts.asType(.float32).flattened(), dtype: .float32)
            .reshaped(B, nT, -1)

        ehs = yEmbedder(ehs)   // [B, 1, N_text, C]

        // Apply text_tokens_zero_pad
        var attnMask = encoderAttentionMask
        if textTokensZeroPad, let m = attnMask {
            var mb = m
            if mb.ndim == 4 {
                mb = mb.squeezed(axis: 1).squeezed(axis: 1)
            }
            ehs = ehs * mb[0..., .newAxis, 0..., .newAxis].asType(ehs.dtype)
            attnMask = MLXArray.ones(like: mb)
        }

        // Pack text across batch: [B, 1, N_text, C] -> [1, sum(valid_per_batch), C]
        let ySeqlens: [Int]
        if let m = attnMask {
            var m2 = m
            if m2.ndim == 4 {
                m2 = m2.squeezed(axis: 1).squeezed(axis: 1)
            } else if m2.ndim == 3 {
                m2 = m2.squeezed(axis: 1)
            }
            var lens: [Int] = []
            lens.reserveCapacity(B)
            for b in 0..<B {
                lens.append(Int(MLX.sum(m2[b]).item(Int32.self)))
            }
            ySeqlens = lens

            let ehs2 = ehs.squeezed(axis: 1)   // [B, N_text, C]
            var packedParts: [MLXArray] = []
            for b in 0..<B {
                packedParts.append(ehs2[b, 0..<lens[b]])
            }
            ehs = MLX.concatenated(packedParts, axis: 0)[.newAxis, 0..., 0...]   // [1, sum_k, C]
        } else {
            let ehs2 = ehs.squeezed(axis: 1)
            ySeqlens = Array(repeating: ehs2.dim(1), count: B)
            ehs = ehs2.reshaped(1, -1, ehs2.dim(-1))
        }

        // Run blocks
        for block in blocks {
            let (out, _) = block(
                hs,
                y: ehs,
                t: t,
                ySeqlen: ySeqlens,
                latentShape: (nT, nH, nW),
                numCondLatents: numCondLatents
            )
            hs = out
        }

        hs = finalLayer(hs, t: t, latentShape: (nT, nH, nW))
        // [B, N, C=T_p*H_p*W_p*C_out] -> [B, C_out, T_p*N_t, H_p*N_h, W_p*N_w]
        return unpatchify(hs, nT: nT, nH: nH, nW: nW).asType(.float32)
    }

    private func unpatchify(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let (tP, hP, wP) = patchSize
        let B = x.dim(0)
        // [B, N_t*N_h*N_w, T_p*H_p*W_p*C_out] -> [B, N_t, N_h, N_w, T_p, H_p, W_p, C_out]
        let shaped = x.reshaped(B, nT, nH, nW, tP, hP, wP, outChannels)
        // Permute to [B, C_out, N_t, T_p, N_h, H_p, N_w, W_p]
        let permuted = shaped.transposed(0, 7, 1, 4, 2, 5, 3, 6)
        return permuted.reshaped(B, outChannels, nT * tP, nH * hP, nW * wP)
    }

    /// Download + load the published DiT weights. Detects quantization
    /// metadata in dit/config.json and applies nn.quantize before loading
    /// the bit-packed shards (mirrors Python L19/L20).
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> LongCatVideoTransformer3DModel {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let dir = try WeightLoader.componentDirectory("dit", under: root)
        let config: LongCatVideoConfig = try WeightLoader.loadConfig(
            LongCatVideoConfig.self,
            from: dir.appendingPathComponent("config.json")
        )
        let model = LongCatVideoTransformer3DModel.fromConfig(config)

        // Sharded safetensors
        let weights = try WeightLoader.loadShardedSafetensors(
            indexURL: dir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )

        // TODO(S3.6): quantization detection — if config.quantization is set,
        // call MLXNN.quantize with skip predicate before update(parameters:).
        // For now, raise if quantized weights are loaded (skip this code path
        // by using the bf16 variant for parity).
        if config.quantization != nil {
            fatalError("S3.6: quantized DiT load path not yet wired through fromPretrained — use the bf16 variant repo id")
        }

        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
