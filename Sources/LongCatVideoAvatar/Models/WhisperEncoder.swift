//
//  WhisperEncoder.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/whisper.py
//
//  Whisper-large-v3 ENCODER ONLY (~0.6 B params, single safetensors).
//  Returns ALL 33 hidden states (input + 32 layer outputs) when called
//  with `returnAllHiddenStates: true` — that's what the LongCat audio
//  pipeline consumes (33 → 5 group pool → 25 Hz interp in S3.5b /
//  AudioProcess.swift).
//
//  2× temporal compression comes from `conv2` (stride=2). For mel input
//  of length `T_mel`, encoder output is `T_enc = T_mel // 2`.
//
//  Property naming uses Swift camelCase with `@ModuleInfo(key:)` to map
//  back to the HF/transformers snake_case keys in the published
//  safetensors (verified by inspecting the published weights' keys).
//
//  See L22 in the Python port's docs/development/skill-lessons.md: the
//  Python-MLX vs Swift-MLX bf16 GPU matmul kernel divergence applies
//  here too, but Whisper has smaller dims (d_model=1280 vs umT5's 4096)
//  and 32 layers instead of 24, so the compounded drift is likely
//  similar in magnitude — verified in the parity test.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - WhisperAttention

/// Whisper self-attention with `q_proj` / `k_proj` (no bias) / `v_proj`
/// / `out_proj`. Uses `MLXFast.scaledDotProductAttention` for the QK^T →
/// softmax → ·V chain — mirrors Python's `mx.fast.scaled_dot_product_attention`.
public final class WhisperAttention: Module, @unchecked Sendable {
    public let dModel: Int
    public let numHeads: Int
    public let headDim: Int

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public init(dModel: Int, numHeads: Int) {
        precondition(dModel % numHeads == 0)
        self.dModel = dModel
        self.numHeads = numHeads
        self.headDim = dModel / numHeads

        // Whisper's HF quirk: no bias on k_proj only.
        self._qProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._kProj.wrappedValue = Linear(dModel, dModel, bias: false)
        self._vProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._outProj.wrappedValue = Linear(dModel, dModel, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)

        let q = qProj(x).reshaped(b, n, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, n, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, n, numHeads, headDim).transposed(0, 2, 1, 3)

        let scale = Float(1.0) / Foundation.sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        let folded = out.transposed(0, 2, 1, 3).reshaped(b, n, dModel)
        return outProj(folded)
    }
}

// MARK: - WhisperEncoderLayer

/// One Whisper encoder layer: pre-LN self-attn + pre-LN FFN (GELU).
public final class WhisperEncoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "self_attn_layer_norm") public var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: WhisperAttention
    @ModuleInfo(key: "final_layer_norm") public var finalLayerNorm: LayerNorm
    public var fc1: Linear
    public var fc2: Linear

    public init(dModel: Int, numHeads: Int, ffnDim: Int) {
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = WhisperAttention(dModel: dModel, numHeads: numHeads)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self.fc1 = Linear(dModel, ffnDim, bias: true)
        self.fc2 = Linear(ffnDim, dModel, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x + selfAttn(selfAttnLayerNorm(x))
        y = y + fc2(gelu(fc1(finalLayerNorm(y))))
        return y
    }
}

// MARK: - WhisperConfig

/// HF-style Whisper config.json. Only the keys the encoder actually uses.
public struct WhisperConfig: Codable, Sendable {
    public var dModel: Int = 1280
    public var encoderLayers: Int = 32
    public var encoderAttentionHeads: Int = 20
    public var encoderFfnDim: Int = 5120
    public var numMelBins: Int = 128
    public var maxSourcePositions: Int = 1500

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        encoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 5120
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
    }
}

// MARK: - WhisperEncoder

public final class WhisperEncoder: Module, @unchecked Sendable {
    public let dModel: Int
    public let maxSourcePositions: Int

    public var conv1: Conv1d
    public var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") public var embedPositions: Embedding
    public var layers: [WhisperEncoderLayer]
    @ModuleInfo(key: "layer_norm") public var layerNorm: LayerNorm

    public init(
        dModel: Int = 1280,
        numLayers: Int = 32,
        numHeads: Int = 20,
        ffnDim: Int = 5120,
        numMelBins: Int = 128,
        maxSourcePositions: Int = 1500
    ) {
        self.dModel = dModel
        self.maxSourcePositions = maxSourcePositions

        self.conv1 = Conv1d(
            inputChannels: numMelBins, outputChannels: dModel,
            kernelSize: 3, padding: 1
        )
        self.conv2 = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: 3, stride: 2, padding: 1
        )
        self._embedPositions.wrappedValue = Embedding(
            embeddingCount: maxSourcePositions, dimensions: dModel
        )
        var ls: [WhisperEncoderLayer] = []
        for _ in 0..<numLayers {
            ls.append(WhisperEncoderLayer(dModel: dModel, numHeads: numHeads, ffnDim: ffnDim))
        }
        self.layers = ls
        self._layerNorm.wrappedValue = LayerNorm(dimensions: dModel)
        super.init()
    }

    public static func fromConfig(_ config: WhisperConfig) -> WhisperEncoder {
        WhisperEncoder(
            dModel: config.dModel,
            numLayers: config.encoderLayers,
            numHeads: config.encoderAttentionHeads,
            ffnDim: config.encoderFfnDim,
            numMelBins: config.numMelBins,
            maxSourcePositions: config.maxSourcePositions
        )
    }

    /// Forward pass.
    /// - melFeatures: `[B, num_mel_bins, T_mel]` (channel-second, matches HF input).
    /// - Returns: post-LayerNorm last hidden state, `[B, T_enc, d_model]` where
    ///   `T_enc = T_mel // 2`. Use `allHiddenStates(...)` to get the 33-tensor stack
    ///   that the LongCat audio pipeline consumes.
    public func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let allHidden = encode(melFeatures, returnAll: false)
        return allHidden.last!
    }

    /// Convenience: returns the stack of 33 hidden states (post conv frontend
    /// + 32 transformer layers, last item unnormalized).
    public func allHiddenStates(_ melFeatures: MLXArray) -> [MLXArray] {
        encode(melFeatures, returnAll: true)
    }

    /// Internal: runs the full forward and optionally collects every layer's
    /// output. Final LayerNorm is applied to the LAST item only when
    /// `returnAll: false` (mirrors HF: `hidden_states[i]` are pre-final-LN;
    /// `.last_hidden_state` is post-LN).
    private func encode(_ melFeatures: MLXArray, returnAll: Bool) -> [MLXArray] {
        // [B, mel, T_mel] → [B, T_mel, mel] for mlx-swift Conv1d (NLC layout)
        var x = melFeatures.transposed(0, 2, 1)
        x = gelu(conv1(x))
        x = gelu(conv2(x))
        // T_enc = T_mel // 2 after stride-2 conv2

        let tEnc = x.dim(1)
        // embed_positions.weight has shape (max_source_positions, d_model)
        let pos = embedPositions.weight[0..<tEnc].expandedDimensions(axis: 0)
        x = x + pos

        var hidden: [MLXArray] = []
        if returnAll {
            hidden.append(x)
        }
        for layer in layers {
            x = layer(x)
            if returnAll {
                hidden.append(x)
            }
        }
        if returnAll {
            return hidden
        }
        return [layerNorm(x)]
    }

    /// Download (if needed) + load the published Whisper encoder weights.
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> WhisperEncoder {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let dir = try WeightLoader.componentDirectory("audio_encoder", under: root)
        let config: WhisperConfig = try WeightLoader.loadConfig(
            WhisperConfig.self,
            from: dir.appendingPathComponent("config.json")
        )
        let model = WhisperEncoder.fromConfig(config)
        let weights = try WeightLoader.loadSafetensors(
            url: dir.appendingPathComponent("model.safetensors")
        )
        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
