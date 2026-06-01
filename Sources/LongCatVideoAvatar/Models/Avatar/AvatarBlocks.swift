//
//  AvatarBlocks.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/blocks.py
//
//  AudioProjModel — 3-layer MLP audio adapter.
//
//  Ingests grouped-pooled Whisper hidden states and produces 32 audio
//  context tokens per VAE latent frame at dim output_dim=768. Two parallel
//  Linear projections (proj1 for first-frame audio window, proj1_vf for
//  VAE-scale-aligned subsequent frames) feed into a shared proj2 → proj3
//  stack, with optional output LayerNorm.
//
//  v1.5 config:
//    seq_len = audio_window = 5         (Whisper feature window per latent)
//    blocks = audio_block = 5           (group-pooled Whisper layers, NOT 12!)
//    channels = audio_channel = 1280    (Whisper-large hidden dim)
//    intermediate_dim = 512
//    output_dim = 768
//    context_tokens = 32
//    vae_scale = 4
//

import Foundation
import MLX
import MLXNN

/// 3-layer MLP audio adapter. Output: `[B, T_latent, context_tokens, output_dim]`.
public final class AudioProjModel: Module, @unchecked Sendable {
    public let seqLen: Int
    public let seqLenVF: Int
    public let blocksCount: Int
    public let channels: Int
    public let intermediateDim: Int
    public let contextTokens: Int
    public let outputDim: Int
    public let inputDim: Int
    public let inputDimVF: Int
    public let normOutputAudio: Bool

    public var proj1: Linear
    @ModuleInfo(key: "proj1_vf") public var proj1VF: Linear
    public var proj2: Linear
    public var proj3: Linear
    /// PT: `self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()`.
    /// When `norm_output_audio=False`, PT uses Identity (no params, no key).
    /// We mirror with an optional — only the `true` path carries weights.
    public var norm: LayerNorm?

    public init(
        seqLen: Int = 5,
        seqLenVF: Int = 8,    // default seqLen + vae_scale - 1 = 5 + 4 - 1
        blocksCount: Int = 5,
        channels: Int = 1280,
        intermediateDim: Int = 512,
        outputDim: Int = 768,
        contextTokens: Int = 32,
        normOutputAudio: Bool = true
    ) {
        self.seqLen = seqLen
        self.seqLenVF = seqLenVF
        self.blocksCount = blocksCount
        self.channels = channels
        self.inputDim = seqLen * blocksCount * channels
        self.inputDimVF = seqLenVF * blocksCount * channels
        self.intermediateDim = intermediateDim
        self.contextTokens = contextTokens
        self.outputDim = outputDim
        self.normOutputAudio = normOutputAudio

        self.proj1 = Linear(self.inputDim, intermediateDim)
        self._proj1VF.wrappedValue = Linear(self.inputDimVF, intermediateDim)
        self.proj2 = Linear(intermediateDim, intermediateDim)
        self.proj3 = Linear(intermediateDim, contextTokens * outputDim)
        if normOutputAudio {
            self.norm = LayerNorm(dimensions: outputDim)
        }
        super.init()
    }

    /// Forward.
    /// - audioEmbeds: `[B, 1, W=seqLen, S=blocksCount, C=channels]` — first-frame audio window
    /// - audioEmbedsVF: `[B, T-1, W'=seqLenVF, S=blocksCount, C=channels]` — subsequent windows
    /// - Returns: `[B, videoLength, contextTokens, outputDim]`
    public func callAsFunction(
        audioEmbeds: MLXArray,
        audioEmbedsVF: MLXArray
    ) -> MLXArray {
        let videoLength = audioEmbeds.dim(1) + audioEmbedsVF.dim(1)
        let B = audioEmbeds.dim(0)

        // First-frame branch: (B, F, W, S, C) → (B*F, W*S*C)
        let bz = audioEmbeds.dim(0), f = audioEmbeds.dim(1)
        let w = audioEmbeds.dim(2), s = audioEmbeds.dim(3), c = audioEmbeds.dim(4)
        var ae = audioEmbeds.reshaped(bz * f, w * s * c)
        ae = relu(proj1(ae))   // [B*F, intermediate]

        // Latter-frame branch
        let bzV = audioEmbedsVF.dim(0), fV = audioEmbedsVF.dim(1)
        let wV = audioEmbedsVF.dim(2), sV = audioEmbedsVF.dim(3), cV = audioEmbedsVF.dim(4)
        var aeVF = audioEmbedsVF.reshaped(bzV * fV, wV * sV * cV)
        aeVF = relu(proj1VF(aeVF))

        // Reshape back to (B, F, intermediate) and concat over time
        ae = ae.reshaped(B, f, intermediateDim)
        aeVF = aeVF.reshaped(B, fV, intermediateDim)
        var aeC = MLX.concatenated([ae, aeVF], axis: 1)   // [B, T_latent, intermediate]

        // Shared proj2 → proj3 (per-token application via flatten)
        let bc = aeC.dim(0), nT = aeC.dim(1), cA = aeC.dim(2)
        aeC = aeC.reshaped(bc * nT, cA)
        aeC = relu(proj2(aeC))
        var ctx = proj3(aeC).reshaped(bc * nT, contextTokens, outputDim)

        if normOutputAudio, let n = norm {
            ctx = n(ctx)
        }

        return ctx.reshaped(B, videoLength, contextTokens, outputDim)
    }
}
