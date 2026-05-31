//
//  LongCatVideoDiTAvatar.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/avatar/
//               longcat_video_dit_avatar.py
//
//  Avatar 1.5 DiT: base DiT + AudioProjModel + per-block audio cross-attn.
//
//  Module hierarchy mirrors PT — blocks.X.attn, blocks.X.cross_attn,
//  blocks.X.audio_cross_attn, blocks.X.audio_adaLN_modulation.1, etc.
//

import Foundation
import MLX
import MLXNN

// MARK: - LongCatAvatarSingleStreamBlock

/// Avatar DiT block: self-attn + text cross-attn + audio cross-attn + FFN.
///
/// AdaLN-Zero on self-attn (6-param) AND on audio cross-attn output
/// (3-param, via `audio_adaLN_modulation`). Text cross-attn is residual-
/// only (no AdaLN modulation).
public final class LongCatAvatarSingleStreamBlock: Module, @unchecked Sendable {
    public let hiddenSize: Int
    public let audioPrenorm: Bool

    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: [Linear?]
    @ModuleInfo(key: "audio_adaLN_modulation") public var audioAdaLNModulation: [Linear?]

    @ModuleInfo(key: "mod_norm_attn") public var modNormAttn: LayerNormFP32
    @ModuleInfo(key: "mod_norm_ffn") public var modNormFFN: LayerNormFP32
    @ModuleInfo(key: "pre_crs_attn_norm") public var preCrsAttnNorm: LayerNormFP32

    @ModuleInfo(key: "pre_video_crs_attn_norm") public var preVideoCrsAttnNorm: LayerNormFP32
    /// Optional pre-norm for audio side; only present when `audioPrenorm == true`.
    @ModuleInfo(key: "pre_audio_crs_attn_norm") public var preAudioCrsAttnNorm: LayerNormFP32?

    public var attn: AvatarAttention
    @ModuleInfo(key: "cross_attn") public var crossAttn: MultiHeadCrossAttention
    @ModuleInfo(key: "audio_cross_attn") public var audioCrossAttn: SingleStreamAttention
    public var ffn: FeedForwardSwiGLU

    public init(
        hiddenSize: Int,
        numHeads: Int,
        mlpRatio: Int,
        adalnTembedDim: Int,
        outputDim: Int = 768,
        audioPrenorm: Bool = false,
        classRange: Int = 24,
        classInterval: Int = 4
    ) {
        self.hiddenSize = hiddenSize
        self.audioPrenorm = audioPrenorm

        self._adaLNModulation.wrappedValue = [
            nil,
            Linear(adalnTembedDim, 6 * hiddenSize, bias: true),
        ]
        self._audioAdaLNModulation.wrappedValue = [
            nil,
            Linear(adalnTembedDim, 3 * hiddenSize, bias: true),
        ]

        self._modNormAttn.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: false)
        self._modNormFFN.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: false)
        self._preCrsAttnNorm.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: true)

        self._preVideoCrsAttnNorm.wrappedValue = LayerNormFP32(dim: hiddenSize, eps: 1e-6, elementwiseAffine: true)
        if audioPrenorm {
            self._preAudioCrsAttnNorm.wrappedValue = LayerNormFP32(dim: outputDim, eps: 1e-6, elementwiseAffine: true)
        }

        self.attn = AvatarAttention(dim: hiddenSize, numHeads: numHeads)
        self._crossAttn.wrappedValue = MultiHeadCrossAttention(dim: hiddenSize, numHeads: numHeads)
        self._audioCrossAttn.wrappedValue = SingleStreamAttention(
            dim: hiddenSize,
            encoderHiddenStatesDim: outputDim,
            numHeads: numHeads,
            qkvBias: true,
            qkNorm: true,
            classRange: classRange,
            classInterval: classInterval
        )
        self.ffn = FeedForwardSwiGLU(dim: hiddenSize, hiddenDim: hiddenSize * mlpRatio)
        super.init()
    }

    private func maybePreAudioNorm(_ a: MLXArray) -> MLXArray {
        if audioPrenorm, let n = preAudioCrsAttnNorm {
            return n(a)
        }
        return a
    }

    public func callAsFunction(
        _ x: MLXArray,
        y: MLXArray,
        t: MLXArray,
        ySeqlen: [Int],
        latentShape: (Int, Int, Int),
        audioHiddenStates: MLXArray,
        numCondLatents: Int? = nil,
        skipCrsAttn: Bool = false,
        humanNum: Int? = nil,
        numRefLatents: Int? = nil,
        refImgIndex: Int? = nil,
        maskFrameRange: Int? = nil,
        refTargetMasks: MLXArray? = nil
    ) -> MLXArray {
        let xDtype = x.dtype
        let B = x.dim(0), N = x.dim(1), C = x.dim(2)
        let T = latentShape.0

        // AdaLN params (fp32) — t already fp32
        let tIn = silu(t)
        let adaRaw = adaLNModulation[1]!(tIn).asType(.float32)
        let ada = adaRaw[0..., 0..., .newAxis, 0...]   // [B, T, 1, 6*C]
        let parts = MLX.split(ada, parts: 6, axis: -1)
        let shiftMSA = parts[0], scaleMSA = parts[1], gateMSA = parts[2]
        let shiftMLP = parts[3], scaleMLP = parts[4], gateMLP = parts[5]

        let ncl = numCondLatents ?? 0

        // Audio AdaLN: only for the noise region (t[:, ncl:])
        let audioTIn = (ncl > 0) ? silu(t[0..., ncl...]) : tIn
        let audioAdaRaw = audioAdaLNModulation[1]!(audioTIn).asType(.float32)
        let audioAda = audioAdaRaw[0..., 0..., .newAxis, 0...]
        let audioParts = MLX.split(audioAda, parts: 3, axis: -1)
        let audioShiftMCA = audioParts[0]
        let audioScaleMCA = audioParts[1]
        let audioGateMCA = audioParts[2]

        // Self-attn (Avatar's Attention)
        let xMReshaped = x.reshaped(B, T, -1, C)
        let xM = modulateFP32(
            { self.modNormAttn($0) },
            xMReshaped, shift: shiftMSA, scale: scaleMSA
        ).reshaped(B, N, C)
        let (xS, _, xRefAttnMap) = self.attn(
            xM,
            shape: latentShape,
            numCondLatents: numCondLatents,
            numRefLatents: numRefLatents,
            refImgIndex: refImgIndex,
            maskFrameRange: maskFrameRange,
            refTargetMasks: refTargetMasks
        )
        let gateMSAf = gateMSA.asType(.float32)
        let xSf = xS.reshaped(B, T, -1, C).asType(.float32)
        var xOut = (x.asType(.float32) + (gateMSAf * xSf).reshaped(B, N, C)).asType(xDtype)

        // Text cross-attn (no AdaLN)
        if !skipCrsAttn {
            xOut = xOut + self.crossAttn(
                self.preCrsAttnNorm(xOut),
                cond: y,
                kvSeqlen: ySeqlen,
                numCondLatents: numCondLatents,
                shape: latentShape
            )
        }

        // Audio cross-attn (AdaLN-gated output)
        if !skipCrsAttn {
            let videoIn = self.preVideoCrsAttnNorm(xOut)
            let audioIn = self.maybePreAudioNorm(audioHiddenStates)
            let (audioOutCond, audioOutNoise) = self.audioCrossAttn(
                videoIn,
                cond: audioIn,
                shape: latentShape,
                numCondLatents: numCondLatents,
                xRefAttnMap: xRefAttnMap,
                humanNum: humanNum
            )
            let TNoise = T - ncl
            let audioOutNoiseReshaped = audioOutNoise.reshaped(B, TNoise, -1, C)
            let modulated = modulateFP32(
                { self.modNormAttn($0) },
                audioOutNoiseReshaped,
                shift: audioShiftMCA, scale: audioScaleMCA
            ).reshaped(B, -1, C)
            let gateAf = audioGateMCA.asType(.float32)
            var audioAdd = (gateAf * modulated.reshaped(B, TNoise, -1, C).asType(.float32))
                .reshaped(B, -1, C)
            if let aoc = audioOutCond {
                audioAdd = MLX.concatenated([aoc.asType(.float32), audioAdd], axis: 1)
            }
            xOut = (xOut.asType(.float32) + audioAdd).asType(xDtype)
        }

        // FFN
        let xMM = modulateFP32(
            { self.modNormFFN($0) },
            xOut.reshaped(B, T, -1, C),
            shift: shiftMLP, scale: scaleMLP
        ).reshaped(B, N, C)
        let xFFN = self.ffn(xMM)
        let gateMLPf = gateMLP.asType(.float32)
        let xFFNf = xFFN.reshaped(B, T, -1, C).asType(.float32)
        xOut = (xOut.asType(.float32) + (gateMLPf * xFFNf).reshaped(B, N, C)).asType(xDtype)

        return xOut
    }
}

// MARK: - LongCatVideoAvatarConfig

/// Avatar DiT config — superset of LongCatVideoConfig with audio fields.
public struct LongCatVideoAvatarConfig: Codable, Sendable {
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
    // Audio config
    public var audioWindow: Int = 5
    public var audioBlock: Int = 5
    public var audioChannel: Int = 1280
    public var intermediateDim: Int = 512
    public var outputDim: Int = 768
    public var contextTokens: Int = 32
    public var vaeScale: Int = 4
    public var audioPrenorm: Bool = false
    public var classRange: Int = 24
    public var classInterval: Int = 4
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
        case audioWindow = "audio_window"
        case audioBlock = "audio_block"
        case audioChannel = "audio_channel"
        case intermediateDim = "intermediate_dim"
        case outputDim = "output_dim"
        case contextTokens = "context_tokens"
        case vaeScale = "vae_scale"
        case audioPrenorm = "audio_prenorm"
        case classRange = "class_range"
        case classInterval = "class_interval"
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
        audioWindow = try c.decodeIfPresent(Int.self, forKey: .audioWindow) ?? 5
        audioBlock = try c.decodeIfPresent(Int.self, forKey: .audioBlock) ?? 5
        audioChannel = try c.decodeIfPresent(Int.self, forKey: .audioChannel) ?? 1280
        intermediateDim = try c.decodeIfPresent(Int.self, forKey: .intermediateDim) ?? 512
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 768
        contextTokens = try c.decodeIfPresent(Int.self, forKey: .contextTokens) ?? 32
        vaeScale = try c.decodeIfPresent(Int.self, forKey: .vaeScale) ?? 4
        audioPrenorm = try c.decodeIfPresent(Bool.self, forKey: .audioPrenorm) ?? false
        classRange = try c.decodeIfPresent(Int.self, forKey: .classRange) ?? 24
        classInterval = try c.decodeIfPresent(Int.self, forKey: .classInterval) ?? 4
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - LongCatVideoAvatarTransformer3DModel

public final class LongCatVideoAvatarTransformer3DModel: Module, @unchecked Sendable {
    public let patchSize: (Int, Int, Int)
    public let inChannels: Int
    public let outChannels: Int
    public let textTokensZeroPad: Bool
    public let depth: Int
    public let vaeScale: Int
    public let audioWindow: Int

    @ModuleInfo(key: "x_embedder") public var xEmbedder: PatchEmbed3D
    @ModuleInfo(key: "t_embedder") public var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "y_embedder") public var yEmbedder: CaptionEmbedder
    public var blocks: [LongCatAvatarSingleStreamBlock]
    @ModuleInfo(key: "audio_proj") public var audioProj: AudioProjModel
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
        textTokensZeroPad: Bool = false,
        audioWindow: Int = 5,
        audioBlock: Int = 5,
        audioChannel: Int = 1280,
        intermediateDim: Int = 512,
        outputDim: Int = 768,
        contextTokens: Int = 32,
        vaeScale: Int = 4,
        audioPrenorm: Bool = false,
        classRange: Int = 24,
        classInterval: Int = 4
    ) {
        precondition(patchSize.0 == 1, "Temporal patchify must be 1")
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.textTokensZeroPad = textTokensZeroPad
        self.depth = depth
        self.vaeScale = vaeScale
        self.audioWindow = audioWindow

        self._xEmbedder.wrappedValue = PatchEmbed3D(
            patchSize: patchSize, inChans: inChannels, embedDim: hiddenSize
        )
        self._tEmbedder.wrappedValue = TimestepEmbedder(
            tEmbedDim: adalnTembedDim,
            frequencyEmbeddingSize: frequencyEmbeddingSize
        )
        self._yEmbedder.wrappedValue = CaptionEmbedder(
            inChannels: captionChannels, hiddenSize: hiddenSize
        )

        var bs: [LongCatAvatarSingleStreamBlock] = []
        for _ in 0..<depth {
            bs.append(LongCatAvatarSingleStreamBlock(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                mlpRatio: mlpRatio,
                adalnTembedDim: adalnTembedDim,
                outputDim: outputDim,
                audioPrenorm: audioPrenorm,
                classRange: classRange,
                classInterval: classInterval
            ))
        }
        self.blocks = bs

        self._audioProj.wrappedValue = AudioProjModel(
            seqLen: audioWindow,
            seqLenVF: audioWindow + vaeScale - 1,
            blocksCount: audioBlock,
            channels: audioChannel,
            intermediateDim: intermediateDim,
            outputDim: outputDim,
            contextTokens: contextTokens
        )

        let numPatch = patchSize.0 * patchSize.1 * patchSize.2
        self._finalLayer.wrappedValue = FinalLayerFP32(
            hiddenSize: hiddenSize,
            numPatch: numPatch,
            outChannels: outChannels,
            adalnTembedDim: adalnTembedDim
        )
        super.init()
    }

    public static func fromConfig(_ config: LongCatVideoAvatarConfig) -> LongCatVideoAvatarTransformer3DModel {
        let ps = config.patchSize
        return LongCatVideoAvatarTransformer3DModel(
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
            textTokensZeroPad: config.textTokensZeroPad,
            audioWindow: config.audioWindow,
            audioBlock: config.audioBlock,
            audioChannel: config.audioChannel,
            intermediateDim: config.intermediateDim,
            outputDim: config.outputDim,
            contextTokens: config.contextTokens,
            vaeScale: config.vaeScale,
            audioPrenorm: config.audioPrenorm,
            classRange: config.classRange,
            classInterval: config.classInterval
        )
    }

    /// Mirror the audio windowing in PT (longcat_video_dit_avatar.py:420-441).
    /// `audioEmbs` shape: `[B, T_frames, W=5, S=blocks, C]`. Split first-frame
    /// vs later frames, re-window per VAE temporal stride, run AudioProjModel,
    /// reshape to per-latent-frame.
    private func prepareAudioHiddenStates(_ audioEmbs: MLXArray, numRefLatents: Int, nT: Int) -> MLXArray {
        let firstFrameAudio = audioEmbs[0..., 0..<1]   // [B, 1, W, S, C]
        let latterFrameAudio = audioEmbs[0..., 1...]   // [B, T-1, W, S, C]

        let v = vaeScale
        let w = audioWindow
        let middle = w / 2

        let b_ = latterFrameAudio.dim(0)
        let t_ = latterFrameAudio.dim(1)
        let w_ = latterFrameAudio.dim(2)
        let s_ = latterFrameAudio.dim(3)
        let c_ = latterFrameAudio.dim(4)
        let nT_ = t_ / v
        let latter = latterFrameAudio.reshaped(b_, nT_, v, w_, s_, c_)

        // First sub-frame: first 'middle+1' tokens of its W window
        let lFirst = latter[0..., 0..., 0..<1, 0..<(middle + 1)]
        let lFirstFlat = lFirst.reshaped(b_, nT_, (middle + 1) * 1, s_, c_)

        // Middle frames: just the middle token of their W window
        let lMid: MLXArray
        if v > 2 {
            let mid = latter[0..., 0..., 1..<(v - 1), middle..<(middle + 1)]
            lMid = mid.reshaped(b_, nT_, 1 * (v - 2), s_, c_)
        } else {
            lMid = MLXArray.zeros([b_, nT_, 0, s_, c_])
        }

        // Last sub-frame: last 'W - middle' tokens
        let lLast = latter[0..., 0..., (v - 1)..., middle...]
        let lLastFlat = lLast.reshaped(b_, nT_, (w_ - middle) * 1, s_, c_)

        let latterFrameAudioS = MLX.concatenated([lFirstFlat, lMid, lLastFlat], axis: 2)

        var audioHiddenStates = audioProj(
            audioEmbeds: firstFrameAudio,
            audioEmbedsVF: latterFrameAudioS
        )

        if numRefLatents > 0 {
            // Pad with a copy of the first frame for the reference latent
            let audioStartRef = audioHiddenStates[0..., 0..<1]
            audioHiddenStates = MLX.concatenated([audioStartRef, audioHiddenStates], axis: 1)
        }

        // Take the last N_t frames
        let len = audioHiddenStates.dim(1)
        return audioHiddenStates[0..., (len - nT)...]
    }

    /// Forward.
    /// - hiddenStates: `[B, C_in, T, H, W]`
    /// - timestep: `[B]` or `[B, T]`
    /// - encoderHiddenStates: `[B, 1, N_text, C_text]`
    /// - audioEmbs: `[B, T_audio, W, S, C_a]` Whisper-pooled audio
    /// - Returns: `[B, C_out, T, H, W]`
    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        encoderHiddenStates: MLXArray,
        audioEmbs: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        numCondLatents: Int = 0,
        numRefLatents: Int? = nil,
        humanNum: Int? = nil,
        refImgIndex: Int? = nil,
        maskFrameRange: Int? = nil
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(2), H = hiddenStates.dim(3), W = hiddenStates.dim(4)
        let nT = T / patchSize.0
        let nH = H / patchSize.1
        let nW = W / patchSize.2

        var ts = timestep
        if ts.ndim == 1 {
            ts = MLX.broadcast(ts[0..., .newAxis], to: [B, nT])
        }

        let dtype = xEmbedder.proj.weight.dtype
        var hs = hiddenStates.asType(dtype)
        ts = ts.asType(dtype)
        var ehs = encoderHiddenStates.asType(dtype)
        let audioEmbsCast = audioEmbs.asType(dtype)

        hs = xEmbedder(hs)   // [B, N, C]

        let t = tEmbedder(ts.asType(.float32).flattened(), dtype: .float32).reshaped(B, nT, -1)

        ehs = yEmbedder(ehs)   // [B, 1, N_text, C]

        // Audio: per-latent-frame audio context tokens
        var audioHiddenStates = prepareAudioHiddenStates(
            audioEmbsCast,
            numRefLatents: numRefLatents ?? 0,
            nT: nT
        )
        // [B, T_lat, M, C] → [B*T_lat, M, C]
        audioHiddenStates = audioHiddenStates.reshaped(
            B * nT,
            audioHiddenStates.dim(2),
            audioHiddenStates.dim(3)
        )

        // text_tokens_zero_pad + pack
        var attnMask = encoderAttentionMask
        if textTokensZeroPad, let m = attnMask {
            var mb = m
            if mb.ndim == 4 { mb = mb.squeezed(axis: 1).squeezed(axis: 1) }
            ehs = ehs * mb[0..., .newAxis, 0..., .newAxis].asType(ehs.dtype)
            attnMask = MLXArray.ones(like: mb)
        }

        let ySeqlens: [Int]
        if let m = attnMask {
            var m2 = m
            if m2.ndim == 4 { m2 = m2.squeezed(axis: 1).squeezed(axis: 1) }
            else if m2.ndim == 3 { m2 = m2.squeezed(axis: 1) }
            var lens: [Int] = []
            lens.reserveCapacity(B)
            for b in 0..<B {
                lens.append(Int(MLX.sum(m2[b]).item(Int32.self)))
            }
            ySeqlens = lens

            let ehs2 = ehs.squeezed(axis: 1)
            var packedParts: [MLXArray] = []
            for b in 0..<B {
                packedParts.append(ehs2[b, 0..<lens[b]])
            }
            ehs = MLX.concatenated(packedParts, axis: 0)[.newAxis, 0..., 0...]
        } else {
            let ehs2 = ehs.squeezed(axis: 1)
            ySeqlens = Array(repeating: ehs2.dim(1), count: B)
            ehs = ehs2.reshaped(1, -1, ehs2.dim(-1))
        }

        for block in blocks {
            hs = block(
                hs,
                y: ehs,
                t: t,
                ySeqlen: ySeqlens,
                latentShape: (nT, nH, nW),
                audioHiddenStates: audioHiddenStates,
                numCondLatents: numCondLatents,
                humanNum: humanNum,
                numRefLatents: numRefLatents,
                refImgIndex: refImgIndex,
                maskFrameRange: maskFrameRange
            )
        }

        hs = finalLayer(hs, t: t, latentShape: (nT, nH, nW))
        return unpatchify(hs, nT: nT, nH: nH, nW: nW).asType(.float32)
    }

    private func unpatchify(_ x: MLXArray, nT: Int, nH: Int, nW: Int) -> MLXArray {
        let (tP, hP, wP) = patchSize
        let B = x.dim(0)
        let shaped = x.reshaped(B, nT, nH, nW, tP, hP, wP, outChannels)
        let permuted = shaped.transposed(0, 7, 1, 4, 2, 5, 3, 6)
        return permuted.reshaped(B, outChannels, nT * tP, nH * hP, nW * wP)
    }

    /// Download + load the full Avatar DiT weights. No filtering needed —
    /// every key in the published checkpoint maps to a slot in this class.
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> LongCatVideoAvatarTransformer3DModel {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let dir = try WeightLoader.componentDirectory("dit", under: root)
        let config: LongCatVideoAvatarConfig = try WeightLoader.loadConfig(
            LongCatVideoAvatarConfig.self,
            from: dir.appendingPathComponent("config.json")
        )
        let model = LongCatVideoAvatarTransformer3DModel.fromConfig(config)
        let weights = try WeightLoader.loadShardedSafetensors(
            indexURL: dir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        )

        if config.quantization != nil {
            fatalError("S3.7: quantized Avatar DiT load not yet wired — use the bf16 variant repo id")
        }

        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
