//
//  LongCatAvatarPipeline.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/pipeline_mlx.py
//
//  High-level inference pipeline for LongCat-Video-Avatar-1.5.
//  Wires:
//   - umT5-XXL text encoder
//   - Whisper-large-v3 audio encoder (with group-mean-pool)
//   - Wan VAE (encode reference image; decode final latent)
//   - LongCatVideoAvatarTransformer3DModel DiT
//   - FlowMatchEulerDiscreteScheduler with DMD distilled sigmas
//   - 3-pass disentangled CFG
//
//  Primary v1 mode: AI2V (Image-Audio-to-Video). 93-frame video at 25 fps.
//
//  Critical (ported lessons):
//   - L17: scheduler appends 1.0 sentinel; we MUST overwrite trailing
//     sigma to 0.0 after `setTimesteps`. Otherwise final step re-adds
//     ~88% of the noise back.
//   - Negative velocity flip: `noisePred = -noisePred` before
//     `scheduler.step`. LongCat DiT outputs `-v`.
//

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Resolved config for a pipeline instance.
public struct PipelineConfig: Sendable {
    // DiT
    public var ditInChannels: Int = 16

    // Sampler
    public var numSamplingSteps: Int = 8
    public var numTrainTimesteps: Int = 1000
    public var schedulerShift: Float = 7.0

    // CFG (per DMD distillation defaults)
    public var textGuidanceScale: Float = 4.0
    public var audioGuidanceScale: Float = 4.0

    // Video
    public var numFrames: Int = 93
    public var targetFps: Int = 25

    // Audio
    public var whisperEncFps: Int = 50

    // Latent space
    public var vaeScaleTemporal: Int = 4
    public var vaeScaleSpatial: Int = 8

    public init() {}
}

public final class LongCatAvatarPipeline {

    public let vae: AutoencoderKLWan
    public let textEncoder: UMT5EncoderModel
    public let audioEncoder: WhisperEncoder
    public let dit: LongCatVideoAvatarTransformer3DModel
    public let config: PipelineConfig
    public let scheduler: FlowMatchEulerDiscreteScheduler

    public init(
        vae: AutoencoderKLWan,
        textEncoder: UMT5EncoderModel,
        audioEncoder: WhisperEncoder,
        dit: LongCatVideoAvatarTransformer3DModel,
        config: PipelineConfig = PipelineConfig(),
        scheduler: FlowMatchEulerDiscreteScheduler? = nil
    ) {
        self.vae = vae
        self.textEncoder = textEncoder
        self.audioEncoder = audioEncoder
        self.dit = dit
        self.config = config
        self.scheduler = scheduler ?? FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: config.numTrainTimesteps,
            shift: config.schedulerShift
        )
    }

    // MARK: - Inference

    /// Sample a Gaussian noise tensor in latent space.
    private func makeInitialNoise(
        batchSize: Int, numFrames: Int, height: Int, width: Int, seed: UInt64
    ) -> MLXArray {
        let v = config.vaeScaleTemporal
        let s = config.vaeScaleSpatial
        let TLat = 1 + (numFrames - 1) / v
        let HLat = height / s
        let WLat = width / s
        MLXRandom.seed(seed)
        return MLXRandom.normal([batchSize, config.ditInChannels, TLat, HLat, WLat])
    }

    private func encodeReferenceImage(_ image: MLXArray) -> MLXArray {
        let rawMu = vae.encode(image)
        return vae.normalizeLatents(rawMu)
    }

    private func prepareAudioEmbs(_ audioMel: MLXArray) -> MLXArray {
        let groups = AudioProcess.whisperEncodeAudioToGroups(
            audioEncoder,
            melFeatures: audioMel
        )
        return AudioProcess.buildAvatarAudioEmbeddings(
            groups,
            fps: config.targetFps,
            encFps: config.whisperEncFps
        )
    }

    /// One CFG step: 3-pass forward + disentangled combine + velocity flip.
    private func cfgForward(
        latents: MLXArray,
        timestep: MLXArray,
        textEmbedsCat: MLXArray,
        textMaskCat: MLXArray,
        audioEmbs: MLXArray,
        uncondTextEmbeds: MLXArray,
        uncondTextMask: MLXArray,
        uncondAudioEmbs: MLXArray,
        numCondLatents: Int
    ) -> MLXArray {
        // Pass 1: batched [latents, latents] with [neg_text, pos_text] and pos audio
        let latents2 = MLX.concatenated([latents, latents], axis: 0)
        var ts = timestep
        if ts.ndim == 0 { ts = ts[.newAxis] }
        let timestep2 = MLX.repeated(ts, count: 2, axis: 0)
        let audioEmbs2 = MLX.repeated(audioEmbs, count: 2, axis: 0)
        let pred2 = dit(
            hiddenStates: latents2,
            timestep: timestep2,
            encoderHiddenStates: textEmbedsCat,
            audioEmbs: audioEmbs2,
            encoderAttentionMask: textMaskCat,
            numCondLatents: numCondLatents
        )
        let (noiseUncondText, noiseCond) = Guidance.cfgSplitOutputs(pred2)

        // Pass 2: fully unconditional (no text, no audio)
        let predUncond = dit(
            hiddenStates: latents,
            timestep: ts,
            encoderHiddenStates: uncondTextEmbeds,
            audioEmbs: uncondAudioEmbs,
            encoderAttentionMask: uncondTextMask,
            numCondLatents: numCondLatents
        )

        let combined = Guidance.disentangledCFGCombine(
            noisePredCond: noiseCond,
            noisePredUncondText: noiseUncondText,
            noisePredUncond: predUncond,
            textGuidanceScale: config.textGuidanceScale,
            audioGuidanceScale: config.audioGuidanceScale
        )
        return Guidance.flipVelocityForScheduler(combined)
    }

    /// Full denoising loop.
    /// - image: `[B=1, 3, 1, H, W]` reference frame in `[-1, 1]`
    /// - audioMel: `[B=1, 128, T_mel]` Whisper mel features
    /// - textEmbeds / textMask: positive prompt embeddings + attention mask
    /// - uncondEmbeds / uncondMask: empty/uncond prompt embeddings + mask
    /// - Returns: video `[1, 3, num_frames, H_out, W_out]` in `[-1, 1]`
    public func callAsFunction(
        image: MLXArray,
        audioMel: MLXArray,
        textEmbeds: MLXArray,
        textMask: MLXArray,
        uncondEmbeds: MLXArray,
        uncondMask: MLXArray,
        numFrames: Int? = nil,
        height: Int = 480,
        width: Int = 832,
        seed: UInt64 = 0
    ) -> MLXArray {
        let nFrames = numFrames ?? config.numFrames

        // 1. Encode reference image to latent
        let refLatent = encodeReferenceImage(image)   // [1, 16, 1, H_lat, W_lat]
        let numCondLatents = 1

        // 2. Prepare audio embeddings
        var audioEmbs = prepareAudioEmbs(audioMel)

        // 3. Initial noise + concat ref
        let noise = makeInitialNoise(
            batchSize: 1, numFrames: nFrames, height: height, width: width, seed: seed
        )
        var latents = MLX.concatenated([refLatent, noise], axis: 2)
        let TLatFull = latents.dim(2)

        // 3a. Trim/pad audio_embs to required_audio_T = 1 + vae_scale * (T_lat_full - 1)
        let v = config.vaeScaleTemporal
        let requiredAudioT = 1 + v * (TLatFull - 1)
        if audioEmbs.dim(1) >= requiredAudioT {
            audioEmbs = audioEmbs[0..., 0..<requiredAudioT]
        } else {
            let shortfall = requiredAudioT - audioEmbs.dim(1)
            let lastFrame = audioEmbs[0..., (audioEmbs.dim(1) - 1)..<audioEmbs.dim(1)]
            var padShape = audioEmbs.shape
            padShape[1] = shortfall
            let pad = MLX.broadcast(lastFrame, to: padShape)
            audioEmbs = MLX.concatenated([audioEmbs, pad], axis: 1)
        }

        // 4. Uncond audio = zeros
        let uncondAudio = MLXArray.zeros(like: audioEmbs)

        // 5. Set up DMD distilled scheduler
        let sigmaArr = Guidance.getDMDDistilledSigmas(
            samplingSteps: config.numSamplingSteps,
            numTrainTimesteps: config.numTrainTimesteps
        )
        let sigmaList: [Float] = sigmaArr.asArray(Float.self)
        scheduler.setTimesteps(config.numSamplingSteps, sigmas: sigmaList)

        // L17 fix: scheduler appends 1.0; overwrite trailing sigma to 0.0
        // so final step actually denoises to clean.
        let nSigmas = scheduler.sigmas.dim(0)
        let frontSigmas = scheduler.sigmas[0..<(nSigmas - 1)]
        scheduler.sigmas = MLX.concatenated(
            [frontSigmas, MLXArray([Float(0)])],
            axis: 0
        ).asType(.float32)

        // 6. Stack uncond + cond text for batched CFG pass
        let textEmbedsCat = MLX.concatenated([uncondEmbeds, textEmbeds], axis: 0)
        let textMaskCat = MLX.concatenated([uncondMask, textMask], axis: 0)

        // 7. Denoising loop
        let timesteps = scheduler.timesteps
        for i in 0..<timesteps.dim(0) {
            let t = timesteps[i]
            let tFloat = t.item(Float.self)
            let noisePred = cfgForward(
                latents: latents,
                timestep: t,
                textEmbedsCat: textEmbedsCat,
                textMaskCat: textMaskCat,
                audioEmbs: audioEmbs,
                uncondTextEmbeds: uncondEmbeds,
                uncondTextMask: uncondMask,
                uncondAudioEmbs: uncondAudio,
                numCondLatents: numCondLatents
            )
            latents = scheduler.step(modelOutput: noisePred, timestep: tFloat, sample: latents)
        }

        // 8. Strip the reference latent before decoding
        let denoised = latents[0..., 0..., numCondLatents...]

        // 9. Decode through VAE (denormalize first)
        let zDenorm = vae.denormalizeLatents(denoised)
        return vae.decode(zDenorm)
    }

    /// Convenience: download + load every component from a single HF repo
    /// and wire the pipeline. Use the bf16-dmd-merged variant for the
    /// recommended default (DMD LoRA already merged into the DiT weights).
    public static func fromPretrained(
        _ repoID: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        config: PipelineConfig = PipelineConfig(),
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> LongCatAvatarPipeline {
        async let vae = AutoencoderKLWan.fromPretrained(repoID, includeEncoder: true, progress: progress)
        async let text = UMT5EncoderModel.fromPretrained(repoID, progress: progress)
        async let whisper = WhisperEncoder.fromPretrained(repoID, progress: progress)
        async let dit = LongCatVideoAvatarTransformer3DModel.fromPretrained(repoID, progress: progress)
        return try await LongCatAvatarPipeline(
            vae: vae,
            textEncoder: text,
            audioEncoder: whisper,
            dit: dit,
            config: config
        )
    }
}
