//
//  LongCatAvatarPipeline.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/pipeline_mlx.py
//
//  High-level inference pipeline for LongCat-Video-Avatar 1.5 / 1.6.
//  Wires:
//   - umT5-XXL text encoder
//   - Whisper-large-v3 audio encoder (with group-mean-pool)
//   - Wan VAE (encode reference image; decode final latent)
//   - LongCatVideoAvatarTransformer3DModel DiT
//   - FlowMatchEulerDiscreteScheduler with DMD distilled sigmas
//   - 3-pass disentangled CFG
//   - Avatar 1.6: the DMD + OPSD LoRAs as runtime side-paths, switched per step
//
//  Modes, each mirroring the PT pipeline method of the same name:
//   - generateAI2V: reference image + speech + prompt → one 93-frame segment
//     at 25 fps (3.72 s). The image is the first frame.
//   - generateAVC: audio-driven video continuation, conditioned on the last
//     13 frames of the previous segment plus the reference latent (no KV cache).
//   - callAsFunction: the whole clip (each continuation adds 80 frames, 3.2 s).
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

    // Avatar 1.6 LoRA schedule, used when the variant ships both LoRAs: the
    // first `mixSteps` steps run mixAlpha*DMD + (1-mixAlpha)*OPSD, the rest
    // OPSD only. Upstream's documented default is 4 / 1.0.
    // nil = the variant's own `pipeline_config.json` value (4 / 1.0 if absent).
    public var mixSteps: Int?
    public var mixAlpha: Float?

    // Guidance. The distilled 8-step path runs WITHOUT classifier-free
    // guidance: upstream's demo forces both scales to 1.0 under
    // `--use_distill`, so each step is one conditional DiT pass. 4.0 / 4.0 is
    // the 50-step undistilled default; a scale above 1.0 brings the 3-pass
    // disentangled CFG back.
    public var textGuidanceScale: Float = 1.0
    public var audioGuidanceScale: Float = 1.0

    // Video
    public var numFrames: Int = 93
    public var targetFps: Int = 25

    // Continuation (PT demo defaults for Avatar 1.6): frames of the previous
    // segment a new one is conditioned on, the reference image's temporal RoPE
    // index, and the Reference Skip half-width (0 = off).
    public var numCondFrames: Int = 13
    public var refImgIndex: Int = 10
    public var maskFrameRange: Int = 0

    // Audio
    public var whisperEncFps: Int = 50

    // Memory. MLX keeps freed buffers in a pool for reuse; left unbounded, a
    // run that churns multi-GB attention slices grows the pool until the
    // process's peak is tens of GB above what is live. The pool is capped at
    // this for the length of a run (and restored after); nil leaves it alone.
    public var cacheLimitBytes: Int? = 2 * 1024 * 1024 * 1024

    // Latent space
    public var vaeScaleTemporal: Int = 4
    public var vaeScaleSpatial: Int = 8

    public init() {}
}

public final class LongCatAvatarPipeline {

    public let vae: AutoencoderKLWan
    /// Nil when the host loads umT5 per request and evicts it (11 GB in bf16); see `PromptEncoding`.
    public let textEncoder: UMT5EncoderModel?
    public let audioEncoder: WhisperEncoder
    public let dit: LongCatVideoAvatarTransformer3DModel
    public let config: PipelineConfig
    public let scheduler: FlowMatchEulerDiscreteScheduler
    /// Avatar 1.6's DMD/OPSD schedule; nil for a 1.5 (pre-merged) variant.
    public private(set) var loraMix: LoRAMix?

    /// Called before each denoising step with `(segment, step, totalSteps)`.
    /// Throwing (e.g. `CancellationError`) stops the run.
    public var onStep: ((Int, Int, Int) throws -> Void)?

    /// Avatar 1.6: attach the DMD and OPSD LoRAs to the DiT as runtime
    /// side-paths and switch between them per denoising step (see `LoRAMix`).
    /// Works over a bf16 or a quantized DiT; call after its weights are loaded.
    /// `mixAlpha` / `mixSteps` are the variant's shipped values; the pipeline
    /// config's, when set, win over them.
    @discardableResult
    public func attachLoRAMix(
        dmd: [String: MLXArray],
        opsd: [String: MLXArray],
        mixAlpha: Float? = nil,
        mixSteps: Int? = nil
    ) throws -> LoRAMix {
        let modules = try LoRA.attachRuntimeLoRAs(
            to: dit, loras: [(LoRAMix.dmdKey, dmd), (LoRAMix.opsdKey, opsd)]
        )
        let mix = try LoRAMix(
            modules: modules,
            mixAlpha: config.mixAlpha ?? mixAlpha ?? 1.0,
            mixSteps: config.mixSteps ?? mixSteps ?? 4
        )
        loraMix = mix
        return mix
    }

    public init(
        vae: AutoencoderKLWan,
        textEncoder: UMT5EncoderModel?,
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

    /// PT: `text_guidance_scale > 1.0 or audio_guidance_scale > 1.0`.
    public var doClassifierFreeGuidance: Bool {
        config.textGuidanceScale > 1.0 || config.audioGuidanceScale > 1.0
    }

    /// Conditioning that only video continuation passes to the DiT.
    struct ContinuationArgs {
        var numRefLatents: Int
        var refImgIndex: Int
        var maskFrameRange: Int
    }

    /// The text conditioning of one clip: positive and unconditional prompt
    /// embeddings `[1, 1, N_text, 4096]` and their masks `[1, N_text]`.
    public struct TextConditioning {
        public var embeds: MLXArray
        public var mask: MLXArray
        public var uncondEmbeds: MLXArray
        public var uncondMask: MLXArray
        public init(embeds: MLXArray, mask: MLXArray, uncondEmbeds: MLXArray, uncondMask: MLXArray) {
            self.embeds = embeds; self.mask = mask; self.uncondEmbeds = uncondEmbeds; self.uncondMask = uncondMask
        }
    }

    /// One denoising forward + velocity flip. With both guidance scales at 1.0
    /// or below (the distilled default) this is a single conditional pass, as
    /// in PT; otherwise the 3-pass disentangled CFG. The first
    /// `numCondLatents` latent frames are clean, so their timestep is 0.
    private func cfgForward(
        latents: MLXArray,
        timestep: MLXArray,
        text: TextConditioning,
        audioEmbs: MLXArray,
        numCondLatents: Int,
        continuation: ContinuationArgs?
    ) -> MLXArray {
        let TLat = latents.dim(2)
        let frameIsNoise = MLXArray((0..<TLat).map { Float($0 >= numCondLatents ? 1 : 0) })
        let ts = (MLX.broadcast(timestep.reshaped(1, 1), to: [1, TLat]) * frameIsNoise[.newAxis, 0...])
        func forward(_ hidden: MLXArray, _ t: MLXArray, _ embeds: MLXArray, _ audio: MLXArray, _ mask: MLXArray) -> MLXArray {
            dit(
                hiddenStates: hidden,
                timestep: t,
                encoderHiddenStates: embeds,
                audioEmbs: audio,
                encoderAttentionMask: mask,
                numCondLatents: numCondLatents,
                numRefLatents: continuation?.numRefLatents,
                refImgIndex: continuation?.refImgIndex,
                maskFrameRange: continuation?.maskFrameRange
            )
        }
        if !doClassifierFreeGuidance {
            return Guidance.flipVelocityForScheduler(forward(latents, ts, text.embeds, audioEmbs, text.mask))
        }
        // Pass 1: batched [latents, latents] with [neg_text, pos_text] and pos audio
        let pred2 = forward(
            MLX.concatenated([latents, latents], axis: 0),
            MLX.repeated(ts, count: 2, axis: 0),
            MLX.concatenated([text.uncondEmbeds, text.embeds], axis: 0),
            MLX.repeated(audioEmbs, count: 2, axis: 0),
            MLX.concatenated([text.uncondMask, text.mask], axis: 0)
        )
        let (noiseUncondText, noiseCond) = Guidance.cfgSplitOutputs(pred2)
        // Pass 2: fully unconditional (no text, no audio)
        let predUncond = forward(latents, ts, text.uncondEmbeds, MLXArray.zeros(like: audioEmbs), text.uncondMask)
        let combined = Guidance.disentangledCFGCombine(
            noisePredCond: noiseCond,
            noisePredUncondText: noiseUncondText,
            noisePredUncond: predUncond,
            textGuidanceScale: config.textGuidanceScale,
            audioGuidanceScale: config.audioGuidanceScale
        )
        return Guidance.flipVelocityForScheduler(combined)
    }

    // MARK: - Audio

    /// Whisper mel `[1, 128, T_mel]` (30 s chunks concatenated) → one feature
    /// per video frame, `[T_video, 5, 1280]`, for the whole audio.
    /// `audioSampleCount` is the driving audio's real length at `sampleRate`;
    /// without it the mel's full (padded) extent is used.
    public func encodeAudio(_ audioMel: MLXArray, audioSampleCount: Int? = nil, sampleRate: Int = 16000) -> MLXArray {
        let groups = AudioProcess.whisperEncodeAudioToGroups(audioEncoder, melFeatures: audioMel)
        let videoLength = audioSampleCount.map { Int(Double($0) / Double(sampleRate) * Double(config.targetFps)) }
        let features = AudioProcess.buildFrameAudioFeatures(
            groups, fps: config.targetFps, encFps: config.whisperEncFps, videoLength: videoLength
        )
        MLX.eval(features)
        return features
    }

    // MARK: - Sampling

    private func setDistilledTimesteps() -> MLXArray {
        let sigmaArr = Guidance.getDMDDistilledSigmas(
            samplingSteps: config.numSamplingSteps,
            numTrainTimesteps: config.numTrainTimesteps
        )
        scheduler.setTimesteps(config.numSamplingSteps, sigmas: sigmaArr.asArray(Float.self))
        // L17: the scheduler appends 1.0; overwrite the trailing sigma to 0.0
        // so the final step denoises to clean.
        let nSigmas = scheduler.sigmas.dim(0)
        scheduler.sigmas = MLX.concatenated(
            [scheduler.sigmas[0..<(nSigmas - 1)], MLXArray([Float(0)])], axis: 0
        ).asType(.float32)
        return scheduler.timesteps
    }

    /// The 8-step loop. The first `numCondLatents` latent frames are clean
    /// conditioning: they run at timestep 0 and the scheduler never touches
    /// them (PT steps `latents[:, :, num_cond_latents:]` only).
    private func denoise(
        latents initial: MLXArray,
        numCondLatents: Int,
        audioEmbs: MLXArray,
        text: TextConditioning,
        continuation: ContinuationArgs? = nil,
        segment: Int
    ) throws -> MLXArray {
        let timesteps = setDistilledTimesteps()
        let total = timesteps.dim(0)
        let cond = initial[0..., 0..., 0..<numCondLatents]
        var latents = initial
        for i in 0..<total {
            let t = timesteps[i]
            loraMix?.setStep(i)
            try onStep?(segment, i, total)
            let noisePred = cfgForward(
                latents: latents, timestep: t, text: text, audioEmbs: audioEmbs,
                numCondLatents: numCondLatents, continuation: continuation
            )
            let stepped = scheduler.step(
                modelOutput: noisePred[0..., 0..., numCondLatents...],
                timestep: t.item(Float.self),
                sample: latents[0..., 0..., numCondLatents...]
            )
            latents = MLX.concatenated([cond, stepped], axis: 2)
            // Materialize per step: the graph otherwise grows across all 8 steps.
            MLX.eval(latents)
        }
        return latents
    }

    private func decode(_ latents: MLXArray) -> MLXArray {
        let video = vae.decode(vae.denormalizeLatents(latents))
        MLX.eval(video)
        return video
    }

    /// One segment from a reference image (PT `generate_ai2v`).
    ///
    /// - image: `[1, 3, 1, H, W]` in `[-1, 1]`
    /// - audioEmbs: `[1, numFrames, 5, 5, 1280]` (`AudioProcess.windowAudioFeatures`)
    /// - initialNoise: `[1, 16, 1 + (numFrames - 1) / 4, H / 8, W / 8]`; drawn
    ///   from `seed` when nil. Its first latent frame is replaced by the
    ///   image's latent, so the image is the clip's first frame.
    /// - Returns: `(video [1, 3, numFrames, H, W] in [-1, 1], latents)`
    public func generateAI2V(
        image: MLXArray,
        audioEmbs: MLXArray,
        text: TextConditioning,
        numFrames: Int,
        height: Int,
        width: Int,
        seed: UInt64 = 0,
        initialNoise: MLXArray? = nil,
        segment: Int = 0
    ) throws -> (video: MLXArray, latents: MLXArray) {
        precondition(audioEmbs.dim(1) == numFrames, "audioEmbs covers \(audioEmbs.dim(1)) frames, not \(numFrames)")
        let noise = initialNoise ?? makeInitialNoise(
            batchSize: 1, numFrames: numFrames, height: height, width: width, seed: seed)
        let refLatent = encodeReferenceImage(image).asType(noise.dtype)
        let start = MLX.concatenated([refLatent, noise[0..., 0..., 1...]], axis: 2)
        let latents = try denoise(
            latents: start, numCondLatents: 1, audioEmbs: audioEmbs, text: text, segment: segment)
        return (decode(latents), latents)
    }

    /// The next segment, continuing `video` (PT `generate_avc` without a KV
    /// cache, the path the 1.6 LoRA mix uses).
    ///
    /// The last `numCondFrames` frames of `video` are re-encoded as the
    /// segment's leading latents; `refLatent` (the reference image's latent)
    /// rides in front of them at temporal RoPE index `refImgIndex` and is
    /// dropped again before decoding. The returned video repeats those
    /// `numCondFrames` frames at its head.
    public func generateAVC(
        video: MLXArray,
        refLatent: MLXArray,
        audioEmbs: MLXArray,
        text: TextConditioning,
        numFrames: Int,
        height: Int,
        width: Int,
        seed: UInt64 = 0,
        initialNoise: MLXArray? = nil,
        segment: Int = 1
    ) throws -> (video: MLXArray, latents: MLXArray) {
        precondition(audioEmbs.dim(1) == numFrames, "audioEmbs covers \(audioEmbs.dim(1)) frames, not \(numFrames)")
        let frames = video.dim(2)
        let condFrames = video[0..., 0..., (frames - config.numCondFrames)...]
        let condLatents = vae.normalizeLatents(vae.encode(condFrames))
        let nCond = 1 + (config.numCondFrames - 1) / config.vaeScaleTemporal
        precondition(condLatents.dim(2) == nCond, "\(config.numCondFrames) frames encoded to \(condLatents.dim(2)) latents, not \(nCond)")
        let noise = initialNoise ?? makeInitialNoise(
            batchSize: 1, numFrames: numFrames, height: height, width: width, seed: seed)
        let start = MLX.concatenated(
            [refLatent.asType(noise.dtype), condLatents.asType(noise.dtype), noise[0..., 0..., nCond...]], axis: 2)
        var latents = try denoise(
            latents: start, numCondLatents: nCond + 1, audioEmbs: audioEmbs, text: text,
            continuation: ContinuationArgs(
                numRefLatents: 1, refImgIndex: config.refImgIndex, maskFrameRange: config.maskFrameRange),
            segment: segment)
        latents = latents[0..., 0..., 1...]   // drop the reference latent
        return (decode(latents), latents)
    }

    /// Frames a clip of `numSegments` segments has.
    public func frames(forSegments numSegments: Int, numFrames: Int? = nil) -> Int {
        let n = numFrames ?? config.numFrames
        return n + max(0, numSegments - 1) * (n - config.numCondFrames)
    }

    /// The whole clip: one image-to-video segment, then `numSegments - 1`
    /// continuations. Segment `i` draws its noise from `seed + i`.
    ///
    /// - image: `[1, 3, 1, H, W]` reference frame in `[-1, 1]`
    /// - audioMel: `[1, 128, T_mel]` Whisper mel features
    /// - initialNoise: per-segment noise overriding the seeded draw (parity:
    ///   Python and Swift then see identical noise)
    /// - onSegment: called with each finished segment's NEW frames
    ///   `[1, 3, F, H, W]`; when set, nothing is accumulated and the return
    ///   value is the last segment only (a long clip need not sit in memory)
    /// - Returns: video `[1, 3, frames(forSegments:), H, W]` in `[-1, 1]`
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
        seed: UInt64 = 0,
        numSegments: Int = 1,
        audioSampleCount: Int? = nil,
        initialNoise: [MLXArray]? = nil,
        onSegment: ((_ segment: Int, _ newFrames: MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let nFrames = numFrames ?? config.numFrames
        let savedCacheLimit = MLX.Memory.cacheLimit
        if let limit = config.cacheLimitBytes { MLX.Memory.cacheLimit = limit }
        defer {
            MLX.Memory.cacheLimit = savedCacheLimit
            MLX.Memory.clearCache()
        }
        let frameAudio = encodeAudio(audioMel, audioSampleCount: audioSampleCount)
        MLX.Memory.clearCache()
        let text = TextConditioning(embeds: textEmbeds, mask: textMask, uncondEmbeds: uncondEmbeds, uncondMask: uncondMask)
        func noise(_ segment: Int) -> MLXArray? {
            guard let initialNoise, segment < initialNoise.count else { return nil }
            return initialNoise[segment]
        }

        let first = try generateAI2V(
            image: image,
            audioEmbs: AudioProcess.windowAudioFeatures(frameAudio, start: 0, numFrames: nFrames),
            text: text, numFrames: nFrames, height: height, width: width,
            seed: seed, initialNoise: noise(0), segment: 0)
        try onSegment?(0, first.video)
        if numSegments <= 1 { return first.video }

        let refLatent = first.latents[0..., 0..., 0..<1]
        let stride = nFrames - config.numCondFrames
        var segments: [MLXArray] = onSegment == nil ? [first.video] : []
        var current = first.video
        for segment in 1..<numSegments {
            current = try generateAVC(
                video: current, refLatent: refLatent,
                audioEmbs: AudioProcess.windowAudioFeatures(frameAudio, start: segment * stride, numFrames: nFrames),
                text: text, numFrames: nFrames, height: height, width: width,
                seed: seed + UInt64(segment), initialNoise: noise(segment), segment: segment
            ).video
            let fresh = current[0..., 0..., config.numCondFrames...]
            if let onSegment { try onSegment(segment, fresh) } else { segments.append(fresh) }
            MLX.Memory.clearCache()
        }
        return onSegment == nil ? MLX.concatenated(segments, axis: 2) : current
    }

    /// Convenience: download + load every component from a single HF repo
    /// and wire the pipeline. A 1.6 variant (`LongCat-Video-Avatar-1.6-*`)
    /// carries both LoRAs unmerged and gets the runtime schedule from its
    /// `pipeline_config.json`; a 1.5 `-dmd-merged` variant needs no LoRA.
    public static func fromPretrained(
        _ repoID: String = LongCatAvatarPipeline.defaultRepoID,
        config: PipelineConfig = PipelineConfig(),
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> LongCatAvatarPipeline {
        async let vae = AutoencoderKLWan.fromPretrained(repoID, includeEncoder: true, progress: progress)
        async let text = UMT5EncoderModel.fromPretrained(repoID, progress: progress)
        async let whisper = WhisperEncoder.fromPretrained(repoID, progress: progress)
        async let dit = LongCatVideoAvatarTransformer3DModel.fromPretrained(repoID, progress: progress)
        let pipeline = try await LongCatAvatarPipeline(
            vae: vae,
            textEncoder: text,
            audioEncoder: whisper,
            dit: dit,
            config: config
        )
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        try pipeline.attachLoRAMixIfShipped(root: root)
        return pipeline
    }

    public static let defaultRepoID = "mlx-community/LongCat-Video-Avatar-1.6-bf16"

    /// Load every component from a local variant directory (the layout the
    /// conversion recipe writes) and wire the pipeline, including the 1.6 LoRA
    /// schedule when the variant ships one.
    public static func fromDirectory(
        _ root: URL,
        config: PipelineConfig = PipelineConfig()
    ) throws -> LongCatAvatarPipeline {
        let pipeline = LongCatAvatarPipeline(
            vae: try AutoencoderKLWan.fromDirectory(root, includeEncoder: true),
            textEncoder: try UMT5EncoderModel.fromDirectory(root),
            audioEncoder: try WhisperEncoder.fromDirectory(root),
            dit: try LongCatVideoAvatarTransformer3DModel.fromDirectory(root),
            config: config
        )
        try pipeline.attachLoRAMixIfShipped(root: root)
        return pipeline
    }

    /// The `lora` block of a 1.6 variant's `pipeline_config.json`.
    struct ShippedLoRAConfig: Decodable {
        let mode: String
        let dmd: String
        let opsd: String
        let mixSteps: Int?
        let mixAlpha: Float?
        enum CodingKeys: String, CodingKey {
            case mode, dmd, opsd
            case mixSteps = "mix_steps"
            case mixAlpha = "mix_alpha"
        }
    }

    struct ShippedPipelineConfig: Decodable {
        let lora: ShippedLoRAConfig?
    }

    /// Attach the runtime LoRA schedule when `root/pipeline_config.json`
    /// declares one.
    public func attachLoRAMixIfShipped(root: URL) throws {
        let url = root.appendingPathComponent("pipeline_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let shipped = try JSONDecoder().decode(ShippedPipelineConfig.self, from: Data(contentsOf: url))
        guard let lora = shipped.lora, lora.mode == "runtime_mix" else { return }
        try attachLoRAMix(
            dmd: try WeightLoader.loadSafetensors(url: root.appendingPathComponent(lora.dmd)),
            opsd: try WeightLoader.loadSafetensors(url: root.appendingPathComponent(lora.opsd)),
            mixAlpha: lora.mixAlpha,
            mixSteps: lora.mixSteps
        )
    }
}
