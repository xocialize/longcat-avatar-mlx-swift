import Foundation
import LongCatVideoAvatar
import MLX
import MLXToolKit
import Tokenizers

public enum LongCatAvatarPackageError: Error, CustomStringConvertible {
    case weightsNotProvisioned(missing: String, repo: String)
    case unsupportedQuant(Quant)

    public var description: String {
        switch self {
        case .weightsNotProvisioned(let missing, let repo):
            return "LongCat-Video-Avatar 1.6 weights not found (\(missing)). Through MLXEngine the lane's repo "
                + "(\(repo)) is downloaded on first use; without an engine-managed store, set modelDirectory on "
                + "LongCatAvatarConfiguration."
        case .unsupportedQuant(let quant):
            return "LongCat-Video-Avatar 1.6 has no \(quant.rawValue) lane (bf16, int8 and int4 only)."
        }
    }
}

/// An MLXEngine `textToVideo` package over **LongCat-Video-Avatar 1.6** (Meituan, MIT): an audio-driven talking
/// avatar. A reference image (`initImage`), speech (`initAudio`) and a scene prompt make a clip of the speech's
/// length in which the person in the image speaks it, at 25 fps, with the speech as its soundtrack.
///
/// The model is a 13.6 B-parameter video DiT with an audio cross-attention path, distilled to 8 steps without
/// classifier-free guidance. 1.6 ships two distillation LoRAs over one base and switches between them part-way through
/// the denoise (four DMD steps, then four OPSD steps), so they run as side-paths rather than merged weights.
///
/// A clip is a chain of segments: the first is 93 frames (3.72 s) from the image; each further one continues from the
/// last 13 frames and adds 80 (3.2 s). The image's size picks the output size from upstream's 480p bucket table
/// (480×832 for 16:9, 832×480 for 9:16, 608×640 near square).
///
/// Lifecycle (C13): `load()` pages in the DiT, both LoRAs, the Whisper encoder and the VAE. umT5-XXL (11 GB) is loaded
/// per request and evicted before denoising; the last prompt's features are kept. Cancellation is checked before
/// every denoising step.
@InferenceActor
public final class LongCatAvatarPackage: ModelPackage {
    public typealias Configuration = LongCatAvatarConfiguration

    public nonisolated static let frameRate = 25.0
    nonisolated static let upstreamRepo = "meituan-longcat/LongCat-Video-Avatar-1.6"

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights are MIT (upstream model card). C8: this port is MIT, as upstream's code is.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: upstreamRepo, revision: "main", tier: 3),
            requirements: RequirementsManifest(
                footprints: LongCatAvatarPackage.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [SpecialtyWeight(.general, strength: 0.3)],
            surfaces: [
                T2VContract.descriptor(
                    name: "longcat-avatar-1.6",
                    summary: "LongCat-Video-Avatar 1.6 audio-driven talking avatar (MLX). `initImage` (the person) and "
                        + "`initAudio` (clean speech) are both required; `prompt` describes the scene. The clip is the "
                        + "speech's length unless `numFrames` says otherwise, at 25 fps, with the speech as its "
                        + "soundtrack. Size follows the image's aspect ratio (480p buckets) unless `width` and "
                        + "`height` name one. Distilled 8-step sampling with no CFG: `steps`, `guidanceScale` and "
                        + "`negativePrompt` are refused.",
                    controls: T2VControls(supportsInitAudio: true, audioOutput: .switchable))
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: LongCatAvatarPipeline?
    private var root: URL?
    private var tokenizer: (any Tokenizer)?
    /// The last prompt's umT5 features: an avatar is usually driven line after line with one scene prompt.
    private var cachedPrompt: (prompt: String, embeds: MLXArray, mask: MLXArray)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        guard Configuration.supportedQuants.contains(configuration.quant) else {
            throw LongCatAvatarPackageError.unsupportedQuant(configuration.quant)
        }
        if let missing = configuration.firstMissingFile() {
            throw LongCatAvatarPackageError.weightsNotProvisioned(missing: missing, repo: configuration.repo)
        }
        let root = configuration.resolvedDirectory()!
        var config = PipelineConfig()
        config.mixSteps = configuration.mixSteps
        config.mixAlpha = configuration.mixAlpha
        // umT5 stays out of the resident set: `textEncoder: nil`, loaded per request in `encodePrompt`.
        let pipeline = LongCatAvatarPipeline(
            vae: try AutoencoderKLWan.fromDirectory(root, includeEncoder: true),
            textEncoder: nil,
            audioEncoder: try WhisperEncoder.fromDirectory(root),
            dit: try LongCatVideoAvatarTransformer3DModel.fromDirectory(root),
            config: config)
        try pipeline.attachLoRAMixIfShipped(root: root)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: root.appendingPathComponent("tokenizer"))
        self.root = root
        self.pipeline = pipeline
    }

    public func unload() async {
        pipeline = nil
        tokenizer = nil
        cachedPrompt = nil
        root = nil
        MLX.Memory.clearCache()
    }

    /// umT5 features of `prompt`: loads the encoder, encodes, and lets it go again.
    private func encodePrompt(_ prompt: String) throws -> LongCatAvatarPipeline.TextConditioning {
        if cachedPrompt?.prompt != prompt {
            let encoder = try UMT5EncoderModel.fromDirectory(root!)
            let (embeds, mask) = PromptEncoding.encode(prompt, encoder: encoder, tokenizer: tokenizer!)
            cachedPrompt = (prompt, embeds, mask)
        }
        MLX.Memory.clearCache()   // umT5 is out of scope: hand its 11 GB back before denoising
        let cached = cachedPrompt!
        // No CFG: the unconditional branch is never evaluated.
        return .init(embeds: cached.embeds, mask: cached.mask,
                     uncondEmbeds: MLXArray.zeros(like: cached.embeds), uncondMask: cached.mask)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()  // CAN-1: first act
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .textToVideo, let req = request as? T2VRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let inputs = try Self.checked(req)

        RunProgress.report(.plan)
        let image = try ImageInput.decode(inputs.image)
        let speechURL = try AudioInput.temporaryFile(for: inputs.audio)
        defer { try? FileManager.default.removeItem(at: speechURL) }
        let speech = try AudioInput.loadSpeech(speechURL)
        let plan = try Self.plan(
            req, config: pipeline.config, imageWidth: image.width, imageHeight: image.height,
            speechSeconds: Double(speech.count) / Double(AudioInput.sampleRate),
            maximumFrames: configuration.maximumFrames)

        RunProgress.report(.encode)
        let text = try encodePrompt(req.prompt)
        try Task.checkCancellation()

        let includeAudio = req.includeAudio ?? true
        let soundtrack = includeAudio ? try LongCatAvatarPipeline.soundtrack(speechURL, seconds: plan.seconds) : nil
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("longcat-avatar-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: output) }
        let writer = try MP4Writer(
            url: output, width: plan.width, height: plan.height, fps: Self.frameRate,
            audio: soundtrack, audioSampleRate: LongCatAvatarPipeline.soundtrackSampleRate, audioChannels: 1)

        // One progress scale over the whole clip: 8 steps per segment.
        pipeline.onStep = { segment, step, total in
            try Task.checkCancellation()
            RunProgress.report(.denoise, step: segment * total + step + 1, totalSteps: plan.segments * total,
                               stage: segment + 1, totalStages: plan.segments)
        }
        defer { pipeline.onStep = nil }
        do {
            try pipeline.generateClip(
                image: image, speech: speech, text: text, plan: plan,
                seed: req.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
            ) { _, frames in
                try writer.append(video: frames)
                try Task.checkCancellation()
            }
        } catch {
            writer.cancel()
            MLX.Memory.clearCache()
            throw error
        }
        RunProgress.report(.postprocess)
        try await writer.finish()
        MLX.Memory.clearCache()
        return T2VResponse(video: Video(
            format: .mp4, data: try Data(contentsOf: output),
            durationSeconds: plan.seconds, frameRate: Self.frameRate,
            width: plan.width, height: plan.height, frameCount: plan.frames,
            audio: VideoAudioFacts(
                hasAudio: soundtrack != nil,
                sampleRate: soundtrack != nil ? Int(LongCatAvatarPipeline.soundtrackSampleRate) : nil,
                channels: soundtrack != nil ? 1 : nil)))
    }

    // MARK: - What a request may ask

    /// The request's two required inputs, after refusing what the model cannot honour.
    public nonisolated static func checked(_ req: T2VRequest) throws -> (image: Data, audio: Data) {
        func refuse(_ why: String) -> PackageError { .unsupportedRequestFeature("LongCat-Video-Avatar: " + why) }
        guard let image = req.initImage else { throw refuse("initImage is required (the person who speaks)") }
        guard image.format == .png || image.format == .jpeg else { throw refuse("initImage must be PNG or JPEG") }
        guard let audio = req.initAudio else { throw refuse("initAudio is required (the speech that drives the clip)") }
        if let refs = req.referenceImages, !refs.isEmpty { throw refuse("referenceImages (the person comes from initImage)") }
        if !req.keyframes.isEmpty { throw refuse("keyframes") }
        if !req.adapters.isEmpty { throw refuse("adapters") }
        if let n = req.negativePrompt, !n.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw refuse("negativePrompt (distilled sampling has no classifier-free guidance)")
        }
        if let g = req.guidanceScale, g != 1 { throw refuse("guidanceScale (distilled sampling has no classifier-free guidance)") }
        if let s = req.steps, s != 8 { throw refuse("steps (fixed distilled schedule: 8)") }
        if let fps = req.fps, abs(fps - frameRate) > 0.001 { throw refuse("fps \(fps) (the model's motion is 25 fps)") }
        if (req.width == nil) != (req.height == nil) { throw refuse("width and height go together") }
        return (image.data, audio.data)
    }

    /// The clip a request asks for. Refuses a size outside upstream's bucket tables and a length outside 1 s…`maximumFrames`.
    nonisolated static func plan(
        _ req: T2VRequest, config: PipelineConfig, imageWidth: Int, imageHeight: Int,
        speechSeconds: Double, maximumFrames: Int
    ) throws -> AvatarClipPlan {
        func refuse(_ why: String) -> PackageError { .unsupportedRequestFeature("LongCat-Video-Avatar: " + why) }
        var size: (width: Int, height: Int)?
        if let width = req.width, let height = req.height {
            guard AvatarResolution.allCases.contains(where: { $0.contains(width: width, height: height) }) else {
                throw refuse("\(width)x\(height) is not one of the model's sizes (for example 832x480, 480x832, 640x608, "
                             + "or 1248x736 at 720p)")
            }
            size = (width, height)
        }
        do {
            return try LongCatAvatarPipeline.planClip(
                config: config, imageWidth: imageWidth, imageHeight: imageHeight, speechSeconds: speechSeconds,
                frames: req.numFrames, resolution: .p480, size: size, maximumFrames: maximumFrames)
        } catch let error as AvatarClipError {
            switch error {
            case .tooShort(let frames, let minimum):
                throw refuse(req.numFrames == nil
                    ? "the speech is \(String(format: "%.2f", speechSeconds)) s; the shortest clip is \(minimum) frames (1 s)"
                    : "numFrames \(frames) (at least \(minimum))")
            case .tooLong(let frames, let maximum):
                throw refuse("\(frames) frames (\(String(format: "%.1f", Double(frames) / frameRate)) s) is over this "
                             + "lane's limit of \(maximum); pass a shorter numFrames or raise maximumFrames")
            case .badSize(let width, let height): throw refuse("\(width)x\(height)")
            }
        }
    }
}

extension LongCatAvatarPackage {
    public nonisolated static var registration: PackageRegistration { .of(LongCatAvatarPackage.self) }

    /// Split footprints — MEASURED over GPP on the signed plugin bundle (`gaea-plugin-check`, 2026-11-24, M5 Max
    /// 128 GB, macOS 27.3, release build, mlx-swift 0.33.1; basis = the worker's `phys_footprint`, floor read after
    /// `prepare`, peak = the host's lifetime-peak reading; one fresh worker per run). The envelope run is one full
    /// 93-frame segment at 768×512 (36,864 tokens; the largest 480p bucket is 37,440):
    ///
    ///   lane   floor (post-load)   lifetime peak   run
    ///   bf16   37.75 GB            57.68 GB        565 s
    ///   int8   23.78 GB            43.87 GB        627 s
    ///   int4   16.32 GB            36.25 GB        622 s
    ///
    /// Resident = the floor. Activation = (peak − floor) × 1.2, about 24 GB in every lane: it is the request's own
    /// working set (umT5 paged in and evicted, the 2 GiB attention slices, the VAE's full-resolution feature maps
    /// for one latent frame), which does not depend on the DiT's precision. A longer clip repeats segments, so the
    /// peak does not grow with length. See `docs/footprints.md`.
    nonisolated static var footprints: [QuantFootprint] {
        [
            QuantFootprint(quant: .bf16, residentBytes: 38_000_000_000, peakActivationBytes: 24_000_000_000),
            QuantFootprint(quant: .int8, residentBytes: 24_000_000_000, peakActivationBytes: 24_500_000_000),
            QuantFootprint(quant: .int4, residentBytes: 16_500_000_000, peakActivationBytes: 24_000_000_000),
        ]
    }
}
