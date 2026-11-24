//
//  AvatarClip.swift
//
//  The whole job in one place: reference image + speech + prompt → a clip of
//  the speech's length, as an MP4 with the speech muxed in.
//
//  A clip is a chain of segments (PT demo): the first is image-to-video,
//  `numFrames` frames; each further one continues from the last
//  `numCondFrames` frames of the one before and adds
//  `numFrames - numCondFrames` new frames. At the defaults that is 93 frames
//  (3.72 s), then 80 (3.2 s) per continuation, at 25 fps.
//

import CoreGraphics
import Foundation
import MLX

/// What a clip will be, decided before any model runs.
public struct AvatarClipPlan: Sendable, Equatable {
    public let width: Int
    public let height: Int
    /// Frames in the finished clip.
    public let frames: Int
    /// Frames per segment (`4k + 1`).
    public let segmentFrames: Int
    public let segments: Int
    public let fps: Int

    public init(width: Int, height: Int, frames: Int, segmentFrames: Int, segments: Int, fps: Int) {
        self.width = width; self.height = height; self.frames = frames
        self.segmentFrames = segmentFrames; self.segments = segments; self.fps = fps
    }

    public var seconds: Double { Double(frames) / Double(fps) }
    /// Frames the model generates; the clip keeps the first `frames` of them.
    public func generatedFrames(numCondFrames: Int) -> Int {
        segmentFrames + max(0, segments - 1) * (segmentFrames - numCondFrames)
    }
}

public enum AvatarClipError: Error, CustomStringConvertible {
    case tooShort(frames: Int, minimum: Int)
    case tooLong(frames: Int, maximum: Int)
    case badSize(width: Int, height: Int)

    public var description: String {
        switch self {
        case .tooShort(let frames, let minimum): return "\(frames) frames is below the minimum of \(minimum)"
        case .tooLong(let frames, let maximum): return "\(frames) frames is above the maximum of \(maximum)"
        case .badSize(let width, let height):
            return "\(width)×\(height) is not a size the model makes (both sides must be multiples of 16)"
        }
    }
}

extension LongCatAvatarPipeline {

    /// The shortest clip: one second.
    public static let minimumClipFrames = 25

    /// Plan a clip.
    ///
    /// - frames: frames wanted; nil = the speech's whole length
    ///   (`Int(speechSeconds * fps)`, as PT's `video_length`).
    /// - size: an explicit `width × height`; nil = the `resolution` bucket
    ///   closest to the image's aspect ratio (PT `get_condition_shape`).
    /// - maximumFrames: refuse anything longer.
    public func planClip(
        imageWidth: Int, imageHeight: Int, speechSeconds: Double,
        frames: Int? = nil, resolution: AvatarResolution = .p480,
        size: (width: Int, height: Int)? = nil, maximumFrames: Int? = nil
    ) throws -> AvatarClipPlan {
        try Self.planClip(
            config: config, imageWidth: imageWidth, imageHeight: imageHeight, speechSeconds: speechSeconds,
            frames: frames, resolution: resolution, size: size, maximumFrames: maximumFrames)
    }

    /// `planClip` without a loaded pipeline: planning reads the config only.
    public static func planClip(
        config: PipelineConfig,
        imageWidth: Int, imageHeight: Int, speechSeconds: Double,
        frames: Int? = nil, resolution: AvatarResolution = .p480,
        size: (width: Int, height: Int)? = nil, maximumFrames: Int? = nil
    ) throws -> AvatarClipPlan {
        let wanted = frames ?? Int(speechSeconds * Double(config.targetFps))
        guard wanted >= Self.minimumClipFrames else {
            throw AvatarClipError.tooShort(frames: wanted, minimum: Self.minimumClipFrames)
        }
        if let maximumFrames, wanted > maximumFrames {
            throw AvatarClipError.tooLong(frames: wanted, maximum: maximumFrames)
        }
        let (height, width): (Int, Int)
        if let size {
            // PT `check_inputs`: both sides divisible by vae_scale_spatial * 2.
            guard size.width > 0, size.height > 0, size.width % 16 == 0, size.height % 16 == 0 else {
                throw AvatarClipError.badSize(width: size.width, height: size.height)
            }
            (height, width) = (size.height, size.width)
        } else {
            (height, width) = resolution.size(imageWidth: imageWidth, imageHeight: imageHeight)
        }
        let full = config.numFrames
        let stride = full - config.numCondFrames
        if wanted <= full {
            // One segment, no longer than it has to be (frame counts are 4k + 1).
            let v = config.vaeScaleTemporal
            let segmentFrames = ((wanted - 1 + v - 1) / v) * v + 1
            return AvatarClipPlan(
                width: width, height: height, frames: wanted, segmentFrames: segmentFrames,
                segments: 1, fps: config.targetFps)
        }
        let segments = 1 + (wanted - full + stride - 1) / stride
        return AvatarClipPlan(
            width: width, height: height, frames: wanted, segmentFrames: full,
            segments: segments, fps: config.targetFps)
    }

    /// Generate `plan`'s clip, handing each segment's new frames
    /// (`[1, 3, F, H, W]` in `[-1, 1]`, already cut to the plan's length) to
    /// `onFrames` as it finishes.
    ///
    /// - speech: the driving audio, 16 kHz mono. Cut or zero-padded to the
    ///   generated length plus one second, as upstream does for Avatar 1.6.
    public func generateClip(
        image: CGImage,
        speech: [Float],
        text: TextConditioning,
        plan: AvatarClipPlan,
        seed: UInt64,
        onFrames: (_ segment: Int, _ frames: MLXArray) throws -> Void
    ) throws {
        let reference = try ImageInput.tensor(image, width: plan.width, height: plan.height)
        let generated = plan.generatedFrames(numCondFrames: config.numCondFrames)
        let fitted = AudioInput.fitted(speech, seconds: Double(generated) / Double(config.targetFps) + 1.0)
        let mel = AudioInput.whisperMel(fitted)

        var remaining = plan.frames
        try withoutActuallyEscaping(onFrames) { onFrames in
        _ = try self(
            image: reference,
            audioMel: mel,
            textEmbeds: text.embeds,
            textMask: text.mask,
            uncondEmbeds: text.uncondEmbeds,
            uncondMask: text.uncondMask,
            numFrames: plan.segmentFrames,
            height: plan.height,
            width: plan.width,
            seed: seed,
            numSegments: plan.segments,
            audioSampleCount: fitted.count,
            onSegment: { segment, frames in
                guard remaining > 0 else { return }
                let keep = min(remaining, frames.dim(2))
                try onFrames(segment, keep < frames.dim(2) ? frames[0..., 0..., 0..<keep] : frames)
                remaining -= keep
            }
        )
        }
    }

    /// `generateClip` into an MP4 at `output`, with the speech from
    /// `speechURL` as its soundtrack (cut to the clip's length).
    @discardableResult
    public func generateMP4(
        image: CGImage,
        speechURL: URL,
        text: TextConditioning,
        output: URL,
        frames: Int? = nil,
        resolution: AvatarResolution = .p480,
        size: (width: Int, height: Int)? = nil,
        maximumFrames: Int? = nil,
        seed: UInt64 = 0
    ) async throws -> AvatarClipPlan {
        let speech = try AudioInput.loadSpeech(speechURL)
        let plan = try planClip(
            imageWidth: image.width, imageHeight: image.height,
            speechSeconds: Double(speech.count) / Double(AudioInput.sampleRate),
            frames: frames, resolution: resolution, size: size, maximumFrames: maximumFrames)
        let soundtrack = try Self.soundtrack(speechURL, seconds: plan.seconds)
        let writer = try MP4Writer(
            url: output, width: plan.width, height: plan.height, fps: Double(plan.fps),
            audio: soundtrack, audioSampleRate: Self.soundtrackSampleRate, audioChannels: 1)
        do {
            try generateClip(image: image, speech: speech, text: text, plan: plan, seed: seed) { _, frames in
                try writer.append(video: frames)
            }
        } catch {
            writer.cancel()
            throw error
        }
        try await writer.finish()
        return plan
    }

    public static let soundtrackSampleRate = 48_000.0

    /// The clip's soundtrack: the speech at 48 kHz mono, cut or zero-padded to `seconds`.
    public static func soundtrack(_ speechURL: URL, seconds: Double) throws -> [Float] {
        let samples = try AudioInput.loadPCM(speechURL, sampleRate: soundtrackSampleRate, channels: 1)
        let count = Int((seconds * soundtrackSampleRate).rounded())
        if samples.count >= count { return Array(samples[0..<count]) }
        return samples + [Float](repeating: 0, count: count - samples.count)
    }
}
