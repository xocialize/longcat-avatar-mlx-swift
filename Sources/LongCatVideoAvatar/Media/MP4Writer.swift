//
//  MP4Writer.swift
//
//  RGB frames (+ the driving speech) → H.264 / AAC MP4.
//
//  The audio track is written BEFORE the first video frame. AVAssetWriter
//  interleaves its inputs, and a video input that fills up while the audio
//  input has nothing yet stops accepting frames for good (the fleet's LTX
//  writer hit this; see ltx-2-mlx-swift FrameCodec). The whole speech track is
//  known up front here, so it goes in first and is closed.
//

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX

public enum MP4WriterError: Error, CustomStringConvertible {
    case setup(String)
    case append(String)
    case incomplete(String)

    public var description: String {
        switch self {
        case .setup(let why): return "MP4 writer setup failed: \(why)"
        case .append(let why): return "MP4 writer could not append: \(why)"
        case .incomplete(let why): return "MP4 was not written: \(why)"
        }
    }
}

/// Streams a clip to an MP4 file one batch of frames at a time, so a long
/// clip's frames need not all sit in memory.
///
/// Single-context: drive one instance from one isolation context.
public final class MP4Writer: @unchecked Sendable {
    public let url: URL
    public let width: Int
    public let height: Int
    public let fps: Double
    public private(set) var frameCount = 0

    private let writer: AVAssetWriter
    private let videoInput: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let audioInput: AVAssetWriterInput?
    private let frameDuration: CMTime
    private var audioWritten = false
    private var finished = false

    /// - audio: interleaved Float32 PCM in `[-1, 1]` at `audioSampleRate`
    ///   with `audioChannels` channels; written whole, before any frame.
    public init(
        url: URL, width: Int, height: Int, fps: Double,
        audio: [Float]? = nil, audioSampleRate: Double = 48_000, audioChannels: Int = 1,
        videoBitRate: Int? = nil
    ) throws {
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        // ~0.2 bits per pixel per frame: visually clean for talking-head content.
        let bitRate = videoBitRate ?? max(2_000_000, Int(Double(width * height) * fps * 0.2))
        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitRate,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ] as [String: Any],
        ])
        videoInput.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoInput, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(videoInput) else { throw MP4WriterError.setup("cannot add the video input") }
        writer.add(videoInput)

        if let audio, !audio.isEmpty {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: audioSampleRate,
                AVNumberOfChannelsKey: audioChannels,
                AVEncoderBitRateKey: audioChannels == 1 ? 96_000 : 192_000,
            ])
            input.expectsMediaDataInRealTime = false
            guard writer.canAdd(input) else { throw MP4WriterError.setup("cannot add the audio input") }
            writer.add(input)
            audioInput = input
        } else {
            audioInput = nil
        }

        guard writer.startWriting() else {
            throw MP4WriterError.setup(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)
        frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)

        if let audio, let audioInput {
            try appendAudio(audio, to: audioInput, sampleRate: audioSampleRate, channels: audioChannels)
        }
        audioWritten = true
    }

    private func waitUntilReady(_ input: AVAssetWriterInput, what: String) throws {
        var waited = 0.0
        while !input.isReadyForMoreMediaData {
            if writer.status == .failed {
                throw MP4WriterError.append("\(what): \(writer.error?.localizedDescription ?? "writer failed")")
            }
            usleep(2000)
            waited += 0.002
            if waited > 90 { throw MP4WriterError.append("\(what): the encoder was not ready for 90 s") }
        }
    }

    private func appendAudio(_ samples: [Float], to input: AVAssetWriterInput, sampleRate: Double, channels: Int) throws {
        var description = AudioStreamBasicDescription(
            mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(4 * channels), mFramesPerPacket: 1, mBytesPerFrame: UInt32(4 * channels),
            mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault, asbd: &description, layoutSize: 0, layout: nil,
            magicCookieSize: 0, magicCookie: nil, extensions: nil, formatDescriptionOut: &format) == noErr,
            let format
        else { throw MP4WriterError.setup("audio format description") }

        // One-second buffers: a single multi-minute buffer stalls the AAC encoder.
        let framesPerBuffer = Int(sampleRate)
        let totalFrames = samples.count / channels
        var frame = 0
        while frame < totalFrames {
            let count = min(framesPerBuffer, totalFrames - frame)
            let byteCount = count * channels * 4
            var block: CMBlockBuffer?
            guard CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: byteCount,
                blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
                dataLength: byteCount, flags: 0, blockBufferOut: &block) == kCMBlockBufferNoErr,
                let block
            else { throw MP4WriterError.append("audio block buffer") }
            try samples.withUnsafeBytes { raw in
                guard CMBlockBufferReplaceDataBytes(
                    with: raw.baseAddress! + frame * channels * 4, blockBuffer: block,
                    offsetIntoDestination: 0, dataLength: byteCount) == kCMBlockBufferNoErr
                else { throw MP4WriterError.append("audio copy") }
            }
            var sampleBuffer: CMSampleBuffer?
            guard CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: format,
                sampleCount: count,
                presentationTimeStamp: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(sampleRate)),
                packetDescriptions: nil, sampleBufferOut: &sampleBuffer) == noErr,
                let sampleBuffer
            else { throw MP4WriterError.append("audio sample buffer") }
            try waitUntilReady(input, what: "audio")
            guard input.append(sampleBuffer) else {
                throw MP4WriterError.append("audio: \(writer.error?.localizedDescription ?? "append failed")")
            }
            frame += count
        }
        input.markAsFinished()
    }

    /// Append frames given as RGB bytes: `count` frames of `height × width × 3`, row-major.
    public func append(rgb: [UInt8], count: Int) throws {
        guard !finished else { throw MP4WriterError.append("append after finish") }
        guard rgb.count == count * width * height * 3 else {
            throw MP4WriterError.append("\(rgb.count) bytes is not \(count) frames of \(width)×\(height) RGB")
        }
        guard let pool = adaptor.pixelBufferPool else { throw MP4WriterError.setup("no pixel buffer pool") }
        let frameBytes = width * height * 3
        try rgb.withUnsafeBufferPointer { source in
            for index in 0..<count {
                var buffer: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let buffer else {
                    throw MP4WriterError.append("pixel buffer")
                }
                CVPixelBufferLockBaseAddress(buffer, [])
                let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
                let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
                let frame = source.baseAddress! + index * frameBytes
                for y in 0..<height {
                    let row = base + y * rowBytes, input = frame + y * width * 3
                    for x in 0..<width {
                        row[x * 4] = input[x * 3 + 2]
                        row[x * 4 + 1] = input[x * 3 + 1]
                        row[x * 4 + 2] = input[x * 3]
                        row[x * 4 + 3] = 255
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                try waitUntilReady(videoInput, what: "frame \(frameCount)")
                guard adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(frameCount))) else {
                    throw MP4WriterError.append("frame \(frameCount): \(writer.error?.localizedDescription ?? "append failed")")
                }
                frameCount += 1
            }
        }
    }

    /// Append the pipeline's output: `[1, 3, F, H, W]` in `[-1, 1]`, at most
    /// `limit` frames of it.
    public func append(video: MLXArray, limit: Int? = nil) throws {
        var frames = video
        if let limit, limit < frames.dim(2) { frames = frames[0..., 0..., 0..<limit] }
        let count = frames.dim(2)
        guard count > 0 else { return }
        // [1, 3, F, H, W] → [F, H, W, 3] uint8
        let bytes = MLX.round((MLX.clip(frames[0].transposed(1, 2, 3, 0), min: -1, max: 1) + 1) * 127.5)
            .asType(.uint8)
        try append(rgb: bytes.asArray(UInt8.self), count: count)
    }

    /// Close the container. Returns the file's URL.
    @discardableResult
    public func finish() async throws -> URL {
        guard !finished else { throw MP4WriterError.incomplete("finish called twice") }
        finished = true
        videoInput.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw MP4WriterError.incomplete(writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")
        }
        return url
    }

    /// Abandon the clip and delete the partial file.
    public func cancel() {
        guard !finished else { return }
        finished = true
        writer.cancelWriting()
        try? FileManager.default.removeItem(at: url)
    }
}
