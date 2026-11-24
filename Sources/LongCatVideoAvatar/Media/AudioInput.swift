//
//  AudioInput.swift
//
//  Driving speech → what the pipeline consumes: 16 kHz mono samples, and the
//  Whisper-large-v3 log-mel in 30 s chunks (PT `get_audio_embedding_whisper`:
//  `MEL_CHUNK = 750 * 640` samples per feature-extractor call, chunks
//  concatenated along time).
//
//  Avatar 1.6 specifics (PT demo, `--model_type avatar-v1.6`):
//  - no loudness normalization;
//  - the audio is cut, or zero-padded, to the clip's duration plus one second.
//
//  Upstream also runs a vocal separator (Kim_Vocal_2) first. It is not ported:
//  give the model clean speech.
//

import AVFoundation
import Foundation
import MLX
import WhisperMLX

public enum AudioInputError: Error, CustomStringConvertible {
    case unreadable(String)
    case empty

    public var description: String {
        switch self {
        case .unreadable(let why): return "the driving audio could not be read: \(why)"
        case .empty: return "the driving audio has no samples"
        }
    }
}

public enum AudioInput {
    public static let sampleRate = 16_000

    /// Any AVFoundation-readable file → interleaved Float32 PCM at
    /// `sampleRate` with `channels` channels.
    public static func loadPCM(_ url: URL, sampleRate: Double, channels: Int) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw AudioInputError.unreadable("\(error)") }
        guard file.length > 0 else { throw AudioInputError.empty }
        guard let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels), interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: outFormat),
              let inBuffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw AudioInputError.unreadable("format setup failed") }
        do { try file.read(into: inBuffer) } catch { throw AudioInputError.unreadable("\(error)") }

        let ratio = sampleRate / file.processingFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(file.length) * ratio) + 4096
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
            throw AudioInputError.unreadable("buffer setup failed")
        }
        var fed = false
        var conversionError: NSError?
        converter.convert(to: out, error: &conversionError) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuffer
        }
        if let conversionError { throw AudioInputError.unreadable(conversionError.localizedDescription) }
        let frames = Int(out.frameLength)
        guard frames > 0, let data = out.floatChannelData else { throw AudioInputError.empty }
        var interleaved = [Float](repeating: 0, count: frames * channels)
        for channel in 0..<channels {
            let source = data[channel]
            for frame in 0..<frames { interleaved[frame * channels + channel] = source[frame] }
        }
        return interleaved
    }

    /// The driving audio as the model hears it: 16 kHz mono.
    public static func loadSpeech(_ url: URL) throws -> [Float] {
        try loadPCM(url, sampleRate: Double(sampleRate), channels: 1)
    }

    /// Bytes of an audio file (WAV, or anything AVFoundation reads) → a temp
    /// file URL to read it from. The caller removes it.
    public static func temporaryFile(for data: Data, pathExtension: String = "wav") throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("longcat-audio-\(UUID().uuidString).\(pathExtension)")
        try data.write(to: url)
        return url
    }

    /// Cut or zero-pad `samples` to exactly `seconds` (PT: the audio covers
    /// the clip's duration plus one second).
    public static func fitted(_ samples: [Float], seconds: Double) -> [Float] {
        let count = Int((seconds * Double(sampleRate)).rounded(.up))
        if samples.count >= count { return Array(samples[0..<count]) }
        return samples + [Float](repeating: 0, count: count - samples.count)
    }

    /// 16 kHz mono samples → Whisper log-mel `[1, 128, 3000 * chunks]`, each
    /// 30 s chunk padded and normalized on its own, as the PT feature extractor does.
    public static func whisperMel(_ samples: [Float]) -> MLXArray {
        let chunkSamples = WhisperAudio.nSamples
        var chunks: [MLXArray] = []
        var start = 0
        repeat {
            let end = min(start + chunkSamples, samples.count)
            let chunk = MLXArray(Array(samples[start..<end]))
            // (3000, 128) → (128, 3000)
            chunks.append(WhisperAudio.logMelSpectrogram(WhisperAudio.padOrTrim(chunk)).transposed(1, 0))
            start = end
        } while start < samples.count
        let mel = chunks.count == 1 ? chunks[0] : MLX.concatenated(chunks, axis: 1)
        return mel.expandedDimensions(axis: 0).asType(.float32)
    }
}
