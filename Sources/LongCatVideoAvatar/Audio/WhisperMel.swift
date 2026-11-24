//
//  WhisperMel.swift
//
//  The Whisper-large-v3 log-mel front end: 16 kHz, n_fft 400, hop 160,
//  128 mel bins, 30 s windows. What PT's `WhisperFeatureExtractor` computes.
//
//  The mel filterbank and the Hann window are a baked resource
//  (`Resources/mel_filters_128.safetensors`: `mel_filters` and `window`),
//  the same values openai/whisper ships in `assets/mel_filters.npz`. The
//  STFT and filterbank primitives are MLXAudioDSP's.
//

import Foundation
import MLX
import MLXAudioDSP

public enum WhisperMel {
    public static let nFFT = 400
    public static let hop = 160
    /// 30 s at 16 kHz: the window every chunk is padded or cut to.
    public static let nSamples = 480_000
    public static let nMels = 128

    struct Filters {
        let mel: MLXArray
        let window: MLXArray
    }

    static let filters: Filters = {
        guard let url = Bundle.module.url(forResource: "mel_filters_128", withExtension: "safetensors"),
              let arrays = try? MLX.loadArrays(url: url),
              let mel = arrays["mel_filters"], let window = arrays["window"]
        else { fatalError("LongCatVideoAvatar: the mel_filters_128.safetensors resource is missing") }
        return Filters(mel: mel, window: window)
    }()

    /// Zero-pad or cut a 1-D 16 kHz waveform to exactly `nSamples`.
    public static func padOrTrim(_ audio: MLXArray) -> MLXArray {
        let count = audio.dim(0)
        if count > nSamples { return audio[0..<nSamples] }
        if count < nSamples { return MLX.padded(audio, widths: [IntOrPair((0, nSamples - count))]) }
        return audio
    }

    /// 1-D waveform (16 kHz) → log-mel `(frames, 128)`; 3000 frames for 30 s.
    /// Reflect-padded STFT with the last frame dropped, power spectrum, mel
    /// filterbank, log10 clamped to 8 below the maximum, then `(x + 4) / 4`.
    public static func logMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        let padded = AudioDSP.reflectPadded(audio, pad: nFFT / 2)
        let frames = AudioDSP.framed(padded, frameLength: nFFT, hop: hop)
        let power = AudioDSP.powerSpectrum(frames, window: filters.window)
        let magnitudes = power[0..<(frames.dim(0) - 1)]
        let mel = AudioDSP.applyMelFilterbank(magnitudes, filters: filters.mel)
        var logSpec = MLX.log(MLX.maximum(mel, 1e-10)) * 0.4342944819
        logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)
        return (logSpec + 4.0) / 4.0
    }
}
