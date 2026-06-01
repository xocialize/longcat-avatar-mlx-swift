//
//  AudioProcess.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/audio_process.py
//
//  Whisper post-processing pipeline:
//   1. Take all 33 Whisper hidden states (input mel + 32 block outputs)
//   2. Mean-pool consecutive groups → 5 groups
//      ([0:8], [8:16], [16:24], [24:32], [32] singleton)
//   3. Linear-interp encoder-rate (50 Hz) → target video fps (25 Hz)
//   4. Window: per-frame fetch 5 surrounding frames (`audio_window=5`)
//      with edge replication on boundaries
//
//  Output shape: `[B, T_video, W=5, S=5, D=1280]` — what
//  `LongCatVideoAvatarTransformer3DModel.audio_embs` expects.
//
//  Raw audio → mel preprocessing (vDSP + Whisper feature extractor) is
//  not yet ported — that lives in S3.9 alongside the demo `run-inference`
//  CLI. For now this module takes mel features as input (matches the
//  Python pipeline's wiring).
//

import Foundation
import MLX

public enum AudioProcess {

    /// Linear-interpolate features along the time axis.
    /// - features: `[B, T, D]`
    /// - Returns: `[B, T_out, D]`
    public static func linearInterpolateFeatures(
        _ features: MLXArray,
        inputFps: Float,
        outputFps: Float,
        outputLen: Int? = nil
    ) -> MLXArray {
        let B = features.dim(0)
        let T = features.dim(1)
        let D = features.dim(2)
        let outLen = outputLen ?? Int(Float(T) / inputFps * outputFps)
        if outLen == T {
            return features
        }
        // Source fractional indices for each output position
        let srcIdx = MLX.linspace(0.0 as Float, Float(T - 1), count: outLen)
        let lo = MLX.minimum(MLX.floor(srcIdx).asType(.int32), MLXArray(Int32(T - 1)))
        let hi = MLX.minimum(lo + Int32(1), MLXArray(Int32(T - 1)))
        let frac = (srcIdx - lo.asType(.float32))
            .asType(features.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: -1)   // [1, T_out, 1]
        // Gather along time axis. We do per-index lookups via broadcasting.
        let featLo = features[0..., lo, 0...]
        let featHi = features[0..., hi, 0...]
        _ = B   // suppress warning if unused above
        _ = D
        return featLo * (Float(1.0) - frac) + featHi * frac
    }

    /// Apply Meituan's specific 33-layer group-mean-pool.
    /// - hiddenStates: 33 arrays, each `[B, T_enc, D=1280]`
    /// - Returns: `[B, T_enc, 5, D]` — 5 groups (4×8 + 1 singleton)
    public static func groupPoolWhisperHiddenStates(_ hiddenStates: [MLXArray]) -> MLXArray {
        precondition(hiddenStates.count == 33, "expected 33 layers, got \(hiddenStates.count)")

        var feats: [MLXArray] = []
        // Four groups of 8 + singleton at index 32
        for (start, end) in [(0, 8), (8, 16), (16, 24), (24, 32)] {
            let stacked = MLX.stacked(Array(hiddenStates[start..<end]), axis: 0)   // [8, B, T, D]
            feats.append(MLX.mean(stacked, axis: 0))
        }
        feats.append(hiddenStates[32])

        // [B, T, 5, D]
        return MLX.stacked(feats, axis: 2)
    }

    /// Run Whisper encoder over mel features (possibly long) and return the
    /// group-pooled 5-channel features.
    /// - melFeatures: `[B, num_mel_bins, T_mel]`
    /// - encChunk: chunk size in mel-frame units (3000 = 30s @ 100 Hz)
    /// - Returns: `[B, T_enc, 5, D]` where `T_enc = T_mel // 2`
    public static func whisperEncodeAudioToGroups(
        _ whisperEncoder: WhisperEncoder,
        melFeatures: MLXArray,
        encChunk: Int = 3000
    ) -> MLXArray {
        let TMel = melFeatures.dim(2)
        var outChunks: [MLXArray] = []
        var start = 0
        while start < TMel {
            let end = min(start + encChunk, TMel)
            let chunk = melFeatures[0..., 0..., start..<end]
            let hidden = whisperEncoder.allHiddenStates(chunk)   // 33 tensors
            let pooled = groupPoolWhisperHiddenStates(hidden)    // [B, T_enc_chunk, 5, D]
            outChunks.append(pooled)
            start = end
        }
        return outChunks.count == 1 ? outChunks[0] : MLX.concatenated(outChunks, axis: 1)
    }

    /// Convert encoder-rate group features to per-video-frame audio embeddings
    /// with a sliding window of size `audioWindow` over the time axis (the W=5
    /// dim the DiT consumes via `AudioProjModel.proj1`).
    /// - audioGroups: `[B, T_enc, 5, D]` (output of `groupPoolWhisperHiddenStates`)
    /// - Returns: `[B, T_video, W=audioWindow, S=5, D]`
    public static func buildAvatarAudioEmbeddings(
        _ audioGroups: MLXArray,
        fps: Int = 25,
        encFps: Int = 50,
        audioWindow: Int = 5
    ) -> MLXArray {
        let B = audioGroups.dim(0)
        let TEnc = audioGroups.dim(1)
        let G = audioGroups.dim(2)
        let D = audioGroups.dim(3)

        // 1. Resample T_enc → T_video. Flatten (G, D) → (G*D) so we interp
        //    each (group, dim) independently.
        let flat = audioGroups.reshaped(B, TEnc, G * D)
        let targetLen = Int(Float(TEnc) / Float(encFps) * Float(fps))
        let resampled = linearInterpolateFeatures(
            flat,
            inputFps: Float(encFps),
            outputFps: Float(fps),
            outputLen: targetLen
        )
        let perFrame = resampled.reshaped(B, targetLen, G, D)

        // 2. Sliding window of size audioWindow with edge replication.
        //    Output: [B, T_video, W, G, D].
        let half = audioWindow / 2
        let padded: MLXArray
        if half > 0 {
            let first = perFrame[0..., 0..<1]
            let last = perFrame[0..., (targetLen - 1)..<targetLen]
            let leftPad = MLX.broadcast(first, to: [B, half, G, D])
            let rightPad = MLX.broadcast(last, to: [B, half, G, D])
            padded = MLX.concatenated([leftPad, perFrame, rightPad], axis: 1)
        } else {
            padded = perFrame
        }

        // Build the window by gathering per-position slices.
        var windows: [MLXArray] = []
        for offset in 0..<audioWindow {
            windows.append(padded[0..., offset..<(offset + targetLen)])   // [B, T_video, G, D]
        }
        return MLX.stacked(windows, axis: 2)   // [B, T_video, W, G, D]
    }
}
