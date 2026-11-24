//
//  WhisperMelSmoke.swift
//
//  The log-mel front end and the audio fitting around it. No weights.
//

import MLX
import XCTest

@testable import LongCatVideoAvatar

final class WhisperMelSmoke: XCTestCase {

    func tone(seconds: Double, hz: Double = 440) -> [Float] {
        (0..<Int(seconds * 16_000)).map { Float(0.5 * sin(2 * Double.pi * hz * Double($0) / 16_000)) }
    }

    func testFilterbankResourceLoads() {
        XCTAssertEqual(WhisperMel.filters.mel.shape.sorted(), [128, 201])
        XCTAssertEqual(WhisperMel.filters.window.dim(0), 400)
    }

    func testThirtySecondWindowIs3000Frames() {
        let mel = WhisperMel.logMelSpectrogram(WhisperMel.padOrTrim(MLXArray(tone(seconds: 2))))
        XCTAssertEqual(mel.shape, [3000, 128])
        // Whisper's normalization: (log10 clamped to max - 8, + 4) / 4, so the range spans exactly 2.
        let (lo, hi) = (mel.min().item(Float.self), mel.max().item(Float.self))
        XCTAssertEqual(hi - lo, 2.0, accuracy: 1e-3)
        XCTAssertTrue(lo.isFinite && hi.isFinite)
        // The tone sits in the low bins of the frames that carry it; the zero padding is at the floor.
        let voiced = mel[0..<200].mean(axis: 0), silent = mel[400..<3000].mean(axis: 0)
        XCTAssertGreaterThan(voiced.max().item(Float.self), silent.max().item(Float.self) + 1.0)
        XCTAssertLessThan(voiced.argMax().item(Int32.self), 40, "440 Hz is a low mel bin")
    }

    func testClipMelIsChunkedPerThirtySeconds() {
        XCTAssertEqual(AudioInput.whisperMel(tone(seconds: 1)).shape, [1, 128, 3000])
        XCTAssertEqual(AudioInput.whisperMel(tone(seconds: 30)).shape, [1, 128, 3000])
        XCTAssertEqual(AudioInput.whisperMel(tone(seconds: 31)).shape, [1, 128, 6000])
    }

    func testFittedCutsAndPads() {
        let samples = tone(seconds: 1)
        XCTAssertEqual(AudioInput.fitted(samples, seconds: 0.5).count, 8_000)
        let padded = AudioInput.fitted(samples, seconds: 2)
        XCTAssertEqual(padded.count, 32_000)
        XCTAssertEqual(padded[16_000...].max(), 0)
        XCTAssertEqual(Array(padded[0..<16_000]), samples)
    }
}
