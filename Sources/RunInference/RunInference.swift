//
//  RunInference.swift
//
//  CLI wrapper that loads pre-processed inputs from a directory and runs
//  the full LongCatAvatarPipeline. The inputs are the same .npy files the
//  PipelineParity test consumes, produced by the Python repo's
//  `scripts/dump_pipeline_swift_fixtures.py` (or any equivalent
//  preprocessing pipeline that produces matching tensors).
//
//  HOW TO BUILD AND RUN
//  --------------------
//  Like the tests (see L21 in the Python port's skill-lessons), the CLI
//  built via `swift build` doesn't pick up MLX's Metal shader library
//  (default.metallib). Use Xcode or xcodebuild instead:
//
//      xcodebuild -scheme run-inference \
//          -destination "platform=macOS" \
//          -configuration Release build
//      ~/Library/Developer/Xcode/DerivedData/longcat-avatar-mlx-swift-*/Build/Products/Release/run-inference \
//          --input-dir <dir> --out output.npy
//
//  Alternatively, open Package.swift in Xcode and run the `run-inference`
//  scheme — same effect.
//
//  Required input files in `--input-dir`:
//    - image.npy           [1, 3, 1, H, W] in [-1, 1]
//    - audio_mel.npy       [1, 128, T_mel] Whisper mel features
//    - text_embeds.npy     [1, 1, N_text, 4096] umT5 hidden states
//    - text_mask.npy       [1, N_text] valid-token mask (int32)
//    - uncond_embeds.npy   [1, 1, N_text, 4096]
//    - uncond_mask.npy     [1, N_text]
//    - initial_noise.npy   [1, 16, T_lat, H_lat, W_lat] (optional)
//
//  Audio file → mel, image PNG → tensor, and prompt → umT5 tokenization
//  are deferred to a follow-up (would need vDSP/Accelerate for mel
//  extraction). Today the CLI consumes the same fixtures the parity
//  test does, which gives the same end-to-end correctness signal.
//

import ArgumentParser
import Foundation
import LongCatVideoAvatar
import MLX

@main
struct RunInference: AsyncParsableCommand {
    @Option(help: "HF repo id of the converted MLX weights (mlx-community/...)")
    var repo: String = "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

    @Option(help: "Directory containing pre-processed .npy inputs")
    var inputDir: String

    @Option(help: "Number of frames to generate (must match initial_noise.npy if provided)")
    var frames: Int = 5

    @Option(help: "Output video height (must match image.npy)")
    var height: Int = 64

    @Option(help: "Output video width (must match image.npy)")
    var width: Int = 64

    @Option(help: "Random seed for noise (ignored if initial_noise.npy is present)")
    var seed: UInt64 = 0

    @Option(help: "Output .npy path for the generated video tensor")
    var out: String = "output.npy"

    func run() async throws {
        let dir = URL(fileURLWithPath: inputDir, isDirectory: true)
        print("=== LongCat-Avatar MLX Swift inference ===")
        print("Repo:      \(repo)")
        print("Inputs:    \(dir.path)")
        print("Frames:    \(frames)")
        print("Size:      \(height) x \(width)")
        print()

        print("[1/4] Loading pipeline (downloads weights on first run, ~46 GB)...")
        let t0 = Date()
        let pipeline = try await LongCatAvatarPipeline.fromPretrained(repo)
        print("       loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")

        print("[2/4] Loading inputs...")
        let image = try loadNpy(dir.appendingPathComponent("image.npy"))
        let audioMel = try loadNpy(dir.appendingPathComponent("audio_mel.npy"))
        let textEmbeds = try loadNpy(dir.appendingPathComponent("text_embeds.npy"))
        let textMask = try loadNpy(dir.appendingPathComponent("text_mask.npy"))
        let uncondEmbeds = try loadNpy(dir.appendingPathComponent("uncond_embeds.npy"))
        let uncondMask = try loadNpy(dir.appendingPathComponent("uncond_mask.npy"))
        let noisePath = dir.appendingPathComponent("initial_noise.npy")
        let initialNoise: MLXArray? = FileManager.default.fileExists(atPath: noisePath.path)
            ? try loadNpy(noisePath)
            : nil
        print("       image:     \(image.shape)")
        print("       audio_mel: \(audioMel.shape)")
        print("       text:      \(textEmbeds.shape)")
        if let n = initialNoise { print("       noise:     \(n.shape) (will override seeded noise)") }

        print("[3/4] Running pipeline...")
        let t1 = Date()
        let video = pipeline(
            image: image,
            audioMel: audioMel,
            textEmbeds: textEmbeds,
            textMask: textMask,
            uncondEmbeds: uncondEmbeds,
            uncondMask: uncondMask,
            numFrames: frames,
            height: height,
            width: width,
            seed: seed,
            initialNoise: initialNoise
        )
        video.eval()
        print("       inference: \(String(format: "%.1f", Date().timeIntervalSince(t1)))s")

        print("[4/4] Saving output to \(out)...")
        try saveNpyFloat32(video.asType(.float32), to: URL(fileURLWithPath: out))
        print("       video shape: \(video.shape)")
        print()
        print("Done. To convert to MP4: post-process the .npy with imageio/ffmpeg.")
    }

    /// Load a `.npy` file as fp32 or int32 MLXArray.
    private func loadNpy(_ url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        guard data.count > 10,
              data[0] == 0x93,
              data[1...5] == Data("NUMPY".utf8) else {
            throw RunInferenceError.invalidNpy(url, "magic bytes mismatch")
        }
        // We support .npy v1 only — same restriction as the test reader.
        let headerLen = Int(data[8]) | (Int(data[9]) << 8)
        let header = String(data: data[10..<(10 + headerLen)], encoding: .utf8) ?? ""

        // Extract descr
        guard let descrRange = header.range(of: "'descr':") else {
            throw RunInferenceError.invalidNpy(url, "missing 'descr'")
        }
        let descr = header[descrRange.upperBound...]
            .split(separator: ",")[0]
            .trimmingCharacters(in: CharacterSet(charactersIn: "' "))
        let supported = ["<f4", "<i4"]
        guard supported.contains(descr) else {
            throw RunInferenceError.invalidNpy(url, "dtype \(descr) not supported (use <f4 or <i4)")
        }

        // Extract shape
        guard let shapeRange = header.range(of: "'shape':") else {
            throw RunInferenceError.invalidNpy(url, "missing 'shape'")
        }
        let tail = header[shapeRange.upperBound...]
        guard let openParen = tail.firstIndex(of: "("),
              let closeParen = tail.firstIndex(of: ")") else {
            throw RunInferenceError.invalidNpy(url, "bad shape tuple")
        }
        let shapeStr = String(tail[openParen...closeParen])
        let shape = shapeStr
            .trimmingCharacters(in: CharacterSet(charactersIn: "() "))
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let payload = data[(10 + headerLen)...]
        let count = shape.reduce(1, *)
        let expected = count * 4
        guard payload.count == expected else {
            throw RunInferenceError.invalidNpy(url, "payload size mismatch: \(payload.count) vs \(expected)")
        }

        if descr == "<f4" {
            let floats: [Float] = payload.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return MLXArray(floats, shape)
        } else {
            let ints: [Int32] = payload.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
            return MLXArray(ints, shape)
        }
    }

    /// Write a `.npy` v1 fp32 file. Stores the array in row-major order.
    private func saveNpyFloat32(_ array: MLXArray, to url: URL) throws {
        let shape = array.shape
        var headerStr = "{'descr': '<f4', 'fortran_order': False, 'shape': "
        if shape.count == 1 {
            headerStr += "(\(shape[0]),)"
        } else {
            headerStr += "(" + shape.map(String.init).joined(separator: ", ") + ")"
        }
        headerStr += ", }"
        // Pad to 64-byte alignment
        let prefixSize = 10
        var headerBytes = Data(headerStr.utf8)
        let targetLen = ((prefixSize + headerBytes.count + 1 + 63) / 64) * 64
        let padding = targetLen - prefixSize - headerBytes.count - 1
        headerBytes.append(Data(repeating: 0x20, count: padding))
        headerBytes.append(0x0A)
        var out = Data()
        out.append(contentsOf: [0x93])
        out.append(contentsOf: "NUMPY".utf8)
        out.append(contentsOf: [0x01, 0x00])
        let hl = UInt16(headerBytes.count)
        out.append(UInt8(hl & 0xff))
        out.append(UInt8((hl >> 8) & 0xff))
        out.append(headerBytes)

        let floats: [Float] = array.asArray(Float.self)
        floats.withUnsafeBufferPointer { buf in
            out.append(buf.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: buf.count * 4) {
                Data(bytes: $0, count: buf.count * 4)
            })
        }
        try out.write(to: url)
    }
}

enum RunInferenceError: LocalizedError {
    case invalidNpy(URL, String)
    var errorDescription: String? {
        switch self {
        case .invalidNpy(let url, let why): return "Bad .npy file \(url.lastPathComponent): \(why)"
        }
    }
}
