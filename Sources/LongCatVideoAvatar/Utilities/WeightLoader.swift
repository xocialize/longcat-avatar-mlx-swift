//
//  WeightLoader.swift
//
//  Runtime HF download + cache helper. No Python equivalent — this is the
//  Swift port's substitute for Python's `huggingface_hub.snapshot_download`
//  and `mx.load`. We ship our own minimal HF Hub client (URLSession +
//  the public /api/models/ REST endpoint) instead of depending on
//  swift-transformers' internal `Hub` target, which isn't exposed as a
//  library product (see Package.swift comment).
//
//  Standard repo IDs (point at the same artifacts as the Python port):
//    - mlx-community/LongCat-Video-Avatar-1.5-bf16
//    - mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged   (recommended)
//    - mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged     (32 GB Macs)
//    - mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged
//
//  Env override:
//    LONGCAT_AVATAR_WEIGHTS_DIR=/path/to/local/weights/<variant-dir>/
//

import Foundation
import MLX

public enum WeightLoaderError: LocalizedError {
    case invalidRepoID(String)
    case httpError(URL, status: Int)
    case decodingError(URL, underlying: Error)
    case missingComponent(component: String, in: URL)

    public var errorDescription: String? {
        switch self {
        case .invalidRepoID(let r):
            return "Invalid HF repo id: \(r). Expected <org>/<name>."
        case .httpError(let url, let s):
            return "HTTP \(s) for \(url.absoluteString)"
        case .decodingError(let url, let err):
            return "Failed to decode JSON at \(url.lastPathComponent): \(err.localizedDescription)"
        case .missingComponent(let c, let dir):
            return "Component '\(c)' not found under \(dir.path) (expected subdir like vae/, dit/, …)"
        }
    }
}

/// Quantization metadata carried inside `dit/config.json` for the q4/q8
/// variants. Mirrors the `quantization` block written by the Python
/// recipe (`recipes/convert_longcat_avatar.py::_write_dit_config_with_quant`).
public struct QuantizationConfig: Codable, Sendable {
    public let method: String           // "mlx.nn.quantize"
    public let bits: Int                // 4 or 8
    public let groupSize: Int           // 64
    public let skipPatterns: [String]   // module-path substrings to keep at full precision

    enum CodingKeys: String, CodingKey {
        case method
        case bits
        case groupSize = "group_size"
        case skipPatterns = "skip_patterns"
    }
}

/// Index file describing a sharded safetensors checkpoint. Matches the
/// format the Python recipe emits (HuggingFace convention).
public struct SafetensorsIndex: Codable, Sendable {
    public struct Metadata: Codable, Sendable {
        public let totalSize: Int?
        enum CodingKeys: String, CodingKey { case totalSize = "total_size" }
    }
    public let metadata: Metadata?
    public let weightMap: [String: String]   // tensor name → shard filename

    enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }
}

public enum WeightLoader {

    // MARK: - Cache location

    /// Root cache directory: `<Caches>/longcat-avatar-mlx-swift/<orgSafeRepo>/`.
    /// Overridable via `LONGCAT_AVATAR_WEIGHTS_DIR` env var (points directly
    /// at an unpacked weights dir; bypasses download entirely).
    public static func cacheDirectory(for repoID: String) throws -> URL {
        if let override = ProcessInfo.processInfo.environment["LONGCAT_AVATAR_WEIGHTS_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let parts = repoID.split(separator: "/")
        guard parts.count == 2 else { throw WeightLoaderError.invalidRepoID(repoID) }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return caches
            .appendingPathComponent("longcat-avatar-mlx-swift", isDirectory: true)
            .appendingPathComponent(String(parts[0]), isDirectory: true)
            .appendingPathComponent(String(parts[1]), isDirectory: true)
    }

    // MARK: - HF Hub download

    /// Snapshot-download a complete HF repo into the local cache. Resumable
    /// per-file: already-present files with matching size are skipped. No
    /// xet / git-lfs negotiation — just direct downloads via the public
    /// `resolve/main/<path>` endpoint, which is what `huggingface_hub`
    /// falls back to anyway when xet isn't available.
    @discardableResult
    public static func snapshotDownload(
        repoID: String,
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> URL {
        // If env override is set, assume the dir already exists with weights
        // and skip network entirely.
        if ProcessInfo.processInfo.environment["LONGCAT_AVATAR_WEIGHTS_DIR"] != nil {
            return try cacheDirectory(for: repoID)
        }

        let cacheDir = try cacheDirectory(for: repoID)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let files = try await listRepoFiles(repoID: repoID)
        var doneCount = 0
        for entry in files {
            let dest = cacheDir.appendingPathComponent(entry.path)
            if let existing = try? FileManager.default.attributesOfItem(atPath: dest.path),
               let size = existing[.size] as? Int,
               let expected = entry.size, size == expected {
                doneCount += 1
                progress?(entry.path, doneCount, files.count)
                continue
            }
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try await downloadFile(repoID: repoID, path: entry.path, to: dest)
            doneCount += 1
            progress?(entry.path, doneCount, files.count)
        }
        return cacheDir
    }

    // MARK: - Sharded safetensors loading

    /// Load every shard listed in a safetensors index into a single
    /// `[String: MLXArray]` dict, matching what the Python port's
    /// `_load_pt_safetensors_sharded` returns. Loads each shard once.
    public static func loadShardedSafetensors(indexURL: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: indexURL)
        let index: SafetensorsIndex
        do {
            index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        } catch {
            throw WeightLoaderError.decodingError(indexURL, underlying: error)
        }

        let baseDir = indexURL.deletingLastPathComponent()
        var grouped: [String: [String]] = [:]
        for (tensor, shard) in index.weightMap {
            grouped[shard, default: []].append(tensor)
        }

        var combined: [String: MLXArray] = [:]
        combined.reserveCapacity(index.weightMap.count)
        for (shard, tensors) in grouped {
            let shardURL = baseDir.appendingPathComponent(shard)
            let shardDict = try MLX.loadArrays(url: shardURL)
            for tensor in tensors {
                guard let arr = shardDict[tensor] else { continue }
                combined[tensor] = arr
            }
        }
        return combined
    }

    /// Convenience: load a single safetensors file (non-sharded components
    /// like vae/, audio_encoder/).
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    // MARK: - Config + quantization detection

    /// Read a JSON config file as a Codable type.
    public static func loadConfig<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw WeightLoaderError.decodingError(url, underlying: error)
        }
    }

    /// Returns the DiT quantization config if the published variant carries one,
    /// or `nil` for bf16 variants.
    public static func detectQuantization(ditConfigURL: URL) throws -> QuantizationConfig? {
        let data = try Data(contentsOf: ditConfigURL)
        guard let top = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let q = top["quantization"] else {
            return nil
        }
        let qData = try JSONSerialization.data(withJSONObject: q)
        do {
            return try JSONDecoder().decode(QuantizationConfig.self, from: qData)
        } catch {
            throw WeightLoaderError.decodingError(ditConfigURL, underlying: error)
        }
    }

    // MARK: - Component path helpers

    /// Resolve `<weightsRoot>/<component>/` and verify it exists. Components
    /// match the Python port's directory layout:
    ///   vae/, text_encoder/, audio_encoder/, dit/, scheduler/, tokenizer/, lora/
    public static func componentDirectory(_ component: String, under root: URL) throws -> URL {
        let url = root.appendingPathComponent(component, isDirectory: true)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            throw WeightLoaderError.missingComponent(component: component, in: root)
        }
        return url
    }
}

// MARK: - Minimal HF Hub REST client (file-private)

/// One file in the HF repo tree.
private struct HFTreeEntry: Sendable {
    let path: String
    let size: Int?
}

extension WeightLoader {
    /// `GET https://huggingface.co/api/models/{repo}/tree/main?recursive=true`
    /// returns a JSON array of file entries with `path`, `size`, `type`.
    /// We filter to leaf files (type == "file") and return paths + sizes.
    fileprivate static func listRepoFiles(repoID: String) async throws -> [HFTreeEntry] {
        let url = URL(
            string: "https://huggingface.co/api/models/\(repoID)/tree/main?recursive=true"
        )!
        let (data, response) = try await URLSession.shared.data(from: url)
        try Self.throwIfNonOK(response: response, url: url)

        struct RawEntry: Decodable {
            let type: String
            let path: String
            let size: Int?
        }
        let raw = try JSONDecoder().decode([RawEntry].self, from: data)
        return raw.compactMap { entry in
            guard entry.type == "file" else { return nil }
            return HFTreeEntry(path: entry.path, size: entry.size)
        }
    }

    /// Download a single file from `https://huggingface.co/{repo}/resolve/main/{path}`
    /// directly to disk. Follows redirects (HF returns 302 to CloudFront / xet).
    fileprivate static func downloadFile(
        repoID: String,
        path: String,
        to destination: URL
    ) async throws {
        let encodedPath = path.split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(encodedPath)")!

        // Stream to a tmp file first, then atomic-move on success.
        let (tempURL, response) = try await URLSession.shared.download(from: url)
        try Self.throwIfNonOK(response: response, url: url)

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    fileprivate static func throwIfNonOK(response: URLResponse, url: URL) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw WeightLoaderError.httpError(url, status: http.statusCode)
        }
    }
}
