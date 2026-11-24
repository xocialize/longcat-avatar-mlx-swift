import Foundation
import LongCatVideoAvatar
import MLXToolKit

/// Init-time configuration for `LongCatAvatarPackage` (C9). The reference image, speech, prompt, length, size and
/// seed ride the canonical `T2VRequest`.
///
/// **One weight source per lane.** A lane is one converted repo holding everything the model needs, in the layout
/// `recipes/convert_longcat_avatar.py` (xocialize/longcat-avatar-mlx) writes:
///
/// | directory | what | dtype |
/// |---|---|---|
/// | `dit/` | the Avatar 1.6 base DiT, no LoRA merged | bf16, or 8-bit / 4-bit Linears |
/// | `lora/` | `dmd_lora.safetensors` (upstream `loraA`) and `opsd_lora.safetensors` (`loraB`) | bf16 |
/// | `text_encoder/`, `tokenizer/` | umT5-XXL | bf16 |
/// | `audio_encoder/` | Whisper-large-v3 encoder | bf16 |
/// | `vae/` | Wan 2.1 VAE | bf16 |
/// | `pipeline_config.json` | the LoRA schedule (`mix_steps`, `mix_alpha`) | |
///
/// Upstream is `meituan-longcat/LongCat-Video-Avatar-1.6` (MIT). An explicit `modelDirectory` wins over the store and
/// never touches the network.
public struct LongCatAvatarConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, Equatable {

    /// The lanes this package has weights for.
    public static let supportedQuants: [Quant] = [.bf16, .int8, .int4]

    /// The converted repo of a lane.
    public static func repo(for quant: Quant) -> String {
        switch quant {
        case .int8: return "mlx-community/LongCat-Video-Avatar-1.6-q8"
        case .int4: return "mlx-community/LongCat-Video-Avatar-1.6-q4"
        default: return "mlx-community/LongCat-Video-Avatar-1.6-bf16"
        }
    }

    /// What a lane's repo must hold before the model can load.
    static let requiredFiles = [
        "dit/config.json",
        "dit/diffusion_pytorch_model.safetensors.index.json",
        "dit/*.safetensors",
        "lora/dmd_lora.safetensors",
        "lora/opsd_lora.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/*.safetensors",
        "audio_encoder/config.json",
        "audio_encoder/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "pipeline_config.json",
    ]

    /// The DiT's precision: `.bf16`, `.int8` or `.int4`. Everything else is bf16 in every lane.
    public var quant: Quant
    /// Leading denoising steps (of 8) that run `mixAlpha * DMD + (1 - mixAlpha) * OPSD`; the rest run OPSD alone.
    /// nil = the lane's shipped value (4, upstream's documented default).
    public var mixSteps: Int?
    /// The DMD LoRA's weight during the mixed steps, 0…1. nil = the lane's shipped value (1.0).
    public var mixAlpha: Float?
    /// The longest clip a request may ask for, in frames at 25 fps. Each 80 frames past the first 93 is one more
    /// continuation segment, so run time grows linearly; the default is one minute.
    public var maximumFrames: Int

    /// A local lane directory; host-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    /// Stamped by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(quant: Quant = .bf16, mixSteps: Int? = nil, mixAlpha: Float? = nil, maximumFrames: Int = 1500,
                modelDirectory: URL? = nil, modelsRootDirectory: URL? = nil) {
        self.quant = quant
        self.mixSteps = mixSteps
        self.mixAlpha = mixAlpha
        self.maximumFrames = maximumFrames
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case quant, mixSteps, mixAlpha, maximumFrames }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(quant: try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .bf16,
                  mixSteps: try c.decodeIfPresent(Int.self, forKey: .mixSteps),
                  mixAlpha: try c.decodeIfPresent(Float.self, forKey: .mixAlpha),
                  maximumFrames: try c.decodeIfPresent(Int.self, forKey: .maximumFrames) ?? 1500)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(quant, forKey: .quant)
        try c.encodeIfPresent(mixSteps, forKey: .mixSteps)
        try c.encodeIfPresent(mixAlpha, forKey: .mixAlpha)
        try c.encode(maximumFrames, forKey: .maximumFrames)
    }

    // MARK: - resolution

    public var repo: String { Self.repo(for: quant) }

    /// Where the lane is read from: the explicit directory, else the store's snapshot of the repo, else the store's
    /// flat repo directory. nil without either.
    public func resolvedDirectory(storeRoot: URL? = nil) -> URL? {
        if let modelDirectory { return modelDirectory }
        let store = ModelStore(root: storeRoot ?? modelsRootDirectory)
        if let snapshot = store.snapshotDirectory(for: repo, revision: "main"),
           FileManager.default.fileExists(atPath: snapshot.path) { return snapshot }
        return store.directory(for: repo)
    }

    /// The first required file that is not on disk, or nil when the lane is complete.
    func firstMissingFile(storeRoot: URL? = nil) -> String? {
        guard let root = resolvedDirectory(storeRoot: storeRoot) else { return "<no modelDirectory and no model store>" }
        return Self.requiredFiles.first { !Self.exists($0, under: root) }.map { root.appendingPathComponent($0).path }
    }

    /// `pattern` is a relative path, or `<directory>/*.safetensors`.
    static func exists(_ pattern: String, under root: URL) -> Bool {
        let fm = FileManager.default
        guard pattern.contains("*") else { return fm.fileExists(atPath: root.appendingPathComponent(pattern).path) }
        let directory = root.appendingPathComponent((pattern as NSString).deletingLastPathComponent)
        let suffix = String((pattern as NSString).lastPathComponent.dropFirst())   // "*.safetensors" → ".safetensors"
        return ((try? fm.contentsOfDirectory(atPath: directory.path)) ?? []).contains { $0.hasSuffix(suffix) }
    }
}

extension LongCatAvatarConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "model", repo: repo, revision: "main", matching: Self.requiredFiles)]
    }

    /// A complete explicit directory is not missing, an incomplete one is; otherwise the default store probe.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if modelDirectory != nil { return firstMissingFile() == nil ? [] : weightSources }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

extension LongCatAvatarConfiguration: WeightPrewarming {
    /// The resident set: DiT, LoRAs, Whisper encoder, VAE. umT5 is paged in per request.
    public var prewarmPaths: [URL] {
        guard let root = resolvedDirectory() else { return [] }
        return ["dit", "lora", "audio_encoder", "vae"].map { root.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }
}
