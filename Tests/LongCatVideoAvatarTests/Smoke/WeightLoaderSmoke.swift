//
//  WeightLoaderSmoke.swift
//
//  Smoke tests for WeightLoader — no network, no weights. Exercises the
//  Codable + JSON parsing paths against synthetic config + index files.
//

import Foundation
import XCTest
import MLX
import MLXNN
@testable import LongCatVideoAvatar

final class WeightLoaderSmoke: XCTestCase {
    var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("WeightLoaderSmoke-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - cacheDirectory

    func testCacheDirectoryUsesEnvOverrideWhenSet() throws {
        let custom = "/tmp/my-local-weights"
        setenv("LONGCAT_AVATAR_WEIGHTS_DIR", custom, 1)
        defer { unsetenv("LONGCAT_AVATAR_WEIGHTS_DIR") }

        let dir = try WeightLoader.cacheDirectory(for: "mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged")
        XCTAssertEqual(dir.path, custom)
    }

    func testCacheDirectoryComposesFromCaches() throws {
        unsetenv("LONGCAT_AVATAR_WEIGHTS_DIR")
        let dir = try WeightLoader.cacheDirectory(for: "mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged")
        XCTAssertTrue(dir.path.contains("longcat-avatar-mlx-swift"))
        XCTAssertTrue(dir.path.contains("mlx-community"))
        XCTAssertTrue(dir.path.hasSuffix("LongCat-Video-Avatar-1.5-q4-dmd-merged"))
    }

    func testCacheDirectoryRejectsMalformedRepoID() {
        unsetenv("LONGCAT_AVATAR_WEIGHTS_DIR")
        XCTAssertThrowsError(try WeightLoader.cacheDirectory(for: "no-slash"))
    }

    // MARK: - SafetensorsIndex decoding

    func testSafetensorsIndexDecodesHuggingFaceFormat() throws {
        let json = """
        {
          "metadata": {"total_size": 32479923456},
          "weight_map": {
            "blocks.0.attn.qkv.weight": "model-00001-of-00003.safetensors",
            "blocks.0.attn.proj.weight": "model-00001-of-00003.safetensors",
            "blocks.47.ffn.0.weight": "model-00003-of-00003.safetensors"
          }
        }
        """.data(using: .utf8)!
        let url = tempDir.appendingPathComponent("model.safetensors.index.json")
        try json.write(to: url)

        let idx: SafetensorsIndex = try WeightLoader.loadConfig(SafetensorsIndex.self, from: url)
        XCTAssertEqual(idx.metadata?.totalSize, 32_479_923_456)
        XCTAssertEqual(idx.weightMap.count, 3)
        XCTAssertEqual(idx.weightMap["blocks.0.attn.qkv.weight"], "model-00001-of-00003.safetensors")
    }

    // MARK: - QuantizationConfig decoding

    func testDetectQuantizationReturnsConfigForQuantizedVariant() throws {
        // Mirrors what _write_dit_config_with_quant emits in the Python recipe.
        let json = """
        {
          "_class_name": "LongCatVideoAvatarTransformer3DModel",
          "hidden_size": 4096,
          "depth": 48,
          "quantization": {
            "method": "mlx.nn.quantize",
            "bits": 4,
            "group_size": 64,
            "skip_patterns": [
              "final_layer.linear",
              "t_embedder.",
              "y_embedder.",
              "adaLN_modulation.",
              "audio_adaLN_modulation."
            ]
          }
        }
        """.data(using: .utf8)!
        let url = tempDir.appendingPathComponent("config.json")
        try json.write(to: url)

        let quant = try WeightLoader.detectQuantization(ditConfigURL: url)
        XCTAssertNotNil(quant)
        XCTAssertEqual(quant?.bits, 4)
        XCTAssertEqual(quant?.groupSize, 64)
        XCTAssertEqual(quant?.method, "mlx.nn.quantize")
        XCTAssertEqual(quant?.skipPatterns.count, 5)
        XCTAssertTrue(quant?.skipPatterns.contains("final_layer.linear") ?? false)
    }

    func testDetectQuantizationReturnsNilForBf16Variant() throws {
        // Mirrors the unmodified config.json shipped by the bf16 variants.
        let json = """
        {
          "_class_name": "LongCatVideoAvatarTransformer3DModel",
          "hidden_size": 4096,
          "depth": 48
        }
        """.data(using: .utf8)!
        let url = tempDir.appendingPathComponent("config.json")
        try json.write(to: url)

        let quant = try WeightLoader.detectQuantization(ditConfigURL: url)
        XCTAssertNil(quant)
    }

    func testDetectQuantizationDecodesQ8Variant() throws {
        // q8 carries the same shape but bits=8.
        let json = """
        {
          "quantization": {
            "method": "mlx.nn.quantize",
            "bits": 8,
            "group_size": 64,
            "skip_patterns": ["final_layer.linear"]
          }
        }
        """.data(using: .utf8)!
        let url = tempDir.appendingPathComponent("config.json")
        try json.write(to: url)

        let quant = try WeightLoader.detectQuantization(ditConfigURL: url)
        XCTAssertEqual(quant?.bits, 8)
    }

    // MARK: - applyDiTQuantization (S3.6.q)

    /// Tiny synthetic module mirroring the DiT name-pattern surface.
    /// `final_layer.linear` and `t_embedder.mlp` should be SKIPPED;
    /// `blocks.0.attn.qkv` should be QUANTIZED.
    final class _TinyDiTLike: Module {
        @ModuleInfo(key: "final_layer") var finalLayer: _FinalLayerStub
        @ModuleInfo(key: "t_embedder") var tEmbedder: _TEmbedderStub
        @ModuleInfo(key: "blocks") var blocks: [_BlockStub]

        override init() {
            self._finalLayer.wrappedValue = _FinalLayerStub()
            self._tEmbedder.wrappedValue = _TEmbedderStub()
            self._blocks.wrappedValue = [_BlockStub()]
            super.init()
        }
    }
    final class _FinalLayerStub: Module {
        @ModuleInfo(key: "linear") var linear: Linear
        override init() {
            self._linear.wrappedValue = Linear(64, 64, bias: false)
            super.init()
        }
    }
    final class _TEmbedderStub: Module {
        @ModuleInfo(key: "mlp") var mlp: Linear
        override init() {
            self._mlp.wrappedValue = Linear(64, 64, bias: false)
            super.init()
        }
    }
    final class _BlockStub: Module {
        @ModuleInfo(key: "attn") var attn: _AttnStub
        override init() {
            self._attn.wrappedValue = _AttnStub()
            super.init()
        }
    }
    final class _AttnStub: Module {
        @ModuleInfo(key: "qkv") var qkv: Linear
        override init() {
            self._qkv.wrappedValue = Linear(64, 64, bias: false)
            super.init()
        }
    }

    func testApplyDiTQuantizationSkipsConfiguredPatternsAndQuantizesRest() throws {
        let model = _TinyDiTLike()
        let cfg = QuantizationConfig(
            method: "mlx.nn.quantize",
            bits: 4,
            groupSize: 64,
            skipPatterns: ["final_layer.linear", "t_embedder."]
        )

        WeightLoader.applyDiTQuantization(to: model, config: cfg)

        // Skip-pattern matches stay regular Linear.
        XCTAssertFalse(model.finalLayer.linear is QuantizedLinear,
                       "final_layer.linear must remain regular Linear (skip pattern)")
        XCTAssertFalse(model.tEmbedder.mlp is QuantizedLinear,
                       "t_embedder.mlp must remain regular Linear (skip pattern)")
        // Everything else under nn.Linear gets quantized.
        XCTAssertTrue(model.blocks[0].attn.qkv is QuantizedLinear,
                      "blocks.0.attn.qkv must be quantized to QuantizedLinear")
    }

    func testApplyDiTQuantizationWithEmptySkipPatternsQuantizesAllLinears() throws {
        let model = _TinyDiTLike()
        let cfg = QuantizationConfig(
            method: "mlx.nn.quantize",
            bits: 4,
            groupSize: 64,
            skipPatterns: []
        )
        WeightLoader.applyDiTQuantization(to: model, config: cfg)
        XCTAssertTrue(model.finalLayer.linear is QuantizedLinear)
        XCTAssertTrue(model.tEmbedder.mlp is QuantizedLinear)
        XCTAssertTrue(model.blocks[0].attn.qkv is QuantizedLinear)
    }

    // MARK: - componentDirectory

    func testComponentDirectoryReturnsURLForExistingSubdir() throws {
        let vaeDir = tempDir.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: vaeDir, withIntermediateDirectories: true)
        let resolved = try WeightLoader.componentDirectory("vae", under: tempDir)
        XCTAssertEqual(resolved.standardizedFileURL.path, vaeDir.standardizedFileURL.path)
    }

    func testComponentDirectoryThrowsForMissingSubdir() {
        XCTAssertThrowsError(try WeightLoader.componentDirectory("dit", under: tempDir)) { err in
            guard case WeightLoaderError.missingComponent(let comp, _) = err else {
                XCTFail("Wrong error type: \(err)")
                return
            }
            XCTAssertEqual(comp, "dit")
        }
    }
}
