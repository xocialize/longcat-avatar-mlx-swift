// LongCatAvatarPackageTests — the package gates, offline: MAT (one source per lane, an explicit directory wins),
// the licence posture (MIT both layers), FIT, CAN-1 and the cadence declaration, request checking and clip planning
// (legible refusals, segments, size buckets), the honest load refusal, and the configuration's host-path-free
// Codable form. No weights and no kernel: the first real forward pass is the CLI's and the engine smoke's job.

import Foundation
import LongCatVideoAvatar
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXLongCatAvatar

final class LongCatAvatarPackageTests: XCTestCase {

    static func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// A lane directory holding every required file (placeholder bytes).
    static func satisfied(quant: Quant = .bf16) throws -> LongCatAvatarConfiguration {
        let root = try tempDir()
        for pattern in LongCatAvatarConfiguration.requiredFiles {
            let file = root.appendingPathComponent(pattern.replacingOccurrences(of: "*", with: "part-00001-of-00001"))
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([0]).write(to: file)
        }
        return LongCatAvatarConfiguration(quant: quant, modelDirectory: root)
    }

    static let image = Image(format: .png, data: Data([1]), width: 832, height: 480)
    static let audio = Audio(format: .wav, data: Data([1]))

    static func request(
        numFrames: Int? = nil, width: Int? = nil, height: Int? = nil, fps: Double? = nil, steps: Int? = nil,
        guidanceScale: Double? = nil, negativePrompt: String? = nil
    ) -> T2VRequest {
        T2VRequest(prompt: "p", negativePrompt: negativePrompt, initImage: image, initAudio: audio,
                   numFrames: numFrames, fps: fps, width: width, height: height, steps: steps,
                   guidanceScale: guidanceScale)
    }

    // MARK: - MAT

    func testMATGate() throws {
        let report = MaterializationConformance.check(
            freshConfiguration: LongCatAvatarConfiguration(), satisfiedConfiguration: try Self.satisfied())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testOneSourcePerLane() {
        XCTAssertEqual(LongCatAvatarConfiguration().weightSources.map(\.repo), ["mlx-community/LongCat-Video-Avatar-1.6-bf16"])
        XCTAssertEqual(LongCatAvatarConfiguration(quant: .int8).repo, "mlx-community/LongCat-Video-Avatar-1.6-q8")
        XCTAssertEqual(LongCatAvatarConfiguration(quant: .int4).repo, "mlx-community/LongCat-Video-Avatar-1.6-q4")
        XCTAssertEqual(LongCatAvatarConfiguration().missingWeightSources(storeRoot: nil).count, 1, "fresh machine")
    }

    func testBothLoRAsAreRequired() throws {
        let cfg = try Self.satisfied()
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        // 1.6 switches LoRA mid-denoise: a lane without the OPSD LoRA is not loadable.
        try FileManager.default.removeItem(at: cfg.modelDirectory!.appendingPathComponent("lora/opsd_lora.safetensors"))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["model"])
        XCTAssertTrue(cfg.firstMissingFile()!.hasSuffix("lora/opsd_lora.safetensors"))
    }

    func testStoreLayoutsResolve() throws {
        let store = try Self.tempDir()
        var cfg = LongCatAvatarConfiguration(modelsRootDirectory: store)
        let flat = store.appendingPathComponent("models--mlx-community--LongCat-Video-Avatar-1.6-bf16")
        XCTAssertEqual(cfg.resolvedDirectory()?.standardizedFileURL.path, flat.standardizedFileURL.path)
        // A hub snapshot behind refs/main wins over the flat directory.
        let snapshot = flat.appendingPathComponent("snapshots/abc123")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: flat.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try "abc123".write(to: flat.appendingPathComponent("refs/main"), atomically: true, encoding: .utf8)
        XCTAssertEqual(cfg.resolvedDirectory()?.standardizedFileURL.path, snapshot.standardizedFileURL.path)
        cfg.modelDirectory = store
        XCTAssertEqual(cfg.resolvedDirectory(), store, "an explicit directory wins")
    }

    func testConfigurationCodableCarriesNoHostPaths() throws {
        var cfg = try Self.satisfied(quant: .int4)
        cfg.mixSteps = 3
        cfg.mixAlpha = 0.5
        let data = try JSONEncoder().encode(cfg)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("/"))
        let back = try JSONDecoder().decode(LongCatAvatarConfiguration.self, from: data)
        XCTAssertNil(back.modelDirectory)
        XCTAssertNil(back.modelsRootDirectory)
        XCTAssertEqual(back.quant, .int4)
        XCTAssertEqual(back.mixSteps, 3)
        XCTAssertEqual(back.mixAlpha, 0.5)
        XCTAssertEqual(back.maximumFrames, 1500)
        XCTAssertEqual(try JSONDecoder().decode(LongCatAvatarConfiguration.self, from: Data("{}".utf8)), LongCatAvatarConfiguration())
    }

    // MARK: - licence

    func testLicenceIsMITBothLayers() async throws {
        let lic = LongCatAvatarPackage.manifest.license
        XCTAssertEqual(lic.weightLicenses, [.mit])
        XCTAssertEqual(lic.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.admits(.mit))
        let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)
        _ = try await engine.register(LongCatAvatarPackage.registration, configuration: LongCatAvatarConfiguration())
    }

    // MARK: - FIT

    func testFITGateEveryLane() {
        for quant in LongCatAvatarConfiguration.supportedQuants {
            let report = FootprintConformance.check(
                manifest: LongCatAvatarPackage.manifest, configuration: LongCatAvatarConfiguration(quant: quant))
            XCTAssertTrue(report.passed, "\(quant): \(report.summary)")
        }
        XCTAssertEqual(Set(LongCatAvatarPackage.manifest.requirements.footprints.map(\.quant)),
                       Set(LongCatAvatarConfiguration.supportedQuants))
    }

    // MARK: - CAN

    func testCANPreCancelledRun() async {
        let report = await CancellationConformance.checkRun(
            package: LongCatAvatarPackage(configuration: LongCatAvatarConfiguration()),
            request: Self.request())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: LongCatAvatarPackage.manifest,
            // before every denoising step (pipeline.onStep), with RunProgress(.denoise) there
            posture: .cadence([.init(phase: .denoise, unit: .step, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - surface

    func testSurfaceDeclaresAudioInAndSwitchableAudioOut() {
        let surface = LongCatAvatarPackage.manifest.surfaces[0]
        XCTAssertEqual(surface.capability, .textToVideo)
        let controls = surface.t2vControls
        XCTAssertEqual(controls?.supportsInitAudio, true, "an ignored initAudio would be silently wrong output")
        XCTAssertEqual(controls?.audioOutput, .switchable)
    }

    // MARK: - request checking

    func testImageAndAudioAreRequired() {
        let bad: [T2VRequest] = [
            T2VRequest(prompt: "p"),
            T2VRequest(prompt: "p", initImage: Self.image),
            T2VRequest(prompt: "p", initAudio: Self.audio),
            T2VRequest(prompt: "p", initImage: Image(format: .rawBGRA8, data: Data(), width: 1, height: 1), initAudio: Self.audio),
        ]
        for r in bad { assertRefused(try LongCatAvatarPackage.checked(r)) }
        XCTAssertNoThrow(try LongCatAvatarPackage.checked(Self.request()))
    }

    func testUnsupportedFieldsAreRefusedLegibly() {
        let bad: [T2VRequest] = [
            Self.request(negativePrompt: "blurry"),
            Self.request(guidanceScale: 4),
            Self.request(steps: 50),
            Self.request(fps: 30),
            Self.request(width: 832),
            Self.request(height: 480),
            T2VRequest(prompt: "p", initImage: Self.image, referenceImages: [Self.image], initAudio: Self.audio),
        ]
        for r in bad { assertRefused(try LongCatAvatarPackage.checked(r)) }
        // native values are accepted, not refused
        XCTAssertNoThrow(try LongCatAvatarPackage.checked(Self.request(width: 832, height: 480, fps: 25, steps: 8, guidanceScale: 1)))
        XCTAssertNoThrow(try LongCatAvatarPackage.checked(Self.request(negativePrompt: "  ")))
    }

    func assertRefused<T>(_ expression: @autoclosure () throws -> T, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try expression(), file: file, line: line) { e in
            guard case PackageError.unsupportedRequestFeature(let why) = e else { return XCTFail("\(e)", file: file, line: line) }
            XCTAssertTrue(why.hasPrefix("LongCat-Video-Avatar: "), why, file: file, line: line)
        }
    }

    // MARK: - clip planning

    /// A pipeline is only consulted for its config; planning runs no model.
    func plan(_ req: T2VRequest, image: (Int, Int) = (1920, 1080), speech: Double, maximum: Int = 1500) throws -> AvatarClipPlan {
        try LongCatAvatarPackage.plan(
            req, config: PipelineConfig(), imageWidth: image.0, imageHeight: image.1,
            speechSeconds: speech, maximumFrames: maximum)
    }

    func testClipFollowsTheSpeech() throws {
        // 3 s of speech: 75 frames, one segment trimmed to 4k + 1 = 77 generated.
        let short = try plan(Self.request(), speech: 3.0)
        XCTAssertEqual(short, AvatarClipPlan(width: 832, height: 480, frames: 75, segmentFrames: 77, segments: 1, fps: 25))
        // 3.72 s is exactly one full segment.
        XCTAssertEqual(try plan(Self.request(), speech: 3.72).segments, 1)
        XCTAssertEqual(try plan(Self.request(), speech: 3.72).segmentFrames, 93)
        // One frame over: a second segment (93 + 80 = 173 generated, 94 kept).
        let two = try plan(Self.request(), speech: 3.76)
        XCTAssertEqual(two.frames, 94)
        XCTAssertEqual(two.segments, 2)
        XCTAssertEqual(two.generatedFrames(numCondFrames: 13), 173)
        // 10 s = 250 frames: 93 + 80 + 80 = 253 ≥ 250.
        let ten = try plan(Self.request(), speech: 10)
        XCTAssertEqual(ten.frames, 250)
        XCTAssertEqual(ten.segments, 3)
        XCTAssertEqual(ten.seconds, 10)
        // Every length gets enough frames and never a whole segment too many.
        for frames in 25...600 {
            let p = try plan(Self.request(numFrames: frames), speech: 60)
            XCTAssertEqual(p.frames, frames)
            let generated = p.generatedFrames(numCondFrames: 13)
            XCTAssertGreaterThanOrEqual(generated, frames)
            XCTAssertLessThan(generated - frames, p.segments == 1 ? 4 : 80)
        }
    }

    func testNumFramesOverridesTheSpeechLength() throws {
        XCTAssertEqual(try plan(Self.request(numFrames: 50), speech: 30).frames, 50)
        // Longer than the speech is allowed: the audio is zero-padded (a silent tail).
        XCTAssertEqual(try plan(Self.request(numFrames: 200), speech: 2).frames, 200)
    }

    func testLengthLimits() {
        assertRefused(try plan(Self.request(), speech: 0.5))
        assertRefused(try plan(Self.request(numFrames: 10), speech: 10))
        assertRefused(try plan(Self.request(), speech: 61))
        assertRefused(try plan(Self.request(numFrames: 1501), speech: 5))
        XCTAssertNoThrow(try plan(Self.request(), speech: 61, maximum: 3000))
    }

    func testSizeFollowsTheImageUnlessNamed() throws {
        func size(_ w: Int, _ h: Int) throws -> [Int] { let p = try plan(Self.request(), image: (w, h), speech: 2); return [p.width, p.height] }
        XCTAssertEqual(try size(1920, 1080), [832, 480])
        XCTAssertEqual(try size(1080, 1920), [480, 832])
        XCTAssertEqual(try size(1024, 1024), [640, 608])   // PT: 0.95 and 1.05 tie at ratio 1.0; the first wins
        XCTAssertEqual(try size(2223, 1431), [768, 512])   // upstream's demo portrait, ratio 0.64
        // A named size must be a bucket (either tier).
        let named = try plan(Self.request(width: 480, height: 832), speech: 2)
        XCTAssertEqual([named.width, named.height], [480, 832])
        XCTAssertNoThrow(try plan(Self.request(width: 1248, height: 736), speech: 2))
        assertRefused(try plan(Self.request(width: 1280, height: 720), speech: 2))
        assertRefused(try plan(Self.request(width: 500, height: 500), speech: 2))
    }

    // MARK: - load refusal

    @InferenceActor
    func testLoadRefusesInsteadOfDownloading() async throws {
        let cfg = try Self.satisfied()
        try FileManager.default.removeItem(at: cfg.modelDirectory!.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let package = LongCatAvatarPackage(configuration: cfg)
        do {
            try await package.load()
            XCTFail("load must refuse a lane with a missing file")
        } catch LongCatAvatarPackageError.weightsNotProvisioned(let missing, let repo) {
            XCTAssertTrue(missing.hasSuffix("vae/diffusion_pytorch_model.safetensors"))
            XCTAssertEqual(repo, "mlx-community/LongCat-Video-Avatar-1.6-bf16")
        }
        do {
            try await LongCatAvatarPackage(configuration: LongCatAvatarConfiguration(quant: .fp16)).load()
            XCTFail("no fp16 lane")
        } catch LongCatAvatarPackageError.unsupportedQuant(let quant) {
            XCTAssertEqual(quant, .fp16)
        }
    }
}
