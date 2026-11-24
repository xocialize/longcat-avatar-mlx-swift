//
//  LoRASmoke.swift
//
//  Mirrors longcat-avatar-mlx/tests/smoke/test_lora_smoke.py: key decoding,
//  merge math, the runtime side-paths, and Avatar 1.6's DMD → OPSD schedule.
//  Synthetic tensors only; no weights.
//

import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest

@testable import LongCatVideoAvatar

private final class TinyBlock: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    override init() {
        self._qkv.wrappedValue = Linear(8, 24)
        self._proj.wrappedValue = Linear(8, 8)
        super.init()
    }
}

private final class TinyModel: Module {
    @ModuleInfo var blocks: [TinyBlock]
    override init() {
        self._blocks.wrappedValue = [TinyBlock(), TinyBlock()]
        super.init()
    }
}

final class LoRASmoke: XCTestCase {

    static let targets: [(path: String, inDim: Int, outDim: Int, splits: Int)] = [
        ("blocks.0.qkv", 8, 24, 3), ("blocks.0.proj", 8, 8, 1),
        ("blocks.1.qkv", 8, 24, 3), ("blocks.1.proj", 8, 8, 1),
    ]

    /// A Meituan-encoded LoRA state dict over `targets`.
    func fakeLoRA(
        _ targets: [(path: String, inDim: Int, outDim: Int, splits: Int)] = LoRASmoke.targets,
        rank: Int = 4, alphaScale: Float = 0.5
    ) -> [String: MLXArray] {
        var sd: [String: MLXArray] = [:]
        for target in targets {
            let name = "lora___lorahyphen___" + target.path.replacingOccurrences(of: ".", with: "___lorahyphen___")
            sd["\(name).lora_down.weight"] = MLXRandom.normal([rank * target.splits, target.inDim])
            if target.splits == 1 {
                sd["\(name).lora_up.weight"] = MLXRandom.normal([target.outDim, rank])
            } else {
                for index in 0..<target.splits {
                    sd["\(name).lora_up.blocks.\(index).weight"] = MLXRandom.normal([target.outDim / target.splits, rank])
                }
            }
            sd["\(name).alpha_scale"] = MLXArray(alphaScale)
        }
        return sd
    }

    /// Metal's fp32 matmul is not bit-faithful (skill-lessons L10): compare
    /// relative to the output's magnitude.
    func assertClose(_ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-3, file: StaticString = #filePath, line: UInt = #line) {
        let scale = Swift.max(1, MLX.abs(want).max().item(Float.self))
        let diff = MLX.abs(got - want).max().item(Float.self)
        XCTAssertLessThanOrEqual(diff, rtol * scale, file: file, line: line)
    }

    func expected(_ linear: Linear, _ x: MLXArray, delta: MLXArray) -> MLXArray {
        MLX.matmul(x, (linear.weight + delta).transposed()) + linear.bias!
    }

    func testDecodeModuleName() throws {
        let down = try LoRA.decodeModuleName(
            "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight")
        XCTAssertEqual(down.modulePath, "blocks.0.attn.qkv")
        XCTAssertEqual(down.tensor, "lora_down.weight")
        let up = try LoRA.decodeModuleName(
            "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight")
        XCTAssertEqual(up.modulePath, "blocks.0.attn.qkv")
        XCTAssertEqual(up.tensor, "lora_up.blocks.1.weight")
        let alpha = try LoRA.decodeModuleName(
            "lora___lorahyphen___blocks___lorahyphen___5___lorahyphen___ffn___lorahyphen___w2.alpha_scale")
        XCTAssertEqual(alpha.modulePath, "blocks.5.ffn.w2")
        XCTAssertEqual(alpha.tensor, "alpha_scale")
        XCTAssertThrowsError(try LoRA.decodeModuleName("blocks.0.qkv.weight"))
    }

    func testMergedDeltaSplitFused() throws {
        MLXRandom.seed(0)
        let group = try LoRA.groupLoRATensors(fakeLoRA())["blocks.0.qkv"]!
        XCTAssertEqual(group.ups.count, 3)
        let delta = LoRA.computeMergedDelta(group, multiplier: 2)
        XCTAssertEqual(delta.shape, [24, 8])
        // Split 1 only sees rows 4..<8 of the shared down projection.
        let want = MLX.matmul(group.ups[1], group.down[4..<8]) * (2 * 0.5)
        assertClose(delta[8..<16], want)
    }

    func testRuntimeMatchesMergedDeltaAndKeepsParameterKeys() throws {
        MLXRandom.seed(1)
        let model = TinyModel()
        let keysBefore = model.parameters().flattened().map(\.0).sorted()
        let lora = fakeLoRA()
        let groups = try LoRA.groupLoRATensors(lora)
        let x = MLXRandom.normal([2, 5, 8])
        let wantQKV = expected(model.blocks[0].qkv, x, delta: LoRA.computeMergedDelta(groups["blocks.0.qkv"]!, multiplier: 0.7))
        let wantProj = expected(model.blocks[1].proj, x, delta: LoRA.computeMergedDelta(groups["blocks.1.proj"]!, multiplier: 0.7))
        let baseOut = model.blocks[0].qkv(x)

        let modules = try LoRA.attachRuntimeLoRAs(to: model, loras: [("dmd", lora)], dtype: .float32)
        XCTAssertEqual(modules.count, 4)
        XCTAssertTrue(model.blocks[0].qkv is LoRALinear)
        // Multiplier 0: exactly the base.
        XCTAssertEqual(MLX.abs(model.blocks[0].qkv(x) - baseOut).max().item(Float.self), 0)
        modules.forEach { $0.lora.setMultiplier(0.7, for: "dmd") }
        assertClose(model.blocks[0].qkv(x), wantQKV)
        assertClose(model.blocks[1].proj(x), wantProj)
        // Adapter tensors stay out of the parameter tree.
        XCTAssertEqual(model.parameters().flattened().map(\.0).sorted(), keysBefore)
    }

    func testMissingTargetThrows() {
        MLXRandom.seed(2)
        XCTAssertThrowsError(try LoRA.attachRuntimeLoRAs(
            to: TinyModel(), loras: [("dmd", fakeLoRA([("blocks.0.nope", 8, 8, 1)]))]))
    }

    func testMixScheduleSwitchesDMDToOPSD() throws {
        MLXRandom.seed(3)
        let model = TinyModel()
        let dmd = fakeLoRA(), opsd = fakeLoRA()
        let x = MLXRandom.normal([1, 3, 8])
        let base = model.blocks[0].qkv
        let wantDMD = expected(base, x, delta: LoRA.computeMergedDelta(try LoRA.groupLoRATensors(dmd)["blocks.0.qkv"]!))
        let wantOPSD = expected(base, x, delta: LoRA.computeMergedDelta(try LoRA.groupLoRATensors(opsd)["blocks.0.qkv"]!))

        let mix = try LoRAMix(modules: LoRA.attachRuntimeLoRAs(
            to: model, loras: [(LoRAMix.dmdKey, dmd), (LoRAMix.opsdKey, opsd)], dtype: .float32))
        XCTAssertEqual(mix.mixSteps, 4)
        XCTAssertEqual(mix.mixAlpha, 1)
        for step in 0..<8 {
            mix.setStep(step)
            assertClose(model.blocks[0].qkv(x), step < 4 ? wantDMD : wantOPSD)
        }
        // The next generation starts over at step 0.
        mix.setStep(0)
        assertClose(model.blocks[0].qkv(x), wantDMD)
    }

    func testMixFractionalAlphaBlendsBoth() throws {
        MLXRandom.seed(4)
        let model = TinyModel()
        let dmd = fakeLoRA(), opsd = fakeLoRA()
        let x = MLXRandom.normal([1, 3, 8])
        let base = model.blocks[1].proj
        let delta = LoRA.computeMergedDelta(try LoRA.groupLoRATensors(dmd)["blocks.1.proj"]!, multiplier: 0.25)
            + LoRA.computeMergedDelta(try LoRA.groupLoRATensors(opsd)["blocks.1.proj"]!, multiplier: 0.75)
        let want = expected(base, x, delta: delta)
        let mix = try LoRAMix(
            modules: LoRA.attachRuntimeLoRAs(to: model, loras: [(LoRAMix.dmdKey, dmd), (LoRAMix.opsdKey, opsd)], dtype: .float32),
            mixAlpha: 0.25, mixSteps: 3)
        XCTAssertEqual(mix.multipliers(forStep: 2).dmd, 0.25)
        XCTAssertEqual(mix.multipliers(forStep: 2).opsd, 0.75)
        XCTAssertEqual(mix.multipliers(forStep: 3).dmd, 0)
        XCTAssertEqual(mix.multipliers(forStep: 3).opsd, 1)
        mix.setStep(1)
        assertClose(model.blocks[1].proj(x), want)
        XCTAssertThrowsError(try LoRAMix(modules: [], mixAlpha: 1.5))
    }

    func testRuntimeOverQuantizedBase() throws {
        MLXRandom.seed(5)
        final class Q: Module {
            @ModuleInfo var proj: Linear
            override init() {
                self._proj.wrappedValue = Linear(64, 64)
                super.init()
            }
        }
        let model = Q()
        MLXNN.quantize(model: model, groupSize: 64, bits: 8)
        let x = MLXRandom.normal([1, 2, 64])
        let baseOut = model.proj(x)
        let lora = fakeLoRA([("proj", 64, 64, 1)])
        let modules = try LoRA.attachRuntimeLoRAs(to: model, loras: [("dmd", lora)], dtype: .float32)
        XCTAssertTrue(model.proj is LoRAQuantizedLinear)
        modules.forEach { $0.lora.setMultiplier(1, for: "dmd") }
        let delta = LoRA.computeMergedDelta(try LoRA.groupLoRATensors(lora)["proj"]!)
        assertClose(model.proj(x), baseOut + MLX.matmul(x, delta.transposed()))
    }

    /// Over the score budget the queries are sliced; rows of an attention
    /// matrix are independent, so the result equals the dense call.
    func testQuerySlicedAttentionMatchesDense() {
        MLXRandom.seed(6)
        let q = MLXRandom.normal([1, 4, 50, 16]), k = MLXRandom.normal([1, 4, 50, 16]), v = MLXRandom.normal([1, 4, 50, 16])
        let dense = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: .none)
        let budget = SlicedAttention.scoreBudgetBytes
        defer { SlicedAttention.scoreBudgetBytes = budget }
        SlicedAttention.scoreBudgetBytes = 4 * 50 * 4 * 7   // 7 query rows per slice (fp32), an uneven last slice
        let sliced = SlicedAttention.attend(q, k, v, scale: 0.25)
        XCTAssertEqual(sliced.shape, dense.shape)
        XCTAssertLessThan(MLX.abs(sliced - dense).max().item(Float.self), 1e-2)
        let fewer = q[0..., 0..., 0..<20]
        let denseFewer = MLXFast.scaledDotProductAttention(queries: fewer, keys: k, values: v, scale: 0.25, mask: .none)
        XCTAssertLessThan(MLX.abs(SlicedAttention.attend(fewer, k, v, scale: 0.25) - denseFewer).max().item(Float.self), 1e-2)
    }
}
