//
//  LoRA.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/lora.py
//
//  Two ways to apply Meituan's distillation LoRAs:
//
//  - Merge math (`computeMergedDelta`), for the 1.5 `bf16` base variant. The
//    recommended 1.5 variants ship the DMD LoRA already merged.
//  - Runtime side-paths (`LoRALinear`, `LoRAQuantizedLinear`, `LoRAMix`), for
//    Avatar 1.6. 1.6 ships two LoRAs over one base DiT (DMD and OPSD) and
//    switches between them during the 8-step denoise: the first `mixSteps`
//    steps run `mixAlpha * DMD + (1 - mixAlpha) * OPSD`, the rest OPSD alone.
//    A merge cannot express that, so each target Linear keeps its base weight
//    and adds `up(down(x)) * multiplier * alphaScale` per active LoRA, which is
//    what PT does (`_create_multi_lora_forward`). The side-path also rides a
//    quantized base.
//
//  Meituan's LoRA key encoding:
//  - Tensor name: `lora___lorahyphen___<MODULE_PATH>.<TENSOR>`, where
//    `___lorahyphen___` stands for `.` inside MODULE_PATH.
//  - TENSOR is `lora_down.weight`, `lora_up.weight`, `alpha_scale`, or for the
//    fused projections (qkv, kv_linear) `lora_up.blocks.{i}.weight`.
//  - A fused target has one shared down projection of `N * rank` rows and one
//    up per split; split i uses `down[i*rank ..< (i+1)*rank]`.
//
//  336 target modules = 7 LoRA targets × 48 blocks.
//

import Foundation
import MLX
import MLXNN

public enum LoRAError: Error, CustomStringConvertible {
    case unexpectedKey(String)
    case missingTensor(module: String, tensor: String)
    case badSplit(module: String, rows: Int, splits: Int)
    case missingTarget(lora: String, module: String)
    case notLinear(lora: String, module: String)
    case invalidMix(String)

    public var description: String {
        switch self {
        case .unexpectedKey(let key): return "unexpected LoRA key: \(key)"
        case .missingTensor(let module, let tensor): return "LoRA target \(module) has no \(tensor)"
        case .badSplit(let module, let rows, let splits):
            return "LoRA target \(module): \(rows) down rows don't divide into \(splits) splits"
        case .missingTarget(let lora, let module): return "LoRA \(lora) targets missing module \(module)"
        case .notLinear(let lora, let module): return "LoRA \(lora) targets \(module), which is not a Linear"
        case .invalidMix(let message): return message
        }
    }
}

/// One LoRA target's tensors, decoded from the Meituan state dict.
public struct LoRAGroup {
    /// `(rank * splits, in)`.
    public let down: MLXArray
    /// One `(out / splits, rank)` per split; a single entry for an unsplit target.
    public let ups: [MLXArray]
    public let alphaScale: Float
}

public enum LoRA {

    static let prefix = "lora___lorahyphen___"
    static let separator = "___lorahyphen___"

    /// Split a Meituan-encoded tensor name into `(modulePath, tensor)`:
    /// `lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight`
    /// → `("blocks.0.attn.qkv", "lora_down.weight")`.
    public static func decodeModuleName(_ key: String) throws -> (modulePath: String, tensor: String) {
        guard key.hasPrefix(prefix) else { throw LoRAError.unexpectedKey(key) }
        let decoded = String(key.dropFirst(prefix.count)).replacingOccurrences(of: separator, with: ".")
        for marker in [".lora_down.", ".lora_up.", ".alpha_scale"] {
            if let range = decoded.range(of: marker, options: .backwards) {
                let tensorStart = decoded.index(after: range.lowerBound)
                return (String(decoded[..<range.lowerBound]), String(decoded[tensorStart...]))
            }
        }
        throw LoRAError.unexpectedKey(key)
    }

    /// Group a LoRA state dict by target module path.
    public static func groupLoRATensors(_ stateDict: [String: MLXArray]) throws -> [String: LoRAGroup] {
        var raw: [String: [String: MLXArray]] = [:]
        for (key, value) in stateDict {
            let (path, tensor) = try decodeModuleName(key)
            raw[path, default: [:]][tensor] = value
        }
        var groups: [String: LoRAGroup] = [:]
        for (path, tensors) in raw {
            guard let down = tensors["lora_down.weight"] else {
                throw LoRAError.missingTensor(module: path, tensor: "lora_down.weight")
            }
            let splitKeys = tensors.keys
                .filter { $0.hasPrefix("lora_up.blocks.") }
                .sorted { splitIndex($0) < splitIndex($1) }
            let ups: [MLXArray]
            if splitKeys.isEmpty {
                guard let up = tensors["lora_up.weight"] else {
                    throw LoRAError.missingTensor(module: path, tensor: "lora_up.weight")
                }
                ups = [up]
            } else {
                ups = splitKeys.map { tensors[$0]! }
            }
            guard down.dim(0) % ups.count == 0 else {
                throw LoRAError.badSplit(module: path, rows: down.dim(0), splits: ups.count)
            }
            let alpha = tensors["alpha_scale"].map { $0.asType(.float32).item(Float.self) } ?? 1.0
            groups[path] = LoRAGroup(down: down, ups: ups, alphaScale: alpha)
        }
        return groups
    }

    private static func splitIndex(_ key: String) -> Int {
        Int(key.split(separator: ".")[2]) ?? 0
    }

    /// `multiplier * alphaScale * (up @ down)` in PT weight layout `(out, in)`,
    /// split-fused targets concatenated along the output axis.
    public static func computeMergedDelta(_ group: LoRAGroup, multiplier: Float = 1.0) -> MLXArray {
        let rank = group.down.dim(0) / group.ups.count
        let parts = group.ups.enumerated().map { index, up in
            MLX.matmul(up, group.down[(index * rank)..<((index + 1) * rank)])
        }
        let delta = parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)
        return delta * (multiplier * group.alphaScale)
    }

    /// Replace every LoRA target Linear of `model` with a `LoRALinear` (or
    /// `LoRAQuantizedLinear`) that carries one adapter per entry of `loras`,
    /// all at multiplier 0. Call after the base weights are loaded, and after
    /// `MLXNN.quantize` for a quantized base: the replacement copies the
    /// module's current arrays.
    ///
    /// Throws when a LoRA targets a module the model lacks: a skipped target
    /// would run that layer undistilled without any sign of it.
    @discardableResult
    public static func attachRuntimeLoRAs(
        to model: Module,
        loras: [(key: String, tensors: [String: MLXArray])],
        dtype: DType = .bfloat16
    ) throws -> [any LoRASwitchable] {
        let leaves = Dictionary(model.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var wrapped: [String: any LoRASwitchable] = [:]
        var replacements: [(String, Module)] = []

        for (key, tensors) in loras {
            for (path, group) in try groupLoRATensors(tensors) {
                let target: any LoRASwitchable
                if let existing = wrapped[path] {
                    target = existing
                } else {
                    guard let module = leaves[path] else {
                        throw LoRAError.missingTarget(lora: key, module: path)
                    }
                    if let already = module as? any LoRASwitchable {
                        target = already
                    } else if let quantized = module as? QuantizedLinear {
                        let replacement = LoRAQuantizedLinear(quantized)
                        replacements.append((path, replacement))
                        target = replacement
                    } else if let linear = module as? Linear {
                        let replacement = LoRALinear(linear)
                        replacements.append((path, replacement))
                        target = replacement
                    } else {
                        throw LoRAError.notLinear(lora: key, module: path)
                    }
                    wrapped[path] = target
                }
                target.lora.add(key: key, group: group, dtype: dtype)
            }
        }
        if !replacements.isEmpty {
            model.update(modules: ModuleChildren.unflattened(replacements))
        }
        return Array(wrapped.values)
    }
}

// MARK: - Runtime side-paths

/// The adapters of one target module and their current multipliers.
///
/// A plain class, not a `Module`: its arrays stay out of the model's parameter
/// tree, so attaching LoRAs changes neither the weight keys nor what
/// `update(parameters:)` verifies.
public final class LoRAAdapterSet: @unchecked Sendable {

    struct Adapter {
        let key: String
        /// `(in, rank * splits)`: the down projection, transposed for `x @ downT`.
        let downT: MLXArray
        /// `(rank, out / splits)` per split.
        let upsT: [MLXArray]
        let alphaScale: Float
        var multiplier: Float
    }

    private(set) var adapters: [Adapter] = []

    public var keys: [String] { adapters.map(\.key) }

    func add(key: String, group: LoRAGroup, dtype: DType) {
        let downT = group.down.transposed().asType(dtype)
        let upsT = group.ups.map { $0.transposed().asType(dtype) }
        MLX.eval([downT] + upsT)
        adapters.removeAll { $0.key == key }
        adapters.append(Adapter(key: key, downT: downT, upsT: upsT, alphaScale: group.alphaScale, multiplier: 0))
    }

    public func setMultiplier(_ multiplier: Float, for key: String) {
        for index in adapters.indices where adapters[index].key == key {
            adapters[index].multiplier = multiplier
        }
    }

    public func multiplier(for key: String) -> Float {
        adapters.first { $0.key == key }?.multiplier ?? 0
    }

    /// `out + Σ up(down(x)) * multiplier * alphaScale` over the active adapters.
    func apply(_ x: MLXArray, to out: MLXArray) -> MLXArray {
        var result = out
        for adapter in adapters where adapter.multiplier != 0 {
            var lx = MLX.matmul(x.asType(adapter.downT.dtype), adapter.downT)
            if adapter.upsT.count == 1 {
                lx = MLX.matmul(lx, adapter.upsT[0])
            } else {
                let chunks = MLX.split(lx, parts: adapter.upsT.count, axis: -1)
                lx = MLX.concatenated(zip(chunks, adapter.upsT).map { MLX.matmul($0, $1) }, axis: -1)
            }
            result = result + lx.asType(result.dtype) * (adapter.multiplier * adapter.alphaScale)
        }
        return result
    }
}

/// A Linear whose LoRA side-paths can be switched at runtime.
public protocol LoRASwitchable: AnyObject {
    var lora: LoRAAdapterSet { get }
}

/// `Linear` + runtime LoRA side-paths. Shares the base module's arrays.
public final class LoRALinear: Linear, LoRASwitchable, @unchecked Sendable {
    public let lora = LoRAAdapterSet()

    public init(_ base: Linear) {
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        lora.apply(x, to: super.callAsFunction(x))
    }
}

/// `QuantizedLinear` + runtime LoRA side-paths. Shares the base module's arrays.
public final class LoRAQuantizedLinear: QuantizedLinear, LoRASwitchable, @unchecked Sendable {
    public let lora = LoRAAdapterSet()

    public init(_ base: QuantizedLinear) {
        super.init(
            weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
            groupSize: base.groupSize, bits: base.bits, mode: base.mode, globalScale: base.globalScale
        )
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        lora.apply(x, to: super.callAsFunction(x))
    }
}

/// Avatar 1.6's per-step DMD/OPSD schedule (PT `enable_lora_mix` /
/// `set_lora_mix_step`).
///
/// Steps below `mixSteps` run `mixAlpha * DMD + (1 - mixAlpha) * OPSD`; later
/// steps run OPSD alone. Upstream's documented default is `mixSteps = 4`,
/// `mixAlpha = 1.0`: four DMD-only steps, then four OPSD-only steps.
public final class LoRAMix: @unchecked Sendable {
    public static let dmdKey = "dmd"
    public static let opsdKey = "opsd"

    public let modules: [any LoRASwitchable]
    public let mixAlpha: Float
    public let mixSteps: Int
    private var active: Bool?

    public init(modules: [any LoRASwitchable], mixAlpha: Float = 1.0, mixSteps: Int = 4) throws {
        guard (0...1).contains(mixAlpha) else {
            throw LoRAError.invalidMix("mixAlpha must be in [0, 1], got \(mixAlpha)")
        }
        guard mixSteps >= 0 else { throw LoRAError.invalidMix("mixSteps must be >= 0, got \(mixSteps)") }
        self.modules = modules
        self.mixAlpha = mixAlpha
        self.mixSteps = mixSteps
        setStep(0)
    }

    /// `(dmd, opsd)` multipliers for a denoising step.
    public func multipliers(forStep step: Int) -> (dmd: Float, opsd: Float) {
        step < mixSteps ? (mixAlpha, 1 - mixAlpha) : (0, 1)
    }

    public func setStep(_ step: Int) {
        let mixed = step < mixSteps
        if mixed == active { return }
        let (dmd, opsd) = multipliers(forStep: step)
        for module in modules {
            module.lora.setMultiplier(dmd, for: Self.dmdKey)
            module.lora.setMultiplier(opsd, for: Self.opsdKey)
        }
        active = mixed
    }
}
