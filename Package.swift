// swift-tools-version: 5.9
//
// Package manifest for longcat-avatar-mlx-swift.
//
// Mirrors the structure of the Python reference port at
// https://github.com/xocialize/longcat-avatar-mlx — see CLAUDE.md for the
// isomorphic-structure rule that motivates the file-name mapping.
//

import PackageDescription

let package = Package(
    name: "LongCatVideoAvatar",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        // The library every consuming app links against. Pulls in MLX +
        // swift-transformers transitively.
        .library(
            name: "LongCatVideoAvatar",
            targets: ["LongCatVideoAvatar"]
        ),
        // CLI smoke / demo binary, mirroring `scripts/run_inference.py`.
        .executable(
            name: "run-inference",
            targets: ["RunInference"]
        ),
    ],
    dependencies: [
        // Core MLX runtime + nn + fast primitives + random.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        // Hugging Face Hub download + tokenizers for umT5.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.18"),
        // Arg parsing for the CLI target.
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "LongCatVideoAvatar",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // swift-transformers exposes Tokenizers (incl. T5 sentencepiece);
                // `Hub` is internal — we ship our own minimal HF-Hub client
                // for weight download in Utilities/WeightLoader.swift (S3.2).
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/LongCatVideoAvatar"
        ),
        .executableTarget(
            name: "RunInference",
            dependencies: [
                "LongCatVideoAvatar",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/RunInference"
        ),
        .testTarget(
            name: "LongCatVideoAvatarTests",
            dependencies: ["LongCatVideoAvatar"],
            path: "Tests/LongCatVideoAvatarTests"
        ),
    ]
)
