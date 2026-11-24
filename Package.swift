// swift-tools-version: 6.2
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
    // macOS 26: the floor of mlx-engine-swift's MLXToolKit, which the engine package target links.
    platforms: [
        .macOS(.v26),
    ],
    products: [
        // The library every consuming app links against. Pulls in MLX +
        // swift-transformers transitively.
        .library(
            name: "LongCatVideoAvatar",
            targets: ["LongCatVideoAvatar"]
        ),
        // The MLXEngine `textToVideo` ModelPackage over the core (image + speech + prompt → MP4).
        .library(
            name: "MLXLongCatAvatar",
            targets: ["MLXLongCatAvatar"]
        ),
        // CLI smoke / demo binary, mirroring `scripts/run_inference.py`.
        .executable(
            name: "run-inference",
            targets: ["RunInference"]
        ),
    ],
    dependencies: [
        // Core MLX runtime + nn + fast primitives + random: the fleet's mlx-swift fork at 0.33.1
        // (upstream 0.33.0 with the split-head attention routes off, AB-D-0264). A root line naming
        // ml-explore's URL is the same package identity and would silently bring the unpatched build back.
        .package(url: "https://github.com/xocialize/mlx-swift", exact: "0.33.1"),
        // Tokenizers for umT5 (1.x: what co-resolves with the engine and the fleet's other packages).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // The Whisper log-mel frontend (128 bins, parity-gated against mlx-whisper) and audio decoding.
        .package(url: "https://github.com/xocialize/whisper-mlx-swift.git", from: "0.3.0"),
        // MLXToolKit: the engine contract the ModelPackage is written against.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.82.1"),
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
                // Tokenizers (incl. T5 sentencepiece). Weight download has its own
                // minimal HF-Hub client in Utilities/WeightLoader.swift.
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "WhisperMLX", package: "whisper-mlx-swift"),
            ],
            path: "Sources/LongCatVideoAvatar",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXLongCatAvatar",
            dependencies: [
                "LongCatVideoAvatar",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXLongCatAvatar",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "RunInference",
            dependencies: [
                "LongCatVideoAvatar",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/RunInference",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "LongCatVideoAvatarTests",
            dependencies: ["LongCatVideoAvatar"],
            path: "Tests/LongCatVideoAvatarTests",
            resources: [
                // Tiny .npy reference outputs from the Python-MLX VAE,
                // consumed by Parity/VAEParity.swift. Total ~470 KB.
                // Regenerate via the companion Python repo's
                // scripts/dump_vae_swift_fixtures.py.
                .copy("Resources"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXLongCatAvatarTests",
            dependencies: [
                "MLXLongCatAvatar",
                "LongCatVideoAvatar",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXLongCatAvatarTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
