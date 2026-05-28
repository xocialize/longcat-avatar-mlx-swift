//
//  RunInference.swift
//
//  CLI wrapper that mirrors `scripts/run_inference.py` from the Python port.
//
//  Usage (once S3.9 lands):
//      swift run run-inference \
//          --repo mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged \
//          --image refs/longcat-video/assets/avatar/single/man.png \
//          --audio refs/longcat-video/assets/avatar/single/man.mp3 \
//          --prompt "A western man stands on stage…" \
//          --frames 29 --width 432 --height 256 \
//          --out output.mp4
//

import ArgumentParser
import Foundation
import LongCatVideoAvatar

@main
struct RunInference: AsyncParsableCommand {
    @Option(help: "HF repo id of the converted MLX weights (mlx-community/...)")
    var repo: String = "mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged"

    @Option(help: "Path to reference portrait image")
    var image: String

    @Option(help: "Path to driving audio")
    var audio: String

    @Option(help: "Scene prompt")
    var prompt: String

    @Option(help: "Number of frames to generate")
    var frames: Int = 29

    @Option(help: "Output video width")
    var width: Int = 432

    @Option(help: "Output video height")
    var height: Int = 256

    @Option(help: "Random seed")
    var seed: UInt64 = 42

    @Option(help: "Output mp4 path")
    var out: String = "output.mp4"

    func run() async throws {
        // TODO(S3.9): wire up pipeline once ports land
        print("Not yet implemented — see TODOs in Sources/LongCatVideoAvatar/")
        throw ExitCode.failure
    }
}
