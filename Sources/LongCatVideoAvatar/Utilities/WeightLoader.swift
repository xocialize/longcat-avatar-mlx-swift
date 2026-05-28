//
//  WeightLoader.swift
//
//  Runtime HF download + cache helper. No Python equivalent — this is the
//  Swift port's substitute for Python's `huggingface_hub.snapshot_download`
//  and `mx.load`. swift-transformers' `Hub` package gives us the download
//  primitives; we layer on:
//    - sharded safetensors index parsing
//    - environment-variable cache root override (matches Python pattern)
//    - quant-config detection (so the loader can apply MLXNN.quantize
//      before loading the bit-packed shards, mirroring Python L19/L20)
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
import MLXNN

public enum WeightLoader {
    // TODO(S3.2): port runtime download + cache
    //   ☐ snapshotDownload(repoId, into cacheDir, with progress)
    //   ☐ parse safetensors.index.json → [shardName: [tensorName]]
    //   ☐ load split safetensors via MLX (swift binding)
    //   ☐ readConfig → typed Config struct per component
    //   ☐ env override LONGCAT_AVATAR_WEIGHTS_DIR for local dev
    //   ☐ detect dit/config.json:quantization block; if present, surface
    //     to caller so they can apply MLXNN.quantize before load_weights
}
