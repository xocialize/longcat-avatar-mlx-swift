//
//  UMT5EncoderModel.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/umt5.py
//
//  umT5-XXL text encoder — 11 B params, sharded as 3 safetensors.
//
//  Architecture (from Python port):
//  - PT key namespace → MLX-compact rename (already done at conversion time):
//    shared → token_embedding, encoder.block.{B}.layer.{0,1}.… →
//    blocks.{B}.{attn,ffn,norm1,norm2}.…
//  - Per-block relative position bias (NOT shared across layers, unlike T5).
//  - Gated GeLU FFN.
//  - Final LayerNorm.
//
//  Loading:
//  - Tokenizer comes from swift-transformers `Tokenizers` (umT5 uses the same
//    sentencepiece vocab as T5). Pull from <repo>/tokenizer/.
//  - Three model shards: model-{00001..00003}-of-00003.safetensors, indexed by
//    model.safetensors.index.json.
//
//  Critical:
//  - All weights bf16 in the published variants. Internal compute is bf16
//    except the final-layer LayerNorm (kept fp32 by upstream `_FP32` convention
//    — see Python `models/umt5.py`).
//

import Foundation
import MLX
import MLXNN

// TODO(S3.4): port UMT5EncoderModel
//   ☐ port T5LayerNorm (RMSNorm variant, no bias, eps=1e-6, kept-fp32 gamma)
//   ☐ port T5DenseGatedActDense (gated GeLU FFN)
//   ☐ port T5Attention with per-block relative position bias (compute_bias)
//   ☐ port T5Block (attn + ffn with pre-norm residuals)
//   ☐ port UMT5Stack (block list + final norm)
//   ☐ from_pretrained: parse safetensors.index.json, sharded mx.load
//   ☐ parity test: encode(demo prompt) max_abs < 1e-3 vs transformers PT
