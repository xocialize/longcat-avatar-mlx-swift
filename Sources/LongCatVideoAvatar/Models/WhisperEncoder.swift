//
//  WhisperEncoder.swift
//
//  Port target: longcat-avatar-mlx/longcat_video_avatar/models/whisper.py
//
//  Whisper-large-v3 ENCODER ONLY. ~0.6 B params, single safetensors file.
//
//  Architecture (from Python port):
//  - 2 Conv1d frontend (mel → embed) with stride 2 in conv2 (downsample).
//  - 32 transformer encoder blocks (no decoder).
//  - Output: 33 hidden states (input mel + 32 block outputs) — we return ALL
//    of them. The audio_process module does the 33→5 mean-pool downstream.
//
//  Conversion-time transforms (already applied in published weights):
//  - `model.encoder.` prefix stripped.
//  - Conv1d weight transpose PT (O,I,K) → MLX (O,K,I).
//
//  Critical:
//  - We need ALL 33 hidden states, not just the last. The audio_process
//    module group-pools (33 → 5) then linear-interps to 25 Hz before feeding
//    AudioProjModel. See Python `audio_process.py` for the post-processing.
//

import Foundation
import MLX
import MLXNN

// TODO(S3.5): port WhisperEncoder
//   ☐ port WhisperConvFrontend (2× Conv1d + GELU)
//   ☐ port WhisperEncoderBlock (pre-norm MHA + FFN)
//   ☐ port WhisperEncoder stack (positional embed + 32 blocks + final norm)
//   ☐ from_pretrained: single safetensors file
//   ☐ port `audio_process` mel preprocessing using Accelerate / vDSP (librosa
//     is Python; we'll re-implement the mel filterbank + STFT)
//   ☐ parity test: encode(demo audio mel) max_abs < 1e-3 vs transformers PT
