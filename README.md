# longcat-avatar-mlx-swift

Swift port of [LongCat-Video-Avatar](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven talking-avatar video diffusion model — on
[mlx-swift](https://github.com/ml-explore/mlx-swift), for Apple Silicon. A reference image,
speech and a scene prompt make a 25 fps clip in which the person in the image speaks, with the
speech as its soundtrack.

**Current: [LongCat-Video-Avatar 1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6).**

Three things are in the package:

| Product | What |
|---|---|
| `LongCatVideoAvatar` | The model and pipeline, plus image / audio / prompt input and MP4 output. |
| `MLXLongCatAvatar` | An [MLXEngine](https://github.com/xocialize/mlx-engine-swift) `textToVideo` `ModelPackage` over it. |
| `run-inference` | A CLI: image + speech + prompt → MP4, and the parity mode the tests use. |

The Python port, [xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx),
is the reference: it converts the weights and its outputs are what this port is gated against.

## What 1.6 is

1.6 keeps 1.5's architecture (a 13.6 B-parameter DiT with an audio cross-attention path, the
Wan 2.1 VAE, umT5-XXL, a Whisper-large-v3 encoder) and ships a new base DiT with two
distillation LoRAs, DMD and OPSD, which are switched **during** the 8-step denoise: by
upstream's documented default, steps 1–4 run DMD and steps 5–8 run OPSD. A merge cannot
express that, so the LoRAs run as side-paths, as they do upstream:
`base(x) + up(down(x)) * multiplier * alphaScale` (`Utilities/LoRA.swift`). The side-path also
rides a quantized base. Sampling is distilled without classifier-free guidance: one DiT pass
per step.

A clip is a chain of segments. The first is 93 frames (3.72 s) from the image, which is its
first frame. Each further one continues from the last 13 frames of the one before, with the
image's latent as a reference, and adds 80 frames (3.2 s).

The output size follows the image: upstream's 480p bucket closest to its aspect ratio
(832×480 for 16:9, 480×832 for 9:16, 640×608 near square), the image scaled to cover it and
centre-cropped.

## CLI

```bash
swift run -c release run-inference \
    --weights-dir /path/to/LongCat-Video-Avatar-1.6-bf16 \
    --image portrait.png --audio speech.wav \
    --prompt "A woman speaks to the camera in a sunlit kitchen" \
    --out clip.mp4
```

- The clip is the speech's length unless `--clip-frames` says otherwise (25 fps, at least 25).
- `--clip-width` / `--clip-height` name a size instead of taking the image's bucket;
  `--resolution 720p` uses upstream's 720p bucket table.
- `--mix-steps` / `--mix-alpha` change the LoRA schedule; `--seed` the noise.
- `--repo` loads a converted repo from the Hub instead of `--weights-dir`.
- Give clean speech: upstream's vocal separator is not ported.

Parity mode (`--input-dir <fixtures> --reference <output.npy>`) runs the pipeline on
pre-computed `.npy` inputs and compares with a Python-MLX output.

## Library

```swift
import LongCatVideoAvatar

let pipeline = try LongCatAvatarPipeline.fromDirectory(weightsDirectory)
let tokenizer = try await AutoTokenizer.from(modelFolder: weightsDirectory.appendingPathComponent("tokenizer"))
let (embeds, mask) = PromptEncoding.encode(prompt, encoder: pipeline.textEncoder!, tokenizer: tokenizer)
let text = LongCatAvatarPipeline.TextConditioning(
    embeds: embeds, mask: mask, uncondEmbeds: MLXArray.zeros(like: embeds), uncondMask: mask)

let plan = try await pipeline.generateMP4(
    image: try ImageInput.decode(imageData), speechURL: speechURL, text: text, output: outputURL)
```

`generateClip` hands each segment's frames to a closure instead of writing a file;
`generateAI2V` and `generateAVC` are the two segment kinds; `onStep` reports progress and may
throw to stop a run.

## MLXEngine package

`LongCatAvatarPackage` serves `textToVideo`: `T2VRequest.initImage` (the person) and
`initAudio` (the speech) are both required, `prompt` describes the scene.

- `numFrames` sets the length (default: the speech's); `width` and `height` name a bucket
  (default: the image's); `includeAudio: false` gives a silent clip; `seed` as usual.
- `steps`, `guidanceScale`, `negativePrompt`, `fps` other than 25, reference images, keyframes
  and adapters are refused with a reason.
- Lanes: `LongCatAvatarConfiguration(quant:)` is `.bf16`, `.int8` or `.int4` (the DiT's
  precision; everything else is bf16). Each lane is one repo,
  `mlx-community/LongCat-Video-Avatar-1.6-{bf16,q8,q4}`, or a local `modelDirectory`.
- umT5-XXL (11 GB) is loaded per request and evicted before denoising; the last prompt's
  features are kept.
- Progress is `RunProgress(.denoise)` per step over the whole clip; cancellation is checked
  before every step.

## Weights

Converted by the Python port's recipe, one directory (or repo) per DiT precision:

```bash
python -m recipes.convert_longcat_avatar --variant v1.6-bf16 --out ./weights   # or v1.6-q8, v1.6-q4
```

| Directory | What | dtype |
|---|---|---|
| `dit/` | the 1.6 base DiT, no LoRA merged | bf16, or 8-bit / 4-bit Linears |
| `lora/` | `dmd_lora.safetensors` (upstream `loraA`), `opsd_lora.safetensors` (`loraB`) | bf16 |
| `text_encoder/`, `tokenizer/` | umT5-XXL | bf16 |
| `audio_encoder/` | Whisper-large-v3 encoder | bf16 |
| `vae/` | Wan 2.1 VAE | bf16 |
| `pipeline_config.json` | the LoRA schedule | |

The 1.5 `-dmd-merged` repos still load (`LongCatAvatarPipeline.fromPretrained`); they carry no
`pipeline_config.json` LoRA block, so no side-paths are attached.

## Speed and memory

M5 Max, 128 GB, macOS 27.3, mlx-swift 0.33.1, release build. See `docs/footprints.md` for the
runs.

One full segment (93 frames, 3.72 s) at 768×512, through the engine package:

| Lane | Weights | Resident | Peak | Time |
|---|---|---|---|---|
| bf16 | 46 GB | 37.8 GB | 57.7 GB | 9.4 min |
| int8 | 33 GB | 23.8 GB | 43.9 GB | 10.4 min |
| int4 | 26 GB | 16.3 GB | 36.3 GB | 10.4 min |

- A longer clip is more segments: add about 9 minutes per 3.2 s. The peak does not grow with length.
- Smaller sizes are much faster: 50 frames at 432×256 take 52 s (5.5 s per step).
- The quantized lanes are not faster here; they are for Macs without the memory.
- Prepare is 5 s on a warm file cache.

## Parity against Python-MLX

| Gate | Result | Notes |
|---|---|---|
| LoRA terms, fp32, real DMD and OPSD files, six target modules | exact on the CPU stream; within 0.1 % as the runtime side-path | The other LoRA's term is 60–130 % away, so a swapped file or a wrong split order cannot pass. |
| Pipeline, bf16, 5 frames at 64×64, default schedule | mean abs 0.0059 to Python's default-schedule output, 0.0078 to its all-DMD output | bf16 kernels drift between the bindings by as much as the LoRA switch changes this tiny clip (max abs 0.14 either way), so the gate is relative. |
| Pipeline, DMD on all 8 steps | mean abs 0.0051 to Python's all-DMD output, 0.0081 to its default | The other way round, as it should be. |
| Components (1.5 weights, opt-in) | VAE encode 3.1e-6, decode 1.76e-3; umT5 0.119; Whisper 0.016; base DiT 0.033; Avatar DiT 0.32 | Unchanged by 1.6: the architecture is the same. |

## Tests

```bash
swift test                                                              # 147 tests, no weights
LONGCAT_AVATAR_V16_DIR=/path/to/LongCat-Video-Avatar-1.6-bf16 swift test --filter AvatarV16Parity
```

`MLXLongCatAvatarTests` are the engine package's offline gates: materialization, licence,
footprints, cancellation, request checking and clip planning.

## Layout

```
Sources/LongCatVideoAvatar/
├── Models/                 # Wan VAE, umT5, Whisper encoder, base DiT, attention, RoPE
│   └── Avatar/             # the Avatar DiT: audio cross-attention, AudioProjModel
├── Pipeline/               # LongCatAvatarPipeline, guidance, scheduler
├── Media/                  # image / audio / prompt input, size buckets, clip planning, MP4 writer
├── Audio/AudioProcess.swift
└── Utilities/              # WeightLoader, LoRA (merge math + runtime side-paths + schedule)
Sources/MLXLongCatAvatar/   # the MLXEngine ModelPackage
Sources/RunInference/       # the CLI
```

File names mirror the Python package so the two trees can be read side by side.

## Requirements

macOS 26 or later, Apple Silicon. The package pins the fleet's mlx-swift fork
(`xocialize/mlx-swift` 0.33.1: upstream 0.33.0 with the split-head attention routes off).

## License

MIT, as upstream Meituan LongCat-Video and the Python port are.
