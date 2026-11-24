# Footprints and speed: LongCat-Video-Avatar 1.6

The numbers behind `LongCatAvatarPackage.footprints` and the README's tables.

## Setup

- M5 Max, 128 GB, macOS 27.3. Release build, mlx-swift 0.33.1 (the fleet fork), mlx-engine-swift 0.82.1.
- Weights on an external USB volume (about 250 MB/s cold); the file cache was warm unless noted.
- Measured on the signed plugin bundle (`gaea-plugin-longcat-avatar`) with `gaea-plugin-check`, which launches the
  worker the way Gaea does and reads its `phys_footprint` from outside. One fresh worker process per run.
- Floor = `phys_footprint` right after `prepare`. Peak = the worker's lifetime peak.
- Input: a 960×618 portrait (bucket 768×512), synthetic speech, seed 7.

## One full segment (93 frames, 3.72 s, 768×512: 36,864 tokens)

| Lane | Floor | Lifetime peak | After the job | Run | Declared (resident + activation) |
|---|---|---|---|---|---|
| bf16 | 37.75 GB | 57.68 GB | 41.1 GB | 565 s | 38.0 + 24.0 GB |
| int8 | 23.78 GB | 43.87 GB | 27.1 GB | 627 s | 24.0 + 24.5 GB |
| int4 | 16.32 GB | 36.25 GB | 19.2 GB | 622 s | 16.5 + 24.0 GB |

Activation is `(peak − floor) × 1.2`. It is about the same in every lane because it is the request's own working
set, not the DiT's: umT5-XXL paged in for the prompt and evicted (11 GB), the attention score slices (2 GiB each),
and the VAE's full-resolution feature maps for one latent frame.

A longer clip is more segments of the same size, so its peak is the same and its time is linear: 93 frames, then 80
per continuation.

## Shorter clips (bf16, 768×512)

| Clip | Generated | Run | Lifetime peak |
|---|---|---|---|
| 1 s (25 frames) | 25 frames | 111 s | 60.2 GB (before the VAE fix below) |
| 2 s (50 frames) | 53 frames | 272 s | 72.1 GB (before the VAE fix below) |

Prepare: 4.9 s on a warm file cache; 65–82 s cold from the USB volume (the engine pages 37.6 GB first).

## What moved the peak

Three changes, each found by measuring:

1. **Query-sliced attention.** One dense attention call over 36,864 tokens makes a 32 × 36,864² bf16 score array,
   about 87 GB. Before the slicing, a 93-frame run swapped for 17 minutes without finishing a step.
2. **VAE decode one latent frame at a time.** Left lazy, the decode graph spanned every frame and was evaluated in
   one go. Peak for one segment: 75.9 GB before, 57.7 GB after. A second-by-second trace showed the worker flat at
   38–39 GB through load, prompt encoding and all eight steps, and the spike only in decode.
3. **Weight loads on the CPU stream.** Not a peak, a crash: the first engine run died in `load()` with
   `kIOGPUCommandBufferCallbackErrorTimeout`.

The MLX buffer pool is capped at 2 GiB during a run (`PipelineConfig.cacheLimitBytes`); on its own that did not move
the peak.

## Quantized lanes

The picture from the 4-bit DiT is close to bf16's on the same seed: same composition and identity, slightly softer
detail. The 8-bit and 4-bit lanes are not faster than bf16 on this machine; they are for Macs without the memory.
