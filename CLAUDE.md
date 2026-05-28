# CLAUDE.md — longcat-avatar-mlx-swift

Operational orientation for Claude when working in this repo.

## What this is

Swift port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video)
using [mlx-swift](https://github.com/ml-explore/mlx-swift). The
**production reference is the Python port at
[xocialize/longcat-avatar-mlx](https://github.com/xocialize/longcat-avatar-mlx)**
— same MLX runtime, same HF weights, same outputs. When in doubt about
model behavior, the Python port is the oracle.

## Where things live

- **[README.md](README.md)** — public-facing scaffold status + quick-start sketch
- **[Sources/LongCatVideoAvatar/](Sources/LongCatVideoAvatar/)** — the library.
  File names and directory layout mirror the Python package 1:1 per the
  `/mlx-porting` skill's isomorphic-structure rule. Each stub file has a
  header comment naming the Python source it ports, an architecture
  summary, and a `TODO(S3.x)` checklist.
- **[Sources/RunInference/](Sources/RunInference/)** — CLI mirror of the
  Python `scripts/run_inference.py`.
- **[Tests/](Tests/)** — smoke + parity. Parity tests compare against
  the Python port's reference outputs (frame dumps in `/tmp/q*_smoke_output.npy`
  if available) or against `transformers` PT when small enough.
- **[docs/port-roadmap.md](docs/port-roadmap.md)** — Stage 3 plan, what's
  done vs pending.

## Running tests

```bash
swift test            # smoke + any parity tests with bundled fixtures
```

## Skill in use

`/mlx-porting` is the operational skill — load it before structural
changes. The skill is written around Python MLX but its core rules
(isomorphic structure, parity-first, never refactor during port) apply
identically to mlx-swift.

The Python port's `docs/development/skill-lessons.md` (L1–L20) is the
canonical list of traps already discovered. Read those before debugging
anything that "looks weird" — it's almost always a known trap.

## Trust the Python port's choices

If the Python port made a specific choice (e.g. routing VAE attention to
CPU stream for fp32 precision, overriding scheduler trailing sigma to
0.0, casting SDPA mask to Q dtype), port the same choice. Each was
debugged the hard way; deviating "to clean up" is how regressions get
reintroduced.
