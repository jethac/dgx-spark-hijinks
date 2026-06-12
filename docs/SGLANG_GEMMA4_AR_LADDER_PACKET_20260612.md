# SGLang Gemma 4 AR Ladder Packet

Scope: live Spark packet for the remaining SGLang Gemma 4 autoregressive ladder
rows after the Ubicloud-built source-stack image.

## Preconditions

- Spark marker `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` absent.
- `docker ps` empty.
- GB10 memory guardrails: one server at a time, `--memory 100g`, no concurrent
  comparators, `MEM_FRACTION_STATIC` at or below `0.72`.
- Use the baked GHCR image, not a Spark-local rebuild:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-spark-u22-torch211-arm64`
  (`sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`).
- The image includes a `linux/arm64` manifest, Ubuntu 22.04 / glibc <= 2.35,
  torch 2.11 + CUDA 13, and passed the Spark E2B readiness smoke; see
  `results/sglang_gemma4_source_stack_image_27428220601/summary.md` and
  `results/sglang_spark_image_smoke_20260613T022153JST/summary.md`. The first
  preflight with the older x64-only image failed before serving; see
  `results/sglang_gemma4_ar_ladder_20260612T183849JST/summary.md`.

## Run

Default queue is 12B, 26B-A4B, then 31B:

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
git fetch origin
git checkout epoch2
git pull --ff-only

bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

To run one model only:

```bash
MODELS=google/gemma-4-12B-it \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

## What The Packet Proves

For each model, the packet runs two sequential servers:

- `bf16`: FlashInfer VO-split, BF16 weights, auto KV.
- `fullnvfp4`: FlashInfer VO-split, BF16 weights, full NVFP4 K+V
  (`--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`).

It records:

- baked-image provenance and package/binary proof lines;
- chat determinism for a low-entropy Tokyo prompt;
- supplied-token PPL using the repository markdown corpus;
- bf16-vs-full-NVFP4 PPL comparison;
- server logs containing Gemma KV pool geometry lines.

## Stop-On-Red

Stop the ladder at the first repeated failure mode and commit the artifacts:

- server not ready or `Not enough memory`: inspect `*_server.log` and the
  `SGLANG_GEMMA_KV_POOL_CONFIG` line before changing memory fraction;
- `Unsupported max_mma_kv: 0`: dispatcher bug evidence, do not work around in
  SGLang;
- incoherent or non-deterministic chat: quality red, do not bless capacity;
- missing input logprobs: PPL gate red, keep any smoke row scoped separately.

The packet is a runtime gate only. It does not modify SGLang or build anything
on the Spark.
