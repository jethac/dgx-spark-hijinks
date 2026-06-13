# SGLang Gemma 4 AR Ladder Packet

Scope: live Spark packet for the remaining SGLang Gemma 4 autoregressive ladder
rows after the Ubicloud-built source-stack image.

## Preconditions

- Spark marker `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` absent.
- `docker ps` empty.
- GB10 memory guardrails: one server at a time, `--memory 100g`, no concurrent
  comparators, `MEM_FRACTION_STATIC` at or below `0.72`.
- Use the baked GHCR image, not a Spark-local rebuild:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-mm-prefix-f920e2d-arm64`
  (`sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`).
- The image includes a `linux/arm64` manifest, Ubuntu 22.04 / glibc <= 2.35,
  torch 2.11 + CUDA 13, SGLang `f920e2d88a`, FlashInfer `f99323bd`, and passed
  the Spark E4B full-NVFP4 FlashInfer mm-prefix baked-image row; see
  `results/sglang_gemma4_source_stack_image_27479559994/summary.md` and
  `results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_baked_20260614T072000JST/STOP_SUMMARY.md`.
  The original U22 carrier and first smoke remain documented in
  `results/sglang_gemma4_source_stack_image_27428220601/summary.md` and
  `results/sglang_spark_image_smoke_20260613T022153JST/summary.md`.

## Current Blocker State

- Do not run the full matched AR ladder for a broad claim until Claude's
  FlashInfer/numerics fix lands for the long-context NVFP4 red. Mail 0138
  reproduced the `+0.40` 12B loss on vLLM, exonerating SGLang radix/merge for
  that red.
- Keep the E4B fp8 comparator scoped red until the FlashInfer dispatcher fix
  for D512/VO256 1-byte KV lands. Re-running the existing fp8 row before that
  fix should reproduce the same dispatcher wall, not create a new claim.
- Short smoke or single-arm diagnostic rows are still allowed when they answer a
  new question, but label them as scoped diagnostics and do not quote capacity or
  quality as ladder-complete.
- The runner enforces this state by refusing known-blocked 12B full-NVFP4 and
  E4B fp8 rows unless `ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1` is set. Use that
  override only after a relevant FlashInfer/SGLang dependency changes or for an
  explicitly labeled diagnostic replay.

## Run

Default queue is 12B, 26B-A4B, then 31B:

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
git fetch origin
git checkout epoch2
git pull --ff-only

bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

After the shared quality or dispatcher fix lands, make the rerun intent explicit:

```bash
ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

To run one model only:

```bash
MODELS=google/gemma-4-12B-it \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

## What The Packet Proves

For each model, the packet runs three sequential servers:

- `bf16`: FlashInfer VO-split, BF16 weights, auto KV.
- `fp8`: FlashInfer VO-split, BF16 weights, fp8 KV.
- `fullnvfp4`: FlashInfer VO-split, BF16 weights, full NVFP4 K+V
  (`--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`).

It records:

- baked-image provenance and package/binary proof lines;
- chat determinism for a low-entropy Tokyo prompt;
- supplied-token PPL using the repository markdown corpus;
- bf16-vs-full-NVFP4 and fp8-vs-full-NVFP4 PPL comparisons;
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
