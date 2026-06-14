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
- The runner enforces this state by refusing full-NVFP4 12B/26B-A4B/31B
  claim-ladder rows while the shared 12B long-context quality blocker is open,
  and by refusing the known-blocked E4B fp8 row. Set
  `ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1` only after a relevant
  FlashInfer/SGLang dependency changes or for an explicitly labeled diagnostic
  replay. Override runs that touch blocked rows must also set
  `SGLANG_AR_LADDER_OVERRIDE_REASON`, which the runner records in preflight
  artifacts. The offline regression check for these guards is
  `bash scripts/test_sglang_gemma4_ar_ladder_guard.sh`.
- To check whether the dependency state has changed enough to justify a
  diagnostic override, run
  `python3 scripts/sglang_gemma4_ar_ladder_blocker_audit.py`. A changed ref is
  not a green result; it only means the smallest matched red row is worth
  replaying with an explicit override reason and fresh package provenance.
  Current audit:
  `results/sglang_gemma4_ar_ladder_blocker_audit_20260614T0826JST.json`.

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
SGLANG_AR_LADDER_OVERRIDE_REASON="flashinfer <ref>: shared quality or fp8 dispatcher fix" \
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

- `blocker_audit.json`, which captures the FlashInfer/SGLang dependency refs
  used to justify the run or diagnostic override. Claim audit hard-fails
  `blocked-known-red-dependencies`; dependency-changed reruns can be audited,
  but unchanged known-red refs cannot become claim-grade by filling in row JSON;
- `fp8_dispatch_audit.json` for `google/gemma-4-E4B-it` fp8 rows, when logs
  are available, to classify the known D512/VO256 `NUM_MMA_KV=1` dispatcher
  red automatically;
- baked-image provenance and package/binary proof lines;
- chat determinism for a low-entropy Tokyo prompt;
- supplied-token PPL using the repository markdown corpus;
- bf16-vs-full-NVFP4 and fp8-vs-full-NVFP4 PPL comparisons;
- server logs containing Gemma KV pool geometry lines.
- exact row identities: `bf16` uses `kv_cache_dtype=auto`, `fp8` uses
  `fp8_e4m3`, and `fullnvfp4` uses `fp4_e2m1`;
- exact comparison coverage: every required comparison must include the same
  contexts as manifest `ctx_list`;
- PPL artifact provenance: every row's `*_ppl.json` must match the manifest
  image, expected KV dtype, tokenizer/model, context list, reuse-prefix length,
  logprob start, max-new-tokens, positive scored-token count, zero missing or
  mismatched tokens, and GB10 `sm_121` hardware evidence.
- chat artifact integrity: every row's repeated `*_chat_1.json` and
  `*_chat_2.json` artifacts must be OpenAI `chat.completion` responses for the
  manifest model, finish with `stop`, include positive usage counts, agree with
  each other, and contain the expected deterministic `Tokyo` answer.
- backing artifact presence: every claim row must retain its chat, PPL,
  preflight, provenance, server, inspect, summary, and comparison JSON/log
  files. The audit accepts Spark absolute paths when run on Spark and falls
  back to manifest-relative paths when a result bundle has been copied into the
  repo.
- preflight integrity: the run-level and per-row preflight logs must match the
  manifest run id, model list, row labels, image digest, context/page/memory
  knobs, row KV dtype, `source_overlay=0`, and empty retracted global-scale
  multiplier knobs.
- container inspect provenance: every row must prove it ran the manifest image
  digest from `/hijinks`, with the 100g Docker memory and swap limits, host
  network/IPC, no source-overlay mount for claim rows, Spark `sm_121a` JIT
  flags, offline HF mode, FlashInfer VO-split enabled, graphs disabled, and the
  expected `--kv-cache-dtype` argument for the row.
- provenance markers: every claim row's provenance/server logs must include
  package versions, `binary_md5 sgl_kernel`, resolved FlashInfer source paths,
  `attention_backend='flashinfer'`, running-model Gemma KV geometry, and the
  VO-split route marker. Full-NVFP4 rows must additionally show FP4 module
  trace markers, scale-factor tensors, and `deswizzle_macro_active=False`.
- capacity provenance: row summaries extract `full_tokens`, `swa_tokens`,
  per-token byte geometry, cell size, pool dtype, and pool class from the
  running server's `SGLANG_GEMMA_KV_POOL_CONFIG` / `SGLANG_GEMMA_KV_SWAKVPOOL`
  lines. The claim audit requires positive capacity fields and the expected
  token-slot ordering `bf16 < fp8 < fullnvfp4` per model.
- corpus/corpus-manifest paths and the exact shape knobs (`ctx_list`,
  `reuse_prefix_len`, `logprob_start_len`, `page_size`, `context_length`,
  `max_new_tokens`, graphs disabled).
- packaged-image provenance: `source_overlay=false` and
  `allow_retracted_global_scale_diagnostic=false` for claim-grade rows.

The runner writes `results/<run_id>/claim_audit.json` after every manifest. For
an explicit post-run gate or to regenerate it manually:

```bash
python3 scripts/sglang_gemma4_ar_claim_audit.py results/<run_id>/manifest.json \
  --max-delta-nats 0.25 \
  --output results/<run_id>/claim_audit.json
```

The default threshold is a conservative mechanical tripwire, not a public
quality promise. Tighten it with `SGLANG_AR_CLAIM_AUDIT_MAX_DELTA_NATS=<n>`
when the publication claim needs a stricter bar.
Set `SGLANG_AR_CLAIM_AUDIT_STRICT=1` when a full claim-grade ladder run should
fail the shell process if the audit is red.
The runner refuses `SGLANG_FP4_KV_*GLOBAL_SCALE_MULTIPLIER` env vars by default:
that calibration hypothesis was retracted in mail/0132. Use
`SGLANG_ALLOW_RETRACTED_GLOBAL_SCALE_DIAGNOSTIC=1` plus
`SGLANG_AR_LADDER_OVERRIDE_REASON=...` only for a clearly labeled diagnostic replay.
Before editing either the runner manifest contract or the claim audit, run:

```bash
bash scripts/test_sglang_gemma4_ar_claim_audit.sh
```

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
