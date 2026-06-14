# SGLang 12B Chunked/Merge Diagnostic Packet

Purpose: scoped Spark packet for replaying the known-red 12B row under an
explicit diagnostic label while we decide whether a real SGLang chunked/merge
path knob exists. This is **not** a claim ladder row and must not be quoted as
broad SGLang Gemma 4 NVFP4 support.

## Background

The matched SGLang 12B row at ctx `8185`, reused prefix `4096`, page size `1`,
and graphs disabled is red by `+0.402969` nats/token:

`results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/STOP_SUMMARY.md`

Mail 0140 reclassifies that red:

- exact HF eager SDPA with nvfp4-qdq K+V: about `+0.1932` nats/token
- vLLM chunked/reuse paged+ragged merge: `+0.1906`
- vLLM single-prefill: `+0.4215`
- current SGLang single-/large-prefill-shaped row: `+0.402969`

So the diagnostic question is narrow: can SGLang be driven, without source
overlays or global-scale changes, through a route that lands near the `+0.19`
reference instead of the large-prefill `+0.40` artifact? The command below is
only the safe replay scaffold using the current runner. If a concrete
SGLang-side chunking knob is identified later, add it to this packet before
running and record it in the override reason.

## Preconditions

- Run only when Spark is free: `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN`
  absent and `docker ps` empty.
- Use the packaged SGLang image, not a Spark-local build and not loose-wheel
  injection:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Keep GB10 guardrails: one server at a time, Docker `--memory 100g`, no
  concurrent comparator servers.
- Do not set any `SGLANG_FP4_KV_*GLOBAL_SCALE_MULTIPLIER` knobs.
- Keep CUDA graphs disabled.

## Candidate Replay Scaffold

This uses the existing AR ladder runner in its explicit known-blocked diagnostic
mode. It runs only the already-red 12B full-NVFP4 arm and records the blocker
audit and override reason, so the artifact cannot be mistaken for a claim row.
As written, it is expected to reproduce the current path; it becomes a
chunked/merge discriminator only after a concrete serving-path knob is added to
the command.

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
git fetch origin
git checkout epoch2
git pull --ff-only

RUN_ID=sglang_12b_chunked_merge_diag_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST) \
MODELS=google/gemma-4-12B-it \
ROW_LABELS=fullnvfp4 \
CTX_LIST=8185 \
REUSE_PREFIX_LEN=4096 \
LOGPROB_START_LEN=4096 \
CONTEXT_LENGTH=8192 \
PAGE_SIZE=1 \
MEM_FRACTION_STATIC=0.72 \
ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
SGLANG_AR_LADDER_OVERRIDE_REASON='mail 0140 scoped replay scaffold; not claim-grade; add concrete chunked/merge knob before using as discriminator' \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

## Stop-On-Red

Stop immediately and commit artifacts if any of these happen:

- server does not reach readiness;
- `cached_tokens` is not `4096`;
- full-NVFP4 no longer serves a coherent `Tokyo` smoke;
- logs show a different failure class than the known 12B quality artifact;
- the diagnostic requires source overlays, global-scale multipliers, disabled
  radix cache, or a second concurrent server.

## Interpretation Rules

- If the replay still reports NLL near `4.974959` / delta near `+0.402969` against
  the banked bf16 baseline, it confirms the current SGLang serving route is
  still exercising the large-prefill artifact at unchanged dependencies.
- If a future run with an explicit chunked/merge path knob lands near the mail
  0140 reference cost (`~+0.19` against the banked bf16 baseline), it is a
  **scoped diagnostic lead only**. Mail Claude with the artifact and do not
  promote it into the ladder until a matched bf16 / full-NVFP4 rerun with the
  same route and claim audit passes.
- If it exposes a new FlashInfer/vLLM-facing red, mail Claude with the verbatim
  error and keep the SGLang ladder blocked.

## Required Artifact Notes

The stop summary must include:

- image digest and SGLang / FlashInfer refs;
- `blocker_audit.json` path and override reason;
- model, ctx, prefix, page size, graph state, and KV dtype;
- chat smoke result;
- `cached_tokens`;
- NLL/PPL result and comparison against the banked bf16 baseline;
- explicit scope label: "diagnostic only; known-blocked 12B row replay; not a
  claim-grade SGLang AR ladder result."
