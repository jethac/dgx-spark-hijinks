# Block E2/E3 — E4B VO-split serving smoke RED; BEFORE row banked (2026-06-11)

## E2: VLLM_FLASHINFER_VOSPLIT=1 crashes at warmup — NEW WALL FOUND
`RuntimeError ... prefill.cuh:2964: Unsupported max_mma_kv: 0` in
`_run_vo_split_prefill -> paged_run`, deterministic, during kernel warmup.
The orchestration itself worked: all three log proofs appeared (Gemma4Config
forced FLASHINFER with VO split; builder logged "FA2 VO split (auto KV):
head_size 512 runs as 2 passes"; zero TRITON_ATTN lines).

Diagnosis: the compile-time trait guard (IsInvalid) passes (512,256) — that is
what P0/Block A validated — but the RUNTIME smem dispatcher computes zero KV
MMA tiles under real E4B GQA geometry. Probes used num_qo_heads=4 /
num_kv_heads=2 (group 2); serving geometry packs a larger GQA group into the
q tile, and D_QK=512 bf16 Q smem leaves no room for KV tiles. Lesson applied:
the campaign's own measured-geometry rule now extends to kernel probes — the
next probe must replicate the serving head counts, page size, soft-cap and
window config exactly.

Consequences:
- bf16 Triton retirement (vllm#38887 answer): RED at serving until a
  FlashInfer dispatch/tiling fix (shrink q tiling so max_mma_kv >= 1) on
  jethac/flashinfer@spark/hijinks-022-fa2-d512. Same bounded class as the
  fp8 dispatch term from E1.
- NVFP4 Block D (31B) risk: Block A's FP4 green also used group-2 GQA. Must
  re-probe with 31B/E4B real geometry BEFORE the serving smoke.

## E3a: BEFORE row (default Triton force), healthy, medians of 3
google/gemma-4-E4B-it bf16, util 0.72, vLLM 0.22.1rc1.dev281+gad2337814:
- decode (256 tok, batch 1): 19.03 tok/s
- TTFT (2083-tok prompt): 0.317 s; prefill 6570 tok/s
- concurrent x4 aggregate decode: 92.04 tok/s
Note vs vllm#38887: GB10's Triton fallback does ~19 tok/s (the ~9 in the
issue is an RTX 4090 row). Benchmark harness: bench_e3.py (streaming TTFT,
temp 0, per-request nonces after v1 medians showed prefix-cache
contamination; v1 JSON retained, marked contaminated).

E3b/after: N/A (E2 unhealthy). Artifacts: this directory (server logs, crash
traceback, benchmark JSONs, harness, runner).
