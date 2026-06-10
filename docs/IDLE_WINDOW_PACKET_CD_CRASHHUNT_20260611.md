# Window packet 2 — Blocks C+D (rebuilt image) + E2 crash hunt

Date authored: 2026-06-11 JST. Supersedes the consumed parts of
`IDLE_WINDOW_PACKET_P1_VOSPLIT_20260610.md` (A/B/E1/E3a done; E2 red).
Protocol and guardrails unchanged (marker handshake, one container, --memory
100g, util <= 0.72, sequential comparators).

Ledger in: A green (aa008d4), B green/M1 (c121c6f), E1 fp8 RED (guard term),
E3a BEFORE row banked (19.03 tok/s), geometry probes (3fc96c0): **Block D
probe-CLEARED at real 31B geometry; E2 crash NOT geometry-driven.**

## Gate: the rebuilt image
Blocks C/D need the image Codex was asked to cut: vLLM
`spark/hijinks-022-gemma4-mixed-kv@ad2337814` with REBUILT `_C` extension
(writer kernel 58d72be7e) + FlashInfer `spark/hijinks-022-fa2-d512@fb7d62ea`.
Check `docs/WHEEL_CONTAINER_MATRIX.md` for the tag. Without it, only the
crash-hunt block below runs (old image suffices).

## Crash hunt (no rebuild needed; probe @ >= 1bd6eb8)
Goal: standalone repro of `max_mma_kv: 0` (prefill.cuh:2964). Sweep on
`--probe fa2-vo-split-d512-vo256 --geometry e4b --skip-reference
--flashinfer-source-root /fisrc`, escalating:
1. `--plan-parity` alone (lean workload) — isolates plan kwargs;
2. warmup regime: `--batch-size 256 --qo-len 32 --kv-len 32 --plan-parity`
   (vLLM dummy-run-like: many short seqs), then `--batch-size 1 --qo-len 8192
   --kv-len 8192 --plan-parity` (one long prefill), then `--batch-size 8
   --qo-len 1024 --kv-len 1024 --plan-parity`;
3. if still green: sm-scale variants (`--sm-scale 0.0625` = Gemma
   query_pre_attn_scalar 256), and rerun the matrix WITHOUT --plan-parity to
   bisect which kwarg (if 1 reproduced, drop kwargs one at a time by editing
   the parity dict).
Record every row; FIRST red row = the repro; stop escalating, bisect to the
minimal trigger, then the dispatcher fix on the flashinfer branch has its
red test.

## Block C — linear-V-SF writer regression (rebuilt image)
`VLLM_NVFP4_KV_LINEAR_V_SF=1` on existing green rungs; numbers must match
swizzled rows: (1) Qwen NVFP4-KV quality gate + prefix-reuse row; (2) Gemma 3
27B first-token + PPL pair.

## Block D — 31B full-NVFP4 serving (rebuilt image; probe-cleared 3fc96c0)
```
VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  ... google/gemma-4-31B-it --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.72
```
NO skip-layers. Proof lines: "FA2 VO split (nvfp4 KV): head_size 512 runs as
2 passes"; per-layer nvfp4 on ALL layers; no Triton/fp8 fallback lines; no
decode-pathway assert. Gates: first-token sanity, then (separate runs)
capacity comparator vs fp8 and PPL pair. CAUTION: E2's crash was at WARMUP on
the bf16 path - if D crashes at warmup with max_mma_kv, capture and feed the
crash hunt; that's the FP4-flavored repro.

## Order
Crash hunt (cheap, old image) -> C -> D smoke -> D capacity/PPL. If the
image isn't ready, crash hunt only and release the box.
