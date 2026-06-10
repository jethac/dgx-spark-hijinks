# Block A — P1 VO-split probe gates: ALL GREEN (2026-06-11 JST)

Window: Codex-granted (`CLAUDE_WINDOW_OPEN`), runs sequential in capped `--rm`
containers (`claude_blockA_1..4`), image
`jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`, FlashInfer
`/fisrc` at `fb7d62ea` (clean tree, verified), probe from this branch @ `fc9088c`.
Gate: cosine >= 0.9999 vs torch fp32 reference. JSONs + full logs in this directory.

| run | config | NHD cosine | HND cosine | pass |
|---|---|---:|---:|---|
| A1 | vo-split 2, d512 FP4-KV, batch 4, kv 96, qo 16, page 16 | 0.9999986 | 0.9999985 | PASS |
| A2 | vo-split 2, d512 FP4-KV, qo_len=1 (decode-as-prefill) | 0.9999985 | 0.9999987 | PASS |
| A3 | vo-split 2, d512 FP4-KV, signed E2M1 values | 0.9999984 | 0.9999983 | PASS |
| A4 | control: d128 FP4-KV linear V-SF, no split (decode+prefill) | 0.9999995–0.9999999 | 0.9999995–0.9999999 | PASS |

Notes:
- A1 `FLASHINFER_PREFILL_DEBUG_ONCE` proof: `head_dim_qk=512;head_dim_vo=256`,
  `dtype_kv=__nv_fp4x2_e2m1`, paged_v_cache `[24,16,2,128]` u8 (half-width view),
  o `[64,4,256]` bf16 — the asymmetric module with zero-copy V halves, post
  out-width fix.
- A2 is the gate the vLLM orchestration's `reorder_batch_threshold=0` decode
  routing depends on: qo_len=1 causal prefill is numerically clean per pass.
- A4 (linear V-SF, no split) backs the `VLLM_NVFP4_KV_LINEAR_V_SF=1` reader path
  at probe level for the standard head size; the WRITER-side regression (C-block)
  still needs the rebuilt `_C` extension.
- HND runs reuse the NHD-JIT'd module (0.02 s) — one module serves both layouts.
- P1 remaining: fp8-KV (512,256) pair (block E1 — needs the fp8 probe variant),
  then serving smokes.
- Session's only failure: first A1 attempt missed sibling module
  `spark_hardware.py` (probe import) — environmental, fixed by shipping the module
  alongside; no probe red anywhere.
