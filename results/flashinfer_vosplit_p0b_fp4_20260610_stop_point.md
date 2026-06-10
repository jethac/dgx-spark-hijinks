# P0b stop point — FP4 VO-split red; head_dim_vo not plumbed through the FP4 path

Date: 2026-06-10 JST. Lane: `spark/hijinks-022-gemma4-mixed-kv`. Runs in Codex-granted
idle window (marker handshake honored; window released after).

## Status
- **P0 (bf16) remains GREEN**: (qk=512, vo=256) two-pass VO split, cosine 0.9999978.
- **P0b (NVFP4 KV) is RED** — but with a precise, fixable diagnosis, not a dead end.

## Runs (image `...e152cf4d-nvfp4kv`, FlashInfer source root `c3dae30f`, GB10)
`flashinfer_nvfp4_kv_probe.py --vo-split 2 --head-dim 512 --kv-container tuple --causal`,
both `--v-scale-layout swizzled` (deswizzle ON) and `linear` (`--no-deswizzle-flag`),
NHD+HND. Artifacts: `p0b_{swizzled,linear}_full.log` under
`/home/jethac/spark_tmp/claude_g4_probe_results/` (stdout JSON; the `--output` host-path
inside the container was invalid — fix the runner next window).

| variant | out shape per split | cosine vs dequant ref |
|---|---|---:|
| zero-copy view halves (strides 256B rows) | `[32, 4, 512]` (expected 256) | ~0.63–0.64 |
| contiguous halves (strides 128B rows) | `[32, 4, 512]` — **bit-identical outputs** | identical |

## Diagnosis
Two facts pin it:
1. `plan(head_dim_qk=512, head_dim_vo=256)` under `kv_data_type=uint8` (FP4) produced
   **512-wide output** — the FP4 branch sizes O from `head_dim_qk`.
2. View halves vs contiguous halves (different shapes AND strides) produced
   **bit-identical results** — the FP4 run path never consumed the passed V tensor's
   geometry; V addressing comes from plan-time constants.

Cause: the campaign's FP4 paged modules were only ever generated/exercised with
`head_dim_qk == head_dim_vo`, so `head_dim_vo` is not plumbed through the FP4 paged-prefill
branch (module generation, output allocation, V/V-SF stride derivation). The bf16 branch
plumbs it correctly (P0 green proves it).

## Consequence for the D=512 plan
K1 stands, but **P2 (FlashInfer enablement) is now real, bounded work** rather than
zero-code: plumb `head_dim_vo` through the FP4 paged path on
`jethac/flashinfer@spark/hijinks-022-fa2-d512` —
- module gen: emit FP4 paged-prefill modules keyed on (qk, vo) like the bf16 path;
- run path: size O from `head_dim_vo`; derive V data + V-SF strides from the passed
  tensors (or explicit args) instead of assuming `head_dim_qk`;
- with explicit V/V-SF strides, the zero-copy view trick should work again (no
  contiguity requirement).
This is Python/codegen + params plumbing — same shape of work as the SWA-prefill SF-stride
fix — NOT kernel math (the kernel templates already take HEAD_DIM_QK/HEAD_DIM_VO).

## Next steps
1. P2: head_dim_vo plumbing for the FP4 paged path (offline code work, no GPU needed).
2. Rerun P0b in the next idle window (fix the runner's --output to a container path).
3. Then P1 correctness gates → P3 vLLM orchestration.
