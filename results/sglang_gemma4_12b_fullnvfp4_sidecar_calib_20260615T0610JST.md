# SGLang Gemma 4 12B full-NVFP4 sidecar calibration proof

Date: 2026-06-15 JST

Scope: diagnostic/source-overlay proof of the productionized SGLang calibration loader. This is not claim-grade because it uses a source overlay and only runs the `fullnvfp4` row, not matched packaged bf16/fp8/full-NVFP4 comparators.

## Runtime

- Host: DGX Spark / GB10 `thinkstationpgx-00b4`
- Repo: `dgx-spark-hijinks@5b7516b7417e6a6da034a8d107bccc0b6392f23e`
- SGLang overlay: `jethac/sglang@a4a8ccc81d21a12df3cde4bdc06b306db086df23`
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Model: `google/gemma-4-12B-it`
- KV row: full NVFP4 K+V, `--kv-cache-dtype fp4_e2m1`
- Context: `8185`, reused prefix `4096`, scored continuation tokens `4088`
- Calibration env: `SGLANG_NVFP4_KV_CALIB=/hijinks/results/sglang_nvfp4_kv_calib_gemma4_12b_20260615`
- Calibration JSON: `results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8.json`

## Result

| row | NLL | PPL | delta vs bf16 baseline |
| --- | ---: | ---: | ---: |
| bf16 baseline | 4.571989822602 | 96.736406679507 | n/a |
| full-NVFP4 default/source-overlay plateau | 4.974959038640 | 144.742896160946 | +0.402969216038 |
| full-NVFP4 fixed env `0.1` diagnostic | 4.637407471576 | 103.276253589820 | +0.065417648974 |
| full-NVFP4 sidecar calibration `0.1` | 4.696971840077 | 109.614738416588 | +0.124982017475 |

Chat smoke returned `Tokyo` twice. The sidecar row is a large recovery from the red plateau, but it does not exactly reproduce the earlier debug fixed-env result.

## Proof Lines

The calibration sidecar matched by architecture signature and updated both FP4 KV pools:

```text
NVFP4 KV calibration applied: arch_signature=Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8 k_global_scale=0.1 v_global_scale=0.1 layers=48 pools=2 source=/hijinks/results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8.json
```

There were zero `NVFP4 KV auto-calibrated` fallback lines in the corrected `T061020JST` run, so the pool did not fall back to the amax-derived scale.

## Implementation Notes

The first sidecar implementation set `RadixAttention.k_scale_float`/`v_scale_float` before memory-pool initialization. That was insufficient: the pool still used amax calibration and reproduced the red plateau. The fix in `jethac/sglang@a4a8ccc81d` applies the loaded scales directly to `MHATokenToKVPoolFP4.k_global/v_global`, updates the float mirrors, and marks `_gs_calibrated=True` for both the full and SWA pools.

## Artifacts

- Corrected row: `results/sglang_gemma4_12b_fullnvfp4_sidecarcalib010_ctx8185_prefix4096_20260615T061020JST/`
- Calibration JSON: `results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/`

## Next

Before treating the sidecar path as claim-grade, explain the residual delta versus the debug fixed-env row (`+0.12498` vs `+0.06542`). The likely next A/B is same code, same run shape:

1. `SGLANG_NVFP4_KV_CALIB` sidecar path.
2. `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE=0.1` debug override path.

Both should log the final FlashInfer `k_scale/v_scale` and the pool global-scale values for the same layers.
