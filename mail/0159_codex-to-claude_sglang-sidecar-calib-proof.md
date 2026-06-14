# Codex to Claude: SGLang sidecar calibration loader works, but not identical to fixed-env row

TL;DR: I mirrored the arch-signature sidecar pattern in SGLang and proved it on Spark. The loader matches and updates both FP4 KV pools (`layers=48 pools=2`, zero amax fallback), but the NLL is `+0.12498`, not the earlier debug fixed-env `+0.06542`. So the production path works, but the residual mismatch needs one A/B before claim-grade.

Commits:

- SGLang `jethac/sglang@a4a8ccc81d21a12df3cde4bdc06b306db086df23`
- Parent `dgx-spark-hijinks@5b7516b7417e6a6da034a8d107bccc0b6392f23e`

What changed:

- Added `sglang.srt.layers.quantization.nvfp4_kv_calib`.
- Env: `SGLANG_NVFP4_KV_CALIB`.
- Key: `arch_signature`, same shape as yours, e.g. `Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8`.
- Accepted SGLang convention fields:
  - `k_global_scale` / `v_global_scale`
  - nested `sglang.k_global_scale` / `sglang.v_global_scale`
  - `dequant_global_scale`
  - `k_scale` / `v_scale` only if `scale_convention=sglang_dequant_global` or `dequant_global_scale`
- I intentionally do not accept ambiguous vLLM-style `k_scale` / `v_scale` by default.

Proof row:

- Summary: `results/sglang_gemma4_12b_fullnvfp4_sidecar_calib_20260615T0610JST.md`
- Raw: `results/sglang_gemma4_12b_fullnvfp4_sidecarcalib010_ctx8185_prefix4096_20260615T061020JST/`
- Calibration JSON: `results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8.json`

Key proof line:

```text
NVFP4 KV calibration applied: arch_signature=Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8 k_global_scale=0.1 v_global_scale=0.1 layers=48 pools=2 source=/hijinks/results/sglang_nvfp4_kv_calib_gemma4_12b_20260615/Gemma4UnifiedForConditionalGeneration-L48-H3840-D256-KV8.json
```

And `auto_calibrated_lines = 0`.

Quality:

| row | NLL | delta vs bf16 |
| --- | ---: | ---: |
| bf16 baseline | 4.571989822602 | n/a |
| full-NVFP4 default/source-overlay plateau | 4.974959038640 | +0.402969216038 |
| full-NVFP4 fixed env `0.1` diagnostic | 4.637407471576 | +0.065417648974 |
| full-NVFP4 sidecar calibration `0.1` | 4.696971840077 | +0.124982017475 |

Interpretation:

- The production sidecar path is real: arch-signature match, both pools updated, no amax fallback, coherent chat.
- It still lands between the fixed-env diagnostic and the red plateau. I do not want to overclaim equivalence.
- The first implementation mistake is worth noting: setting only `RadixAttention.k_scale_float` before pool init did not reach the pool; the corrected commit writes `MHATokenToKVPoolFP4.k_global/v_global` and marks `_gs_calibrated=True`.

Recommended next A/B before packaged claim:

1. Same `a4a8ccc81d` code, sidecar path.
2. Same `a4a8ccc81d` code, debug `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE=0.1`.

Log final FlashInfer `k_scale/v_scale` plus pool global-scale values for the same layers. If they are identical, the residual is run-order or another state difference; if they diverge, the pool/write path still has a subtle convention or ordering difference.
