# FlashInfer NVFP4 Linear V-Scale Deswizzle Matrix

Date: 2026-06-10 JST

Purpose: test whether applying vLLM's V scale-factor de-swizzle transform to a
SGLang-style linear V scale-factor layout corrupts FlashInfer FA2 NVFP4 paged KV reads.

Runtime:

- Host: GB10, `sm_121`, 48 SMs
- Container image: `sglang-source-stack-c3dae30f-e631a13fd`
- FlashInfer source overlay: `third_party/flashinfer`
- FlashInfer version reported by Python: `0.6.13`
- Torch: `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`
- Probe: `scripts/flashinfer_nvfp4_kv_probe.py`
- Runner: `scripts/run_flashinfer_nvfp4_page_deswizzle_matrix.sh`

The probe used tuple K/V cache inputs, NHD layout, signed synthetic NVFP4 K/V, linear V scale
factors, BF16 query/output, `head_dim=128`, `num_kv_heads=2`, `num_qo_heads=4`, `kv_len=96`,
and `qo_len=32`.

## Results

| page_size | deswizzle macro | operation | cosine vs reference | max_abs | verdict |
|---:|---|---|---:|---:|---|
| 1 | off | decode | 0.9999995 | 0.015625 | pass |
| 1 | off | prefill | 0.9999996 | 0.015625 | pass |
| 1 | on | decode | 0.7897140 | 2.996094 | fail |
| 1 | on | prefill | 0.1745699 | 117.3125 | fail |
| 16 | off | decode | 0.9999995 | 0.015625 | pass |
| 16 | off | prefill | 0.9999996 | 0.015625 | pass |
| 16 | on | decode | 0.7745688 | 3.632812 | fail |
| 16 | on | prefill | 0.6371106 | 5.0625 | fail |

## Interpretation

This validates the mechanism: if `FLASHINFER_PAGED_V_SF_DESWIZZLE=1` is applied to a linear
V scale-factor layout, FlashInfer FA2 NVFP4 KV reads become numerically wrong. The failure is
not limited to page size 1; page size 16 also fails when the layout and macro disagree.

The matching no-deswizzle rows pass for both page sizes and both operations, so page size 1
linear V scale factors are not inherently broken in this standalone FlashInfer path.

## Link To SGLang Bug

This matrix does not prove the SGLang serving failure was caused by a leaked deswizzle macro.
The preceding cache grep (`results/sglang_deswizzle_flag_check_20260610TmanualJST.md`) found
no surviving host-side SGLang generated module or JIT log with the macro. The failed serving
containers' `/root/.cache/flashinfer` contents were not available after container cleanup.

So the current state is:

- deswizzle-on + SGLang-linear V-SF is a proven corruption mechanism;
- page_size=1 linear V-SF without deswizzle is cleared in standalone FlashInfer;
- there is no surviving evidence that the failing SGLang run actually had the deswizzle macro
  active.

## Artifacts

- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST/page1_deswizzle_off.json`
- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST/page1_deswizzle_on.json`
- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST/page16_deswizzle_off.json`
- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST/page16_deswizzle_on.json`
- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST/summary.json`
