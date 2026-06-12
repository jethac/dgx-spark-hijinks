# SGLang DiffusionGemma DG-R3 VO-Split Smoke

Status: RED

## Scope

BF16/no-KV-quant DiffusionGemma 26B-A4B text-only serving through the experimental SGLang FlashInfer VO-split opt-in. This is not an NVFP4 KV or capacity row.

## Provenance

- Run: `sglang_dgemma_dgr3_vosplit_smoke_20260612T111121JST`
- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `dec4c040a8ede4561c1f26cccc599286643b49fd`
- FlashInfer: `f99323bd7d1c`
- Launch: `--dllm-algorithm Gemma4Renoise --attention-backend flashinfer --dtype bfloat16 --page-size 256 --disable-cuda-graph --disable-piecewise-cuda-graph`
- Environment: `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, offline HF mode

## Gates

- Revised DG-R2 text quality gate: FAIL
- Opt-in policy warning present: PASS
- D=512 geometry routes with `vo_split=True`: FAIL
- D=512 VO-split exposes `head_dim_vo=256`: FAIL

## Red Reasons

- revised text quality gate failed or is missing
- no D=512 geometry line with vo_split=True

## Geometry Evidence


## Quality Checks

- `capital_japan_direct`: stable=True non_empty=True answer_ok=True text='The capital of Japan is **Tokyo**.'
- `arithmetic_2_plus_2_direct`: stable=True non_empty=True answer_ok=True text='2 + 2 = 4'
- `dgx_spark_use`: stable=False non_empty=True answer_ok=True text='The NVIDIA DGX Spark desktop is used for developing, testing, and deploying AI models and machine learning workflows in a compact, desktop-grade form factor.'
