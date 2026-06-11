# SGLang Overnight Ladder

Finished: `2026-06-11T17:50:20Z`

Rows attempted: `12`
Green rows: `2`
Red rows: `10`

| Row | Model | KV | Verdict | Notes |
|---|---|---|---|---|
| `12b_bf16` | `google/gemma-4-12B-it` | `auto` | RED |  |
| `12b_fp8` | `google/gemma-4-12B-it` | `fp8_e4m3` | RED |  |
| `12b_nvfp4` | `google/gemma-4-12B-it` | `fp4_e2m1` | RED |  |
| `26b_a4b_bf16` | `google/gemma-4-26B-A4B-it` | `auto` | RED |  |
| `26b_a4b_fp8` | `google/gemma-4-26B-A4B-it` | `fp8_e4m3` | RED |  |
| `26b_a4b_nvfp4` | `google/gemma-4-26B-A4B-it` | `fp4_e2m1` | RED |  |
| `31b_bf16` | `google/gemma-4-31B-it` | `auto` | RED |  |
| `31b_fp8` | `google/gemma-4-31B-it` | `fp8_e4m3` | RED |  |
| `31b_nvfp4` | `google/gemma-4-31B-it` | `fp4_e2m1` | RED |  |
| `e2b_bf16` | `google/gemma-4-E2B-it` | `auto` | GREEN | C1 3.923662672115819 |
| `e2b_fp8` | `google/gemma-4-E2B-it` | `fp8_e4m3` | RED | spark_smoke_rc=2, c1a_rc=128, c1b_rc=128, c2_rc=128, c3_rc=128, tokyo_smoke_not_coherent, c1_determinism_fail |
| `e2b_nvfp4` | `google/gemma-4-E2B-it` | `fp4_e2m1` | GREEN | C1 3.9244879447009984 |
