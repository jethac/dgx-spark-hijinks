## strata_fp8_vs_bf16_c1

baseline `claude_bf16_vosplit_c1_ctx8191` vs comparator `claude_fp8_c1_ctx8191` — 8190 tokens, mean delta **-0.1392** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 3923 | +0.1174 | -40.4% |
| [0.5,1) | 261 | +0.2851 | -6.5% |
| [1,2) | 322 | +0.3360 | -9.5% |
| [2,4) | 568 | +0.0015 | -0.1% |
| [4,8) | 1024 | -0.3115 | +28.0% |
| >=8 | 2092 | -0.7004 | +128.5% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | -0.3701 | +33.2% |
| 1023-2045 | 1023 | -0.0518 | +4.6% |
| 2046-3068 | 1023 | -0.0810 | +7.3% |
| 3069-4091 | 1023 | -0.2840 | +25.5% |
| 4092-5114 | 1023 | -0.0492 | +4.4% |
| 5115-6137 | 1023 | -0.0700 | +6.3% |
| 6138-7160 | 1023 | -0.1631 | +14.6% |
| 7161-8190 | 1030 | -0.0453 | +4.1% |

sign: 54.3% improved / 45.3% worsened / 30 tied; mean |delta| 0.7939 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 1309 | `eth` | 0.008 | 22.129 | +22.121 | `` \| ported in `j[eth]ac/vllm@` |
| 3514 | `results` | 21.274 | 0.023 | -21.251 | ` `spark_doctor`: `[results]/spark_doctor_ae` |
| 6272 | `---\|` | 19.498 | 0.000 | -19.498 | ` tok/s \|\n\|[---\|]---:\|---:\|---:\|` |
| 1939 | `ang` | 6.046 | 24.341 | +18.296 | ` output sanity in Gemma SGL[ang] row \| include chat smoke and` |
| 4877 | `ang` | 1.025 | 18.137 | +17.112 | `: `results/sgl[ang]_20260` |
| 3482 | `flash` | 16.026 | 0.884 | -15.142 | `gemma26_d[flash]_20260` |
| 1324 | ` evidence` | 14.887 | 0.016 | -14.870 | `04e1b`;[ evidence]: `third_party/` |
| 1458 | `ll` | 0.038 | 14.265 | +14.227 | `/vllm/v[ll]m/config/compilation` |
| 5868 | `fp` | 4.511 | 18.591 | +14.080 | `_1_5b_[fp]8kv_202` |
| 6888 | `fp` | 25.126 | 11.599 | -13.527 | `315JST_[fp]8_raw_2plus` |
| 3432 | `_` | 13.473 | 0.002 | -13.471 | `: `results/aeon[_]gemma26_d` |
| 5796 | `-` | 0.188 | 13.619 | +13.431 | `_smoke.json`\n[-] fp8 benchmark: `results` |
| 2287 | `\n\n` | 4.334 | 17.487 | +13.153 | `fter/model pair exists \|[\n\n]## Still Needed Counterparts\n\n` |
| 4295 | ` \|` | 1.016 | 13.993 | +12.977 | `\| `medium_decode`[ \|] 36 \| 1` |
| 4546 | ` `` | 15.939 | 3.052 | -12.887 | `abi3.so`, and[ `]_vllm_fa` |
| 5995 | `-` | 18.731 | 5.896 | -12.835 | `_startup.log`\n[-] patched fp4 FlashInfer startup` |
| 3477 | `emma` | 0.002 | 12.790 | +12.789 | `results/aeon_g[emma]26_dflash_` |
| 7443 | `\n\n` | 17.540 | 4.780 | -12.760 | `5`\n\nArtifacts:[\n\n]- summary: `results/` |
| 4237 | ` \|` | 12.718 | 0.061 | -12.657 | ` seconds \| decode tok/s[ \|]\n\|---\|---:\|---` |
| 6859 | `ang` | 12.575 | 0.003 | -12.572 | `- `results/sgl[ang]_qwen_fp4` |
| 6861 | `q` | 24.970 | 12.560 | -12.410 | `results/sglang_[q]wen_fp4kv_` |
| 6809 | ` `` | 12.462 | 0.120 | -12.343 | `manifest.json`\n-[ `]results/sglang_` |
| 6907 | `_` | 2.574 | 14.636 | +12.061 | ` `results/sglang[_]qwen_fp4kv` |
| 6995 | `fp` | 9.281 | 21.204 | +11.923 | `glang_qwen_[fp]4kv_autosafe` |
| 5859 | `wen` | 1.041 | 12.859 | +11.818 | `/sglang_q[wen]25_1_5` |
| 4609 | `\n\n` | 12.076 | 0.286 | -11.790 | ` settings, and server API.[\n\n]## 2026` |
| 3103 | `it` | 15.316 | 3.599 | -11.717 | `B-A4B-[it]-Uncensored-NV` |
| 5410 | `Flash` | 1.669 | 13.377 | +11.708 | ` in audited SGLang/[Flash]Infer objects.\n- The` |
| 6985 | ` `` | 2.660 | 14.156 | +11.496 | `benchmark.json`\n-[ `]results/sglang_` |
| 7740 | `5` | 11.368 | 0.000 | -11.368 | `08T145[5]JST_nvfp4` |
| 7010 | `8` | 2.540 | 13.892 | +11.353 | `026060[8]T1315J` |
| 6872 | `2` | 11.516 | 22.844 | +11.327 | `kv_autosafe_[2]026060` |
| 2264 | `.` | 11.351 | 0.045 | -11.307 | ` DFlash literally; evaluate llama[.]cpp-native speculative/draft` |
| 7809 | `flash` | 1.107 | 12.287 | +11.180 | `nvfp4_kv_[flash]infer_server.log`` |
| 3498 | `ST` | 11.593 | 0.440 | -11.153 | `T0436J[ST]_container_versions.json` |
| 5726 | `2` | 11.063 | 0.001 | -11.062 | `_040mem_[2]026060` |
| 3391 | `on` | 0.035 | 11.003 | +10.968 | ` log: `results/ae[on]_gemma26_` |
| 5869 | `8` | 0.000 | 10.950 | +10.950 | `1_5b_fp[8]kv_2026` |
| 4069 | `\n\n` | 0.001 | 10.926 | +10.925 | `\n```\n\nArtifacts:[\n\n]- environment: `results/` |
| 3430 | `ae` | 13.911 | 2.990 | -10.921 | ` log lines: `results/[ae]on_gemma26` |
