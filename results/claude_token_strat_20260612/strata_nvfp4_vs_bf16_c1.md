## strata_nvfp4_vs_bf16_c1

baseline `claude_bf16_vosplit_c1_ctx8191` vs comparator `claude_nvfp4_c1_ctx8191` — 8190 tokens, mean delta **-0.3318** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 3923 | +0.1604 | -23.1% |
| [0.5,1) | 261 | +0.3105 | -3.0% |
| [1,2) | 322 | +0.1212 | -1.4% |
| [2,4) | 568 | +0.0047 | -0.1% |
| [4,8) | 1024 | -0.6282 | +23.7% |
| >=8 | 2092 | -1.3510 | +104.0% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | -0.2712 | +10.2% |
| 1023-2045 | 1023 | -0.5932 | +22.3% |
| 2046-3068 | 1023 | -0.5002 | +18.8% |
| 3069-4091 | 1023 | -0.5006 | +18.8% |
| 4092-5114 | 1023 | -0.2966 | +11.2% |
| 5115-6137 | 1023 | -0.2280 | +8.6% |
| 6138-7160 | 1023 | -0.1971 | +7.4% |
| 7161-8190 | 1030 | -0.0694 | +2.6% |

sign: 54.7% improved / 45.1% worsened / 24 tied; mean |delta| 1.1434 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 5094 | ` pass` | 25.876 | 2.442 | -23.434 | `60.50 \|[ pass] \|\n\| `medium_` |
| 5165 | `a` | 19.812 | 0.001 | -19.811 | ` \| 1 \| n/[a] \| 0.03` |
| 6272 | `---\|` | 19.498 | 0.061 | -19.437 | ` tok/s \|\n\|[---\|]---:\|---:\|---:\|` |
| 1276 | `py` | 20.504 | 1.811 | -18.693 | `rope_text_fallback.[py]` \| text-only M` |
| 1723 | `GU` | 19.112 | 0.608 | -18.503 | ` is not blessed \| current G[GU]F Q4_0 path` |
| 6861 | `q` | 24.970 | 7.314 | -17.656 | `results/sglang_[q]wen_fp4kv_` |
| 6953 | `fp` | 23.990 | 7.376 | -16.614 | `glang_qwen_[fp]4kv_autosafe` |
| 6906 | `ang` | 19.793 | 3.469 | -16.325 | `- `results/sgl[ang]_qwen_fp4` |
| 7553 | `wen` | 2.669 | 18.860 | +16.192 | `/vllm_q[wen]_nvfp4_kv` |
| 5956 | `_` | 0.000 | 16.077 | +16.077 | ` `results/sglang[_]qwen25_1` |
| 6862 | `wen` | 15.720 | 0.012 | -15.708 | `/sglang_q[wen]_fp4kv_aut` |
| 1443 | `1` | 0.000 | 15.268 | +15.268 | `@6804e[1]b`; evidence: `third` |
| 5581 | `\n\n` | 15.232 | 0.002 | -15.230 | `8`\n\nArtifacts:[\n\n]- summary: `results/` |
| 1831 | `quant` | 15.172 | 0.009 | -15.163 | ` AEON weights unless re-[quant]izing; run `scripts/` |
| 6819 | `fp` | 21.756 | 6.997 | -14.758 | `glang_qwen_[fp]4kv_autosafe` |
| 5419 | `ang` | 0.025 | 14.771 | +14.746 | `.\n- The SGL[ang] log labeled the GB10` |
| 1250 | `ang` | 0.029 | 14.621 | +14.592 | `` failure; current SGL[ang] already has hybrid/spec paths` |
| 3349 | `g` | 17.795 | 3.288 | -14.507 | ` `results/aeon_[g]emma26_dflash` |
| 1317 | `8` | 14.348 | 0.000 | -14.348 | `/vllm@6[8]04e1b`;` |
| 3434 | `emma` | 17.735 | 3.434 | -14.301 | `results/aeon_g[emma]26_dflash_` |
| 6951 | `wen` | 7.882 | 22.112 | +14.230 | `/sglang_q[wen]_fp4kv_aut` |
| 3477 | `emma` | 0.002 | 14.142 | +14.140 | `results/aeon_g[emma]26_dflash_` |
| 3433 | `g` | 18.295 | 4.203 | -14.092 | ` `results/aeon_[g]emma26_dflash` |
| 5836 | `ST` | 14.043 | 0.002 | -14.041 | `T0332J[ST]_openai_benchmark.json` |
| 6316 | `decode` | 11.498 | 25.476 | +13.979 | ` \|\n\| `medium_[decode]` \| 56 \|` |
| 7443 | `\n\n` | 17.540 | 3.899 | -13.640 | `5`\n\nArtifacts:[\n\n]- summary: `results/` |
| 3432 | `_` | 13.473 | 0.000 | -13.472 | `: `results/aeon[_]gemma26_d` |
| 4798 | `\n\n` | 13.537 | 0.302 | -13.235 | ` left running\n\nArtifacts:[\n\n]- smoke: `results/` |
| 4546 | ` `` | 15.939 | 2.723 | -13.216 | `abi3.so`, and[ `]_vllm_fa` |
| 6961 | `2` | 0.063 | 13.203 | +13.140 | `kv_autosafe_[2]026060` |
| 6877 | `6` | 18.110 | 5.147 | -12.963 | `_20260[6]08T131` |
| 1305 | ` ported` | 17.129 | 4.174 | -12.955 | `SupportsMRoPE` \|[ ported] in `jethac/` |
| 7012 | `1` | 0.083 | 12.960 | +12.877 | `60608T[1]315JST_` |
| 1319 | `4` | 12.865 | 0.000 | -12.865 | `llm@680[4]e1b`; evidence:` |
| 6941 | `\n` | 16.007 | 3.160 | -12.847 | `row_manifest.json`[\n]- `results/sgl` |
| 1938 | `GL` | 13.163 | 0.419 | -12.744 | ` deterministic output sanity in Gemma S[GL]ang row \| include chat smoke` |
| 1324 | ` evidence` | 14.887 | 2.185 | -12.702 | `04e1b`;[ evidence]: `third_party/` |
| 4237 | ` \|` | 12.718 | 0.039 | -12.679 | ` seconds \| decode tok/s[ \|]\n\|---\|---:\|---` |
| 7710 | ` `` | 16.870 | 4.223 | -12.647 | `FP4-KV benchmark:[ `]results/vllm_` |
| 6010 | `_` | 12.551 | 0.016 | -12.535 | ` `results/sglang[_]qwen25_1` |
