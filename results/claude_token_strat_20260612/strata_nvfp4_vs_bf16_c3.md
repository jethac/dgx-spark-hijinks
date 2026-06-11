## strata_nvfp4_vs_bf16_c3

baseline `claude_bf16_vosplit_c3_ctx8191` vs comparator `claude_nvfp4_c3_ctx8191` — 8190 tokens, mean delta **-0.0468** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 5291 | +0.1702 | -235.0% |
| [0.5,1) | 230 | +0.5403 | -32.4% |
| [1,2) | 279 | +0.4744 | -34.5% |
| [2,4) | 429 | +0.1019 | -11.4% |
| [4,8) | 616 | -0.3819 | +61.4% |
| >=8 | 1345 | -1.0027 | +352.0% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | -0.1340 | +35.7% |
| 1023-2045 | 1023 | +0.0532 | -14.2% |
| 2046-3068 | 1023 | +0.0907 | -24.2% |
| 3069-4091 | 1023 | +0.0940 | -25.1% |
| 4092-5114 | 1023 | -0.1505 | +40.2% |
| 5115-6137 | 1023 | -0.2949 | +78.8% |
| 6138-7160 | 1023 | +0.0779 | -20.8% |
| 7161-8190 | 1030 | -0.1103 | +29.7% |

sign: 46.7% improved / 52.4% worsened / 73 tied; mean |delta| 0.8410 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 5953 | `4` | 0.000 | 24.632 | +24.632 | `ang_qwen_fp[4]kv_clean*.log",` |
| 6056 | `gl` | 25.944 | 2.266 | -23.678 | `\n            "results/s[gl]ang_qwen_fp` |
| 5944 | `/` | 0.000 | 20.938 | +20.938 | `json",\n            "results[/]sglang_qwen` |
| 6813 | `records` | 21.891 | 2.066 | -19.825 | ` serves, "\n            "[records] backend/DFlash evidence,` |
| 7845 | `amac` | 20.441 | 1.637 | -18.804 | `\n            "results/ll[amac]pp_nvfp4_` |
| 3756 | `        ` | 0.313 | 18.463 | +18.149 | `"architecture": architecture,\n[        ]"image_tags": image` |
| 8042 | `m` | 18.695 | 0.550 | -18.145 | `results/llamacpp*[m]xfp4*row_` |
| 5900 | `q` | 0.043 | 17.000 | +16.957 | `results/sglang_[q]wen25_1_` |
| 6185 | `ang` | 16.079 | 0.000 | -16.079 | `            "results/sgl[ang]_qwen_fp4` |
| 6072 | `\n` | 16.330 | 0.542 | -15.788 | `autosafe*.json",[\n]            "results/sgl` |
| 2521 | `8` | 15.781 | 0.000 | -15.781 | `text(encoding="utf-[8]", errors="replace").strip` |
| 6076 | `/` | 15.533 | 0.041 | -15.492 | `json",\n            "results[/]sglang_qwen` |
| 7844 | `ll` | 23.766 | 9.000 | -14.766 | `",\n            "results/[ll]amacpp_nvfp4` |
| 4366 | `compute` | 0.429 | 15.063 | +14.633 | `121a", "[compute]_121a",` |
| 4973 | `sm` | 16.131 | 1.640 | -14.491 | ` arch_list_has_[sm]120),\n            ` |
| 6054 | `/` | 27.772 | 13.292 | -14.480 | `log",\n            "results[/]sglang_qwen` |
| 3682 | `architecture` | 15.900 | 1.649 | -14.251 | ` data[0]\n        [architecture] = first.get("Architecture` |
| 6218 | `q` | 16.384 | 2.468 | -13.915 | `results/sglang_[q]wen_fp4kv_` |
| 5812 | `ang` | 14.090 | 0.282 | -13.808 | `                "results/sgl[ang]*fp4_e2` |
| 4971 | `has` | 14.256 | 0.583 | -13.672 | `x or arch_list_[has]_sm120),` |
| 6294 | `"` | 13.313 | 0.000 | -13.312 | `ile*.md",\n            ["]results/sglang_` |
| 6436 | `            ` | 1.316 | 14.396 | +13.080 | `\n            "_startup",\n[            ]"_autosafe",\n` |
| 4082 | `    ` | 13.004 | 0.014 | -12.990 | `", type=Path)\n[    ]parser.add_argument("--` |
| 8131 | `native` | 15.662 | 2.701 | -12.962 | `results/llamacpp*[native]_fp4*build_` |
| 5041 | ` arch` | 0.728 | 13.658 | +12.930 | `_or_ptx or[ arch]_list_has_sm` |
| 4967 | ` arch` | 12.773 | 0.000 | -12.773 | `_or_ptx or[ arch]_list_has_sm` |
| 7778 | `results` | 12.656 | 0.010 | -12.647 | `_patterns=(\n            "[results]/llamacpp_q` |
| 4954 | `evidence` | 8.328 | 20.875 | +12.547 | `_or_ptx_[evidence]": bool(has_family` |
| 6976 | `\n` | 0.087 | 12.542 | +12.454 | `stop_point.md",[\n]            "results/aeon` |
| 6272 | `gl` | 14.261 | 1.912 | -12.349 | `\n            "results/s[gl]ang_qwen_fp` |
| 6494 | `model` | 6.930 | 19.278 | +12.348 | `sglang",\n        [model]_family="qwen",` |
| 8126 | `/` | 1.450 | 13.761 | +12.311 | `json",\n                "results[/]llamacpp*native_` |
| 8101 | `                ` | 2.466 | 14.632 | +12.166 | `_audit.json",\n[                ]"results/llamacpp` |
| 5132 | `n` | 12.094 | 0.055 | -12.039 | `keys=True) + "\[n]", encoding="utf-8` |
| 4032 | `    ` | 2.529 | 14.497 | +11.968 | `": torch_cuda,\n[    ]}\n\n\ndef main() ->` |
| 5545 | `        ` | 12.551 | 0.659 | -11.891 | `audit.json",),\n[        ]),\n        partial_patterns` |
| 4918 | `native` | 0.314 | 12.125 | +11.811 | `,\n            "explicit_[native]_sm121":` |
| 6078 | `gl` | 13.309 | 1.652 | -11.657 | `\n            "results/s[gl]ang_qwen_fp` |
| 7673 | `_` | 12.510 | 0.895 | -11.615 | `\n        ),\n        claim[_]groups=(\n            (\n` |
| 7246 | `quality` | 19.343 | 7.742 | -11.601 | ` capacity, "\n            "[quality], and serving manifests for both` |
