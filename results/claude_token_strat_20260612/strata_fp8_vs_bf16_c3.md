## strata_fp8_vs_bf16_c3

baseline `claude_bf16_vosplit_c3_ctx8191` vs comparator `claude_fp8_c3_ctx8191` — 8190 tokens, mean delta **-0.0184** nats (negative = quantized better)

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 5291 | +0.1226 | -429.2% |
| [0.5,1) | 230 | +0.3077 | -46.8% |
| [1,2) | 279 | +0.1852 | -34.2% |
| [2,4) | 429 | +0.0483 | -13.7% |
| [4,8) | 616 | -0.4101 | +167.2% |
| >=8 | 1345 | -0.5131 | +456.7% |

| positions | count | mean delta | share of total |
|---|---|---|---|
| 0-1022 | 1022 | -0.0987 | +66.8% |
| 1023-2045 | 1023 | +0.0071 | -4.8% |
| 2046-3068 | 1023 | +0.1362 | -92.2% |
| 3069-4091 | 1023 | +0.1992 | -134.9% |
| 4092-5114 | 1023 | -0.0316 | +21.4% |
| 5115-6137 | 1023 | -0.1701 | +115.1% |
| 6138-7160 | 1023 | +0.0053 | -3.6% |
| 7161-8190 | 1030 | -0.1939 | +132.2% |

sign: 49.9% improved / 48.8% worsened / 112 tied; mean |delta| 0.6157 nats

| pos | token | base NLL | comp NLL | delta | context |
|---|---|---|---|---|---|
| 8042 | `m` | 18.695 | 0.639 | -18.055 | `results/llamacpp*[m]xfp4*row_` |
| 6098 | `/` | 16.245 | 0.034 | -16.211 | `log",\n            "results[/]sglang_qwen` |
| 6185 | `ang` | 16.079 | 0.000 | -16.079 | `            "results/sgl[ang]_qwen_fp4` |
| 5013 | `1` | 0.000 | 15.108 | +15.107 | `_has_sm12[1])\n                else "family` |
| 7028 | `/` | 4.774 | 19.349 | +14.575 | `log",\n            "results[/]aeon_qwen3` |
| 5812 | `ang` | 14.090 | 0.338 | -13.752 | `                "results/sgl[ang]*fp4_e2` |
| 6302 | `wen` | 0.000 | 13.655 | +13.655 | `/sglang_q[wen]_fp4kv_prompt` |
| 6054 | `/` | 27.772 | 14.458 | -13.314 | `log",\n            "results[/]sglang_qwen` |
| 6294 | `"` | 13.313 | 0.009 | -13.304 | `ile*.md",\n            ["]results/sglang_` |
| 3682 | `architecture` | 15.900 | 3.130 | -12.770 | ` data[0]\n        [architecture] = first.get("Architecture` |
| 3780 | `        ` | 0.670 | 13.376 | +12.706 | ` image_digests,\n[        ]"evidence": evidence,\n` |
| 6223 | `kv` | 2.019 | 14.670 | +12.651 | `_qwen_fp4[kv]_d7d93` |
| 6322 | `/` | 0.570 | 12.776 | +12.206 | `log",\n            "results[/]sglang_nvfp` |
| 6906 | `/*` | 14.154 | 2.370 | -11.784 | `json",\n                "results[/*]qwen3.6*` |
| 6097 | `results` | 23.844 | 12.219 | -11.625 | `*.log",\n            "[results]/sglang_q` |
| 2433 | `EX` | 0.021 | 11.634 | +11.613 | `\n        list(DEFAULT_[EX]CLUDE_SUBSTRINGS)` |
| 7748 | `ll` | 15.815 | 4.203 | -11.613 | `",\n                "results/[ll]amacpp*qwen3` |
| 3289 | `path` | 0.152 | 11.519 | +11.367 | `    return json.loads([path].read_text(encoding` |
| 3771 | `dig` | 8.492 | 19.629 | +11.136 | `,\n        "image_[dig]ests": image_digests` |
| 3657 | `data` | 0.062 | 11.132 | +11.070 | ` []\n    if isinstance([data], list) and data and` |
| 6684 | `eagle` | 16.265 | 5.419 | -10.846 | `results/sglang*[eagle]*.md",\n            "` |
| 5132 | `n` | 12.094 | 1.502 | -10.592 | `keys=True) + "\[n]", encoding="utf-8` |
| 179 | `_` | 0.107 | 10.628 | +10.520 | `ZY", "_C_stable[_]libtorch", "import_` |
| 6976 | `\n` | 0.087 | 10.495 | +10.408 | `stop_point.md",[\n]            "results/aeon` |
| 5966 | `gl` | 3.885 | 14.242 | +10.357 | `\n            "results/s[gl]ang_qwen_fp` |
| 2623 | `header` | 0.281 | 10.625 | +10.344 | `total_chars += len([header]) + len(used_` |
| 7111 | `ae` | 0.474 | 10.813 | +10.339 | `",\n            "results/[ae]on_qwen36` |
| 5841 | `            ` | 0.160 | 10.439 | +10.279 | `        partial_patterns=(\n[            ]"results/flashinfer_` |
| 6572 | `(` | 0.074 | 10.340 | +10.266 | `claim_groups=(\n            [(]\n                "results/s` |
| 5545 | `        ` | 12.551 | 2.558 | -9.993 | `audit.json",),\n[        ]),\n        partial_patterns` |
| 4053 | `add` | 0.107 | 10.064 | +9.957 | `ArgumentParser()\n    parser.[add]_argument("--image-inspect` |
| 8075 | `),` | 11.068 | 1.448 | -9.620 | `manifest.json",\n            [),]\n            (\n                "` |
| 3949 | `exists` | 0.160 | 9.313 | +9.153 | `        "exists": path.[exists](),\n        "arch_` |
| 6266 | `\n` | 0.058 | 9.176 | +9.118 | `reconcile*.json",[\n]            "results/sgl` |
| 6923 | `            ` | 0.314 | 9.428 | +9.114 | `_benchmark.json",\n[            ]),\n        ),\n        ` |
| 6078 | `gl` | 13.309 | 4.255 | -9.054 | `\n            "results/s[gl]ang_qwen_fp` |
| 5943 | `results` | 17.531 | 8.500 | -9.031 | `*.json",\n            "[results]/sglang_q` |
| 4180 | `inspect` | 0.311 | 9.325 | +9.015 | ` is not None:\n            [inspect]_records.append(collect` |
| 5475 | `                ` | 17.978 | 9.015 | -8.963 | `groups=(\n            (\n[                ]"results/sglang` |
| 3264 | `as` | 14.185 | 22.844 | +8.659 | `:\n        return path.[as]_posix()\n\n\ndef` |
