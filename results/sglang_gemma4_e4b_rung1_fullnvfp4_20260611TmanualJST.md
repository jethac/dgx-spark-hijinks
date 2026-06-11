# SGLang Gemma 4 E4B Rung 1 Full-NVFP4 Checkpoint

Status: GREEN for short coherence, short prefix-reuse PPL, and allocator capacity
after the hybrid full-NVFP4 denominator fix. fp8 serving remains a red comparator
path and is tracked separately.

Runtime:

- Model: `google/gemma-4-E4B-it`
- SGLang: `jethac/sglang@96a9ff9ce` for the current denominator-fixed row
  (`9d78a007f` was the pre-fix quality checkpoint)
- FlashInfer: `jethac/flashinfer@76af798243d11c4910eaceaf1d62ba4227656d4a`
- Image: `sglang-source-stack-c3dae30f-e631a13fd:latest`
- Serving: `SGLANG_FLASHINFER_VOSPLIT=1`, page size `1`, graphs disabled,
  `--mem-fraction-static 0.40`, Docker `--memory 100g --memory-swap 100g`

## Coherence Smoke

Artifact: `results/sglang_gemma4_e4b_rung1_fullnvfp4_denfix_20260611T190114JST/summary.md`

Full NVFP4 K+V (`--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`)
serves the OpenAI chat smoke and returns:

> The capital of Japan is Tokyo.

The dispatch evidence shows the D=512 global layers use the VO-split paged-prefill
path for both prefill and decode-as-prefill:

- SWA/local layers: `head_dim=256`, cached KV dtype `torch.uint8`
- Global layers: `head_dim=512`, `head_dim_vo=256`, `extend_paged_vosplit*`
- Decode: `decode_as_prefill_vosplit*` on `BatchPrefillWithPagedKVCacheWrapper`
- `Unsupported max_mma_kv: False`

This row proves the r9/`76af7982` dispatcher fix plus
`jethac/sglang@96a9ff9ce` are sufficient for SGLang E4B full-NVFP4 short chat
serving.

## Short PPL Gate

Artifact prefix:
`results/sglang_gemma4_e4b_rung1_fullnvfp4_denfix_ppl_ctx512_prefix256_20260611T190622JST_*`

Method:

- Sequential servers, never concurrent.
- Baseline: bf16/auto KV, graphs disabled.
- Candidate: full NVFP4 K+V, graphs disabled.
- Native `/generate` supplied-token prompt-logprob scoring with exact token IDs.
- Context: `512` tokens.
- Reused prefix: `256` tokens.
- Eval text: deterministic repository markdown corpus slice, not a cleaned benchmark
  corpus. The matched delta is the claim, not the absolute PPL.

Result:

| ctx | reused prefix | PPL bf16 | PPL full NVFP4 | delta nats/token |
|---:|---:|---:|---:|---:|
| 512 | 256 | `174.041493` | `143.900217` | `-0.190174` |

Both reports scored successfully (`bf16_ok=true`, `candidate_ok=true`). This is a
short quality gate with prefix reuse, not a long-context blessing.

## Capacity

At the same `--mem-fraction-static 0.40`, the denominator-fixed full NVFP4 row
raises allocator token count by `3.5668x` versus the bf16/auto baseline:

| KV mode | max_total_num_tokens | SWA tokens | full tokens | log KV memory |
|---|---:|---:|---:|---:|
| bf16/auto | `357187` | `285749` | `357187` | `28.61 GB` |
| full NVFP4 K+V | `1274008` | `1019206` | `1274008` | `28.70 GB` |

Ratio versus bf16/auto: `1274008 / 357187 = 3.5668x`.

The denominator fix is `jethac/sglang@96a9ff9ce`, which adds scale-aware full-NVFP4
byte sizing to `HybridSWAPoolConfigurator`. Before that fix, SGLang treated hybrid
full NVFP4 like fp8 for sizing and left physical KV memory unused:

| KV mode | max_total_num_tokens | SWA tokens | full tokens | log KV memory |
|---|---:|---:|---:|---:|
| pre-fix full NVFP4 K+V | `716992` | `573593` | `716992` | `16.15 GB` |
| fixed full NVFP4 K+V | `1274008` | `1019206` | `1274008` | `28.70 GB` |

Ratio fixed versus pre-fix: `1274008 / 716992 = 1.7770x`, matching the expected
FP4-data-plus-scale storage ratio.

## fp8 Comparator

The required fp8 comparator is RED and is not a valid quality baseline yet:

- First fp8 run:
  `results/sglang_gemma4_e4b_rung1_fp8_20260611T182428JST/summary.md`
  timed out after the old 180 second client limit.
- Retry with 30 minute client timeout:
  `results/sglang_gemma4_e4b_rung1_fp8_retry1_20260611T183110JST/summary.md`
  started, allocated fp8 KV, routed D=512 prefill through VO-split, then SGLang's
  warmup/request path timed out internally after 600 seconds and returned
  `Internal Server Error`.
- fp8 logs also warn: `Using FP8 KV cache but no scaling factors provided. Defaulting
  to scaling factors of 1.0.`

The fp8 allocator row itself reports about the same token count as the pre-fix full
NVFP4 row:

| KV mode | max_total_num_tokens | SWA tokens | full tokens | log KV memory |
|---|---:|---:|---:|---:|
| fp8_e4m3 | `715185` | `572148` | `715185` | `28.65 GB` |
| fixed full NVFP4 K+V | `1274008` | `1019206` | `1274008` | `28.70 GB` |

Ratio versus fp8 allocator tokens: `1274008 / 715185 = 1.7814x`.

Interpretation: the allocator capacity ratio is now where it should be, but fp8
serving is still red as a quality comparator for E4B. The claim-ready row is:
full-NVFP4 E4B short quality green versus bf16/auto, with `1.7814x` allocator-token
capacity versus the red fp8 allocator row. The fp8 quality baseline remains open.
