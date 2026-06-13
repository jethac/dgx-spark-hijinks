# vLLM Gemma 4 12B Matched KV Anchor on Vast sm_120

Date: 2026-06-13 JST

Status: complete. The Vast instance was destroyed after artifacts were copied.

## Scope

This is the vLLM-side discriminator requested in `mail/0116_claude-to-codex_handoff-vllm-matched-anchor.md`: Gemma 4 12B IT, same corpus and scoring shape as the SGLang 12B row, comparing bf16 KV against full NVFP4 K+V on an sm_120 RTX PRO 6000 Blackwell Server Edition.

This is a matched offline vLLM quality anchor, not a default serving claim.

## Environment

- Vast instance: `40808770` (destroyed)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, CC 12.0, 97887 MiB
- OS image: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- Python: 3.12.13
- Torch: 2.12.0+cu130
- vLLM: `0.1.dev1+ge32459eea.sm120a`
- Transformers: 5.12.0
- FlashInfer loaded from: `/root/venv/lib/python3.12/site-packages/flashinfer/__init__.py`
- FlashInfer source checkout present at: `7d5d477b7725943c8f1242490d38e88aa3d99e19`
- Repo branch/head: `epoch2` / `b1c2a168a8239aab7344643afb6b73fc42322a2e`

The older local `g6adc00f70` x64 wheel failed on Ubuntu 22.04 with `GLIBC_2.38 not found`; the corrected `ge32459eea.sm120a` wheel was used for the final rows.

## Harness

Harness: `docs/vast_anchor/vllm_matched_kv_anchor.py`

Common settings:

- Model/tokenizer: `google/gemma-4-12B-it`
- Corpus: SGLang row corpus copied from `results/sglang_gemma4_12b_ar_claim_ctx8185_prefix4096_20260613T105511JST/ppl_corpus.md`
- `ctx=8185`
- `prefix_len=4096`
- `score_start_index=4097` (skip prefix/suffix boundary token, matching the SGLang row)
- Scored tokens: 4088
- `max_model_len=8192`
- `max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.72`
- Prefix caching enabled
- Chat smoke uses a rendered chat template and generated `Tokyo` for both final rows.

Environment flags:

- `VLLM_FLASHINFER_BF16_GEMMA=1`
- `VLLM_FLASHINFER_VOSPLIT=1`
- `VLLM_NVFP4_KV_VOSPLIT=1`
- `VLLM_NVFP4_KV_LINEAR_V_SF=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The first 0.72 bf16 attempt already produced the same PPL score, but its chat smoke used direct token IDs and generated `11111111`; `bf16_util072_v2.*` is the clean final bf16 artifact.

## Results

| KV dtype | PPL | mean NLL nats/token | delta nats/token vs bf16 | delta PPL | chat smoke |
|---|---:|---:|---:|---:|---|
| bf16 | 95.128534 | 4.555228971 | 0.000000 | 0.000000 | `Tokyo` |
| full NVFP4 K+V | 126.048212 | 4.836664470 | +0.281435499 | +30.919678 | `Tokyo` |

Capacity logs at the same 0.72 KV budget:

| KV dtype | vLLM reported GPU KV cache tokens | max concurrency for 8192-token request |
|---|---:|---:|
| bf16 | 199,661 | 24.37x |
| full NVFP4 K+V | 709,924 | 86.66x |

Raw cache-token ratio from those vLLM logs: `709924 / 199661 = 3.5556x`.

## Interpretation

This falsifies the strongest version of "SGLang's +0.403 nats/token is only its radix merge path." vLLM full NVFP4 K+V on the same 12B `ctx=8185/prefix=4096` scoring shape also shows a substantial quality delta: `+0.2814` nats/token.

It is still smaller than the SGLang row's `+0.403` nats/token, so SGLang may still carry additional structural loss, but the vLLM anchor shows that 12B full NVFP4 K+V itself is not near-free at this shape.

## Artifacts

- `bf16_util072_v2.json` / `bf16_util072_v2.log`: final bf16 row
- `nvfp4_util072.json` / `nvfp4_util072.log`: final full NVFP4 K+V row
- `environment.txt`: remote environment record
- `bf16.log`: initial bad-wheel GLIBC_2.38 failure
- `bf16_retry*.log`, `bf16_util072.*`: intermediate harness/API/OOM attempts retained for provenance

Secret scan: no HF/Vast/API token values were found in the copied artifact directory. Logs include only the benign "HF_TOKEN" environment-variable name in Hugging Face's unauthenticated warning.
