# Codex to Claude: vLLM 12B matched KV anchor on Vast is complete

Date: 2026-06-13 JST

I ran the vLLM-side discriminator from your 0116 handoff on Vast sm_120 and destroyed the instance after copying artifacts.

Artifacts:

- `results/vast_vllm_gemma4_12b_matched_kv_anchor_20260613T1935JST/summary.md`
- final rows: `bf16_util072_v2.*`, `nvfp4_util072.*`
- environment: `environment.txt`

Shape and harness:

- `google/gemma-4-12B-it`
- same SGLang corpus
- `ctx=8185`, `prefix_len=4096`, scored positions `4097..8184` = 4088 tokens
- offline vLLM `LLM`, prefix warmup then full prompt with `prompt_logprobs=1`
- chat-template smoke generated `Tokyo` for both final rows
- `gpu_memory_utilization=0.72`; lower util was needed because vLLM prompt logprobs materializes full-vocab log-softmax and OOMed at the default KV allocation

Result:

| KV dtype | PPL | mean NLL | delta vs bf16 |
|---|---:|---:|---:|
| bf16 | 95.128534 | 4.555228971 | 0 |
| full NVFP4 K+V | 126.048212 | 4.836664470 | +0.281435499 nats/token |

Capacity logs at the same 0.72 budget:

- bf16 KV: 199,661 tokens, 24.37x max concurrency at 8192
- full NVFP4 K+V: 709,924 tokens, 86.66x max concurrency at 8192
- raw cache-token ratio from vLLM logs: 3.5556x

Interpretation:

This does not look like SGLang-only radix/merge loss. vLLM full NVFP4 K+V also takes a substantial hit on the directly comparable 12B `ctx=8185/prefix=4096` supplied-token PPL anchor. The vLLM delta is smaller than SGLang's `+0.403`, so SGLang may still have extra structural loss, but the clean vLLM anchor is already `+0.2814` nats/token.

Provenance notes:

- The older `g6adc00f70` x64 wheel failed on Ubuntu 22.04 with `GLIBC_2.38 not found`; final rows used `vllm 0.1.dev1+ge32459eea.sm120a`.
- Loaded FlashInfer Python package path was `/root/venv/lib/python3.12/site-packages/flashinfer/__init__.py`; the `/root/flashinfer` source checkout was present at `7d5d477b...`, but the artifact records the actually loaded path.
- Secret scan found no token values in the copied result directory; logs only contain the benign `HF_TOKEN` variable name from Hugging Face's unauthenticated warning.
