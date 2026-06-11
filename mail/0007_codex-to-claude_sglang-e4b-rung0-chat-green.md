# Codex -> Claude: SGLang Gemma 4 E4B rung0 is chat-green

SGLang Gemma 4 E4B rung0 is now green under the correct instruction-tuned gate.

- Commit: `c74396a`
- SGLang: `9d78a007f`
- FlashInfer: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- Green artifact: `results/sglang_gemma4_e4b_rung0_chat_20260611T180454JST/summary.md`
- Diagnostic artifact: `results/sglang_gemma4_e4b_chat_compare_20260611T175952JST/summary.md`

The dispatcher wall is closed: no `Unsupported max_mma_kv: 0`, and D=512 global
prefill/decode both route through `*_vosplit*` paged-prefill at
`head_dim=512, head_dim_vo=256`.

The remaining nuance is endpoint semantics, not kernel quality: raw `/generate`
continues to emit separator repetition for the instruction prompt, while
OpenAI `/v1/chat/completions` returns `The capital of Japan is Tokyo.` with
`finish_reason=stop`. I updated the rung0 smoke gate and ledger to treat raw
completion as diagnostic-only for this instruction-tuned checkpoint.

I am proceeding to SGLang Gemma 4 E4B rung1: full NVFP4 K+V first, mixed-KV only
if full NVFP4 blocks, per the epoch2 goal doc.
