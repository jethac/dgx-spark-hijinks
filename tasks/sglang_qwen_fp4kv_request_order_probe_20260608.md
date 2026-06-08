# SGLang Qwen FP4-KV Request-Order Probe, 2026-06-08

Run this on the GB10 host after the quant-error trace.

Purpose: prove whether the FP4-KV OpenAI/native split is actually request-order and radix
reuse state. The prior rows show:

- OpenAI first, native second: native reuses a 55-token FP4 radix prefix and fails.
- `--disable-radix-cache`: native has `cached_tokens=0` and passes.
- Quant-error and global-scale traces are identical for the failing default row and the
  passing radix-off row.

Use the same server shape as the quant-error row, but keep only above-attention/radix logs
unless deeper traces are needed:

```bash
SGLANG_FP4_KV_TRACE_RADIX=1
SGLANG_FP4_KV_TRACE_BACKEND=1
SGLANG_FP4_KV_TRACE_LAYERS=0
```

Keep tensor/logit traces off for the first pass.

## Cases

Use explicit request IDs and the exact 56 prompt IDs already proven equivalent by
`scripts/sglang_openai_native_reconcile.py`.

1. Baseline order:
   - `POST /flush_cache`
   - OpenAI `/v1/chat/completions`, `rid="openai-first"`, one token.
   - Native `/generate`, `rid="native-second"`, same `input_ids`, one token.

2. Reverse order:
   - `POST /flush_cache`
   - Native `/generate`, `rid="native-first"`, one token.
   - OpenAI `/v1/chat/completions`, `rid="openai-second"`, one token.

3. Flush-between isolation:
   - `POST /flush_cache`
   - OpenAI one token.
   - `POST /flush_cache`
   - Native one token.

4. Namespace isolation, if accepted by the payload path:
   - No flush between OpenAI and native.
   - Give different `extra_key`/`cache_salt` values to OpenAI and native.

## Capture

- `FP4 KV radix trace rid=...`
- `FP4 KV ForwardBatch trace rids=...`
- response `cached_tokens`
- first token and top logprobs
- startup calibration lines

## Decision Rule

- If native first after `flush_cache` has `cached_tokens=0` and produces `**`, the old
  native failure was sequencing: native was second and reused OpenAI's FP4 radix prefix.
- If OpenAI second after native also diverges with `cached_tokens=55`, the defect is
  endpoint-independent cached-prefix reuse.
- If OpenAI second remains correct while native second diverges with the same prefix length,
  compare `extend_prefix_lens_cpu`, `extend_seq_lens_cpu`, `input_ids`, `positions`, and
  `req_pool_indices`.
- If different `extra_key`/`cache_salt` makes native miss radix and pass, radix namespace
  matching is confirmed as the switch.
