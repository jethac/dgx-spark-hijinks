# SGLang Qwen FP4-KV OpenAI/Native First-Token Pair, 2026-06-08 20:21 JST

Status: red quality diagnostic, stronger localization.

Purpose: rerun the SGLang Qwen FP4-KV first-token localization with the endpoint metadata
probe inside the SGLang container so `transformers` is available before requests are sent.
This fixes the previous partial row, which captured only native `/generate`.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- probe case: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Artifacts:

- endpoint probe: `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST.json`
- server log: `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_fp4_server.log`
- container inspect:
  `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_fp4_container_inspect.json`
- dump file list:
  `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_dump_filelist.txt`
- compact dump summary:
  `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_dump_summary.md`
- machine-readable dump summary:
  `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_dump_summary.json`

Endpoint result:

- Prompt reconciliation passes:
  - local prompt token count: `56`
  - local prompt SHA256:
    `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`
  - OpenAI prompt SHA256:
    `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`
  - `openai_matches_local=true`
- OpenAI Chat first token: `**`, logprob `-0.7235294580459595`.
- Native `/generate` first token: `ark`, token id `838`, logprob `-0.5874708890914917`.
- `same_text=false`, `same_token_id=false`.
- Server log confirms both target requests completed:
  - `POST /v1/chat/completions`
  - `POST /generate`

Dump result:

- Dump files: `75`
- Grouped forward passes: `15`
- Real request groups: `13`
- Health-check groups: `2`
- The pre-probe warmup/native-like request group `6bdcac9f...` is excluded from the endpoint
  interpretation below because it completed before the paired OpenAI/native requests.

Endpoint candidate groups:

- OpenAI candidate:
  - rid: `703d87f46be3448a84c7723fbe942dc6`
  - prefill forward pass: `12`, `forward_mode=1`, `seq_lens=[56]`
  - prefill argmax before/after preprocessing: `334` / `334`
  - decode forward pass: `13`, `forward_mode=2`, `seq_lens=[57]`
  - decode argmax before/after preprocessing: `86000` / `86000`
  - decode preprocessing delta: `max_abs_delta=0.886364`, same argmax, top-20 Jaccard `1.0`
- Native candidate:
  - rid: `fade953138da4989bcffc0e167af9002`
  - prefill forward pass: `14`, `forward_mode=1`, `seq_lens=[56]`
  - prefill argmax before/after preprocessing: `838` / `838`
  - decode forward pass: `15`, `forward_mode=2`, `seq_lens=[57]`
  - decode argmax before/after preprocessing: `198` / `198`
  - decode preprocessing delta: `max_abs_delta=0`, same argmax, top-20 Jaccard `1.0`

Interpretation:

- This is not a pass. SGLang FP4 KV still fails the quality gate.
- The split is present at the first-token prefill logits for same-length, same-hash prompt
  requests: OpenAI candidate prefill argmax `334`; native candidate prefill argmax `838`.
- The native candidate prefill argmax `838` matches the native `/generate` first token id
  for `ark`, so the dump is aligned with the endpoint-visible native output.
- `_preprocess_logits()` is not the primary cause of the OpenAI/native first-token split.
  It either leaves the relevant prefill logits unchanged or preserves the same argmax/top-20
  on decode.
- The remaining bug is now localized earlier: endpoint metadata before model execution,
  request construction/scheduler state, KV cache state after the pre-probe request, or
  FP4-KV numerics triggered by endpoint-specific request handling. The next probe should
  avoid the pre-probe warmup request or tag endpoint requests explicitly, then compare fp8
  and FP4-KV under the same OpenAI/native sequencing.

