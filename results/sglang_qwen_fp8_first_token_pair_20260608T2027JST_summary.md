# SGLang Qwen fp8-KV OpenAI/Native First-Token Control, 2026-06-08 20:27 JST

Status: green diagnostic control.

Purpose: run the same OpenAI/native first-token pair used for the FP4-KV failure with
`--kv-cache-dtype fp8_e4m3`. This tests whether the endpoint split is generic SGLang
OpenAI/native behavior or specific to the FP4-KV path.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp8_e4m3`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- probe case: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Artifacts:

- endpoint probe: `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST.json`
- server log: `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_fp8_server.log`
- container inspect:
  `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_fp8_container_inspect.json`
- dump file list:
  `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_dump_filelist.txt`
- compact dump summary:
  `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_dump_summary.md`
- machine-readable dump summary:
  `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_dump_summary.json`

Endpoint result:

- Prompt reconciliation passes with the same 56-token prompt hash used by the FP4 row:
  `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`.
- OpenAI Chat first token: `**`, logprob `-0.7235294580459595`.
- Native `/generate` first token: `**`, token id `334`, logprob `-0.7641105651855469`.
- `same_text=true`.
- Server log confirms both target requests completed:
  - `POST /v1/chat/completions`
  - `POST /generate`

Dump result:

- Dump files: `75`
- Grouped forward passes: `15`
- Real request groups: `13`
- Health-check groups: `2`

Endpoint candidate groups:

- OpenAI candidate:
  - rid: `84a72209798b417f905efc771b5f5240`
  - prefill forward pass: `12`, `forward_mode=1`, `seq_lens=[56]`
  - prefill argmax before/after preprocessing: `334` / `334`
  - decode forward pass: `13`, `forward_mode=2`, `seq_lens=[57]`
  - decode argmax before/after preprocessing: `86000` / `86000`
  - decode preprocessing delta: `max_abs_delta=0.332386`, same argmax, top-20 Jaccard `1.0`
- Native candidate:
  - rid: `8e511a5075d34cf4b83c0ba7357a7696`
  - prefill forward pass: `14`, `forward_mode=1`, `seq_lens=[56]`
  - prefill argmax before/after preprocessing: `334` / `334`
  - decode forward pass: `15`, `forward_mode=2`, `seq_lens=[57]`
  - decode argmax before/after preprocessing: `86000` / `86000`
  - decode preprocessing delta: `max_abs_delta=0`, same argmax, top-20 Jaccard `1.0`

Interpretation:

- The OpenAI/native first-token split is not a generic endpoint mismatch. Under the same
  prompt, same sequencing, same no-graph policy, and fp8 KV, both endpoints choose `**`.
- The FP4 row's native prefill argmax `838` is therefore a FP4-KV-specific regression under
  this endpoint sequence.
- `_preprocess_logits()` remains cleared: both fp8 endpoint candidates preserve the same
  argmax/top-20 through preprocessing.
- Next probe: compare fp8 and FP4 tensor state at or before the first prefill attention/KV
  write for the native endpoint, ideally with explicit endpoint request tags and without
  the unrelated pre-probe warmup `/generate` request.

