# SGLang Qwen FP4-KV First-Token Logits Dump, 2026-06-08 20:08 JST

Status: partial diagnostic, not a quality pass.

Purpose: localize the remaining SGLang Qwen FP4-KV quality regression after prompt
reconciliation proved that the OpenAI Chat and native `/generate` paths use matching prompt
IDs.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- dump hook: `scripts/sglang_fp4_first_token_dump_patch.yaml` around
  `ModelRunner.sample()` before and after `_preprocess_logits()`

Artifacts:

- server log: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_fp4_server.log`
- dump file list: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_dump_filelist.txt`
- compact dump summary: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_dump_summary.md`
- machine-readable summary: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_dump_summary.json`

Result:

- The first attempt used `DUMPER_CLEANUP_PREVIOUS=1` and crashed because the dumper tried to
  remove the active bind-mounted dump directory.
- The successful capture used `DUMPER_CLEANUP_PREVIOUS=0`.
- The host-side endpoint metadata probe then failed before completing because the host
  Python environment lacked `transformers`.
- The server log confirms only one completed `POST /generate` request and no completed
  `POST /v1/chat/completions` request, so this row is not an OpenAI-vs-native comparator.
- The dump contains `55` `.pt` files, `11` grouped forward passes, `9` real-request groups,
  and `2` health-check groups.
- For all captured real-request groups, `next_token_logits` before and after
  `_preprocess_logits()` are identical: same shape, same argmax, zero max absolute delta,
  zero mean absolute delta, and top-20 Jaccard `1.0`.

Interpretation:

- This row does not bless SGLang FP4 KV. The known standardized-output quality regression
  remains red.
- The captured native `/generate` path does not show corruption introduced by
  `_preprocess_logits()` for the dumped request. If the OpenAI-vs-native split still
  reproduces, the cause is earlier than this hook, endpoint metadata before model execution,
  or a decode/KV-state difference not exposed by comparing pre/post preprocessing inside a
  single native request.
- The next diagnostic must run the endpoint probe in an environment with `transformers`, or
  run separate OpenAI-only and native-only request-tagged dumps with explicit endpoint labels.

