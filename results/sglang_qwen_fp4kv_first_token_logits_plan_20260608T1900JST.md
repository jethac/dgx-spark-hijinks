# SGLang Qwen FP4-KV First-Token Logits Dump Plan, 2026-06-08 19:00 JST

Purpose: define the next SGLang Qwen FP4-KV quality localization step after the endpoint
metadata probe showed matching prompt hashes but different first-token behavior.

Current localization state:

- FP4 OpenAI Chat and native `/generate` share the same 56-token prompt hash.
- FP4 OpenAI first token differs from FP4 native `/generate`.
- Existing backend traces cover decode and `extend_merge_paged`, but they are not
  request-tagged.
- Next proof should isolate whether divergence is already present in model/attention
  logits or introduced by logits preprocessing/request metadata.

Hook point:

- File: `third_party/sglang/python/sglang/srt/model_executor/model_runner.py`
- Symbol: `sglang.srt.model_executor.model_runner.ModelRunner.sample()`
- Narrow hook location: immediately around:

```python
self._preprocess_logits(logits_output, forward_batch.sampling_info)
```

Why this hook:

- `logits_output.next_token_logits` is available before and after preprocessing.
- `forward_batch` carries request metadata needed to tag the dump:
  - `forward_mode`
  - `input_ids`
  - `seq_lens`
  - `positions`
  - `rids`
  - `sampling_info`

Patch:

- Use `scripts/sglang_fp4_first_token_dump_patch.yaml`.
- The patch should dump:
  - `next_token_logits` before `_preprocess_logits`
  - `next_token_logits` after `_preprocess_logits`
  - `input_ids`
  - `positions`
  - `seq_lens`
  - request id when available

Server environment additions:

```bash
export DUMPER_ENABLE=1
export DUMPER_NON_INTRUSIVE_MODE=off
export DUMPER_DIR=/tmp/sglang_fp4_first_token_dump
export DUMPER_SOURCE_PATCHER_CONFIG=/workspace/scripts/sglang_fp4_first_token_dump_patch.yaml
export SGLANG_FP4_FIRST_TOKEN_DUMP=1
```

Probe command:

```bash
python scripts/sglang_fp4_endpoint_metadata_probe.py \
  --url http://127.0.0.1:30013 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --case medium_decode \
  --max-new-tokens 1 \
  --run-id sglang_qwen_fp4kv_first_token_logits_YYYYMMDDTHHMMJST \
  --output results/sglang_qwen_fp4kv_first_token_logits_YYYYMMDDTHHMMJST.json
```

Artifacts to collect:

- `results/sglang_qwen_fp4kv_first_token_logits_YYYYMMDDTHHMMJST.json`
- `results/sglang_qwen_fp4kv_first_token_logits_YYYYMMDDTHHMMJST_dump/`
- optional summary:
  `results/sglang_qwen_fp4kv_first_token_logits_YYYYMMDDTHHMMJST_summary.md`

Expected dump naming:

- `fp4_first_token__next_token_logits`
- `fp4_first_token__input_ids`
- `fp4_first_token__positions`
- `fp4_first_token__seq_lens`

Expected tags:

- `phase`
- `forward_mode`
- `rid`
- `forward_pass_id`

Risks:

- Full-vocabulary logits dumps are large; keep `max-new-tokens=1` and run only the FP4
  server first.
- `rids` may be absent or endpoint-specific. If so, align by `forward_pass_id`, route order,
  and dump phase.
- If batching overlaps two requests, rerun with a single request in flight or gate by
  `SGLANG_FP4_FIRST_TOKEN_DUMP_RID`.
- This does not test CUDA graphs. Keep the no-graph FP4 flags from the reconciliation row.
