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

## Concrete GB10 Run Packet

Run this on the Linux GB10 host from a current `dgx-spark-hijinks` checkout. It is SGLang
only: one FP4 Qwen server on port `30013`, no fp8 comparator and no vLLM work.

Assumptions:

- Docker with GPU runtime works.
- `nvcr.io/nvidia/sglang:26.05-py3` is present or pullable.
- `Qwen/Qwen2.5-1.5B-Instruct` is cached or reachable.
- SGLang source overlay is `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`.
- CUDA graphs remain disabled for this quality-localization row.

```bash
cd /path/to/dgx-spark-hijinks

RUN_ID=sglang_qwen_fp4kv_first_token_logits_$(TZ=Asia/Tokyo date +%Y%m%dT%H%MJST)
CONTAINER=sglang-fp4-first-token-logits
SRC=/home/jethac/spark_tmp/sglang_matched_d7d931f_20260608T1545JST/sglang
REPO="$PWD"

test -d "$SRC" || {
  mkdir -p "$(dirname "$SRC")"
  git clone https://github.com/jethac/sglang "$SRC"
  git -C "$SRC" checkout d7d931f530160ba86a2d55b4636d64baaeda3bec
}

mkdir -p "results/${RUN_ID}_dump"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d --rm --gpus all --ipc=host --network=host \
  --name "$CONTAINER" \
  -e PYTHONPATH=/workspace/sglang/python \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  -e SGLANG_FP4_KV_TRACE_BACKEND=1 \
  -e SGLANG_FP4_KV_AUTOCALIB=1 \
  -e SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=0 \
  -e DUMPER_ENABLE=1 \
  -e DUMPER_NON_INTRUSIVE_MODE=off \
  -e DUMPER_DIR=/tmp/sglang_fp4_first_token_dump \
  -e DUMPER_EXP_NAME="$RUN_ID" \
  -e DUMPER_CLEANUP_PREVIOUS=0 \
  -e DUMPER_SOURCE_PATCHER_CONFIG=/workspace/dgx-spark-hijinks/scripts/sglang_fp4_first_token_dump_patch.yaml \
  -e SGLANG_FP4_FIRST_TOKEN_DUMP=1 \
  -v "$SRC:/workspace/sglang:ro" \
  -v "$REPO:/workspace/dgx-spark-hijinks:ro" \
  -v "$REPO/results/${RUN_ID}_dump:/tmp/sglang_fp4_first_token_dump/${RUN_ID}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/sglang:26.05-py3 \
  bash -lc 'python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 30013 \
    --tp 1 \
    --dtype bfloat16 \
    --kv-cache-dtype fp4_e2m1 \
    --attention-backend flashinfer \
    --page-size 1 \
    --mem-fraction-static 0.40 \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph'

until curl -fsS http://127.0.0.1:30013/health >/dev/null; do sleep 5; done

python3 scripts/sglang_fp4_endpoint_metadata_probe.py \
  --url http://127.0.0.1:30013 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --case medium_decode \
  --max-new-tokens 1 \
  --run-id "$RUN_ID" \
  --output "results/${RUN_ID}.json"

docker logs "$CONTAINER" > "results/${RUN_ID}_fp4_server.log" 2>&1
docker inspect "$CONTAINER" > "results/${RUN_ID}_fp4_container_inspect.json"
find "results/${RUN_ID}_dump" -type f -name '*.pt' | sort > "results/${RUN_ID}_dump_filelist.txt"

docker rm -f "$CONTAINER"
```

Additional expected artifacts:

- `results/${RUN_ID}_fp4_server.log`
- `results/${RUN_ID}_fp4_container_inspect.json`
- `results/${RUN_ID}_dump_filelist.txt`

Known remaining gap after capture:

There is not yet a repo script that pairs `.pt` dumps by `rid` / `forward_pass_id` and emits
an OpenAI-vs-native logits delta/top-k summary. If `rid` is missing or insufficient to
separate endpoint calls, add an endpoint tag or split the client into separate OpenAI-only
and native-only requests.

Live correction:

The first live attempt with `DUMPER_CLEANUP_PREVIOUS=1` crashed because the dumper tried to
remove the active bind-mounted dump directory and hit `OSError: [Errno 16] Device or
resource busy`. Use `DUMPER_CLEANUP_PREVIOUS=0` with the per-run mounted directory above.

Live partial result:

- artifact: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_summary.md`
- compact dump summary:
  `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_dump_summary.md`
- result: tensor dump capture succeeded and `scripts/sglang_first_token_dump_summary.py`
  now summarizes grouped request tensors. The host-side endpoint probe failed because the
  host Python environment lacked `transformers`, and the server log shows only one completed
  `POST /generate` request, so this row is not an OpenAI-vs-native comparator.
- narrow finding: for the captured native `/generate` request, logits before and after
  `_preprocess_logits()` are identical across all real request groups (`max_abs_delta=0`,
  `same_argmax=True`, top-20 Jaccard `1.0`).
- next correction: run the probe in the SGLang container or another environment with
  `transformers`, and add endpoint labels or separate request runs so OpenAI and native
  dumps can be paired explicitly.
