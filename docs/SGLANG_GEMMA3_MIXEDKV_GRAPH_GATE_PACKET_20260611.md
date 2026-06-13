# SGLang Gemma 3 Mixed-KV CUDA Graph Gate Packet

Date: 2026-06-11 JST

Purpose: record the SGLang Gemma 3 27B mixed FP8-K + NVFP4-V CUDA graph
re-enable gate packet that was staged after `jethac/flashinfer@f99323bd`
temporarily accepted the validated mixed pair
`(k_data_type=torch.float8_e4m3fn, v_data_type=torch.uint8)`.

Status update, 2026-06-14: **parked, do not run as written.** FlashInfer branch
`spark/hijinks-022-fa2-d512` later moved to `3fa0775c`, intentionally removing
the temporary `k_data_type` / `v_data_type` kwargs from the public `plan()` API.
That does not affect full-NVFP4 K+V (`kv_data_type=torch.uint8`), but it means
mixed FP8-K/NVFP4-V still requires real split-K/V module keying before this
graph gate can become a claim-grade runtime row. A setup-only rerun must either
pin FlashInfer to `f99323bd` to reproduce the old API experiment, or wait for
the proper split-dtype implementation.

This packet is for the next free Spark window. Do not run while
`/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` exists.

## Preconditions

- Branch: `epoch2`
- Main repo head includes the FlashInfer submodule pointer bump to
  `jethac/flashinfer@spark/hijinks-022-fa2-d512@f99323bd`.
  Do not silently substitute the current branch tip (`3fa0775c` as of
  2026-06-14) for this packet: that tip removed the split-dtype kwargs and
  should fail old mixed-KV callers immediately.
- SGLang submodule remains on the DiffusionGemma/SGLang lane head
  `jethac/sglang@spark/hijinks-023-gemma4-fullnvfp4-denominator@3a2e15153`.
- Spark memory guardrails: single server at a time, Docker cgroup
  `--memory 100g --memory-swap 100g`, `--mem-fraction-static 0.60`.
- Hugging Face/model cache must be B-backed or Spark-local, not WSL ext4.

## Why This Retry Is Different

The last retry, `results/sglang_gemma3_27b_mixedkv_graph_gate_retry3_sitecustomize_20260611T094007JST.md`,
closed source-tree packaging but still used `jethac/flashinfer@fb7d62ea`.
It failed when mixed CUDA graph capture reached:

```text
BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'k_data_type'
BatchDecodeWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'k_data_type'
```

Mail `0016_claude-to-codex_graph-gate-unparked.md` reported that the FlashInfer
branch accepted this exact mixed pair at plan level at `f99323bd`. That was a
temporary API experiment and is now superseded by the split-dtype de-scope
decision. If this historical packet is deliberately replayed, it must prove both
the Python package and JIT source are the pinned `f99323bd` branch copy:

- `flashinfer.__file__` must resolve under `/work/third_party/flashinfer`.
- `flashinfer.jit.env.FLASHINFER_DATA` and `FLASHINFER_CSRC_DIR` must resolve
  under `/work/third_party/flashinfer`.

The `sitecustomize` JIT-source redirect alone is insufficient; the process must
import FlashInfer Python from the source branch, not the installed wheel.

## Run Command

Run from the repo root on the Spark:

```bash
set -euo pipefail

test ! -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
docker ps

git pull --rebase
git submodule update --init third_party/sglang third_party/flashinfer
git -C third_party/flashinfer fetch origin spark/hijinks-022-fa2-d512
git -C third_party/flashinfer checkout f99323bd
git -C third_party/flashinfer rev-parse HEAD
git -C third_party/sglang rev-parse HEAD

STAMP=$(date +%Y%m%dT%H%M%SJST)
RUN=sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_f99323bd_${STAMP}

CTX_LIST=8192 \
RUN="${RUN}" \
MODEL=google/gemma-3-27b-it \
MEM_FRACTION_STATIC=0.60 \
PAGE_SIZE=1 \
DISABLE_GRAPHS=0 \
ENABLE_FP4_CUDA_GRAPH=1 \
REUSE_PREFIX_LEN=4096 \
LOGPROB_START_LEN=4096 \
MAX_NEW_TOKENS=1 \
PPL_TIMEOUT=1800 \
READY_TIMEOUT_S=900 \
GB10_DOCKER_MEMORY=100g \
GB10_DOCKER_MEMORY_SWAP=100g \
EXTRA_SERVER_ENVS='PYTHONPATH=/work/python_sitecustomize:/work/third_party/flashinfer:/work/third_party/sglang/python:/tmp/flashinfer-python-path SGLANG_GEMMA3_ENABLE_HYBRID_SWA=1 SPARK_FLASHINFER_SOURCE_ROOT=/work/third_party/flashinfer SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 FLASHINFER_PREFILL_DEBUG_ONCE=1' \
  bash scripts/run_sglang_qwen_ppl_pair.sh \
  2>&1 | tee "results/${RUN}_runner.log"

python3 - <<PY
import json
from pathlib import Path

run = "${RUN}"
root = Path("results")
manifest = json.loads((root / f"{run}_manifest.json").read_text())
compare = json.loads((root / f"{run}_compare.json").read_text())
install_logs = {
    label: (root / f"{run}_{label}_install.log").read_text(errors="replace")
    for label in ("fp8", "mixed")
}
server_logs = {
    label: (root / f"{run}_{label}_server.log").read_text(errors="replace")
    for label in ("fp8", "mixed")
}

errors = []
if not compare.get("ok"):
    errors.append("compare.json ok=false")
for label, text in install_logs.items():
    if "/work/third_party/flashinfer/flashinfer/__init__.py" not in text:
        errors.append(f"{label}: flashinfer.__file__ did not resolve to source branch")
for label, text in server_logs.items():
    if "cuda graph: True" not in text:
        errors.append(f"{label}: no cuda graph: True proof in server log")
    if "#cached-token: 4096" not in text:
        errors.append(f"{label}: no 4096-token radix reuse proof")
if "TypeError: Batch" in server_logs["mixed"]:
    errors.append("mixed: FlashInfer plan TypeError still present")

print(json.dumps({"run": run, "errors": errors, "rows": compare.get("rows")}, indent=2))
if errors:
    raise SystemExit(1)
PY
```

## Green Criteria

- fp8 row completes with CUDA graphs enabled.
- mixed row completes with CUDA graphs enabled.
- Both rows prove radix reuse at `#cached-token: 4096`.
- `flashinfer.__file__` proves source Python import from `/work/third_party/flashinfer`.
- JIT source paths prove `SPARK_FLASHINFER_SOURCE_ROOT=/work/third_party/flashinfer`.
- No FlashInfer `k_data_type` / `v_data_type` `TypeError`.
- `compare.json` is `ok: true`; delta nats/token should stay in the same small range as
  the no-graph deep-prefix row unless graph replay changes model output.

## Red Handling

If the old `k_data_type` TypeError returns, stop: the container imported the wrong
FlashInfer Python or the wrong branch.

If the run reaches readiness but quality diverges, preserve both server logs and
`*_ppl.json`; that is a real graph-replay correctness failure, not a setup failure.

If the run times out during model load or graph capture, preserve `docker inspect`,
server logs, and `free -h`; do not start a second large server concurrently.
