# SGLang Gemma 4 E4B Rung 0 Run Packet

Date authored: 2026-06-11 JST. Owner: Codex SGLang lane.

Run only when `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` is absent, `docker ps`
is empty, and `free -h` shows normal GB10 OS headroom. This packet is for
`docs/CODEX_GOAL_SGLANG_GEMMA4_RUNGS.md` objective 1 only: Gemma 4 E4B
text-only bring-up on SGLang, bf16/no KV quantization, with FlashInfer VO-split
handling the D=512 global layers.

## Target State

- Parent repo: `jethac/dgx-spark-hijinks@90db5e3` or newer on
  `docs/codex-direction-nvfp4-kv`.
- SGLang fork: `jethac/sglang@a5c71988c`.
- FlashInfer fork: `jethac/flashinfer@8d85fff9`.
- Model: `google/gemma-4-E4B-it` from the local HF cache.
- KV mode: default/no KV quantization. Do **not** pass `--kv-cache-dtype`.
- Attention: `--attention-backend flashinfer` with
  `SGLANG_FLASHINFER_VOSPLIT=1`.
- Graphs: disabled for the smoke (`--disable-cuda-graph
  --disable-piecewise-cuda-graph`).

## Stop-On-Red Rule

Stop at the first failure that blocks readiness or coherent output. If the server
hits `Unsupported max_mma_kv: 0`, preserve `server.log` with
`FLASHINFER_PREFILL_DEBUG_ONCE=1` output and do not attempt local kernel work; that
is the shared FlashInfer dispatcher wall from the goal packet.

## Host Prep

The commands below are captured as an executable helper:

```bash
REPO_ROOT=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live \
  bash scripts/run_sglang_gemma4_e4b_rung0_smoke.sh
```

The expanded form is kept here for review/debugging.

```bash
set -euo pipefail

REPO=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
cd "$REPO"

if [ -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN ]; then
  echo "CLAUDE_WINDOW_OPEN present; yield"
  exit 99
fi
if [ "$(docker ps -q | wc -l)" != "0" ]; then
  echo "docker is not empty; yield"
  docker ps
  exit 99
fi

git fetch origin
git checkout docs/codex-direction-nvfp4-kv
git pull --ff-only

git -C third_party/sglang fetch origin
git -C third_party/sglang checkout 0cb1c2936

git -C third_party/flashinfer fetch origin
git -C third_party/flashinfer checkout 8d85fff9
git -C third_party/flashinfer submodule update --init --recursive \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

git rev-parse HEAD
git -C third_party/sglang rev-parse HEAD
git -C third_party/flashinfer rev-parse HEAD
free -h
```

## Launch

```bash
RUN=sglang_gemma4_e4b_rung0_$(date +%Y%m%dT%H%M%SJST)
OUT="$REPO/results/$RUN"
CACHE="/tmp/flashinfer-cache-$RUN"
mkdir -p "$OUT"

docker run --rm --gpus all --memory=100g --memory-swap=100g --ipc=host \
  -w /work \
  -v "$REPO:/work" \
  -v "$REPO/third_party/flashinfer:/flashinfer-src" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR="$CACHE" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
  -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
  -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
  -e SGLANG_FLASHINFER_VOSPLIT=1 \
  -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -p 30000:30000 \
  sglang-source-stack-c3dae30f-e631a13fd:latest \
  bash -lc '
    set -euo pipefail
    rm -rf /root/.cache/flashinfer "$FLASHINFER_CACHE_DIR"
    mkdir -p /tmp/flashinfer-python-path
    ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer

    python - <<PY
import hashlib
import importlib
import pathlib
from flashinfer.jit import env as jit_env

for name in ("sgl_kernel", "sgl_kernel.common_ops"):
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        print(f"binary_missing {name} {exc}", flush=True)
        continue
    path = pathlib.Path(getattr(mod, "__file__", "")).resolve()
    if path.is_file():
        print(f"binary_md5 {name} {path} {hashlib.md5(path.read_bytes()).hexdigest()}", flush=True)
print("flashinfer_data", jit_env.FLASHINFER_DATA, flush=True)
print("flashinfer_csrc", jit_env.FLASHINFER_CSRC_DIR, flush=True)
print("flashinfer_include", jit_env.FLASHINFER_INCLUDE_DIR, flush=True)
print("flashinfer_cutlass", jit_env.CUTLASS_INCLUDE_DIRS, flush=True)
print("flashinfer_cccl", jit_env.CCCL_INCLUDE_DIRS, flush=True)
print("flashinfer_spdlog", jit_env.SPDLOG_INCLUDE_DIR, flush=True)
PY

    exec python3 -m sglang.launch_server \
      --model-path google/gemma-4-E4B-it \
      --dtype bfloat16 \
      --attention-backend flashinfer \
      --page-size 1 \
      --mem-fraction-static 0.40 \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --host 0.0.0.0 \
      --port 30000
  ' > "$OUT/server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"
```

## Readiness And Request

```bash
set -euo pipefail

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:30000/health >/dev/null; then
    echo ready | tee "$OUT/ready.txt"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness" | tee "$OUT/server_exited.txt"
    break
  fi
  sleep 10
done

curl -sS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "In one short sentence, name the capital of Japan.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 16
    }
  }' | tee "$OUT/generate.json"
```

## Teardown

```bash
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
docker ps
```

## Gates

Green rung-0 smoke requires all of:

- `server.log` includes `binary_md5` proof lines and FlashInfer source-tree
  paths resolving to `/flashinfer-src`.
- `server.log` includes `SGLang FlashInfer wrapper geometries` and
  `SGLang Gemma4 FlashInfer geometry` lines from serving dispatch.
- Sliding-wrapper proof lines show the SWA/local head dimension, not global
  D=512.
- Global/full-wrapper proof lines show D=512 with VO-split enabled.
- `generate.json` is coherent and answers Tokyo/Japan correctly.
- No `Unsupported max_mma_kv: 0`, trait rejection, raw-byte FP4 path, or
  `FLASHINFER_PAGED_V_SF_DESWIZZLE` in the SGLang FlashInfer build flags.

If green, write `results/<RUN>/summary.md` and update
`docs/RESULTS_LEDGER.md` as a checkpoint row. If red, write the exact first
failure and preserve `server.log`.
