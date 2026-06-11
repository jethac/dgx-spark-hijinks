# llama.cpp supplied-token loglikelihood live prep stop point

Date: 2026-06-10 22:24 JST

Status: prepared; live GB10 build/run deferred because `CLAUDE_WINDOW_OPEN` stayed present
for the full 30-minute wait window.

## Local repo state

- Parent commit with endpoint runner: `2988f96 Add llama.cpp loglikelihood endpoint runner`
- New runner: `scripts/llamacpp_supplied_loglikelihood_endpoint_runner.py`
- Runner syntax check: `python -m py_compile ...` passed.
- Dry-run shape over `tasks/llamacpp_loglikelihood_smoke.jsonl` produced the expected
  request rows. The contract auditor correctly rejects dry-run rows because they contain
  no token logprobs; this was only a schema/path smoke.

## Remote prep

Host: `jethac@thinkstationpgx-00b4`

Prepared checkout:

```text
/home/jethac/spark_tmp/dgx-spark-hijinks-llamacpp-accuracy
```

Pinned state:

```text
parent: 2988f96 Add llama.cpp loglikelihood endpoint runner
llama.cpp: aa6a59619 Add supplied-token loglikelihood endpoint
branch: jethac/llama.cpp@spark/hijinks-008-supplied-loglikelihood
```

Known model path:

```text
/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Important remote hygiene note:

- The shared checkout `/home/jethac/spark_tmp/dgx-spark-hijinks-022` had untracked
  Claude/Codex artifacts and could not safely switch branches. It was left untouched.
- A separate throwaway checkout was created for this lane instead.
- A first attempt used `--recurse-submodules` and began cloning unrelated nested
  FlashInfer/vLLM submodules; those processes were killed by path. The checkout was then
  recovered enough for this lane by resetting the parent branch and using only
  `third_party/llama.cpp`.

## Blocker

The marker remained open:

```text
/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
```

Last poll result:

```text
TIMEOUT_MARKER_STILL_OPEN
```

No llama.cpp CMake build, llama-server, Docker container, or GPU work was started from this
lane.

## Resume commands

Run only after the marker is absent and `docker ps` is clean:

```bash
set -euo pipefail

REPO=/home/jethac/spark_tmp/dgx-spark-hijinks-llamacpp-accuracy
MODEL=/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN=llamacpp_supplied_loglikelihood_$(date +%Y%m%dT%H%M%SJST)
OUT=$REPO/results/$RUN
PORT=18084

test ! -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
docker ps
free -h

cd "$REPO"
git fetch origin
git checkout docs/codex-direction-nvfp4-kv
git pull --ff-only

git -C third_party/llama.cpp fetch origin
git -C third_party/llama.cpp checkout spark/hijinks-008-supplied-loglikelihood
git -C third_party/llama.cpp reset --hard aa6a5961977139f23ae54dc8279fdac3d1494a77

cmake -S third_party/llama.cpp -B /home/jethac/spark_tmp/llama-loglikelihood-build \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja
cmake --build /home/jethac/spark_tmp/llama-loglikelihood-build --target llama-server -j "$(nproc)"

mkdir -p "$OUT"
/home/jethac/spark_tmp/llama-loglikelihood-build/bin/llama-server \
  -m "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -c 8192 \
  -ngl 999 \
  --no-warmup \
  --cache-ram 0 \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

python3 - <<PY
import time, urllib.request
url = "http://127.0.0.1:${PORT}/health"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("ready")
                raise SystemExit(0)
    except Exception:
        pass
    time.sleep(1)
raise SystemExit("server did not become ready")
PY

python3 scripts/llamacpp_supplied_loglikelihood_endpoint_runner.py \
  --url "http://127.0.0.1:${PORT}" \
  --endpoint /loglikelihood \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output "$OUT/contract_artifact.json"

python3 scripts/llamacpp_loglikelihood_contract_audit.py \
  --artifact "$OUT/contract_artifact.json" \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output "$OUT/contract_audit.json"

/home/jethac/spark_tmp/llama-loglikelihood-build/bin/llama-server --version \
  > "$OUT/llama_server_version.txt" 2>&1 || true
git -C third_party/llama.cpp rev-parse HEAD > "$OUT/llama_cpp_commit.txt"
```

Pass condition:

- `contract_audit.json` has `"ok": true`.
- The `zebra_unlikely` row has finite direct logprobs for every continuation token, not a
  top-N miss.
- `lm_eval_loglikelihood_tuple` is `[finite_number, bool]` for every row.
