# SGLang Gemma 4 VO-split validation packet

Date authored: 2026-06-11 JST. Owner: Codex SGLang lane. Run only inside a granted GPU
window. GB10 memory rules apply throughout: single server at a time, conservative
`--mem-fraction-static`, Docker `--memory 100g` for serving containers, sequential
comparators, and no concurrent fp8/mixed/NVFP4 servers.

This packet validates the offline scaffold recorded in
`results/sglang_gemma4_vosplit_authoring_20260611.md`.

## Code state this packet validates

- Parent repo: `jethac/dgx-spark-hijinks@docs/codex-direction-nvfp4-kv`, at or after
  `ed70eb5`.
- SGLang fork: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`, at or after
  `cf7414f80`.
- FlashInfer fork: `jethac/flashinfer@spark/hijinks-022-fa2-d512`, at or after
  `fb7d62ea`.
- Feature flag under test: `SGLANG_FLASHINFER_VOSPLIT=1`.
- Do **not** compile SGLang's FlashInfer modules with
  `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`; that flag is vLLM-layout-specific.

## Stop-on-red ordering

Stop at the first red block and write a summary before trying the next block.

1. Source/import and module-cache hygiene.
2. Existing SGLang writer/reader regression at head 256.
3. New head-512 VO-split writer-roundtrip harness.
4. Wrapper-plan dry smoke: prove the paged-prefill wrapper requests `head_dim_qk=512`,
   `head_dim_vo=256` only for full/global layers.
5. Gemma 4 31B text-only serve smoke.
6. Matched fp8-vs-mixed/NVFP4 comparator rows.

## Host prep

```bash
set -euo pipefail

REPO=/home/jethac/spark_tmp/dgx-spark-hijinks
cd "$REPO"
git fetch origin
git checkout docs/codex-direction-nvfp4-kv
git pull --ff-only

git -C third_party/sglang fetch origin
git -C third_party/sglang checkout cf7414f80

git -C third_party/flashinfer fetch origin
git -C third_party/flashinfer checkout fb7d62ea

docker ps
free -h
```

Gate:

- `docker ps` is empty except intentional short probes.
- `free -h` has normal OS headroom.
- No `CLAUDE_WINDOW_OPEN` marker is present unless this lane owns the current window.

## Block A — source/import and cache hygiene

Purpose: prove the container sees the intended SGLang/FlashInfer sources and no stale
FlashInfer module cache is carrying a previous mode.

```bash
RUN=sglang_gemma4_vosplit_blockA_$(date +%Y%m%dT%H%M%SJST)
OUT="$REPO/results/$RUN"
mkdir -p "$OUT"

docker run --rm --gpus all --memory=16g --memory-swap=16g --ipc=host \
  -w /work \
  -v "$REPO:/work" \
  -v "$REPO/third_party/flashinfer:/flashinfer-src" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR="/tmp/flashinfer-cache-$RUN" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  -e PYTHONPATH=/work/third_party/sglang/python:/tmp/flashinfer-python-path \
  sglang-source-stack-c3dae30f-e631a13fd:latest \
  bash -lc '
    set -euo pipefail
    rm -rf /root/.cache/flashinfer /tmp/flashinfer-cache-*
    mkdir -p /tmp/flashinfer-python-path
    ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
    python -m py_compile /work/third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
    python - <<PY
import flashinfer, os, sglang
from sglang.srt.layers.attention import flashinfer_backend as fb
print("flashinfer", getattr(flashinfer, "__file__", None))
print("sglang", getattr(sglang, "__file__", None))
print("vosplit_enabled_default", fb._flashinfer_vo_split_enabled())
print("extra_flags", os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""))
assert "FLASHINFER_PAGED_V_SF_DESWIZZLE" not in os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
PY
  ' | tee "$OUT/import_probe.log"
```

Gate:

- `py_compile` passes.
- `flashinfer` path resolves to `/flashinfer-src/flashinfer`.
- `sglang` path resolves to `/work/third_party/sglang/python/...`.
- `FLASHINFER_PAGED_V_SF_DESWIZZLE` is absent.

## Block B — head-256 writer/reader regression

Purpose: make sure the already-green SGLang linear-SF path remains green before testing
head 512.

```bash
REPO_ROOT="$REPO" \
IMAGE=sglang-source-stack-c3dae30f-e631a13fd:latest \
OUT_DIR="$REPO/results/sglang_fp4_kv_writer_roundtrip_vosplit_regression_$(date +%Y%m%dT%H%M%SJST)" \
bash "$REPO/scripts/run_sglang_fp4_kv_writer_roundtrip_container.sh"
```

Gate:

- Match or exceed the existing receipt:
  `results/sglang_fp4_kv_writer_roundtrip_20260611Tprobe2JST/summary.md`.
- Expected cosines:
  - global: around `0.99999118`
  - SWA/window: around `0.99974465`
  - decode-as-prefill: around `0.99999142`
- Any large drop means stop; do not climb to Gemma 4.

## Block C — head-512 VO-split writer-roundtrip harness

Purpose: this is the first new SGLang-specific gate. It must use the real
`MHATokenToKVPoolFP4.set_kv_buffer()` writer, then read through the two-pass FlashInfer
FA2 path with SGLang's **linear** V scale factors.

Required harness delta before running:

- Extend `scripts/sglang_fp4_kv_writer_roundtrip_probe.py` with a `--vo-split 2` mode.
- For `--head-dim 512 --vo-split 2`, plan the paged wrapper with:
  - `head_dim_qk=512`
  - `head_dim_vo=256`
  - unchanged K data and K scale factors
  - V data sliced into two 256-wide views
  - V scale factors sliced along the same last dimension for SGLang linear SF layout
  - output concatenated across the last dimension

Run:

```bash
RUN=sglang_fp4_kv_writer_roundtrip_head512_vosplit_$(date +%Y%m%dT%H%M%SJST)
OUT="$REPO/results/$RUN"
CACHE="/tmp/flashinfer-cache-$RUN"
mkdir -p "$OUT"

docker run --rm --gpus all --memory=16g --memory-swap=16g --ipc=host \
  -w /work \
  -v "$REPO:/work" \
  -v "$REPO/third_party/flashinfer:/flashinfer-src" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR="$CACHE" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  -e PYTHONPATH=/work/third_party/sglang/python:/tmp/flashinfer-python-path \
  -e SGLANG_FLASHINFER_VOSPLIT=1 \
  sglang-source-stack-c3dae30f-e631a13fd:latest \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/flashinfer-python-path
    ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
    python - <<PY
import hashlib
import importlib
import pathlib

modules = [
    "sgl_kernel",
    "sgl_kernel.common_ops",
]
for name in modules:
    mod = importlib.import_module(name)
    path = getattr(mod, "__file__", None)
    if not path:
        continue
    path = pathlib.Path(path).resolve()
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"binary_md5 {name} {path} {digest}")
PY
    python /work/scripts/sglang_fp4_kv_writer_roundtrip_probe.py \
      --head-dim 512 \
      --num-qo-heads 32 \
      --num-kv-heads 16 \
      --kv-len 384 \
      --qo-len 16 \
      --window-left 255 \
      --vo-split 2
  ' > "$OUT/container.stdout" 2> "$OUT/run.log"
awk 'found || /^\{/ { found = 1; print }' "$OUT/container.stdout" > "$OUT/output.json"
```

Gate:

- `container.stdout` includes `binary_md5` proof lines for loaded native SGLang
  binaries before the JSON result. If a loaded `.so` is not the image-installed copy
  expected for the current blessed stack, stop and fix provenance before trusting
  any cosine.
- FlashInfer generated module proves `head_dim_qk=512;head_dim_vo=256`.
- Cosine versus dequantized-pool reference is at least `0.9999`.
- Kernel-side comparison target from Claude Block A is `>=0.9999983` for pure probe
  shapes; the writer-roundtrip may be slightly lower, but anything below `0.9999` is red.
- If this fails, do not start a model server. File the result as a writer/reader or
  wrapper orchestration defect.

## Block D — wrapper-plan dry smoke

Purpose: prove SGLang constructs different plans for Gemma 4 SWA and global wrappers.
Gemma 4 normalizes config as:

- SWA/local layers: `swa_head_dim`, `swa_v_head_dim`
- full/global layers: `head_dim`, `v_head_dim`

Expected plan behavior under `SGLANG_FLASHINFER_VOSPLIT=1`:

- SWA wrapper: no VO split when the running layer is `D=256`.
- Full/global wrapper: `head_dim_qk=512`, `head_dim_vo=256`.
- No ctor-time `jit_args` pins symmetric head dims.

Run a minimal source-overlay launch with trace flags and a short timeout. Stop as soon as
the model reaches readiness and the first plan lines are captured.

```bash
RUN=sglang_gemma4_31b_vosplit_plan_smoke_$(date +%Y%m%dT%H%M%SJST)
OUT="$REPO/results/$RUN"
mkdir -p "$OUT"

docker run --rm --gpus all --memory=100g --memory-swap=100g --ipc=host \
  -w /work \
  -v "$REPO:/work" \
  -v "$REPO/third_party/flashinfer:/flashinfer-src" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR="/tmp/flashinfer-cache-$RUN" \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  -e PYTHONPATH=/work/third_party/sglang/python:/tmp/flashinfer-python-path \
  -e SGLANG_FLASHINFER_VOSPLIT=1 \
  -e SGLANG_FP4_KV_TRACE_MODULE=1 \
  -e SGLANG_FP4_KV_TRACE_RADIX=1 \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 30000:30000 \
  sglang-source-stack-c3dae30f-e631a13fd:latest \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/flashinfer-python-path
    ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
    exec python3 -m sglang.launch_server \
      --model-path google/gemma-4-31B-it \
      --attention-backend flashinfer \
      --kv-cache-dtype fp4_e2m1 \
      --page-size 1 \
      --mem-fraction-static 0.60 \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --host 0.0.0.0 \
      --port 30000
  ' > "$OUT/server.log" 2>&1
```

Gate:

- Server reaches readiness or fails only after emitting enough plan lines to diagnose.
- Logs show the VO split warning from the SGLang scaffold.
- Logs prove full/global D=512 layers request `head_dim_vo=256`.
- Logs prove SWA D=256 layers do not accidentally use global D=512 planning.
- If all wrapper plans use `model_config.head_dim=512`, stop and fix wrapper-id geometry
  before serving.

## Block E — Gemma 4 31B text-only smoke

Purpose: first serving smoke for SGLang Gemma 4 Rung 2. This is not a capacity row.

Use the same launch as Block D, but keep the server running and send one short text-only
request. Record:

- model id: `google/gemma-4-31B-it`
- SGLang commit
- FlashInfer commit
- `SGLANG_FLASHINFER_VOSPLIT=1`
- KV dtype actually selected per local/global layer
- whether radix cache is on
- first token and short completion text
- server logs around first prefill and decode

Gate:

- Text-only request completes without gibberish or obvious first-token corruption.
- No D=512 FlashInfer trait rejection.
- No raw-byte or missing-scale FP4 prefill path.
- No `FLASHINFER_PAGED_V_SF_DESWIZZLE` in SGLang modules.

If green, proceed to a short fp8 comparator in a later row. Do not run fp8 and FP4 servers
concurrently.

## Block F — comparator row, only after smoke is green

Purpose: claim-grade row. Run sequential servers with matched memory budget and page size:

1. fp8 KV comparator
2. mixed FP8-K + NVFP4-V
3. full NVFP4 K+V only if the structural route is explicitly under test

Required gates:

- Matched prompt corpus and max token settings.
- Supplied-token PPL or first-token/logprob quality gate.
- Capacity denominator audited as physical K+V bytes/token, not raw allocator token count.
- CUDA graphs disabled until the SGLang prefix-cache graph-write caveat is closed for this
  model.

Expected capacity labels:

- mixed FP8-K + NVFP4-V: about `1.28x` byte-ratio class.
- full NVFP4 K+V: `1.78x` class only if both K and V are actually NVFP4 and the row reads
  FP4 K in attention.

## Evidence to write after the window

Create:

- `results/sglang_gemma4_vosplit_validation_<stamp>/summary.md`
- raw container logs for every block attempted
- JSON probe outputs
- exact commits/images
- `docker ps` before/after
- explicit stop-on-red reason if any block fails

Update:

- `docs/RESULTS_LEDGER.md` only for green or checkpoint rows.
- `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md` with any new blocker.
- `results/sglang_gemma4_vosplit_authoring_20260611.md` if the scaffold scope changes.
