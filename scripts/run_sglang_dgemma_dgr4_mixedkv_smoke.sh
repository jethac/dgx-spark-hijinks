#!/usr/bin/env bash
# DG-R4 Spark packet: DiffusionGemma 26B-A4B mixed-KV + FlashInfer VO-split opt-in.
#
# Scope: serving/routing/capacity-proof smoke only. This enables SGLang's
# conservative mixed-KV path (FP8-K + NVFP4-V), not full NVFP4 K+V.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
SOURCE_BRANCH="${SOURCE_BRANCH:-epoch2}"
IMAGE="${IMAGE:-sglang-source-stack-dgemma-024-0705924c-f99323bd:latest}"
MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
PORT="${PORT:-30125}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1200}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
SGLANG_COMMIT="${SGLANG_COMMIT:-dec4c040a8ede4561c1f26cccc599286643b49fd}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-f99323bd7d1c}"
RUN_ID="${RUN_ID:-sglang_dgemma_dgr4_mixedkv_smoke_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/tmp/flashinfer-cache-${RUN_ID}}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
CLAUDE_MARKER="${CLAUDE_MARKER:-/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN}"

if [[ -e "${CLAUDE_MARKER}" ]]; then
  echo "CLAUDE_WINDOW_OPEN present; yielding" >&2
  exit 99
fi
if [[ "$(docker ps -q | wc -l)" != "0" ]]; then
  echo "docker is not empty; yielding" >&2
  docker ps >&2
  exit 99
fi

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "repo_root=${REPO_ROOT}"
  echo "source_branch=${SOURCE_BRANCH}"
  echo "image=${IMAGE}"
  echo "model=${MODEL}"
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "sglang_commit=${SGLANG_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "scope=mixed-KV DiffusionGemma FlashInfer VO-split smoke"
  echo "kv_cache_dtype=fp4_e2m1"
  echo "sglang_fp4_kv_mixed_kv=1"
  echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  free -h
} | tee "${OUT_DIR}/preflight.log"

git fetch origin
git checkout "${SOURCE_BRANCH}"
git pull --ff-only
git -C third_party/sglang fetch origin
git -C third_party/sglang checkout "${SGLANG_COMMIT}"
git -C third_party/flashinfer fetch origin
git -C third_party/flashinfer checkout "${FLASHINFER_COMMIT}"
git -C third_party/flashinfer submodule update --init --recursive \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog

{
  echo "parent_commit=$(git rev-parse HEAD)"
  echo "sglang_commit=$(git -C third_party/sglang rev-parse HEAD)"
  echo "flashinfer_commit=$(git -C third_party/flashinfer rev-parse HEAD)"
  git status --short
  git -C third_party/sglang status --short
  git -C third_party/flashinfer status --short
} | tee "${OUT_DIR}/checkout.log"

container="${RUN_ID}"
cid_file="${OUT_DIR}/container_id.txt"
server_log="${OUT_DIR}/server.log"
quality_json="${OUT_DIR}/revised_text_quality.json"
summary_md="${OUT_DIR}/summary.md"
dllm_config="${OUT_DIR}/dllm_config.yaml"

cat >"${dllm_config}" <<'EOF'
max_denoising_steps: 48
seed: 1234
sampler_config:
  entropy_bound: 0.1
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
EOF

capture_docker_logs() {
  local name="$1"
  local tmp_log="${server_log}.tmp"
  if docker logs "${name}" >"${tmp_log}" 2>&1; then
    mv "${tmp_log}" "${server_log}"
  else
    rm -f "${tmp_log}"
  fi
}

cleanup() {
  local status=$?
  if docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    capture_docker_logs "${container}" || true
    docker rm -f "${container}" >/dev/null 2>&1 || true
  elif [[ -s "${cid_file}" ]]; then
    capture_docker_logs "$(cat "${cid_file}")" || true
    docker rm -f "$(cat "${cid_file}")" >/dev/null 2>&1 || true
  fi
  docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
  free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

docker rm -f "${container}" >/dev/null 2>&1 || true
cid=$(
  docker run -d --name "${container}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -w /work \
    -v "${REPO_ROOT}:/work" \
    -v "${REPO_ROOT}/third_party/flashinfer:/flashinfer-src" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e MODEL="${MODEL}" \
    -e PORT="${PORT}" \
    -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR}" \
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
    -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
    -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
    -e SGLANG_FLASHINFER_VOSPLIT=1 \
    -e SGLANG_FP4_KV_MIXED_KV=1 \
    -e SGLANG_FP4_KV_TRACE_MODULE=1 \
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 \
    -e SGLANG_GEMMA_KV_GEOMETRY=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${IMAGE}" \
    bash -lc '
      set -euo pipefail
      rm -rf /root/.cache/flashinfer "${FLASHINFER_CACHE_DIR}"
      mkdir -p /tmp/flashinfer-python-path
      ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer

      python - <<'"'"'PY'"'"'
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
        --model-path "${MODEL}" \
        --dllm-algorithm Gemma4Renoise \
        --dllm-algorithm-config "/work/results/'"${RUN_ID}"'/dllm_config.yaml" \
        --trust-remote-code \
        --dtype bfloat16 \
        --attention-backend flashinfer \
        --kv-cache-dtype fp4_e2m1 \
        --context-length 8192 \
        --page-size 256 \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        --host 0.0.0.0 \
        --port "${PORT}"
    '
)
echo "${cid}" >"${cid_file}"

ready=0
attempts=$((READY_TIMEOUT_S / 5))
if (( attempts < 1 )); then
  attempts=1
fi
for _ in $(seq 1 "${attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    break
  fi
  sleep 5
done

capture_docker_logs "${container}" || true

if [[ "${ready}" != "1" ]]; then
  {
    echo "# SGLang DiffusionGemma DG-R4 Mixed-KV Smoke"
    echo
    echo "Status: RED - server did not reach readiness."
    echo
    echo "- Run: \`${RUN_ID}\`"
    echo "- Server log: \`server.log\`"
    if grep -q "Unsupported max_mma_kv: 0" "${server_log}"; then
      echo "- First blocker: \`Unsupported max_mma_kv: 0\`"
    fi
  } >"${summary_md}"
  exit 1
fi

set +e
python3 scripts/diffusion_gemma_dgr2_revised_text_quality_client.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${MODEL}" \
  --out "${quality_json}" \
  --repeats 2 \
  --ready-timeout-s 30 \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  >"${OUT_DIR}/quality_client.stdout" \
  2>"${OUT_DIR}/quality_client.stderr"
quality_status=$?
set -e
echo "${quality_status}" >"${OUT_DIR}/quality_status.txt"

capture_docker_logs "${container}" || true

python3 - "${OUT_DIR}" "${RUN_ID}" "${MODEL}" "${IMAGE}" "${SGLANG_COMMIT}" "${FLASHINFER_COMMIT}" <<'PY'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
run_id, model, image, sglang_commit, flashinfer_commit = sys.argv[2:7]
server_log = out_dir / "server.log"
quality_json = out_dir / "revised_text_quality.json"
summary_md = out_dir / "summary.md"

log = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
quality = {}
if quality_json.exists():
    quality = json.loads(quality_json.read_text(encoding="utf-8"))

quality_ok = bool(quality.get("all_ok"))
policy_ok = "DiffusionGemma is using the experimental FlashInfer VO-split path" in log
mixed_mode_ok = (
    "SGLang FP4 KV mixed mode enabled" in log
    and "K cache uses FP8 e4m3" in log
    and "V cache uses packed NVFP4" in log
)
kv_dtype_ok = "kv_cache_dtype='fp4_e2m1'" in log or 'kv_cache_dtype="fp4_e2m1"' in log
pool_mixed_ok = "mixed_kv=True" in log
pool_class_ok = "full_pool=MHATokenToKVPoolFP4" in log and "swa_pool=MHATokenToKVPoolFP4" in log
geometry_lines = [
    line for line in log.splitlines()
    if "SGLang Gemma4 FlashInfer geometry" in line
]
vosplit_lines = [
    line for line in geometry_lines
    if "layer_head_dim=512" in line
    and (
        "vo_split=True" in line
        or "label=extend_paged_vosplit" in line
        or "label=decode_as_prefill_vosplit" in line
    )
]
head_dim_vo_ok = any(
    re.search(r"head_dim_vo=256|head_dim_vo': 256|head_dim_vo\": 256", line)
    for line in vosplit_lines
)
route_ok = policy_ok and bool(vosplit_lines) and head_dim_vo_ok
kv_ok = mixed_mode_ok and kv_dtype_ok and pool_mixed_ok and pool_class_ok

status = "GREEN" if quality_ok and route_ok and kv_ok else "RED"
reasons = []
if not quality_ok:
    reasons.append("revised text quality gate failed or is missing")
if not policy_ok:
    reasons.append("DiffusionGemma FlashInfer VO-split policy warning missing")
if not mixed_mode_ok:
    reasons.append("mixed-KV backend warning missing")
if not kv_dtype_ok:
    reasons.append("server args do not prove kv_cache_dtype='fp4_e2m1'")
if not pool_mixed_ok:
    reasons.append("pool configurator did not report mixed_kv=True")
if not pool_class_ok:
    reasons.append("hybrid SWA pool did not use MHATokenToKVPoolFP4 for both subpools")
if not vosplit_lines:
    reasons.append("no D=512 geometry line with VO-split trace label")
if vosplit_lines and not head_dim_vo_ok:
    reasons.append("D=512 VO-split geometry did not expose head_dim_vo=256")

lines = [
    "# SGLang DiffusionGemma DG-R4 Mixed-KV Smoke",
    "",
    f"Status: {status}",
    "",
    "## Scope",
    "",
    "DiffusionGemma 26B-A4B text-only serving through SGLang's conservative mixed-KV path: FP8-K + NVFP4-V, with the experimental FlashInfer VO-split opt-in for D=512 global layers. This is not a full NVFP4 K+V row.",
    "",
    "## Provenance",
    "",
    f"- Run: `{run_id}`",
    f"- Model: `{model}`",
    f"- Image: `{image}`",
    f"- SGLang: `{sglang_commit}`",
    f"- FlashInfer: `{flashinfer_commit}`",
    "- Launch: `--dllm-algorithm Gemma4Renoise --dllm-algorithm-config dllm_config.yaml --attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --dtype bfloat16 --page-size 256 --disable-cuda-graph --disable-piecewise-cuda-graph`",
    "- Environment: `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_FP4_KV_MIXED_KV=1`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, offline HF mode",
    "",
    "## Gates",
    "",
    f"- Revised DG-R2 text quality gate: {'PASS' if quality_ok else 'FAIL'}",
    f"- Opt-in policy warning present: {'PASS' if policy_ok else 'FAIL'}",
    f"- Mixed-KV backend warning present: {'PASS' if mixed_mode_ok else 'FAIL'}",
    f"- Server args prove `kv_cache_dtype='fp4_e2m1'`: {'PASS' if kv_dtype_ok else 'FAIL'}",
    f"- Pool configurator reports `mixed_kv=True`: {'PASS' if pool_mixed_ok else 'FAIL'}",
    f"- Hybrid subpools are `MHATokenToKVPoolFP4`: {'PASS' if pool_class_ok else 'FAIL'}",
    f"- D=512 geometry routes through VO-split trace labels: {'PASS' if bool(vosplit_lines) else 'FAIL'}",
    f"- D=512 VO-split exposes `head_dim_vo=256`: {'PASS' if head_dim_vo_ok else 'FAIL'}",
]
if reasons:
    lines += ["", "## Red Reasons", ""]
    lines += [f"- {reason}" for reason in reasons]
if geometry_lines:
    lines += ["", "## Geometry Evidence", ""]
    for line in vosplit_lines[:8]:
        lines.append(f"- `{line[:500]}`")
pool_lines = [
    line for line in log.splitlines()
    if "SGLANG_GEMMA_KV_POOL_CONFIG" in line
    or "SGLANG_GEMMA_KV_SWAKVPOOL" in line
    or "KV Cache is allocated" in line
    or "max_total_num_tokens=" in line
    or "SGLang FP4 KV mixed mode enabled" in line
]
if pool_lines:
    lines += ["", "## Mixed-KV Evidence", ""]
    for line in pool_lines[:16]:
        lines.append(f"- `{line[:500]}`")
if quality.get("checks"):
    lines += ["", "## Quality Checks", ""]
    for check in quality["checks"]:
        text0 = (check.get("texts") or [""])[0].replace("\n", " ")
        if len(text0) > 180:
            text0 = text0[:177] + "..."
        lines.append(
            f"- `{check.get('prompt_id')}`: stable={check.get('stable')} "
            f"non_empty={check.get('non_empty')} answer_ok={check.get('answer_ok')} text={text0!r}"
        )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": status, "quality_ok": quality_ok, "route_ok": route_ok, "kv_ok": kv_ok}, sort_keys=True))
if status != "GREEN":
    raise SystemExit(1)
PY
