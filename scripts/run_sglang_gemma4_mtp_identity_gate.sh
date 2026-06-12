#!/usr/bin/env bash
# SGLang Gemma 4 MTP identity gate.
#
# Runs spec-off and spec-on sequentially under GB10 memory guardrails, then
# compares greedy outputs. No two model servers are kept alive together.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
SOURCE_BRANCH="${SOURCE_BRANCH:-epoch2}"
IMAGE="${IMAGE:-sglang-source-stack-dgemma-024-0705924c-f99323bd:latest}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
DRAFT_MODEL="${DRAFT_MODEL:-google/gemma-4-E2B-it-assistant}"
PORT="${PORT:-30135}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
SGLANG_COMMIT="${SGLANG_COMMIT:-dec4c040a8ede4561c1f26cccc599286643b49fd}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-f99323bd7d1c}"
RUN_ID="${RUN_ID:-sglang_gemma4_mtp_identity_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
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
  echo "draft_model=${DRAFT_MODEL}"
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "sglang_commit=${SGLANG_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "hf_cache=${HF_CACHE}"
  echo "hf_local_files_only=${HF_LOCAL_FILES_ONLY}"
  echo "hf_hub_offline=${HF_HUB_OFFLINE}"
  echo "scope=bf16 greedy spec-off/spec-on MTP identity, sequential servers"
  echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  free -h
} | tee "${OUT_DIR}/preflight.log"

git fetch origin
git checkout "${SOURCE_BRANCH}"
git pull --ff-only

checkout_submodule_commit() {
  local path="$1"
  local commit="$2"
  if ! git -C "${path}" rev-parse --verify "${commit}^{commit}" >/dev/null 2>&1; then
    git -C "${path}" fetch origin
  fi
  git -C "${path}" checkout "${commit}"
}

checkout_submodule_commit third_party/sglang "${SGLANG_COMMIT}"
checkout_submodule_commit third_party/flashinfer "${FLASHINFER_COMMIT}"
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

probe_flags=()
if [[ "${HF_LOCAL_FILES_ONLY}" == "1" ]]; then
  probe_flags+=(--local-files-only)
fi
python3 scripts/hf_model_access_probe.py \
  --model "${MODEL}" \
  --cache-dir "${HF_CACHE}" \
  --output "${OUT_DIR}/hf_access_target.json" \
  "${probe_flags[@]}" || true
python3 scripts/hf_model_access_probe.py \
  --model "${DRAFT_MODEL}" \
  --cache-dir "${HF_CACHE}" \
  --output "${OUT_DIR}/hf_access_draft.json" \
  "${probe_flags[@]}" || true

container=""
cid_file="${OUT_DIR}/container_id.txt"

capture_docker_logs() {
  local name="$1"
  local log_path="$2"
  local tmp_log="${log_path}.tmp"
  if [[ -n "${name}" ]] && docker logs "${name}" >"${tmp_log}" 2>&1; then
    mv "${tmp_log}" "${log_path}"
  else
    rm -f "${tmp_log}"
  fi
}

cleanup_container() {
  local name="$1"
  local log_path="$2"
  if [[ -n "${name}" ]] && docker ps --format '{{.Names}}' | grep -qx "${name}"; then
    capture_docker_logs "${name}" "${log_path}" || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
  elif [[ -n "${name}" ]]; then
    capture_docker_logs "${name}" "${log_path}" || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local status=$?
  if [[ -n "${container}" ]]; then
    cleanup_container "${container}" "${OUT_DIR}/${container}.server.log" || true
  fi
  docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
  free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

launch_and_capture() {
  local label="$1"
  local extra_args="$2"
  local flashinfer_cache_dir="/tmp/flashinfer-cache-${RUN_ID}-${label}"
  local server_log="${OUT_DIR}/${label}.server.log"
  local capture_json="${OUT_DIR}/${label}.capture.json"

  container="${RUN_ID}-${label}"
  docker rm -f "${container}" >/dev/null 2>&1 || true
  local cid
  cid=$(
    docker run -d --name "${container}" --gpus all --ipc=host --network=host \
      --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
      -w /work \
      -v "${REPO_ROOT}:/work" \
      -v "${REPO_ROOT}/third_party/flashinfer:/flashinfer-src" \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      -e MODEL="${MODEL}" \
      -e DRAFT_MODEL="${DRAFT_MODEL}" \
      -e PORT="${PORT}" \
      -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
      -e TORCH_CUDA_ARCH_LIST=12.1a \
      -e FLASHINFER_CACHE_DIR="${flashinfer_cache_dir}" \
      -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
      -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
      -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
      -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
      -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
      -e SGLANG_FLASHINFER_VOSPLIT=1 \
      -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 \
      -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
      -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
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
          --dtype bfloat16 \
          --attention-backend flashinfer \
          --page-size 1 \
          --mem-fraction-static "${MEM_FRACTION_STATIC}" \
          --disable-cuda-graph \
          --disable-piecewise-cuda-graph \
          --host 0.0.0.0 \
          --port "${PORT}" '"${extra_args}"'
      '
  )
  echo "${cid}" >"${cid_file}"

  local attempts=$((READY_TIMEOUT_S / 5))
  if (( attempts < 1 )); then
    attempts=1
  fi
  local ready=0
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 ||
      curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
      break
    fi
    sleep 5
  done
  capture_docker_logs "${container}" "${server_log}" || true
  if [[ "${ready}" != "1" ]]; then
    echo "${label} server did not reach readiness" >&2
    return 1
  fi

  python3 scripts/sglang_mtp_identity_client.py \
    --mode capture \
    --base-url "http://127.0.0.1:${PORT}" \
    --label "${label}" \
    --model "${MODEL}" \
    --out "${capture_json}" \
    --ready-timeout-s 30 \
    --request-timeout-s "${REQUEST_TIMEOUT_S}" \
    >"${OUT_DIR}/${label}.capture.stdout" \
    2>"${OUT_DIR}/${label}.capture.stderr"
  capture_docker_logs "${container}" "${server_log}" || true
  cleanup_container "${container}" "${server_log}" || true
  container=""
}

launch_and_capture "spec_off" ""

launch_and_capture "spec_on" "--speculative-algorithm NEXTN --speculative-draft-model-path \"${DRAFT_MODEL}\" --speculative-num-steps 1 --speculative-num-draft-tokens 1 --speculative-eagle-topk 1 --speculative-draft-model-quantization unquant"

set +e
python3 scripts/sglang_mtp_identity_client.py \
  --mode compare \
  --spec-off-artifact "${OUT_DIR}/spec_off.capture.json" \
  --spec-on-artifact "${OUT_DIR}/spec_on.capture.json" \
  --out "${OUT_DIR}/identity_comparison.json" \
  >"${OUT_DIR}/identity_compare.stdout" \
  2>"${OUT_DIR}/identity_compare.stderr"
compare_status=$?
set -e
echo "${compare_status}" >"${OUT_DIR}/identity_compare_status.txt"

python3 - "${OUT_DIR}" "${RUN_ID}" "${MODEL}" "${DRAFT_MODEL}" "${IMAGE}" "${SGLANG_COMMIT}" "${FLASHINFER_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
run_id, model, draft_model, image, sglang_commit, flashinfer_commit = sys.argv[2:8]
comparison_path = out_dir / "identity_comparison.json"
comparison = {}
if comparison_path.exists():
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
cmp = comparison.get("comparison", {})
status = "GREEN" if cmp.get("all_ok") else "RED"
reasons = []
if not cmp.get("text_identity_ok"):
    reasons.append("spec-on/spec-off text identity failed")
if not cmp.get("any_token_ids"):
    reasons.append("no token IDs exposed by captured endpoints; zero-bug token identity gate cannot pass")
elif not cmp.get("token_identity_ok"):
    reasons.append("spec-on/spec-off token identity failed")

lines = [
    "# SGLang Gemma 4 MTP Identity Gate",
    "",
    f"Status: {status}",
    "",
    "## Scope",
    "",
    "BF16 greedy spec-off vs spec-on identity for SGLang Gemma 4 Frozen-KV MTP. Servers are run sequentially; this is not a speedup claim and not an NVFP4 row.",
    "",
    "## Provenance",
    "",
    f"- Run: `{run_id}`",
    f"- Target model: `{model}`",
    f"- Draft model: `{draft_model}`",
    f"- Image: `{image}`",
    f"- SGLang: `{sglang_commit}`",
    f"- FlashInfer: `{flashinfer_commit}`",
    "- Spec flags: `--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-num-draft-tokens 1 --speculative-eagle-topk 1 --speculative-draft-model-quantization unquant`",
    "- Graphs disabled for the first identity gate.",
    "",
    "## Gates",
    "",
    f"- Text identity: {'PASS' if cmp.get('text_identity_ok') else 'FAIL'}",
    f"- Token IDs exposed: {'PASS' if cmp.get('any_token_ids') else 'FAIL'}",
    f"- Token identity: {'PASS' if cmp.get('token_identity_ok') else 'FAIL'}",
]
if reasons:
    lines += ["", "## Red Reasons", ""]
    lines += [f"- {reason}" for reason in reasons]
checks = cmp.get("checks") or []
if checks:
    lines += ["", "## Prompt Checks", ""]
    for check in checks:
        off = (check.get("off_chat_text") or "").replace("\n", " ")
        on = (check.get("on_chat_text") or "").replace("\n", " ")
        if len(off) > 140:
            off = off[:137] + "..."
        if len(on) > 140:
            on = on[:137] + "..."
        lines.append(
            f"- `{check.get('prompt_id')}`: chat_text_match={check.get('chat_text_match')} "
            f"native_text_match={check.get('native_text_match')} "
            f"chat_token_ids_match={check.get('chat_token_ids_match')} "
            f"native_token_ids_match={check.get('native_token_ids_match')} "
            f"off={off!r} on={on!r}"
        )
lines += [
    "",
    "## Artifacts",
    "",
    "- `spec_off.capture.json`",
    "- `spec_on.capture.json`",
    "- `identity_comparison.json`",
    "- `spec_off.server.log`",
    "- `spec_on.server.log`",
    "- `hf_access_target.json`",
    "- `hf_access_draft.json`",
]
(out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_dir / "summary.md")
if status != "GREEN":
    raise SystemExit(1)
PY
