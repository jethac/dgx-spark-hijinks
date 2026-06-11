#!/usr/bin/env bash
# SGLang Spark ladder block for docs/OVERNIGHT_LADDER_PLAN_20260612.md.
#
# This is deliberately conservative: one server at a time, write marker first,
# refuse to run if Docker is already busy, and record RED rows instead of
# promoting weak evidence.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live}"
SOURCE_BRANCH="${SOURCE_BRANCH:-epoch2}"
IMAGE="${IMAGE:-sglang-source-stack-dgemma-024-0705924c-f99323bd:latest}"
PORT="${PORT:-30000}"
CTX="${CTX:-8191}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.40}"
GB10_DOCKER_MEMORY="${GB10_DOCKER_MEMORY:-100g}"
GB10_DOCKER_MEMORY_SWAP="${GB10_DOCKER_MEMORY_SWAP:-100g}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1800}"
PPL_TIMEOUT_S="${PPL_TIMEOUT_S:-3600}"
CLAUDE_MARKER="${CLAUDE_MARKER:-/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN}"
RUN_ID="${RUN_ID:-sglang_overnight_ladder_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${RUN_ID}}"
STATUS_FILE="${OUT_DIR}/status.txt"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
FLASHINFER_SRC="${FLASHINFER_SRC:-${REPO_ROOT}/third_party/flashinfer}"
FLASHINFER_CACHE_BASE="${FLASHINFER_CACHE_BASE:-/tmp}"
CORPUS_SRC_DIR="${CORPUS_SRC_DIR:-/home/jethac/spark_tmp/claude_overnight_ladder_20260612/docs}"

MODEL_MATRIX=(
  "e2b|google/gemma-4-E2B-it|gemma4-e2b-it|0.35"
  "12b|google/gemma-4-12B-it|gemma4-12b-it|0.38"
  "26b_a4b|google/gemma-4-26B-A4B-it|gemma4-26b-a4b-it|0.38"
  "31b|google/gemma-4-31B-it|gemma4-31b-it|0.36"
)

ROW_MATRIX=(
  "bf16|auto|0"
  "nvfp4|fp4_e2m1|0"
  "fp8|fp8_e4m3|0"
)

log() {
  mkdir -p "${OUT_DIR}"
  printf '%s %s\n' "$(TZ=Asia/Tokyo date -Is)" "$*" | tee -a "${STATUS_FILE}"
}

next_mail_path() {
  python3 - "$REPO_ROOT" <<'PY'
import re
import sys
from pathlib import Path

mail = Path(sys.argv[1]) / "mail"
mail.mkdir(exist_ok=True)
max_id = 0
for path in mail.glob("*.md"):
    match = re.match(r"(\d+)_", path.name)
    if match:
        max_id = max(max_id, int(match.group(1)))
print(mail / f"{max_id + 1:04d}_codex-to-claude_sglang-overnight-stop.md")
PY
}

write_stop_mail() {
  local verdict="$1"
  local mail_path
  mail_path="$(next_mail_path)"
  cat >"${mail_path}" <<EOF
# $(basename "${mail_path}") Codex -> Claude: SGLang overnight ladder stop point

Date: $(TZ=Asia/Tokyo date -Is)

Verdict: ${verdict}

Artifacts:
- Run directory: \`${OUT_DIR}\`
- Status file: \`${STATUS_FILE}\`
- Summary: \`${OUT_DIR}/summary.md\`

Spark state after stop:

\`\`\`
$(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null || true)
\`\`\`

Marker: $(if [[ -e "${CLAUDE_MARKER}" ]]; then echo present; else echo absent; fi)

Notes:
- Rows are RED unless their own row manifests prove readiness, coherent smoke,
  C1 deterministic PPL, and bounded quantized-vs-bf16 deltas.
- DG-R2 remains deferred behind the Gemma ladder per mail/0029.
EOF
  log "MAIL_WRITTEN ${mail_path}"
}

cleanup_marker() {
  local status=$?
  if [[ -f "${CLAUDE_MARKER}" ]] && grep -q "owner=codex-sglang-overnight" "${CLAUDE_MARKER}" 2>/dev/null; then
    rm -f "${CLAUDE_MARKER}"
  fi
  exit "${status}"
}
trap cleanup_marker EXIT

claim_window() {
  if [[ -e "${CLAUDE_MARKER}" ]]; then
    echo "CLAUDE_WINDOW_OPEN present; yielding" >&2
    exit 99
  fi
  mkdir -p "$(dirname "${CLAUDE_MARKER}")"
  {
    echo "owner=codex-sglang-overnight"
    echo "run_id=${RUN_ID}"
    echo "repo=${REPO_ROOT}"
    echo "started_at=$(TZ=Asia/Tokyo date -Is)"
  } >"${CLAUDE_MARKER}"

  if [[ "$(docker ps -q | wc -l)" != "0" ]]; then
    mkdir -p "${OUT_DIR}"
    docker ps >"${OUT_DIR}/docker_ps_busy_at_claim.txt" 2>&1 || true
    log "DOCKER_BUSY_AFTER_MARKER; yielding without row work"
    rm -f "${CLAUDE_MARKER}"
    exit 99
  fi
}

prepare_repo() {
  cd "${REPO_ROOT}" || exit 2
  mkdir -p "${OUT_DIR}" "${OUT_DIR}/docs" "${OUT_DIR}/rows"
  log "RUN_START run_id=${RUN_ID} image=${IMAGE}"
  log "SPARK_DATE $(date -Is)"
  log "DOCKER_IMAGE_ID $(docker images --format '{{.ID}}' "${IMAGE}" 2>/dev/null | head -1)"
  free -h | tee "${OUT_DIR}/free_before.txt" >/dev/null
  docker ps >"${OUT_DIR}/docker_ps_before.txt" 2>&1 || true

  git fetch origin >>"${OUT_DIR}/git_setup.log" 2>&1
  git checkout "${SOURCE_BRANCH}" >>"${OUT_DIR}/git_setup.log" 2>&1
  git pull --ff-only >>"${OUT_DIR}/git_setup.log" 2>&1
  git submodule update --init third_party/sglang third_party/flashinfer >>"${OUT_DIR}/git_setup.log" 2>&1
  git -C third_party/flashinfer submodule update --init --recursive \
    3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog >>"${OUT_DIR}/git_setup.log" 2>&1

  {
    echo "parent_commit=$(git rev-parse HEAD)"
    echo "sglang_commit=$(git -C third_party/sglang rev-parse HEAD)"
    echo "flashinfer_commit=$(git -C third_party/flashinfer rev-parse HEAD)"
    git status --short
    git -C third_party/sglang status --short
    git -C third_party/flashinfer status --short
  } | tee "${OUT_DIR}/checkout.log" >/dev/null

  for corpus in c1_ppl_corpus.md c2_pride_prejudice_60k.txt c3_hijinks_code_60k.py; do
    cp "${CORPUS_SRC_DIR}/${corpus}" "${OUT_DIR}/docs/${corpus}"
  done
  md5sum "${OUT_DIR}/docs/"* | tee "${OUT_DIR}/corpus_md5.txt" >/dev/null
  if ! grep -q "abb63f0e" "${OUT_DIR}/corpus_md5.txt"; then
    log "CORPUS_MD5_RED c1"
    write_summary || true
    write_stop_mail "corpus_md5_red"
    exit 2
  fi
  if ! grep -q "1686a33b" "${OUT_DIR}/corpus_md5.txt"; then
    log "CORPUS_MD5_RED c2"
    write_summary || true
    write_stop_mail "corpus_md5_red"
    exit 2
  fi
  if ! grep -q "28dfeba9" "${OUT_DIR}/corpus_md5.txt"; then
    log "CORPUS_MD5_RED c3"
    write_summary || true
    write_stop_mail "corpus_md5_red"
    exit 2
  fi
  log "CORPUS_MD5_GREEN source=${CORPUS_SRC_DIR}"
}

run_access_probe() {
  local slug="$1" model="$2"
  local out="${OUT_DIR}/preflight_${slug}_hf_access.json"
  local cache_dir="${FLASHINFER_CACHE_BASE}/flashinfer-cache-${RUN_ID}-preflight-${slug}"
  local env_args=(
    -e TORCH_CUDA_ARCH_LIST=12.1a
    -e FLASHINFER_CACHE_DIR="${cache_dir}"
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a"
    -e FLASHINFER_PREFILL_DEBUG_ONCE=1
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src
    -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path
    -e SGLANG_FLASHINFER_VOSPLIT=1
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1
    -e SGLANG_GEMMA_KV_GEOMETRY=1
    -e SGLANG_FP4_KV_MIXED_KV=0
    -e HF_TOKEN="${HF_TOKEN:-}"
  )
  log "PREFLIGHT_MODEL_ACCESS ${slug} ${model}"
  docker run --rm --memory=8g --memory-swap=8g --ipc=host \
    -w /work \
    -v "${REPO_ROOT}:/work" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${FLASHINFER_SRC}:/flashinfer-src" \
    "${env_args[@]}" \
    "${IMAGE}" \
    bash -lc "python3 scripts/hf_model_access_probe.py --model '${model}' --cache-dir /root/.cache/huggingface --output '${out#${REPO_ROOT}/}'" \
    >"${OUT_DIR}/preflight_${slug}_hf_access.stdout" \
    2>"${OUT_DIR}/preflight_${slug}_hf_access.stderr"
  local rc=$?
  log "PREFLIGHT_MODEL_ACCESS_RC ${slug} ${rc}"
  return "${rc}"
}

wait_ready() {
  local name="$1"
  local attempts=$((READY_TIMEOUT_S / 5))
  if (( attempts < 1 )); then attempts=1; fi
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
    if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${name}"; then return 3; fi
    sleep 5
  done
  return 1
}

capture_logs() {
  local name="$1" row_dir="$2"
  docker logs "${name}" >"${row_dir}/server.log" 2>&1 || true
}

run_ppl_cell() {
  local name="$1" model="$2" served="$3" row_label="$4" kv_label="$5" cell="$6" corpus="$7" prefix_len="$8"
  docker exec "${name}" python3 /work/scripts/sglang_prompt_ppl_sweep.py \
    --url "http://127.0.0.1:${PORT}" \
    --tokenizer "${model}" \
    --text-file "/work/results/${RUN_ID}/docs/${corpus}" \
    --ctx "${CTX}" \
    --run-id "${RUN_ID}_${row_label}_${cell}_ctx${CTX}" \
    --kv-cache-dtype "${kv_label}" \
    --runtime-ref "SGLang ${IMAGE}; overnight ladder row=${row_label}; graphs disabled; full NVFP4 rows use K+V fp4_e2m1" \
    --container-image "${IMAGE}" \
    --scope "SGLang overnight Gemma ladder supplied-token PPL; cell=${cell}; zero-bug deterministic gates" \
    --max-new-tokens 1 \
    --reuse-prefix-len "${prefix_len}" \
    --logprob-start-len "${prefix_len}" \
    --timeout "${PPL_TIMEOUT_S}" \
    --output "/work/results/${RUN_ID}/rows/${row_label}/${cell}_ppl.json" \
    >"${OUT_DIR}/rows/${row_label}/${cell}_ppl_stdout.json" \
    2>"${OUT_DIR}/rows/${row_label}/${cell}_ppl_stderr.log"
}

json_mean_nll() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
    print(repr(data["contexts"][0]["score"]["mean_nll_nats"]))
except Exception:
    print("")
PY
}

run_row() {
  local slug="$1" model="$2" served="$3" memfrac="$4" row_kind="$5" kv_dtype="$6"
  local row_label="${slug}_${row_kind}"
  local row_dir="${OUT_DIR}/rows/${row_label}"
  local container="codex_lad_${row_label}"
  local cache_dir="${FLASHINFER_CACHE_BASE}/flashinfer-cache-${RUN_ID}-${row_label}"
  local kv_args=()
  local env_args=(
    -e TORCH_CUDA_ARCH_LIST=12.1a
    -e FLASHINFER_CACHE_DIR="${cache_dir}"
    -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a"
    -e FLASHINFER_PREFILL_DEBUG_ONCE=1
    -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src
    -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1
    -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path
    -e SGLANG_FLASHINFER_VOSPLIT=1
    -e SGLANG_GEMMA4_TRACE_GEOMETRY=1
    -e SGLANG_GEMMA_KV_GEOMETRY=1
    -e SGLANG_FP4_KV_MIXED_KV=0
    -e HF_TOKEN="${HF_TOKEN:-}"
  )
  local prefix_len=4096
  mkdir -p "${row_dir}"

  if [[ "${kv_dtype}" != "auto" ]]; then
    kv_args+=(--kv-cache-dtype "${kv_dtype}")
  fi
  if [[ "${model}" == *"E2B"* ]]; then
    prefix_len=2048
  fi

  log "ROW_START ${row_label} model=${model} kv=${kv_dtype} mem_fraction=${memfrac}"
  docker rm -f "${container}" >/dev/null 2>&1 || true
  rm -rf "${cache_dir}"

  docker run -d --name "${container}" --gpus all --ipc=host --network=host \
    --memory "${GB10_DOCKER_MEMORY}" --memory-swap "${GB10_DOCKER_MEMORY_SWAP}" \
    -w /work \
    -v "${REPO_ROOT}:/work" \
    -v "${FLASHINFER_SRC}:/flashinfer-src" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${env_args[@]}" \
    "${IMAGE}" \
    bash -lc "
      set -euo pipefail
      rm -rf /root/.cache/flashinfer '${cache_dir}'
      mkdir -p /tmp/flashinfer-python-path
      ln -sfn /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
      python3 - <<'PY'
import hashlib
import importlib
import importlib.metadata as md
import pathlib
import flashinfer
from flashinfer.jit import env as jit_env
print('python_flashinfer', getattr(flashinfer, '__file__', None), flush=True)
for dist in ('sglang', 'sglang-kernel', 'flashinfer_python'):
    try:
        print('dist_version', dist, md.version(dist), flush=True)
    except Exception as exc:
        print('dist_version_error', dist, repr(exc), flush=True)
for name in ('sgl_kernel', 'sgl_kernel.common_ops'):
    try:
        mod = importlib.import_module(name)
        path = pathlib.Path(getattr(mod, '__file__', '')).resolve()
        if path.is_file():
            print('binary_md5', name, path, hashlib.md5(path.read_bytes()).hexdigest(), flush=True)
    except Exception as exc:
        print('binary_probe_error', name, repr(exc), flush=True)
print('flashinfer_data', jit_env.FLASHINFER_DATA, flush=True)
print('flashinfer_csrc', jit_env.FLASHINFER_CSRC_DIR, flush=True)
print('flashinfer_include', jit_env.FLASHINFER_INCLUDE_DIR, flush=True)
print('flashinfer_cutlass', jit_env.CUTLASS_INCLUDE_DIRS, flush=True)
print('flashinfer_cccl', jit_env.CCCL_INCLUDE_DIRS, flush=True)
print('flashinfer_spdlog', jit_env.SPDLOG_INCLUDE_DIR, flush=True)
PY
      exec python3 -m sglang.launch_server \
        --model-path '${model}' \
        --served-model-name '${served}' \
        --dtype bfloat16 \
        --attention-backend flashinfer \
        --page-size 1 \
        --mem-fraction-static '${memfrac}' \
        --disable-cuda-graph \
        --disable-piecewise-cuda-graph \
        --host 0.0.0.0 \
        --port '${PORT}' \
        ${kv_args[*]}
    " >/dev/null

  local ready_rc
  wait_ready "${container}"
  ready_rc=$?
  echo "${ready_rc}" >"${row_dir}/ready_rc.txt"
  capture_logs "${container}" "${row_dir}"
  grep -n "binary_md5\|flashinfer_csrc\|flashinfer_include\|SGLang Gemma4 FlashInfer geometry\|SGLang FlashInfer wrapper geometries\|vo_split\|head_dim_vo\|KV Cache is allocated\|max_total_num_tokens\|SWAKVPool mem usage\|kv_cache_dtype\|fp4\|fp8\|Traceback\|ERROR\|Unsupported max_mma_kv\|FLASHINFER_PAGED_V_SF_DESWIZZLE" \
    "${row_dir}/server.log" >"${row_dir}/proof_lines.txt" 2>&1 || true

  if [[ "${ready_rc}" != "0" ]]; then
    log "ROW_RED ${row_label} readiness_rc=${ready_rc}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    cat >"${row_dir}/manifest.json" <<EOF
{"schema":"sglang-overnight-row/v1","row":"${row_label}","model":"${model}","kv_cache_dtype":"${kv_dtype}","ok":false,"red_reason":"server_not_ready","ready_rc":${ready_rc}}
EOF
    return 0
  fi

  docker exec "${container}" python3 /work/scripts/openai_chat_smoke.py \
    --url "http://127.0.0.1:${PORT}" \
    --model "${served}" \
    --timeout "${REQUEST_TIMEOUT_S}" \
    --max-tokens 12 \
    --output "/work/results/${RUN_ID}/rows/${row_label}/smoke_sparkok.json" \
    >"${row_dir}/smoke_sparkok.stdout" \
    2>"${row_dir}/smoke_sparkok.stderr"
  local spark_rc=$?

  docker exec "${container}" python3 /work/scripts/openai_chat_smoke.py \
    --url "http://127.0.0.1:${PORT}" \
    --model "${served}" \
    --timeout "${REQUEST_TIMEOUT_S}" \
    --prompt "In one short sentence, name the capital of Japan." \
    --max-tokens 24 \
    --output "/work/results/${RUN_ID}/rows/${row_label}/smoke_tokyo.json" \
    >"${row_dir}/smoke_tokyo.stdout" \
    2>"${row_dir}/smoke_tokyo.stderr"
  local tokyo_rc=$?
  if grep -qi "tokyo" "${row_dir}/smoke_tokyo.json"; then
    echo "tokyo=true" >"${row_dir}/smoke_tokyo_gate.txt"
  else
    echo "tokyo=false" >"${row_dir}/smoke_tokyo_gate.txt"
  fi

  run_ppl_cell "${container}" "${model}" "${served}" "${row_label}" "${kv_dtype}" c1a c1_ppl_corpus.md "${prefix_len}"
  local c1a_rc=$?
  run_ppl_cell "${container}" "${model}" "${served}" "${row_label}" "${kv_dtype}" c1b c1_ppl_corpus.md "${prefix_len}"
  local c1b_rc=$?
  local c1a_mean c1b_mean c1_det
  c1a_mean="$(json_mean_nll "${row_dir}/c1a_ppl.json")"
  c1b_mean="$(json_mean_nll "${row_dir}/c1b_ppl.json")"
  if [[ -n "${c1a_mean}" && "${c1a_mean}" == "${c1b_mean}" ]]; then
    c1_det=true
  else
    c1_det=false
  fi
  local c1_det_py
  if [[ "${c1_det}" == "true" ]]; then
    c1_det_py=True
  else
    c1_det_py=False
  fi

  run_ppl_cell "${container}" "${model}" "${served}" "${row_label}" "${kv_dtype}" c2 c2_pride_prejudice_60k.txt "${prefix_len}"
  local c2_rc=$?
  run_ppl_cell "${container}" "${model}" "${served}" "${row_label}" "${kv_dtype}" c3 c3_hijinks_code_60k.py "${prefix_len}"
  local c3_rc=$?

  capture_logs "${container}" "${row_dir}"
  docker rm -f "${container}" >/dev/null 2>&1 || true

  python3 - "${row_dir}/manifest.json" <<PY
import json
from pathlib import Path

row_dir = Path("${row_dir}")
tokyo = (row_dir / "smoke_tokyo_gate.txt").read_text().strip().endswith("true")
manifest = {
    "schema": "sglang-overnight-row/v1",
    "run_id": "${RUN_ID}",
    "row": "${row_label}",
    "model": "${model}",
    "served_model": "${served}",
    "kv_cache_dtype": "${kv_dtype}",
    "image": "${IMAGE}",
    "mem_fraction_static": float("${memfrac}"),
    "ctx": int("${CTX}"),
    "reuse_prefix_len": int("${prefix_len}"),
    "ready_rc": int("${ready_rc}"),
    "spark_smoke_rc": int("${spark_rc}"),
    "tokyo_smoke_rc": int("${tokyo_rc}"),
    "tokyo_contains_tokyo": tokyo,
    "c1a_rc": int("${c1a_rc}"),
    "c1b_rc": int("${c1b_rc}"),
    "c2_rc": int("${c2_rc}"),
    "c3_rc": int("${c3_rc}"),
    "c1a_mean_nll_repr": "${c1a_mean}",
    "c1b_mean_nll_repr": "${c1b_mean}",
    "c1_deterministic": ${c1_det_py},
}
manifest["ok"] = (
    manifest["ready_rc"] == 0
    and manifest["spark_smoke_rc"] == 0
    and manifest["tokyo_contains_tokyo"]
    and manifest["c1a_rc"] == 0
    and manifest["c1b_rc"] == 0
    and manifest["c2_rc"] == 0
    and manifest["c3_rc"] == 0
    and manifest["c1_deterministic"]
)
if not manifest["ok"]:
    reasons = []
    for key in ("ready_rc", "spark_smoke_rc", "c1a_rc", "c1b_rc", "c2_rc", "c3_rc"):
        if manifest[key] != 0:
            reasons.append(f"{key}={manifest[key]}")
    if not manifest["tokyo_contains_tokyo"]:
        reasons.append("tokyo_smoke_not_coherent")
    if not manifest["c1_deterministic"]:
        reasons.append("c1_determinism_fail")
    manifest["red_reasons"] = reasons
Path("${row_dir}/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

  if grep -q '"ok": true' "${row_dir}/manifest.json"; then
    log "ROW_GREEN ${row_label} c1=${c1a_mean}"
  else
    log "ROW_RED ${row_label} manifest=${row_dir}/manifest.json"
  fi
}

compare_model_rows() {
  local slug="$1"
  python3 - "${OUT_DIR}" "${slug}" <<'PY'
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
slug = sys.argv[2]
rows = {}
for kind in ("bf16", "nvfp4", "fp8"):
    path = out / "rows" / f"{slug}_{kind}" / "manifest.json"
    if path.exists():
        rows[kind] = json.loads(path.read_text())
summary = {"schema": "sglang-overnight-model-summary/v1", "model_slug": slug, "rows": rows, "delta_gates": []}
bf16 = rows.get("bf16")
if bf16 and bf16.get("ok") and bf16.get("c1a_mean_nll_repr"):
    base = float(bf16["c1a_mean_nll_repr"])
    for kind in ("nvfp4", "fp8"):
        row = rows.get(kind)
        if not row or not row.get("ok") or not row.get("c1a_mean_nll_repr"):
            continue
        delta = float(row["c1a_mean_nll_repr"]) - base
        gate = {"kind": kind, "delta_nats_vs_bf16_c1": delta, "ok": abs(delta) <= 0.5}
        summary["delta_gates"].append(gate)
        if not gate["ok"]:
            row["ok"] = False
            row.setdefault("red_reasons", []).append(f"delta_vs_bf16_gt_0.5:{delta}")
            (out / "rows" / f"{slug}_{kind}" / "manifest.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
summary["ok"] = bool(rows) and all(r.get("ok") for r in rows.values()) and all(g["ok"] for g in summary["delta_gates"])
(out / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
}

write_summary() {
  python3 - "${OUT_DIR}" <<'PY'
import json
from pathlib import Path
import time

out = Path(__import__("sys").argv[1])
rows = []
for manifest in sorted((out / "rows").glob("*/manifest.json")):
    rows.append(json.loads(manifest.read_text()))
green = [r for r in rows if r.get("ok")]
red = [r for r in rows if not r.get("ok")]
lines = [
    "# SGLang Overnight Ladder",
    "",
    f"Finished: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
    "",
    f"Rows attempted: `{len(rows)}`",
    f"Green rows: `{len(green)}`",
    f"Red rows: `{len(red)}`",
    "",
    "| Row | Model | KV | Verdict | Notes |",
    "|---|---|---|---|---|",
]
for row in rows:
    notes = ", ".join(row.get("red_reasons", [])) if not row.get("ok") else f"C1 {row.get('c1a_mean_nll_repr')}"
    lines.append(
        f"| `{row.get('row')}` | `{row.get('model')}` | `{row.get('kv_cache_dtype')}` | "
        f"{'GREEN' if row.get('ok') else 'RED'} | {notes} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

main() {
  claim_window
  prepare_repo

  for entry in "${MODEL_MATRIX[@]}"; do
    IFS='|' read -r slug model served memfrac <<<"${entry}"
    run_access_probe "${slug}" "${model}"
    access_rc=$?
    if [[ "${access_rc}" != "0" ]]; then
      log "MODEL_RED ${slug} hf_access_rc=${access_rc}; skipping server rows"
      continue
    fi
    for row in "${ROW_MATRIX[@]}"; do
      IFS='|' read -r row_kind kv_dtype _mixed <<<"${row}"
      run_row "${slug}" "${model}" "${served}" "${memfrac}" "${row_kind}" "${kv_dtype}"
    done
    compare_model_rows "${slug}"
  done

  write_summary
  docker ps >"${OUT_DIR}/docker_ps_after.txt" 2>&1 || true
  free -h >"${OUT_DIR}/free_after.txt" 2>&1 || true
  write_stop_mail "ladder_attempted"
  log "RUN_DONE ${OUT_DIR}"
}

main "$@"
