#!/usr/bin/env bash
# DG-R6 Spark packet: DiffusionGemma 26B-A4B stock-vs-GB10 tuned performance pair.
#
# Scope: text-only serving performance row. This runs two servers sequentially:
#   before: stock DiffusionGemma policy, Triton attention, BF16/auto KV
#   after: FlashInfer VO-split opt-in, full NVFP4 K+V
# It does not claim image quality, CUDA graph safety, or long-context quality.
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
SGLANG_COMMIT="${SGLANG_COMMIT:-98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-f99323bd7d1c}"
PAIR_RUN_ID="${PAIR_RUN_ID:-sglang_dgemma_dgr6_perf_pair_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/${PAIR_RUN_ID}}"
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
  echo "pair_run_id=${PAIR_RUN_ID}"
  echo "repo_root=${REPO_ROOT}"
  echo "source_branch=${SOURCE_BRANCH}"
  echo "image=${IMAGE}"
  echo "model=${MODEL}"
  echo "port=${PORT}"
  echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
  echo "sglang_commit=${SGLANG_COMMIT}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "scope=DiffusionGemma DG-R6 stock-vs-full-NVFP4 performance pair"
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

write_dllm_config() {
  local path="$1"
  cat >"${path}" <<'EOF'
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
}

capture_docker_logs() {
  local name="$1"
  local server_log="$2"
  local tmp_log="${server_log}.tmp"
  if docker logs "${name}" >"${tmp_log}" 2>&1; then
    mv "${tmp_log}" "${server_log}"
  else
    rm -f "${tmp_log}"
  fi
}

run_one() {
  local mode="$1"
  local phase="$2"
  local mode_dir="${OUT_DIR}/${mode}"
  local run_id="${PAIR_RUN_ID}_${mode}"
  local container="${run_id}"
  local cid_file="${mode_dir}/container_id.txt"
  local server_log="${mode_dir}/server.log"
  local quality_json="${mode_dir}/revised_text_quality.json"
  local benchmark_json="${mode_dir}/openai_benchmark.json"
  local dllm_config="${mode_dir}/dllm_config.yaml"
  local flashinfer_cache_dir="/tmp/flashinfer-cache-${run_id}"

  mkdir -p "${mode_dir}"
  write_dllm_config "${dllm_config}"

  local description launch_args env_args
  case "${mode}" in
    before_stock_triton_bf16)
      description="stock DiffusionGemma policy: Triton attention, BF16/auto KV"
      launch_args='--context-length 8192'
      env_args=''
      ;;
    after_flashinfer_fullnvfp4)
      description="GB10 tuned path: FlashInfer VO-split, full NVFP4 K+V"
      launch_args='--attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --context-length 8192 --page-size 256'
      env_args='-e SGLANG_FLASHINFER_VOSPLIT=1 -e SGLANG_FP4_KV_MIXED_KV=0 -e SGLANG_FP4_KV_TRACE_MODULE=1 -e SGLANG_GEMMA4_TRACE_GEOMETRY=1 -e SGLANG_GEMMA_KV_GEOMETRY=1'
      ;;
    *)
      echo "unknown mode ${mode}" >&2
      return 2
      ;;
  esac

  {
    echo "run_id=${run_id}"
    echo "mode=${mode}"
    echo "phase=${phase}"
    echo "description=${description}"
    echo "started_at=$(TZ=Asia/Tokyo date -Is)"
    echo "launch_args=${launch_args}"
    free -h
  } | tee "${mode_dir}/preflight.log"

  docker rm -f "${container}" >/dev/null 2>&1 || true
  local cid
  # shellcheck disable=SC2086
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
      -e FLASHINFER_CACHE_DIR="${flashinfer_cache_dir}" \
      -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
      -e FLASHINFER_PREFILL_DEBUG_ONCE=1 \
      -e LAUNCH_ARGS="${launch_args}" \
      -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
      -e SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 \
      -e PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path \
      -e TRANSFORMERS_OFFLINE=1 \
      -e HF_HUB_OFFLINE=1 \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      ${env_args} \
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

        # shellcheck disable=SC2086
        exec python3 -m sglang.launch_server \
          --model-path "${MODEL}" \
          --dllm-algorithm Gemma4Renoise \
          --dllm-algorithm-config "/work/results/'"${PAIR_RUN_ID}"'/'"${mode}"'/dllm_config.yaml" \
          --trust-remote-code \
          --dtype bfloat16 \
          ${LAUNCH_ARGS} \
          --mem-fraction-static "${MEM_FRACTION_STATIC}" \
          --disable-cuda-graph \
          --disable-piecewise-cuda-graph \
          --host 0.0.0.0 \
          --port "${PORT}"
      '
  )
  echo "${cid}" >"${cid_file}"

  local ready=0
  local attempts=$((READY_TIMEOUT_S / 5))
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

  capture_docker_logs "${container}" "${server_log}" || true
  if [[ "${ready}" != "1" ]]; then
    echo "RED: ${mode} server did not reach readiness" | tee "${mode_dir}/status.txt"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    docker ps >"${mode_dir}/docker_ps_after.txt" 2>&1 || true
    free -h >"${mode_dir}/free_after.txt" 2>&1 || true
    return 1
  fi

  set +e
  python3 scripts/diffusion_gemma_dgr2_revised_text_quality_client.py \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --out "${quality_json}" \
    --repeats 2 \
    --ready-timeout-s 30 \
    --request-timeout-s "${REQUEST_TIMEOUT_S}" \
    >"${mode_dir}/quality_client.stdout" \
    2>"${mode_dir}/quality_client.stderr"
  local quality_status=$?

  python3 scripts/openai_serving_benchmark.py \
    --url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --backend sglang \
    --phase "${phase}" \
    --run-id "${run_id}" \
    --case short_decode \
    --case medium_decode \
    --case long_prefill \
    --case natural_long_prefill \
    --timeout "${REQUEST_TIMEOUT_S}" \
    --output "${benchmark_json}" \
    >"${mode_dir}/openai_benchmark.stdout" \
    2>"${mode_dir}/openai_benchmark.stderr"
  local benchmark_status=$?
  set -e

  echo "${quality_status}" >"${mode_dir}/quality_status.txt"
  echo "${benchmark_status}" >"${mode_dir}/benchmark_status.txt"
  capture_docker_logs "${container}" "${server_log}" || true
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker ps >"${mode_dir}/docker_ps_after.txt" 2>&1 || true
  free -h >"${mode_dir}/free_after.txt" 2>&1 || true

  if [[ "${quality_status}" != "0" || "${benchmark_status}" != "0" ]]; then
    echo "RED: ${mode} quality_status=${quality_status} benchmark_status=${benchmark_status}" | tee "${mode_dir}/status.txt"
    return 1
  fi
  echo "GREEN: ${mode}" | tee "${mode_dir}/status.txt"
}

status=0
run_one before_stock_triton_bf16 before || status=$?
if [[ "${status}" == "0" ]]; then
  run_one after_flashinfer_fullnvfp4 after || status=$?
fi

python3 - "${OUT_DIR}" "${PAIR_RUN_ID}" "${MODEL}" "${IMAGE}" "${SGLANG_COMMIT}" "${FLASHINFER_COMMIT}" "${status}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
pair_run_id, model, image, sglang_commit, flashinfer_commit, status_arg = sys.argv[2:9]
summary_md = out_dir / "summary.md"

modes = [
    ("before_stock_triton_bf16", "before", "stock Triton, BF16/auto KV"),
    ("after_flashinfer_fullnvfp4", "after", "FlashInfer VO-split, full NVFP4 K+V"),
]

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None

rows = []
all_ok = status_arg == "0"
for mode, phase, label in modes:
    mode_dir = out_dir / mode
    log = (mode_dir / "server.log").read_text(encoding="utf-8", errors="replace") if (mode_dir / "server.log").exists() else ""
    quality = load_json(mode_dir / "revised_text_quality.json")
    bench = load_json(mode_dir / "openai_benchmark.json")
    quality_ok = bool(quality.get("all_ok"))
    bench_ok = bool(bench.get("ok"))
    status_text = (mode_dir / "status.txt").read_text(encoding="utf-8", errors="replace").strip() if (mode_dir / "status.txt").exists() else "missing"
    full_tokens = first_match(r"max_total_num_tokens=(\d+)", log)
    swa_tokens = first_match(r"SGLANG_GEMMA_KV_SWAKVPOOL.*?swa_tokens=(\d+)", log)
    policy_stock = "Attention backend forced to triton for DiffusionGemma" in log
    policy_vosplit = "DiffusionGemma is using the experimental FlashInfer VO-split path" in log
    full_nvfp4 = "mixed_kv=False" in log and "full_pool=MHATokenToKVPoolFP4" in log and "swa_pool=MHATokenToKVPoolFP4" in log
    case_rows = []
    for case in bench.get("cases", []):
        usage = case.get("usage") or {}
        case_rows.append(
            {
                "case": case.get("case"),
                "ok": case.get("ok"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "ttft_s": case.get("ttft_s"),
                "total_s": case.get("total_s"),
                "decode_tok_s": case.get("decode_tok_s"),
                "completion_tok_s_total": case.get("completion_tok_s_total"),
            }
        )
    rows.append(
        {
            "mode": mode,
            "phase": phase,
            "label": label,
            "quality_ok": quality_ok,
            "bench_ok": bench_ok,
            "status_text": status_text,
            "full_tokens": int(full_tokens) if full_tokens else None,
            "swa_tokens": int(swa_tokens) if swa_tokens else None,
            "policy_stock": policy_stock,
            "policy_vosplit": policy_vosplit,
            "full_nvfp4": full_nvfp4,
            "cases": case_rows,
        }
    )
    all_ok = all_ok and quality_ok and bench_ok

before = rows[0]
after = rows[1] if len(rows) > 1 else None
status = "GREEN" if all_ok and after else "RED"

def fmt(v, digits=3):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{digits}f}"
    return str(v)

lines = [
    "# SGLang DiffusionGemma DG-R6 Performance Pair",
    "",
    f"Status: {status}",
    "",
    "## Scope",
    "",
    "Text-only serving performance comparison on GB10. The before row is the stock DiffusionGemma policy path: Triton attention with BF16/auto KV. The after row is the GB10-tuned path: FlashInfer VO-split with full NVFP4 K+V. Quality and capacity claims remain separate from this speed row.",
    "",
    "## Provenance",
    "",
    f"- Pair run: `{pair_run_id}`",
    f"- Model: `{model}`",
    f"- Image: `{image}`",
    f"- SGLang: `{sglang_commit}`",
    f"- FlashInfer: `{flashinfer_commit}`",
    "- Common launch: `--dllm-algorithm Gemma4Renoise --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.55 --disable-cuda-graph --disable-piecewise-cuda-graph`",
    "- Prompt set: `short_decode`, `medium_decode`, `long_prefill`, `natural_long_prefill` from `scripts/openai_serving_benchmark.py`",
    "",
    "## Gates",
    "",
]
for row in rows:
    lines += [
        f"- `{row['mode']}` quality gate: {'PASS' if row['quality_ok'] else 'FAIL'}",
        f"- `{row['mode']}` OpenAI benchmark: {'PASS' if row['bench_ok'] else 'FAIL'}",
    ]
if before:
    lines.append(f"- before stock Triton policy proof: {'PASS' if before['policy_stock'] else 'FAIL'}")
if after:
    lines.append(f"- after FlashInfer VO-split policy proof: {'PASS' if after['policy_vosplit'] else 'FAIL'}")
    lines.append(f"- after full-NVFP4 K+V pool proof: {'PASS' if after['full_nvfp4'] else 'FAIL'}")

lines += ["", "## Capacity Context", ""]
lines.append("| Row | Full-layer tokens | SWA tokens |")
lines.append("|---|---:|---:|")
for row in rows:
    lines.append(f"| {row['label']} | {fmt(row['full_tokens'], 0)} | {fmt(row['swa_tokens'], 0)} |")
if before and after and before.get("full_tokens") and after.get("full_tokens"):
    lines.append("")
    lines.append(f"Capacity ratio in this performance pair: `{after['full_tokens']} / {before['full_tokens']} = {after['full_tokens'] / before['full_tokens']:.4f}x` full-layer tokens.")

lines += ["", "## Throughput", ""]
lines.append("| Case | Before TTFT s | Before total s | Before total tok/s | After TTFT s | After total s | After total tok/s | After/Before total tok/s |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
before_cases = {case["case"]: case for case in before.get("cases", [])} if before else {}
after_cases = {case["case"]: case for case in after.get("cases", [])} if after else {}
for case_name in sorted(set(before_cases) | set(after_cases)):
    b = before_cases.get(case_name, {})
    a = after_cases.get(case_name, {})
    b_tps = b.get("completion_tok_s_total")
    a_tps = a.get("completion_tok_s_total")
    ratio = a_tps / b_tps if a_tps and b_tps else None
    lines.append(
        f"| `{case_name}` | {fmt(b.get('ttft_s'))} | {fmt(b.get('total_s'))} | {fmt(b_tps)} | "
        f"{fmt(a.get('ttft_s'))} | {fmt(a.get('total_s'))} | {fmt(a_tps)} | {fmt(ratio, 4)} |"
    )

lines += ["", "## Non-Claims", ""]
lines += [
    "- No image/multimodal quality claim.",
    "- No CUDA graph safety claim.",
    "- No long-context quality/PPL claim.",
    "- The speed comparison is a combined stack comparison, not an isolated kernel-only attribution.",
]

lines += ["", "## Artifacts", ""]
for row in rows:
    lines += [
        f"- `{row['mode']}/server.log`",
        f"- `{row['mode']}/revised_text_quality.json`",
        f"- `{row['mode']}/openai_benchmark.json`",
    ]

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": status, "rows": rows}, indent=2, sort_keys=True))
if status != "GREEN":
    raise SystemExit(1)
PY

docker ps >"${OUT_DIR}/docker_ps_after_pair.txt" 2>&1 || true
free -h >"${OUT_DIR}/free_after_pair.txt" 2>&1 || true
exit "${status}"
