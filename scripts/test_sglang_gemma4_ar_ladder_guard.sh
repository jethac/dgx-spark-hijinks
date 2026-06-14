#!/usr/bin/env bash
# Offline regression checks for the SGLang Gemma 4 AR ladder safety guards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT}/scripts/run_sglang_gemma4_ar_ladder_pair.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

mkdir -p "${TMP}/bin"
cat >"${TMP}/bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "ps" && "${2:-}" == "-q" ]]; then
  if [[ "${FAKE_DOCKER_EMPTY:-0}" == "1" ]]; then
    exit 0
  fi
  echo fake-container
  exit 0
fi
if [[ "${1:-}" == "ps" && "${2:-}" == "--format" ]]; then
  if [[ "${FAKE_DOCKER_EMPTY:-0}" == "1" ]]; then
    exit 0
  fi
  echo fake-container
  exit 0
fi
if [[ "${1:-}" == "ps" ]]; then
  echo "CONTAINER ID   IMAGE   NAMES"
  if [[ "${FAKE_DOCKER_EMPTY:-0}" != "1" ]]; then
    echo "fake           fake    fake-container"
  fi
  exit 0
fi
if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
  echo "[]"
  exit 0
fi
if [[ "${1:-}" == "rm" ]]; then
  exit 0
fi
if [[ "${1:-}" == "run" ]]; then
  echo fake-container-id
  exit 0
fi
if [[ "${1:-}" == "logs" ]]; then
  if [[ "${FAKE_DOCKER_E4B_FP8_RED:-0}" == "1" ]]; then
    cat <<'LOG'
SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5
planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256)
[flashinfer][prefill-debug] call_id=1 path=paged compiled={dtype_kv=__nv_fp8_e4m3,head_dim_qk=512,head_dim_vo=256} runtime={page_size=1,split_kv=1,cta_tile_q=64}
tvm.error.InternalError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
FlashInfer Internal Error: Invalid configuration :
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
LOG
  else
    echo "fake server log"
  fi
  exit 0
fi
if [[ "${1:-}" == "inspect" ]]; then
  echo "{}"
  exit 0
fi
echo "unexpected docker invocation: $*" >&2
exit 97
SH
chmod +x "${TMP}/bin/docker"
cat >"${TMP}/bin/git" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "ls-remote" && "${2:-}" == "origin" && "${3:-}" == "spark/hijinks-022-fa2-d512" ]]; then
  echo "3fa0775cafaf88da5e0fc3b987afa6bd75d9510c	refs/heads/spark/hijinks-022-fa2-d512"
  exit 0
fi
if [[ "${1:-}" == "ls-remote" && "${2:-}" == "origin" && "${3:-}" == "spark/hijinks-025-sglang-0.5.13-rebase" ]]; then
  echo "f920e2d88af68031b745494f5435efb71ac93562	refs/heads/spark/hijinks-025-sglang-0.5.13-rebase"
  exit 0
fi
if [[ "${1:-}" == "rev-parse" && "${2:-}" == "HEAD" ]]; then
  echo test-head
  exit 0
fi
echo "unexpected git invocation: $*" >&2
exit 98
SH
chmod +x "${TMP}/bin/git"

run_case() {
  local name="$1"
  local expected_status="$2"
  local expected_pattern="$3"
  shift 3

  local out="${TMP}/${name}.out"
  set +e
  PATH="${TMP}/bin:${PATH}" \
    REPO_ROOT="${ROOT}" \
    CLAUDE_MARKER="${TMP}/no_marker" \
    "$@" >"${out}" 2>&1
  local status=$?
  set -e

  if [[ "${status}" != "${expected_status}" ]]; then
    echo "FAIL ${name}: expected status ${expected_status}, got ${status}" >&2
    cat "${out}" >&2
    exit 1
  fi
  if ! grep -Eq "${expected_pattern}" "${out}"; then
    echo "FAIL ${name}: missing pattern ${expected_pattern}" >&2
    cat "${out}" >&2
    exit 1
  fi
  echo "PASS ${name}"
}

run_case "fullnvfp4_block" 2 "SGLANG_AR_LADDER_OVERRIDE_REASON='flashinfer <ref>: shared quality fix'" \
  env MODELS="google/gemma-4-26B-A4B-it" ROW_LABELS="fullnvfp4" bash "${RUNNER}"

run_case "e4b_fp8_block" 2 "D512 fp8 dispatcher fix" \
  env MODELS="google/gemma-4-E4B-it" ROW_LABELS="fp8" bash "${RUNNER}"

run_case "override_requires_reason" 2 "SGLANG_AR_LADDER_OVERRIDE_REASON is empty" \
  env MODELS="google/gemma-4-26B-A4B-it" ROW_LABELS="fullnvfp4" \
    ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 bash "${RUNNER}"

run_case "override_with_reason_reaches_docker_gate" 99 "docker is not empty; yielding" \
  env MODELS="google/gemma-4-26B-A4B-it" ROW_LABELS="fullnvfp4" \
    ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
    SGLANG_AR_LADDER_OVERRIDE_REASON="test dependency change" bash "${RUNNER}"

OUT_DIR_CAPTURE="${TMP}/audit_capture"
mkdir -p "${OUT_DIR_CAPTURE}"
printf 'short corpus\n' >"${TMP}/corpus.md"
run_case "override_writes_blocker_audit_before_serving" 1 '"blocker_audit":' \
  env MODELS="google/gemma-4-26B-A4B-it" ROW_LABELS="fullnvfp4" \
    ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
    SGLANG_AR_LADDER_OVERRIDE_REASON="test dependency change" \
    FAKE_DOCKER_EMPTY=1 \
    OUT_DIR="${OUT_DIR_CAPTURE}" \
    RUN_ID="test_audit_capture" \
    CORPUS="${TMP}/corpus.md" \
    READY_TIMEOUT_S=1 \
    bash "${RUNNER}"

python3 - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR_CAPTURE}")
audit = json.loads((out / "blocker_audit.json").read_text(encoding="utf-8"))
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
claim_audit = json.loads((out / "claim_audit.json").read_text(encoding="utf-8"))
assert audit["ladder_status"] == "blocked-known-red-dependencies"
assert manifest["blocker_audit"].endswith("/blocker_audit.json")
assert manifest["claim_audit"].endswith("/claim_audit.json")
assert claim_audit["ok"] is False
assert "google/gemma-4-12B-it: missing model row" in claim_audit["findings"]
PY
echo "PASS blocker_and_claim_audit_artifacts"

OUT_DIR_STRICT="${TMP}/claim_audit_strict_capture"
mkdir -p "${OUT_DIR_STRICT}"
run_case "strict_claim_audit_red_fails" 1 "SGLang Gemma 4 AR claim audit is not green" \
  env MODELS="google/gemma-4-26B-A4B-it" ROW_LABELS="fullnvfp4" \
    ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
    SGLANG_AR_LADDER_OVERRIDE_REASON="test dependency change" \
    SGLANG_AR_CLAIM_AUDIT_STRICT=1 \
    FAKE_DOCKER_EMPTY=1 \
    OUT_DIR="${OUT_DIR_STRICT}" \
    RUN_ID="test_claim_audit_strict_capture" \
    CORPUS="${TMP}/corpus.md" \
    READY_TIMEOUT_S=1 \
    bash "${RUNNER}"

python3 - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR_STRICT}")
claim_audit = json.loads((out / "claim_audit.json").read_text(encoding="utf-8"))
status = (out / "claim_audit_status.txt").read_text(encoding="utf-8").strip()
assert status == "1"
assert claim_audit["ok"] is False
assert "google/gemma-4-12B-it: missing model row" in claim_audit["findings"]
PY
echo "PASS strict_claim_audit_artifacts"

OUT_DIR_E4B="${TMP}/e4b_fp8_dispatch_capture"
mkdir -p "${OUT_DIR_E4B}"
run_case "e4b_fp8_override_writes_dispatch_audit" 1 '"model": "google/gemma-4-E4B-it"' \
  env MODELS="google/gemma-4-E4B-it" ROW_LABELS="fp8" \
    ALLOW_KNOWN_BLOCKED_SGLANG_AR_LADDER=1 \
    SGLANG_AR_LADDER_OVERRIDE_REASON="test dispatcher fix replay" \
    FAKE_DOCKER_EMPTY=1 \
    FAKE_DOCKER_E4B_FP8_RED=1 \
    OUT_DIR="${OUT_DIR_E4B}" \
    RUN_ID="test_e4b_fp8_dispatch_capture" \
    CORPUS="${TMP}/corpus.md" \
    READY_TIMEOUT_S=1 \
    bash "${RUNNER}"

python3 - <<PY
import json
from pathlib import Path
path = Path("${OUT_DIR_E4B}") / "google-gemma-4-e4b-it" / "fp8_dispatch_audit.json"
audit = json.loads(path.read_text(encoding="utf-8"))
assert audit["known_red"] is True
assert audit["signals"]["matching_trait_count"] >= 1
assert audit["signals"]["matching_geometry_count"] >= 1
PY
echo "PASS e4b_fp8_dispatch_audit_artifact"
