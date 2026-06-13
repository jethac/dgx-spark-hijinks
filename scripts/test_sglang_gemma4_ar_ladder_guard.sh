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
  echo fake-container
  exit 0
fi
if [[ "${1:-}" == "ps" ]]; then
  echo "CONTAINER ID   IMAGE   NAMES"
  echo "fake           fake    fake-container"
  exit 0
fi
echo "unexpected docker invocation: $*" >&2
exit 97
SH
chmod +x "${TMP}/bin/docker"

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
