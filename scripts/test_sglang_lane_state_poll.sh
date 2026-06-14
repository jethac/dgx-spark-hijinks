#!/usr/bin/env bash
# Regression tests for the SGLang lane coordination poll.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

KNOWN_FLASHINFER_REF="3fa0775cafaf88da5e0fc3b987afa6bd75d9510c"
KNOWN_SGLANG_REF="f920e2d88af68031b745494f5435efb71ac93562"

python3 scripts/sglang_lane_state_poll.py \
  --epoch2-only-mail \
  --flashinfer-ref "${KNOWN_FLASHINFER_REF}" \
  --sglang-ref "${KNOWN_SGLANG_REF}" \
  >"${TMP_DIR}/known.json"

python3 - "${TMP_DIR}/known.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["lane_status"] == "blocked-known-red-dependencies", payload
assert payload["mail"]["new_remote_mail"] is False, payload["mail"]
assert payload["mail"]["remote_refs_scanned"] == ["origin/epoch2"], payload["mail"]
assert all(not item["dependency_changed"] for item in payload["dependencies"]), payload
print("PASS known_blocked_state")
PY

python3 scripts/sglang_lane_state_poll.py \
  --epoch2-only-mail \
  --flashinfer-ref "ffffffffffffffffffffffffffffffffffffffff" \
  --sglang-ref "${KNOWN_SGLANG_REF}" \
  >"${TMP_DIR}/changed.json"

python3 - "${TMP_DIR}/changed.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["lane_status"] == "dependency-changed-review-before-rerun", payload
deps = {item["name"]: item for item in payload["dependencies"]}
assert deps["flashinfer"]["dependency_changed"] is True, payload
assert deps["sglang"]["dependency_changed"] is False, payload
print("PASS dependency_change_state")
PY
