#!/usr/bin/env bash
# Regression tests for the SGLang Gemma 4 AR blocker-audit state machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

KNOWN_FLASHINFER_REF="3fa0775cafaf88da5e0fc3b987afa6bd75d9510c"
KNOWN_SGLANG_REF="f920e2d88af68031b745494f5435efb71ac93562"
CHANGED_REF="ffffffffffffffffffffffffffffffffffffffff"

python3 scripts/sglang_gemma4_ar_ladder_blocker_audit.py \
  --repo-root "${TMP}" \
  --flashinfer-ref "${KNOWN_FLASHINFER_REF}" \
  --sglang-ref "${KNOWN_SGLANG_REF}" \
  --output "${TMP}/known.json"

python3 - "${TMP}/known.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["ladder_status"] == "blocked-known-red-dependencies", payload
assert payload["can_run_claim_ladder"] is False, payload
assert payload["diagnostic_override_allowed"] is False, payload
assert payload["diagnostic_override_reason"] == "", payload
assert all(not item["dependency_changed"] for item in payload["dependencies"]), payload
assert "Do not rerun" in payload["next_action"], payload["next_action"]
print("PASS known_blocked_without_override")
PY

python3 scripts/sglang_gemma4_ar_ladder_blocker_audit.py \
  --repo-root "${TMP}" \
  --flashinfer-ref "${KNOWN_FLASHINFER_REF}" \
  --sglang-ref "${KNOWN_SGLANG_REF}" \
  --diagnostic-override-reason "mail 0140 scoped chunked diagnostic" \
  --output "${TMP}/known_diag.json"

python3 - "${TMP}/known_diag.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["ladder_status"] == "blocked-known-red-dependencies", payload
assert payload["can_run_claim_ladder"] is False, payload
assert payload["diagnostic_override_allowed"] is True, payload
assert payload["diagnostic_override_reason"] == "mail 0140 scoped chunked diagnostic", payload
assert all(not item["dependency_changed"] for item in payload["dependencies"]), payload
assert "diagnostic-only" in payload["next_action"], payload["next_action"]
print("PASS known_blocked_with_diagnostic_override")
PY

python3 scripts/sglang_gemma4_ar_ladder_blocker_audit.py \
  --repo-root "${TMP}" \
  --flashinfer-ref "${CHANGED_REF}" \
  --sglang-ref "${KNOWN_SGLANG_REF}" \
  --output "${TMP}/changed.json"

python3 - "${TMP}/changed.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["ladder_status"] == "dependency-changed-review-before-rerun", payload
assert payload["can_run_claim_ladder"] is False, payload
assert payload["diagnostic_override_allowed"] is True, payload
assert payload["diagnostic_override_reason"] == "", payload
deps = {item["name"]: item for item in payload["dependencies"]}
assert deps["flashinfer"]["dependency_changed"] is True, payload
assert deps["sglang"]["dependency_changed"] is False, payload
assert "Review the dependency delta" in payload["next_action"], payload["next_action"]
print("PASS dependency_changed_review_state")
PY
