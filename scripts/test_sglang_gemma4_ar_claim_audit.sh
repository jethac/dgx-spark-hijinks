#!/usr/bin/env bash
# Regression tests for the SGLang Gemma 4 AR ladder claim-readiness audit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

RED_MANIFEST="results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/manifest.json"

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${RED_MANIFEST}" \
  >"${TMP_DIR}/red_audit.json" 2>"${TMP_DIR}/red_audit.err"; then
  echo "FAIL known-red manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/red_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
expected = [
    "exceeds threshold",
    "google/gemma-4-26B-A4B-it: missing model row",
    "google/gemma-4-31B-it: missing model row",
]
missing = [item for item in expected if item not in findings]
if missing:
    raise SystemExit(f"missing expected findings: {missing}\n{findings}")
print("PASS known_red_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/blocker.json" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
blocker_path = pathlib.Path(sys.argv[2])
corpus_path = manifest_path.parent / "corpus.md"
corpus_manifest_path = manifest_path.parent / "corpus_manifest.json"
models = [
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]
rows = []
for model in models:
    row = {
        "model": model,
        "dir": model.lower().replace("/", "-").replace(".", "-"),
        "mixedkv": None,
        "compare_bf16_vs_mixedkv": None,
    }
    for label, dtype in [
        ("bf16", "auto"),
        ("fp8", "fp8_e4m3"),
        ("fullnvfp4", "fp4_e2m1"),
    ]:
        row[label] = {
            "model": model,
            "label": label,
            "kv_cache_dtype": dtype,
            "ppl_ok": True,
            "chat_transport_ok": True,
            "chat_content_equal": True,
        }
    for name, delta in [
        ("compare_bf16_vs_fullnvfp4", 0.02),
        ("compare_fp8_vs_fullnvfp4", 0.03),
    ]:
        row[name] = {
            "ok": True,
            "rows": [{"ctx": 8185, "delta_nats_per_token": delta}],
        }
    rows.append(row)

blocker_path.write_text(
    json.dumps(
        {
            "can_run_claim_ladder": False,
            "ladder_status": "dependency-changed-review-before-rerun",
        }
    ),
    encoding="utf-8",
)
corpus_path.write_text("short deterministic corpus\n", encoding="utf-8")
corpus_manifest_path.write_text(
    json.dumps({"schema": "ppl-corpus-manifest/v1", "actual_chars": 27}),
    encoding="utf-8",
)
manifest_path.write_text(
    json.dumps(
        {
            "schema": "sglang-gemma4-ar-ladder-pair/v1",
            "run_id": "synthetic",
            "image": "image",
            "image_digest": "image@sha256:abc",
            "row_labels": ["bf16", "fp8", "fullnvfp4"],
            "blocker_audit": blocker_path.name,
            "corpus": corpus_path.name,
            "corpus_manifest": corpus_manifest_path.name,
            "ctx_list": [8185],
            "reuse_prefix_len": 4096,
            "logprob_start_len": 4096,
            "max_new_tokens": 1,
            "context_length": 8192,
            "page_size": 1,
            "graphs": "disabled",
            "source_overlay": False,
            "allow_retracted_global_scale_diagnostic": False,
            "models": models,
            "rows": rows,
        }
    ),
    encoding="utf-8",
)
PY

python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/manifest.json" \
  >"${TMP_DIR}/pass_audit.json"

python3 - "${TMP_DIR}/pass_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("ok"):
    raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))
if payload.get("findings"):
    raise SystemExit(f"unexpected findings: {payload['findings']}")
print("PASS synthetic_complete_manifest_passes")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/overlay_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
payload["source_overlay"] = True
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/overlay_manifest.json" \
  >"${TMP_DIR}/overlay_audit.json" 2>"${TMP_DIR}/overlay_audit.err"; then
  echo "FAIL source-overlay manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/overlay_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "manifest source_overlay must be false for claim-grade rows"
if needle not in findings:
    raise SystemExit(f"missing expected source-overlay finding\n{findings}")
print("PASS source_overlay_manifest_fails")
PY
