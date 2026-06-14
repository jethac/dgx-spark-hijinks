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
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    row_dir = manifest_path.parent / slug
    row_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model,
        "dir": str(row_dir),
        "mixedkv": None,
        "compare_bf16_vs_mixedkv": None,
    }
    for label, dtype in [
        ("bf16", "auto"),
        ("fp8", "fp8_e4m3"),
        ("fullnvfp4", "fp4_e2m1"),
    ]:
        capacity_total = {
            "bf16": 1000,
            "fp8": 1900,
            "fullnvfp4": 3550,
        }[label]
        row[label] = {
            "model": model,
            "label": label,
            "kv_cache_dtype": dtype,
            "ppl_ok": True,
            "chat_transport_ok": True,
            "chat_content_equal": True,
            "kv_capacity": {
                "full_tokens": capacity_total // 2,
                "swa_tokens": capacity_total - capacity_total // 2,
                "total_token_slots": capacity_total,
                "full_per_token_bytes": {
                    "bf16": 2048,
                    "fp8": 1024,
                    "fullnvfp4": 576,
                }[label],
                "swa_per_token_bytes": {
                    "bf16": 8192,
                    "fp8": 4096,
                    "fullnvfp4": 2304,
                }[label],
                "cell_size_bytes": {
                    "bf16": 278528,
                    "fp8": 139264,
                    "fullnvfp4": 78336,
                }[label],
            },
        }
        for suffix in (
            "summary.json",
            "ppl.json",
            "chat_1.json",
            "chat_2.json",
            "preflight.log",
            "provenance.log",
            "server.log",
            "container_inspect.json",
        ):
            artifact = row_dir / f"{label}_{suffix}"
            if suffix == "ppl.json":
                artifact.write_text(
                    json.dumps(
                        {
                            "schema": "sglang-prompt-ppl-sweep/v1",
                            "ok": True,
                            "tokenizer": model,
                            "kv_cache_dtype": dtype,
                            "container_image": "image",
                            "hardware": {
                                "cuda_available": True,
                                "devices": [
                                    {
                                        "comparison_key": "NVIDIA_GB10:sm_121:sms_48",
                                    }
                                ],
                            },
                            "contexts": [
                                {
                                    "ctx": 8185,
                                    "payload": {
                                        "prompt_token_count": 8185,
                                        "max_new_tokens": 1,
                                        "reuse_prefix_len": 4096,
                                        "logprob_start_len": 4096,
                                        "score_start_index": 4096,
                                    },
                                    "score": {
                                        "ok": True,
                                        "cached_tokens": 4096,
                                        "num_scored_tokens": 4088,
                                        "num_missing_tokens": 0,
                                        "num_mismatched_tokens": 0,
                                    },
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            elif suffix.endswith(".json"):
                artifact.write_text("{}\n", encoding="utf-8")
            elif suffix == "provenance.log":
                artifact.write_text(
                    "\n".join(
                        [
                            "transformers 5.11.0",
                            "sglang 0.0.0 /work/third_party/sglang/python/sglang/__init__.py",
                            "flashinfer 0.6.13 /work/third_party/flashinfer/flashinfer/__init__.py",
                            "flashinfer_python 0.6.13",
                            "sglang_kernel 0.4.3",
                            "binary_md5 sgl_kernel /usr/local/lib/python3.12/dist-packages/sgl_kernel/__init__.py abc",
                            "flashinfer_data /work/third_party/flashinfer/flashinfer/data",
                            "flashinfer_csrc /work/third_party/flashinfer/flashinfer/data/csrc",
                            "flashinfer_include /work/third_party/flashinfer/flashinfer/data/include",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            elif suffix == "server.log":
                lines = [
                    "server_args=ServerArgs(attention_backend='flashinfer')",
                    "SGLANG_GEMMA_KV_GEOMETRY layer=0 heads=16 kv_heads=8 head_dim=256 v_head_dim=256",
                    "SGLang FlashInfer VO split enabled: D=512 paged prefill and decode-as-prefill use two D_VO=256 passes.",
                ]
                if label == "fullnvfp4":
                    lines.extend(
                        [
                            "FP4 KV FlashInfer module trace label=extend_paged deswizzle_macro_active=False k_sf={} v_sf={}",
                            "[flashinfer][prefill-debug] compiled={require_fp4_kv=1,fp4_kv=1}",
                        ]
                    )
                artifact.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                artifact.write_text("ok\n", encoding="utf-8")
    for name, delta in [
        ("compare_bf16_vs_fullnvfp4", 0.02),
        ("compare_fp8_vs_fullnvfp4", 0.03),
    ]:
        row[name] = {
            "ok": True,
            "rows": [{"ctx": 8185, "delta_nats_per_token": delta}],
        }
        (row_dir / f"{name}.json").write_text(json.dumps(row[name]), encoding="utf-8")
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

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/known_blocked_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
blocker_path = dst.parent / "known_blocked_blocker.json"
blocker_path.write_text(
    json.dumps(
        {
            "can_run_claim_ladder": False,
            "ladder_status": "blocked-known-red-dependencies",
        }
    ),
    encoding="utf-8",
)
payload["blocker_audit"] = blocker_path.name
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/known_blocked_manifest.json" \
  >"${TMP_DIR}/known_blocked_audit.json" 2>"${TMP_DIR}/known_blocked_audit.err"; then
  echo "FAIL known-blocked complete manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/known_blocked_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "blocker_audit still records known-blocked dependency refs"
if needle not in findings:
    raise SystemExit(f"missing expected blocker finding\n{findings}")
print("PASS known_blocked_complete_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/copied_bundle_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
for row in payload["rows"]:
    row["dir"] = f"/home/jethac/spark_tmp/copied-results/{pathlib.Path(row['dir']).name}"
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/copied_bundle_manifest.json" \
  >"${TMP_DIR}/copied_bundle_audit.json"

python3 - "${TMP_DIR}/copied_bundle_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("ok"):
    raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))
print("PASS copied_bundle_absolute_paths_fallback")
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

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/wrong_dtype_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
payload["rows"][0]["fp8"]["kv_cache_dtype"] = "auto"
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/wrong_dtype_manifest.json" \
  >"${TMP_DIR}/wrong_dtype_audit.json" 2>"${TMP_DIR}/wrong_dtype_audit.err"; then
  echo "FAIL wrong-dtype manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/wrong_dtype_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "google/gemma-4-12B-it: fp8 kv_cache_dtype is not fp8_e4m3"
if needle not in findings:
    raise SystemExit(f"missing expected dtype finding\n{findings}")
print("PASS wrong_dtype_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/missing_ctx_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
payload["ctx_list"] = [512, 8185]
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/missing_ctx_manifest.json" \
  >"${TMP_DIR}/missing_ctx_audit.json" 2>"${TMP_DIR}/missing_ctx_audit.err"; then
  echo "FAIL missing-ctx manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/missing_ctx_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "ctx coverage [8185] does not match manifest ctx_list [512, 8185]"
if needle not in findings:
    raise SystemExit(f"missing expected ctx coverage finding\n{findings}")
print("PASS missing_ctx_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/missing_artifact_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
row_dir = pathlib.Path(payload["rows"][0]["dir"])
(row_dir / "fullnvfp4_provenance.log").unlink()
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/missing_artifact_manifest.json" \
  >"${TMP_DIR}/missing_artifact_audit.json" 2>"${TMP_DIR}/missing_artifact_audit.err"; then
  echo "FAIL missing-artifact manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/missing_artifact_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "google/gemma-4-12B-it: missing fullnvfp4 artifact fullnvfp4_provenance.log"
if needle not in findings:
    raise SystemExit(f"missing expected artifact finding\n{findings}")
print("PASS missing_artifact_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/missing_marker_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
row_dir = pathlib.Path(payload["rows"][0]["dir"])
path = row_dir / "bf16_provenance.log"
text = path.read_text(encoding="utf-8").replace("binary_md5 sgl_kernel", "binary_md5_missing")
path.write_text(text, encoding="utf-8")
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/missing_marker_manifest.json" \
  >"${TMP_DIR}/missing_marker_audit.json" 2>"${TMP_DIR}/missing_marker_audit.err"; then
  echo "FAIL missing-marker manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/missing_marker_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "google/gemma-4-12B-it: bf16 provenance log missing marker 'binary_md5 sgl_kernel '"
if needle not in findings:
    raise SystemExit(f"missing expected marker finding\n{findings}")
print("PASS missing_marker_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/bad_capacity_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
payload["rows"][0]["fullnvfp4"]["kv_capacity"]["total_token_slots"] = 900
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/bad_capacity_manifest.json" \
  >"${TMP_DIR}/bad_capacity_audit.json" 2>"${TMP_DIR}/bad_capacity_audit.err"; then
  echo "FAIL bad-capacity manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/bad_capacity_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "google/gemma-4-12B-it: capacity token slots must increase bf16 < fp8 < fullnvfp4"
if needle not in findings:
    raise SystemExit(f"missing expected capacity finding\n{findings}")
print("PASS bad_capacity_manifest_fails")
PY

python3 - "${TMP_DIR}/manifest.json" "${TMP_DIR}/bad_ppl_manifest.json" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
row_dir = pathlib.Path(payload["rows"][0]["dir"])
path = row_dir / "fp8_ppl.json"
report = json.loads(path.read_text(encoding="utf-8"))
report["kv_cache_dtype"] = "auto"
path.write_text(json.dumps(report), encoding="utf-8")
dst.write_text(json.dumps(payload), encoding="utf-8")
PY

if python3 scripts/sglang_gemma4_ar_claim_audit.py "${TMP_DIR}/bad_ppl_manifest.json" \
  >"${TMP_DIR}/bad_ppl_audit.json" 2>"${TMP_DIR}/bad_ppl_audit.err"; then
  echo "FAIL bad-ppl manifest unexpectedly passed claim audit" >&2
  exit 1
fi

python3 - "${TMP_DIR}/bad_ppl_audit.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
findings = "\n".join(payload.get("findings", []))
needle = "google/gemma-4-12B-it: fp8 ppl report kv_cache_dtype is not fp8_e4m3"
if needle not in findings:
    raise SystemExit(f"missing expected ppl finding\n{findings}")
print("PASS bad_ppl_manifest_fails")
PY
