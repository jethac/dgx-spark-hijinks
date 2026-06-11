#!/usr/bin/env bash
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%dT%H%MJST)}
RUN=${RUN:-sglang_qwen_mixedkv_reuse_prefix_sweep_${STAMP}}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
CTX=${CTX:-8192}
REUSE_PREFIX_LIST=${REUSE_PREFIX_LIST:-"0 1024 2048 4096 6144"}
CORPUS=${CORPUS:-${RESULTS_DIR}/${RUN}_corpus.md}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-${RESULTS_DIR}/${RUN}_corpus_manifest.json}

mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
  python3 "${REPO_ROOT}/scripts/build_ppl_corpus.py" \
    --repo-root "${REPO_ROOT}" \
    --output "${CORPUS}" \
    --manifest "${CORPUS_MANIFEST}" \
    --max-chars "${CORPUS_MAX_CHARS:-250000}"
fi

for prefix in ${REUSE_PREFIX_LIST}; do
  if (( prefix >= CTX )); then
    echo "reuse prefix ${prefix} must be smaller than ctx ${CTX}" >&2
    exit 2
  fi
  child_run="${RUN}_ctx${CTX}_prefix${prefix}"
  RUN="${child_run}" \
    CTX_LIST="${CTX}" \
    REUSE_PREFIX_LEN="${prefix}" \
    LOGPROB_START_LEN="${prefix}" \
    CORPUS="${CORPUS}" \
    CORPUS_MANIFEST="${CORPUS_MANIFEST}" \
    "${REPO_ROOT}/scripts/run_sglang_qwen_ppl_pair.sh"
done

python3 - <<PY
import json
from pathlib import Path

results = Path("${RESULTS_DIR}")
run = "${RUN}"
ctx = int("${CTX}")
prefixes = [int(x) for x in "${REUSE_PREFIX_LIST}".split()]
rows = []
artifacts = {}
for prefix in prefixes:
    child = f"{run}_ctx{ctx}_prefix{prefix}"
    compare_path = results / f"{child}_compare.json"
    manifest_path = results / f"{child}_manifest.json"
    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not compare.get("rows"):
        raise SystemExit(f"{compare_path} has no rows")
    row = dict(compare["rows"][0])
    row["reuse_prefix_len"] = prefix
    row["scored_tokens_expected"] = ctx - max(1, prefix)
    row["fp8_kv_tokens"] = manifest.get("fp8_kv_tokens")
    row["mixed_kv_tokens"] = manifest.get("mixed_kv_tokens")
    if row["fp8_kv_tokens"] and row["mixed_kv_tokens"]:
        row["allocator_token_ratio"] = row["mixed_kv_tokens"] / row["fp8_kv_tokens"]
    else:
        row["allocator_token_ratio"] = None
    for label in ("fp8", "mixed"):
        ppl = json.loads((results / f"{child}_{label}_ppl.json").read_text(encoding="utf-8"))
        contexts = ppl.get("contexts") or []
        if contexts:
            score = contexts[0].get("score", {})
            row[f"{label}_cached_tokens"] = score.get("cached_tokens")
            row[f"{label}_scored_tokens"] = score.get("num_scored_tokens")
    rows.append(row)
    artifacts[str(prefix)] = {
        "run_id": child,
        "compare": str(compare_path),
        "manifest": str(manifest_path),
        "fp8_ppl": str(results / f"{child}_fp8_ppl.json"),
        "mixed_ppl": str(results / f"{child}_mixed_ppl.json"),
        "fp8_server_log": str(results / f"{child}_fp8_server.log"),
        "mixed_server_log": str(results / f"{child}_mixed_server.log"),
    }
report = {
    "schema": "sglang-qwen-reuse-prefix-sweep/v1",
    "run_id": run,
    "ctx": ctx,
    "reuse_prefix_list": prefixes,
    "corpus": "${CORPUS}",
    "corpus_manifest": "${CORPUS_MANIFEST}",
    "artifacts": artifacts,
    "rows": rows,
    "ok": all(row.get("fp8_ok") and row.get("candidate_ok") for row in rows),
}
(results / f"{run}_manifest.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
PY
