#!/usr/bin/env bash
# P520 MTP/spec-decode greedy identity ladder (zero-bug bar gate).
#
# Rows (each: spec-OFF run, spec-ON run, string-identity compare):
#   A  gemma-3-1b-it  + gemma-3-270m-it drafter (draft_model)   bf16 KV
#   B  gemma-3-1b-it  + gemma-3-270m-it drafter (draft_model)   nvfp4 KV
#   C  gemma-4-E2B-it + gemma-4-E2B-it-assistant (native MTP)   bf16 KV
#   D  gemma-4-E2B-it + gemma-4-E2B-it-assistant (native MTP)   nvfp4 KV + VO split
#
# GREEN = spec-on output token-identical to spec-off at temp 0, per row.
# Any identity failure = RED with the divergence banked in the verdict JSON.
#
# Run from WSL (Ubuntu) on the P520:
#   bash /mnt/b/workshop/worktrees/dgx-spark-hijinks/\
#        spark-hijinks-022-gemma4-mixed-kv/scripts/p520_mtp_identity_ladder.sh
#
# Pre-conditions (checked below): GPU free, HF access to all 4 model names,
# vLLM importable from $VENV, and for row D's drafter-pin coverage the
# spark/hijinks-e2-mtp commit (2d3411c331) present in the vLLM install --
# rows A-D do not strictly need it (no mixed-KV row here), so a missing
# commit only WARNs.
set -u

VENV="${VENV:-$HOME/sm120env}"
PY="$VENV/bin/python"
CAMPAIGN="${CAMPAIGN:-/mnt/b/workshop/worktrees/dgx-spark-hijinks/spark-hijinks-022-gemma4-mixed-kv}"
RUNNER="$CAMPAIGN/scripts/mtp_identity_run.py"
STAMP="$(date +%Y%m%d)"
OUT="${OUT:-$CAMPAIGN/results/p520_mtp_$STAMP}"
SPECTOK="${SPECTOK:-3}"
mkdir -p "$OUT"
LOG="$OUT/ladder.log"
exec > >(tee -a "$LOG") 2>&1

echo "== p520 mtp identity ladder $(date -Is) =="

# ---- preflight ------------------------------------------------------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 1500 ]; then
  echo "PREFLIGHT FAIL: GPU busy (${USED} MiB used). Another agent owns the GPU; rerun later."
  exit 2
fi
"$PY" -c "import vllm; print('vllm', vllm.__version__)" || { echo "PREFLIGHT FAIL: vllm import"; exit 2; }
for m in google/gemma-3-1b-it google/gemma-3-270m-it \
         google/gemma-4-E2B-it google/gemma-4-E2B-it-assistant; do
  "$PY" "$CAMPAIGN/scripts/hf_model_access_probe.py" --model "$m" \
      --output "$OUT/access_$(echo "$m" | tr '/' '_').json" \
      || { echo "PREFLIGHT FAIL: HF model access: $m"; exit 2; }
done
"$PY" -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('google/gemma-4-E2B-it-assistant')
assert c.model_type in ('gemma4_assistant', 'gemma4_unified_assistant'), c.model_type
print('assistant config OK:', c.model_type)
" || { echo "PREFLIGHT FAIL: transformers lacks gemma4_assistant"; exit 2; }
if ! "$PY" -c "
import inspect, vllm.model_executor.models.gemma4_mtp as m
assert 'gemma4_global_attn_backend_override' in inspect.getsource(m)
" 2>/dev/null; then
  echo "WARN: vLLM install lacks spark/hijinks-e2-mtp drafter pin (2d3411c331);"
  echo "      rows A-D unaffected, mixed-KV MTP rows would need it."
fi

FAILED=0
run_row() {
  local name="$1" target="$2" kvdtype="$3" speccfg="$4"; shift 4
  local extra_env=("$@")
  echo "---- row $name (kv=$kvdtype) ----"
  env "${extra_env[@]}" "$PY" "$RUNNER" run --target "$target" \
      --kv-cache-dtype "$kvdtype" --out "$OUT/${name}_specoff.json" \
      || { echo "row $name spec-OFF RED (run failure)"; FAILED=1; return; }
  env "${extra_env[@]}" "$PY" "$RUNNER" run --target "$target" \
      --kv-cache-dtype "$kvdtype" --spec-config "$speccfg" \
      --out "$OUT/${name}_specon.json" \
      || { echo "row $name spec-ON RED (run failure)"; FAILED=1; return; }
  "$PY" "$RUNNER" compare --baseline "$OUT/${name}_specoff.json" \
      --spec "$OUT/${name}_specon.json" --out "$OUT/${name}_verdict.json" \
      || { echo "row $name IDENTITY RED"; FAILED=1; }
}

G3_SPEC="{\"model\": \"google/gemma-3-270m-it\", \"method\": \"draft_model\", \"num_speculative_tokens\": $SPECTOK}"
G4_SPEC="{\"model\": \"google/gemma-4-E2B-it-assistant\", \"num_speculative_tokens\": $SPECTOK}"

run_row a_g3_1b_bf16  google/gemma-3-1b-it  auto  "$G3_SPEC"
run_row b_g3_1b_nvfp4 google/gemma-3-1b-it  nvfp4 "$G3_SPEC"
run_row c_g4_e2b_bf16 google/gemma-4-E2B-it auto  "$G4_SPEC"
run_row d_g4_e2b_nvfp4 google/gemma-4-E2B-it nvfp4 "$G4_SPEC" \
  VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1

echo "== ladder done $(date -Is), FAILED=$FAILED; verdicts: =="
grep -H '"verdict"' "$OUT"/*_verdict.json 2>/dev/null || true
exit $FAILED
