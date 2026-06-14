#!/usr/bin/env bash
# Local sm_120 (5060 Ti, 16GB) repro: does the single-vs-chunked nvfp4 long-ctx inflation
# (+0.42 vs +0.19 on 12B) reproduce on E2B? If yes -> bug is general, fixable locally; if the
# nvfp4 single ~= chunked here -> the inflation is VO-split-D512-specific (needs the 96GB box).
set -uo pipefail
export HF_TOKEN="$1"
W=/mnt/b/workshop/worktrees/flashinfer/spark-hijinks-022-fa2-d512
export PYTHONPATH=$W
export PATH=$HOME/sm120env/bin:/usr/local/cuda-13.0/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
PY=$HOME/vllm_wheel_env/bin/python
R=/mnt/b/workshop/worktrees/dgx-spark-hijinks/spark-hijinks-022-gemma4-mixed-kv/docs/vast_anchor
cd "$R"
CORPUS=/tmp/wikitext_8k.txt
[ -f "$CORPUS" ] || HF_TOKEN="$1" $PY - <<PY
import os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
fp=hf_hub_download("Salesforce/wikitext","wikitext-2-raw-v1/test-00000-of-00001.parquet",repo_type="dataset",token=os.environ["HF_TOKEN"])
open("$CORPUS","w",encoding="utf-8").write("\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip()))
print("corpus written")
PY
MODEL=google/gemma-4-E2B-it
mkdir -p /tmp/e2bout
run(){ # kv nbt name
  echo "### ARM $3 (kv=$1 nbt=$2) ###"
  $PY vllm_matched_kv_anchor.py --model "$MODEL" --tokenizer "$MODEL" --corpus "$CORPUS" \
    --kv-cache-dtype "$1" --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
    --max-num-batched-tokens "$2" --skip-warmup --gpu-memory-utilization 0.45 \
    --output /tmp/e2bout/$3.json --enforce-eager > /tmp/e2bout/$3.log 2>--output /tmp/e2bout/$4.json --enforce-eager > /tmp/e2bout/$4.log 2>&11
  echo "rc=$?"
}
run auto   8192 single_bf16
run nvfp4  8192 single_nvfp4
run auto   4096 chunk_bf16
run nvfp4  4096 chunk_nvfp4
echo "===== E2B DELTAS ====="
$PY - <<'PY'
import json,os
o="/tmp/e2bout"
def n(x):
    p=os.path.join(o,x+".json")
    return json.load(open(p))["score"]["mean_nll_nats"] if os.path.exists(p) else None
for path,b,q in [("single","single_bf16","single_nvfp4"),("chunk","chunk_bf16","chunk_nvfp4")]:
    bv,qv=n(b),n(q)
    if bv and qv: print(f"{path:7s}: bf16={bv:.4f} nvfp4={qv:.4f} delta={qv-bv:+.4f}")
    else: print(f"{path:7s}: MISSING b={bv} q={qv}")
PY
echo DONE_E2B
