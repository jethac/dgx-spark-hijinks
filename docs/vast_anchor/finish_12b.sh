#!/usr/bin/env bash
# Finish the 12B green row: chunked bf16 baseline (nbt=4096 clears the mm floor) + both candidate
# calibrations ((0.1,0.1) conservative joint, (0.1,0.06) the 2-D argmin where V wants a larger global
# scale) across single-pass(8192) and chunked(4096). bf16_single=8.2816 already measured.
set -u
HF_TOKEN="$1"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root; mkdir -p fin12b; M=google/gemma-4-12b-it
anchor(){ env $4 python vllm_matched_kv_anchor.py --model $M --tokenizer $M --corpus wikitext_8k.txt \
  --kv-cache-dtype $2 --ctx 8185 --prefix-len 4096 --max-model-len 8192 --max-num-batched-tokens $3 \
  --skip-warmup --gpu-memory-utilization 0.5 --output fin12b/$1.json --enforce-eager > fin12b/$1.log 2>&1
  python3 -c "import json;print(round(json.load(open('fin12b/$1.json'))['score']['mean_nll_nats'],4))" 2>/dev/null||echo FAIL; }
BF_C=$(anchor bf16_chunk auto 4096 "")
C1S=$(anchor c0101_single nvfp4 8192 "VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1")
C1C=$(anchor c0101_chunk  nvfp4 4096 "VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.1")
C2S=$(anchor c0106_single nvfp4 8192 "VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.06")
C2C=$(anchor c0106_chunk  nvfp4 4096 "VLLM_FIX_K_SCALE=0.1 VLLM_FIX_V_SCALE=0.06")
echo "================= 12B FINISH TABLE (bf16_single=8.2816) ================="
python3 - "$BF_C" "$C1S" "$C1C" "$C2S" "$C2C" <<'PY'
import sys
bfc,c1s,c1c,c2s,c2c=[None if x=='FAIL' else float(x) for x in sys.argv[1:6]]
bfs=8.2816
def d(nv,bf): return None if (nv is None or bf is None) else round(nv-bf,4)
print(f"  bf16              single=8.2816  chunk={bfc}")
print(f"  nvfp4 (0.1,0.1)   single={c1s} (d={d(c1s,bfs)})  chunk={c1c} (d={d(c1c,bfc)})")
print(f"  nvfp4 (0.1,0.06)  single={c2s} (d={d(c2s,bfs)})  chunk={c2c} (d={d(c2c,bfc)})")
PY
echo DONE_FIN12B
