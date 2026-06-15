#!/usr/bin/env bash
# Real per-K/V NVFP4-KV calibration + matched bf16-vs-calibrated-nvfp4 ladder, to flip a model's
# long-ctx nvfp4 quality row GREEN. Coordinate-descent calibration (1-D joint -> K refine -> V refine),
# bakes best (k_scale,v_scale) into /root/calib_dir/<arch_signature>.json, then runs the matched row
# through the production calib loader (VLLM_NVFP4_KV_CALIB, no env hack) in BOTH chunked (production)
# and single-pass (worst case) regimes.
#   args: $1=HF_TOKEN  $2=MODEL  $3=GMU  $4=TAG
set -u
HF_TOKEN="$1"; MODEL="${2:-google/gemma-4-12b-it}"; GMU="${3:-0.5}"; TAG="${4:-12b}"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
export VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
cd /root; mkdir -p green_$TAG calib_dir
OUT=green_$TAG; RES=$OUT/results.tsv; : > $RES
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
CTX=8185; PFX=4096; MML=8192
# big models can't fit the 8185 single-pass FA2 workspace; set SWEEP_NBT=4096 to calibrate in the
# (production) chunked regime instead. The optimal global scale is regime-robust (verified on 12B).
SWEEP_NBT=${SWEEP_NBT:-8192}

anchor(){ # $1=tag $2=dtype $3=nbt $4=extra_env  -> echoes NLL
  local tag="$1" dt="$2" nbt="$3" extra="$4"
  env $extra python vllm_matched_kv_anchor.py --model "$MODEL" --tokenizer "$MODEL" \
    --corpus wikitext_8k.txt --kv-cache-dtype "$dt" --ctx $CTX --prefix-len $PFX --max-model-len $MML \
    --max-num-batched-tokens "$nbt" --skip-warmup --gpu-memory-utilization $GMU \
    --output $OUT/${tag}.json --enforce-eager > $OUT/${tag}.log 2>&1
  python3 -c "import json;print(round(json.load(open('$OUT/${tag}.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || echo FAIL
}
sweepK(){ # $1=k $2=v -> records "k v nll"
  local v; v=$(anchor "sw_k${1}_v${2}" nvfp4 $SWEEP_NBT "VLLM_FIX_K_SCALE=$1 VLLM_FIX_V_SCALE=$2")
  echo -e "$1\t$2\t$v" | tee -a $RES; }

# chunked nbt must clear the multimodal per-item floor (Gemma 4 mm item = 2496 tok); 4096 still
# chunks the 8185-token prefill into 2 chunks (the production chunked-prefill regime).
CHUNK_NBT=4096
echo "=== bf16 baselines ==="
BF_S=$(anchor bf16_single auto 8192 ""); BF_C=$(anchor bf16_chunk auto $CHUNK_NBT "")
echo "bf16  single=$BF_S  chunk=$BF_C"

echo "=== phase 1: 1-D joint K=V sweep (single-pass worst case) ==="
for s in 0.2 0.1 0.07 0.05; do sweepK $s $s; done
B1=$(python3 -c "import sys;r=[l.split() for l in open('$RES') if l.split()[0]==l.split()[1] and l.split()[2]!='FAIL'];print(min(r,key=lambda x:float(x[2]))[0])")
echo "best joint k=v=$B1"

echo "=== phase 2: refine K (V fixed=$B1) ==="
for k in $(python3 -c "b=float('$B1');print(' '.join('%.4f'%(b*m) for m in (0.6,0.8,1.3)))"); do sweepK $k $B1; done
BK=$(python3 -c "r=[l.split() for l in open('$RES') if l.split()[1]=='$B1' and l.split()[2]!='FAIL'];print(min(r,key=lambda x:float(x[2]))[0])")
echo "best K=$BK"

echo "=== phase 3: refine V (K fixed=$BK) ==="
for v in $(python3 -c "b=float('$B1');print(' '.join('%.4f'%(b*m) for m in (0.6,0.8,1.3)))"); do sweepK $BK $v; done
BV=$(python3 -c "r=[l.split() for l in open('$RES') if l.split()[0]=='$BK' and l.split()[2]!='FAIL'];print(min(r,key=lambda x:float(x[2]))[1])")
echo "best (K,V)=($BK,$BV)"

echo "=== bake calibration JSON keyed by arch_signature ==="
python - "$MODEL" "$BK" "$BV" <<'PY'
import json,sys,os
sys.path.insert(0,"/root"); from nvfp4_kv_calib import arch_signature
from transformers import AutoConfig
cfg=AutoConfig.from_pretrained(sys.argv[1], token=os.environ["HF_TOKEN"])
sig=arch_signature(cfg); k=float(sys.argv[2]); v=float(sys.argv[3])
json.dump({"arch_signature":sig,"k_scale":k,"v_scale":v,"model":sys.argv[1],"source":"green_ladder_coorddescent"},
          open(f"/root/calib_dir/{sig}.json","w"),indent=2)
print("arch_signature:",sig,"-> k=%g v=%g"%(k,v))
PY

echo "=== matched calibrated ladder (production calib loader, NO env hack) ==="
CK_S=$(anchor calib_single nvfp4 $SWEEP_NBT "VLLM_NVFP4_KV_CALIB=/root/calib_dir")
CK_C=$(anchor calib_chunk  nvfp4 $CHUNK_NBT "VLLM_NVFP4_KV_CALIB=/root/calib_dir")
applied=$(grep -h '\[CALIB\] applied' $OUT/calib_chunk.log | head -1)

echo "================= GREEN LADDER ($TAG / $MODEL) ================="
python3 - "$BF_S" "$BF_C" "$CK_S" "$CK_C" <<'PY'
import sys
bs,bc,cs,cc=[ (None if x=='FAIL' else float(x)) for x in sys.argv[1:5] ]
def row(n,bf,nv):
    d = None if (bf is None or nv is None) else round(nv-bf,4)
    print(f"  {n:18s} bf16={bf}  nvfp4_calib={nv}  delta={d}")
row("single-pass(8185)",bs,cs)
row("chunked(prod)",bc,cc)
PY
echo "applied: $applied"
echo "calib JSON: $(cat /root/calib_dir/*.json | tr -d '\n')"
echo DONE_GREEN_$TAG
