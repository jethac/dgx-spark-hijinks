#!/usr/bin/env bash
# Repro probe for the sm_120 nvfp4-KV read-path defect (Gemma 3 270M minimal repro: d256/kv=1/SWA-512).
# Runs the matched PPL anchor at the 270M geometry across bf16 (truth), FlashInfer-bf16, FlashInfer-nvfp4.
# On the P520 (5060 Ti / GB206 / WSL2) nvfp4 = deterministic gibberish (+8 nats). This box is native
# Linux; comparing isolates the WSL/environment confound (and, vs the PRO 6000, the GB206-vs-GB202 die).
#   args: $1=HF_TOKEN  $2=MODEL(default 270m)  $3=GMU(default 0.6)
set -u
HF_TOKEN="$1"; MODEL="${2:-google/gemma-3-270m-it}"; GMU="${3:-0.6}"
export HF_TOKEN PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1
cd /root; mkdir -p g3repro
[ -f wikitext_8k.txt ] || python corpus_fetch.py >/dev/null 2>&1
CTX=8191; PFX=4096; MML=8192; NBT=2048   # chunked prefill: small 16GB cards crash on single-pass 8191
anchor(){ # $1=tag $2=dtype $3=extra_env -> NLL
  env $3 python vllm_matched_kv_anchor.py --model "$MODEL" --tokenizer "$MODEL" \
    --corpus wikitext_8k.txt --kv-cache-dtype "$2" --ctx $CTX --prefix-len $PFX --max-model-len $MML \
    --max-num-batched-tokens $NBT --skip-warmup --gpu-memory-utilization $GMU \
    --output g3repro/$1.json --enforce-eager > g3repro/$1.log 2>&1
  python3 -c "import json;print(round(json.load(open('g3repro/$1.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || echo FAIL
}
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=== bf16 (auto) truth ==="; BF=$(anchor bf16 auto ""); echo "  bf16 NLL=$BF"
echo "=== nvfp4 (+LINEAR_V_SF) ==="; NV=$(anchor nvfp4 nvfp4 "VLLM_NVFP4_KV_LINEAR_V_SF=1"); echo "  nvfp4 NLL=$NV"
echo "================= G3 NVFP4 REPRO ($MODEL) ================="
python3 - "$BF" "$NV" <<'PY'
import sys
bf,nv=[None if x=='FAIL' else float(x) for x in sys.argv[1:3]]
d=None if (bf is None or nv is None) else round(nv-bf,4)
print(f"  bf16={bf}  nvfp4={nv}  delta={d}")
print("  VERDICT:", "GIBBERISH/REPRO (defect present)" if (d is not None and d>1.0) else
      ("COHERENT (defect ABSENT on this box -> P520 was WSL/env artifact)" if d is not None else "INCONCLUSIVE(FAIL)"))
PY
echo DONE_G3REPRO
