#!/usr/bin/env bash
# $1=HF_TOKEN  $2=patch_name  ; applies a sed patch (defined below) to prefill.cuh, clears JIT, runs single nvfp4
export HF_TOKEN="$1" PATH=/root/v/bin:$PATH PYTHONPATH=/root/flashinfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_FLASHINFER_MM_PREFIX=1 VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1
F=/root/flashinfer/include/flashinfer/attention/prefill.cuh
cp "$F" /root/prefill.bak
case "$2" in
  vscale_fp32)
    # apply V SF in fp32: replace __hmul2(packed2 bf16) with float-precision multiply via expansion
    python3 - "$F" <<'PY'
import sys,re
F=sys.argv[1]; s=open(F).read()
# wrap each "*(packed2_*)&b_frag[i] = __hmul2(*(packed2_*)&b_frag[i], scale_X);" to do fp32 mul
def repl(m):
    idx=m.group(1); sc=m.group(2)
    return (f"{{ float2 _bf=__bfloat1622float2(*(__nv_bfloat162*)&b_frag[{idx}]); "
            f"float2 _sc=__bfloat1622float2(*(__nv_bfloat162*)&{sc}); "
            f"_bf.x*=_sc.x; _bf.y*=_sc.y; *(__nv_bfloat162*)&b_frag[{idx}]=__float22bfloat162_rn(_bf); }}")
s2=re.sub(r"\*\(packed2_\*\)&b_frag\[(\d)\] = __hmul2\(\*\(packed2_\*\)&b_frag\[\d\], (scale_[a-z]+)\);", repl, s)
open(F,"w").write(s2); print("patched vscale_fp32 count diff:", s2.count("float22bfloat162_rn"))
PY
    ;;
esac
rm -rf ~/.cache/flashinfer /root/.cache/flashinfer 2>/dev/null
cd /root; mkdir -p pfx_out
python vllm_matched_kv_anchor.py --model google/gemma-4-12b-it --tokenizer google/gemma-4-12b-it \
  --corpus wikitext_8k.txt --kv-cache-dtype nvfp4 --ctx 8185 --prefix-len 4096 --max-model-len 8192 \
  --max-num-batched-tokens 8192 --skip-warmup --gpu-memory-utilization 0.5 \
  --output pfx_out/kpatch_$2.json --enforce-eager > kpatch_$2.log 2>&1
echo "rc=$?"
python3 -c "import json;print('KPATCH $2 single NLL=',round(json.load(open('pfx_out/kpatch_$2.json'))['score']['mean_nll_nats'],4))" 2>/dev/null || (echo NOJSON; grep -iE "error|assert|Invalid" kpatch_$2.log | tail -4)
cp /root/prefill.bak "$F"
echo "DONE_KPATCH_$2 (baseline single=8.7031, target chunked=8.467)"
