# DG-V5: vLLM DiffusionGemma 26B-A4B NVFP4-KV on Spark (sm_121 / GB10) -- GREEN
Image: jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11 (r10 + e2-dgv vLLM wheel, swap-only)
Serve: FLASHINFER, --kv-cache-dtype nvfp4, VLLM_NVFP4_KV_LINEAR_V_SF=1, VO-split, enforce-eager, util 0.6, max-len 4096
Proofs: "Using nvfp4 data type to store kv cache"; "VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors are linear"
Coherence (chat, temp 0): "The capital of Japan is Tokyo." (Tokyo/2+2/Spark coherent)
Capacity:
  bf16/auto KV:  81,203 tokens (17.1 GiB avail, 19.83x concurrency)
  full-NVFP4 KV: 360,143 tokens (21.33 GiB avail, 87.93x concurrency)
  raw token ratio 4.43x; format-exact per-byte 3.556x (=32/9); >=3.56x headline holds
Parity: SGLang DG-R5 (coherent, 3.56x). NVFP4-KV code identical to the proven AR ladder.
