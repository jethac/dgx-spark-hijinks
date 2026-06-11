TL;DR: The combined window landed. (1) 31B quality table COMPLETE -
bf16-FlashInfer anchor = 4.613163 nats/tok; the anomaly is now
MATCHED-BACKEND confirmed: nvfp4 -0.33, fp8 -0.14 vs bf16 on the SAME
FlashInfer path (and bf16-Triton vs bf16-FlashInfer differ by only
+0.04 - the "Triton quality tax" hypothesis is dead; the quantized-KV-
beats-bf16 phenomenon is real on three platforms/two runtimes and now
needs its own investigation). (2) DG-0 baseline: official gemma-
aarch64-cu130 image serves DiffusionGemma on GB10 clean; coherent;
25.6 tok/s sustained (4-7x below NVIDIA's 150 claim - mean 44-51
denoise steps/canvas at temp-0 vs the docs' "typical 12-16 adaptive";
sampler settings are the suspect); KV pool 1.63M tokens bf16; curve
FLAT 50->6000 prompt tokens, so the KV-read-amplification thesis test
moves to much longer contexts (max-model-len is 262144).

For your DG-S rungs: confirmed serving geometry - 30 layers = 25
sliding (D=256, 8 KV heads, window 1024) + 5 global (D=512, 2 KV
heads = GQA group 8 vs 16 Q heads), MoE 128 experts top-8. Both
groups TRITON_ATTN in the baseline image (our upgrade target). Full
artifacts: results/claude_dg0_anchor_window_20260611/.
