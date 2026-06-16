TL;DR: Deep layer capture is complete for 26B full-NVFP4 `base_k100`; drift is not readout-only and accumulates through the middle/deep decoder stack. Full 26B NVFP4 K+V remains RED; next best discriminator is FP8-K or mixed-K deep capture at the same taps.

Artifact:

- `results/vast_26b_deep_layer_capture_20260616T034500Z/summary.md`
- Split tarball parts under the same results dir; raw tree is inside the reconstructed tarball. Small reports are copied out beside it.
- Bug doc updated: `docs/BUG_NVFP4_KV_GEMMA4_26B_A4B.md`

Run:

- Vast RTX PRO 6000 Blackwell Max-Q WS, sm120.
- Container: `nvidia/cuda:13.0.1-devel-ubuntu22.04`.
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`.
- FlashInfer ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.
- Model: `google/gemma-4-26B-A4B-it`, ctx 8185, prefix 4096, bf16 vs NVFP4 `base_k100`.
- Requested layers: `0,4,8,12,16,20,24,28,32,36,40`; emitted layers: `0,4,8,12,16,20,24,28`.
- Vast instance `41134294` was destroyed after artifact pull; `vastai show instances` was empty.

Rows:

```text
label       mean_nll      delta_vs_bf16  ppl          readout_calls  layer_calls
bf16        7.933360410   +0.000000000   2788.782530  4              40
base_k100   7.815396153   -0.117964257   2478.468612  4              40
```

High-signal nonzero layer trend (`all` bucket, rel-L2):

```text
layer  input     attention  moe       output    router    router_top1
0      0.000000  0.060646   0.085261  0.037239  0.023325  0.960938
4      0.053169  0.111771   0.141606  0.062629  0.026655  0.902344
8      0.097611  0.191567   0.208625  0.120631  0.051521  0.886719
12     0.154590  0.316604   0.422642  0.162863  0.084292  0.808594
16     0.208418  0.313635   0.468606  0.204483  0.118676  0.687500
20     0.323810  0.270761   0.385327  0.297251  0.119152  0.671875
24     0.368328  0.255562   0.409719  0.385123  0.119406  0.710938
28     0.400963  0.270923   0.371407  0.380915  0.192314  0.792969
```

Interpretation:

- Layer-0 input is still identical; first perturbation is layer-0 attention.
- The layer-4-to-final gap is now explained: output rel-L2 grows to about `0.38` by layers 24/28.
- Router is not the initial cause, but it becomes a downstream casualty: top-1 match drops to `0.67-0.71` around layers 16-24.
- Final hidden rel-L2 stays `0.1965`, below the peak mid-stack output rel-L2, so final norm/readout dampens some drift rather than creating it.

I did not run another Vast box just to capture adjacent layer 29. This result is enough to unpark the next discriminator: run the same deep taps with FP8-K or mixed-K so we can separate K-driven attention drift from V/downstream MoE amplification before more global/layer-band NVFP4 scale tuning.
