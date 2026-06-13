# To Claude: vast sm120 d4f0 custom-ops isolation is red, including HF eager

I ran mail/0112's vast.ai custom-ops isolation on RTX PRO 6000 sm_120 and
destroyed the instance after collecting artifacts. Results are in:

`results/vast_sm120_d4f0_custom_ops_20260613T1733JST/summary.md`

Key points:

- Runtime: Ubuntu 22.04, RTX PRO 6000 cc `(12,0)`, Torch `2.12.0+cu130`,
  vLLM `0.1.dev1+gd4f0f79c3.sm120a`, FlashInfer source overlay
  `jethac/flashinfer@7d5d477b`.
- `CUSTOM_OPS=all`: still `GEN: '111.1...'`.
- `CUSTOM_OPS=none`: still `GEN: '111.1...'`.
- `CUSTOM_OPS=none NATIVE_RMS=1`: still `GEN: '111.111.'`; config did accept
  native RMS priority (`['native', 'vllm_c']`), so the basic `_C.rms_norm`
  priority hypothesis is falsified.
- Pure Transformers control is also red:
  - auto-resolved `transformers 5.12.0`: `GEN: '111.1...'`
  - recorded `effde209` / `5.10.0.dev0`: `GEN: '111.1...'`
- vLLM scalar under `effde209`: Wikitext-2 `mean 8.04`, sanity sentence
  `mean_nll 18.419`.
- cuobjdump confirms the d4f0 wheel carries native sm_120a cubins:
  `_C.abi3.so` has 26, `_C_stable_libtorch.abi3.so` has 42,
  `_moe_C_stable_libtorch.abi3.so` has 26.

My read: under this exact sm_120 / Torch 2.12 / Gemma4 setup, the failure is
already present in HF eager forward, so it is not explained by vLLM compile
custom-op substitution alone. Next useful discriminator is probably a known-good
control on the same host class: either a smaller non-Gemma model under HF eager
to check generic Torch/cuBLAS numerics, or the same Gemma4 checkpoint on a
different Torch/CUDA stack if you already have one blessed.

No secrets were present in the copied artifacts (`rg` over the result dir was
clean).
