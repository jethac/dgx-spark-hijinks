TL;DR: llama.cpp arm complete (results/p520_llamacpp_kvquant_20260611):
DIRECTION replicates (q4 < q8 < f16 PPL, monotonic, two corpora, all 72
checkpoints below baseline) but MAGNITUDE does not - 0.004-0.021 nats vs
our 0.14-0.33. Two-component model: small general effect + large
stack-specific component. The stratification arm should now ALSO diff
token-level deltas between stacks. Practical notes for the capstone era:
llama.cpp mixed K/V types collapse to ~8 tg t/s (no fast FA path for
heterogeneous pairs); q4 cache GAINS decode speed there (52->63 t/s);
no NVFP4 GGUF exists yet for gemma-4-E4B.
