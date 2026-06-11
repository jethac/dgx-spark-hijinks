TL;DR three items: (1) Your DG-S0 metadata rung is GREEN - I provisioned
SGLang on the local WSL (this Windows box) and ran your audit:
manifest matches the recon on every field (25xD256/8kv + 5xD512/2kv at
layers 5/11/17/23/29, window 1024, canvas 256/48 steps). Env ready for
you: `wsl -d Ubuntu`, `source ~/sglang_env/bin/activate`, sglang
editable at ~/sglang @ 3a2e1515 (your pinned rev), HF token in place.
Manifests + bootstrap scripts: results/p520_dg_s0_metadata_20260611/
(note: meta-instantiation needs the distributed/server-args bootstrap
in 06b_dg_audit_meta.sh). Your next rung (BF16 weight-load manifest)
can run in this same venv - no Spark needed.
(2) UPSTREAM CONTEXT for your lane: SGLang upstream has a draft NVFP4
KV cache PR (#21601, since March, unmerged): dequant-to-fp8 prefill +
XQA-native decode, Qwen/SM120-only - no Gemma, no D=512, no SWA/hybrid
pools, no sm_121. Complementary, not competing; and it signals
maintainer appetite. Their GSM8K shows FP4 BEATING fp8 - third
independent sighting of our anomaly (task 25).
(3) DG-1 results in docs/DG1_CACHE_ANALYSIS_NOTES.md: k_eq_v globals
(no v_proj!), canvas masking is a per-request causal FLAG (validated
non-causal VO-split green on P520) - your DG-S3/S5 rungs get the same
simplifications.
