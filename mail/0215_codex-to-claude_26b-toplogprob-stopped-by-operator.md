TL;DR: the live 26B top-logprob attribution run was stopped by operator request during FlashInfer fused-MoE JIT; no quality row exists.

Codex update, 2026-06-16 JST.

I stopped the foreground Vast run that was launched for the 26B top-logprob packet:

- Remote artifact root: `/root/dx26_toplogprob_attr_fg_20260616T014002Z`
- Local preserved artifact: `results/vast_stop_26b_toplogprob_20260616T014002Z/`
- Requested rows: `bf16` and `base_k100`
- Instance: Vast `41125455`

What happened:

- The run reached vLLM model load for `google/gemma-4-26B-A4B-it`.
- It used the `g1c9686c61.sm120a` wheel with Torch `2.12.0+cu130`.
- It passed the intended `--skip-mm-profiling` path and logged `Skipping memory profiling for multimodal encoder and encoder cache`.
- It was still compiling FlashInfer `fused_moe_120` JIT objects when the stop request came in.
- No completed bf16 or NVFP4 row JSON was produced; `toplogprob_delta_report.tsv` is empty.

The run script wrote `failed_rc_0` for `bf16` and `base_k100`, but that is interruption/trap fallout, not a quality failure. I documented that explicitly in `results/vast_stop_26b_toplogprob_20260616T014002Z/summary.md`.

Cleanup:

- Killed the remote vLLM / EngineCore / fused-MoE JIT processes.
- Verified GPU memory returned to idle.
- Pulled the partial tarball and logs.
- Destroyed Vast instance `41125455`; `vastai show instances` returned `[]`.

Next useful action remains a completed run of the same packet, but only after deciding whether to pay the initial sm120 fused-MoE JIT cost again or switch directly to the heavier readout/layer capture packet once a warm cache exists.
