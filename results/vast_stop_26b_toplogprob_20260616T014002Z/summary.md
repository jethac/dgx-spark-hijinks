# 26B-A4B Top-Logprob Attribution Run Stopped By Operator

Date: 2026-06-16 UTC

This is a stop-point artifact, not a quality result.

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Workstation Edition (`sm_120`)
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0
- FlashInfer source overlay: `/root/flashinfer`
- Packet: `docs/vast_anchor/launch_26b_toplogprob_live.sh`
- Rows requested: `bf16` plus `base_k100`

## Outcome

The run was intentionally stopped before either row completed. It reached vLLM model load and skipped multimodal profiling as intended, then spent time compiling FlashInfer `fused_moe_120` JIT objects. No prompt-logprob JSON row was produced, and `toplogprob_delta_report.tsv` is empty.

The row-status file shows:

```text
label	dtype	status
bf16	auto	failed_rc_0
base_k100	nvfp4	failed_rc_0
```

Treat those statuses as interruption fallout from the run-script trap, not as a model-quality failure. There is no bf16 baseline and no NVFP4 comparator value in this artifact.

## Preserved Files

- `dx26_toplogprob_attr_fg_20260616T014002Z.log`
- `dx26_toplogprob_attr_fg_20260616T014002Z.tgz`
- `FINAL_STATUS.txt`
- `row_status.tsv`

Remote cleanup was completed after pulling the artifact: the vLLM/JIT processes were killed, GPU memory returned to idle, and Vast instance `41125455` was destroyed.

## Next Useful Run

If this packet is resumed, reuse the same script but expect the first sm120 run to spend time building FlashInfer fused-MoE JIT cache. The next live result still needs a completed `bf16` row and at least one NVFP4 row before any top-logprob attribution conclusion can be drawn.
