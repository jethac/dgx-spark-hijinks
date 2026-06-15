# 0204 Codex -> Claude: 26B block-attribution run stopped by operator

Jetha issued `$codex-autoresearch stop`, so I stopped the active 26B-A4B block-attribution run rather than letting the ladder continue.

State:

- Vast instance `41100426` (`codex-26b-block-attr`) was running the first `bf16` baseline row.
- The row was still in startup/cold FlashInfer MoE JIT after weight load and encoder profiling; it had not produced a JSON result.
- I terminated the remote processes, pulled the partial bundle, and destroyed the instance.
- Only the pre-existing unlabeled Vast instance `41040954` remained active afterward.

Artifacts:

- `results/vast_26b_block_attr_20260615T2050Z_stopped/summary.md`
- `results/vast_26b_block_attr_20260615T2050Z_stopped/dx26_block_attr_20260615T2050Z/RUN_INFO.txt`
- `results/vast_26b_block_attr_20260615T2050Z_stopped/dx26_block_attr_20260615T2050Z/run.log`
- `results/vast_26b_block_attr_20260615T2050Z_stopped/dx26_block_attr_20260615T2050Z/rows/bf16.log`

No scored quality/capacity row exists from this stopped run.
