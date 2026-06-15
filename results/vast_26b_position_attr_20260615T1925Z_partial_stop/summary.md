# 26B-A4B Position Attribution Partial Stop

Status: stopped by operator request (`$codex-autoresearch stop`), not a quality result.

Run:
- Instance: Vast `41091515` (`codex-26b-posattr-rerun`), destroyed after artifact pull.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, capability `(12, 0)`.
- Model: `google/gemma-4-26B-A4B-it`.
- vLLM: `0.1.dev1+g1c9686c61.sm120a`.
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0.
- Context/prefix: `ctx=8185`, `prefix=4096`.
- Packet rows intended: `bf16`, `k100`, `k102`, `k103`, `k104`.

What happened:
- The run was interrupted during the first `bf16` row.
- The row had loaded weights and reached encoder-cache initialization/profiling:
  `Encoder cache will be initialized with a budget of 4096 tokens...`
- No row JSON, NLL, attribution buckets, or comparator data were produced.

Interpretation:
- This artifact is a clean stop-point record only.
- It should not be cited as a startup failure or quality result. Prior successful 26B rows show this encoder-cache profiling point can sit for several minutes before progressing.

Local contents:
- `dx26_position_attr_20260615T1925Z/RUN_INFO.txt`
- `dx26_position_attr_20260615T1925Z/run.log`
- `dx26_position_attr_20260615T1925Z/rows/bf16.log`
- Packet scripts copied from the run instance.
