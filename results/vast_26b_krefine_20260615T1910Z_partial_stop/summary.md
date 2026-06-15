# vLLM Gemma 4 26B-A4B NVFP4 K-refinement partial stop

Status: **INTERRUPTED / partial discriminator only**.

This artifact preserves the stopped Vast sm120 K-refinement run for
`google/gemma-4-26B-A4B-it`. The run was intentionally stopped before completion and the
Vast instance was destroyed after artifact pull.

## Scope

- Instance label: `codex-26b-krefine`
- Device: RTX PRO 6000 Blackwell, capability `(12, 0)`
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Model: `google/gemma-4-26B-A4B-it`
- Context / reused prefix: `ctx=8185`, `prefix=4096`
- Scored tokens: inherited from the prior packet shape, Wikitext slice
- Baseline: vLLM bf16 mean NLL `7.933360410453398`
- Calibration shape:
  - full-attention layers fixed at `k=0.07`, `v=0.05`
  - late sliding layers `[24-28]` fixed at `k=0.07`, `v=0.05`
  - early+mid sliding blocks `[0-4]`, `[6-10]`, `[12-16]`, `[18-22]` fixed at `v=0.08`
  - early+mid sliding K swept narrowly

## Captured rows

| row | early+mid K | early+mid V | mean NLL | delta vs bf16 | PPL | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `replay_k100_v08` | `0.100` | `0.080` | `7.815396153` | `-0.117964257` | `2478.468612492168` | ok |
| `k102_v08` | `0.102` | `0.080` | `8.136488139` | `+0.203127729` | `3416.89710602089` | ok |
| `k103_v08` | `0.103` | `0.080` | `6.246659276` | `-1.686701135` | `516.2851739467054` | ok |
| `k104_v08` | `0.104` | `0.080` | `6.791663277` | `-1.141697134` | `890.393301455291` | ok |
| `k105_v08` | `0.105` | `0.080` | NA | NA | NA | failed / incomplete |

The `k106_v08` row had started when the stop command was issued, but it did not enter
`summary.tsv` and is not a quality result.

## Interpretation

The prior sub-band run suggested a possible smooth crossing between `k=0.100` and `k=0.110`.
This partial run falsifies that simple interpolation: `k=0.102` already overshoots positive,
while adjacent `k=0.103` and `k=0.104` collapse to strongly negative deltas. The K response is
non-monotonic and sharp enough that a scalar early+mid K refinement is not a reliable path to
a claim-grade full-NVFP4 26B row.

This does not prove full NVFP4 26B is impossible. It does narrow the next useful work away from
smooth scalar interpolation and toward either:

- a targeted per-layer/per-block attribution of the low-NLL collapse source, or
- accepting fp8 KV as the 26B-A4B ship path while preserving full NVFP4 as research.

## Files

- `dx26_krefine_20260615T1910Z/RUN_INFO.txt`
- `dx26_krefine_20260615T1910Z/summary.tsv`
- `dx26_krefine_20260615T1910Z/rows/`
- `dx26_krefine_20260615T1910Z/calib/`
