# Preserved Scratch From Reused Vast Instance `41128851`

Date: 2026-06-16 UTC

Before reusing idle Vast instance `41128851` for the 26B readout capture, I preserved the visible `mp_*` scratch files and setup logs under `mp_scratch_41128851.tgz`.

This is not interpreted evidence and should not be cited as a result. The visible `mp_result.txt` on the instance contained only headings and blank row values:

```text
=== MIXED-PRECISION 26B (HF truth=7.9923) ===
bf16:
nvfp4 all:
nvfp4 +L0-4 fp8:
DONE_MP
```

The archive is retained only so the instance could be safely destroyed after the readout-capture run without silently discarding scratch state.
