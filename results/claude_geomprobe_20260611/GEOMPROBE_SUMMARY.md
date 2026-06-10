# Geometry-truth probes (2026-06-11): Block D CLEARED; GQA hypothesis FALSIFIED

4 weight-free runs, image/source as Block A (fb7d62ea), scripts @ 3e35599.

| run | probe | heads (group) | result |
|---|---|---|---|
| 1 | bf16 vo-split (512,256), e4b geometry | 8/2 (4) | **PASS 0.9999979 — expected FAIL** |
| 2 | bf16 vo-split (512,256), 31b geometry | 32/16 (2) | PASS 0.9999980 |
| 3 | NVFP4 vo-split, 31b geometry, NHD+HND | 32/16 (2) | PASS 0.9999986 both |
| 4 | NVFP4 vo-split, e4b geometry, NHD+HND | 8/2 (4) | PASS 0.9999986 both |

## Consequences
1. **Block D (31B full-NVFP4 serving) is probe-cleared** at real 31B global
   geometry, both layouts. Schedule it (with Block C) in the next rebuild window.
2. **FP4 passes even at GQA group 4** — E4B/12B NVFP4 not kernel-blocked either.
3. **The E2 serving-crash diagnosis in BLOCKE23_SUMMARY.md is FALSIFIED**: the
   exact E4B head geometry passes at probe level. "max_mma_kv: 0" is NOT driven
   by GQA group size alone. Remaining suspects, in likelihood order:
   - the warmup/dummy-run WORKLOAD regime (vLLM warms up with
     max_num_batched_tokens-scale qo_len vs the probe's qo=16; large qo drives
     bigger q-tile selection in the dispatcher);
   - plan-kwargs parity (vLLM passes o_data_type, window_left=-1,
     logits_soft_cap, fixed_split_size/disable_split_kv; probe omits them);
   - split-kv scheduling path differences at prefill.cuh:2964.
   Next red-test iteration: extend the probe with --qo-len/--kv-len/--batch and
   vLLM-parity plan kwargs; sweep the warmup regime until the crash reproduces.
   The measured-geometry rule extends again: match the WORKLOAD and PLAN
   SIGNATURE, not just head shapes.
