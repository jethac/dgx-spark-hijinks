# 4-box parallel vLLM Gemma-4 anchor ladder on vast.ai

**Trigger:** Codex delivers a clean x86 sm_120 vLLM image (green sanity on sm_120 —
"Paris" + wikitext NLL ~2–3). Then this fans out 4 vast.ai sm_120 boxes in parallel to
produce the full vLLM anchor ladder + the +0.403 triangulation, in one ~40-min window.

## Safety gate (do first, 1 box, ~5 min)
Before spending on 4 boxes, run the **green-gate** on a single cheap 5090:
`docker run <IMAGE>` → `gen_test.py`. Require: generation coherent ("Paris"),
wikitext mean NLL ~2–3, sane top-1 prefill. **Only fan out if green.** A broken image
× 4 is the failure mode we are avoiding.

## Sharding (4 boxes, parallel)

| box | rung | model | GPU (filter) | shape | env extras |
|---|---|---|---|---|---|
| 1 | 12B matched | `google/gemma-4-12B-it` | RTX 5090 32GB | ctx 8185, prefix 4096, ntok 4096 | — |
| 2 | 12B prefix sweep | `google/gemma-4-12B-it` | RTX 5090 32GB | prefix ∈ {0,1024,2048,4096} | — |
| 3 | 31B matched | `google/gemma-4-31B-it` | RTX PRO 6000 96GB | ctx 8185, prefix 4096 | VO-split knobs if head=512 (`VLLM_NVFP4_KV_VOSPLIT=1`,`VLLM_NVFP4_KV_LINEAR_V_SF=1`) |
| 4 | E4B matched | `google/gemma-4-E4B-it` | RTX PRO 6000 96GB | ctx 8185, prefix 4096 | `--enforce-eager`; expect ~20min MoE CUTLASS JIT (warm cache in image if possible) |

Each box, per arm `kv ∈ {bfloat16, nvfp4}` (full-NVFP4 K+V, matching Codex's row):
1. `docker run` the clean image (mount/copy `eval_harness.py`).
2. run both arms → `out_<kv>.json` (mean_nll, ppl).
3. Δ = nll_nvfp4 − nll_bf16. scp results back. **destroy the box.**

## Readout
- Box 1 Δ vs Codex SGLang 0110 (+0.403): vLLM clean ⇒ SGLang-structural; vLLM ≈+0.4 ⇒ inherent NVFP4.
- Box 2: Δ vs prefix length — rising ⇒ the cached-prefix merge path (SGLang-side); flat ⇒ not the merge.
- Boxes 3/4: the 31B and E4B anchor rows for the ladder (match vLLM ⇄ SGLang per size).

## Controller
`run_parallel_ladder.sh <IMAGE_REF>` — provisions, runs, collects, destroys all boxes.
Finalize the 2 marked TODOs against the real image interface (entrypoint + how the harness
is invoked inside the container) once the image exists. Reuses the solved box runbook in
`SM120_NUMERICS_PLAN.md §5` (ssh direct endpoint, key in-memory only, destroy on bank).

## Cost / hygiene
~$2.9/hr combined (2×5090 + 2×PRO6000); ~40-min ladder ≈ $2. Hard rule: **every box
destroyed the moment its JSON is banked** (and an unconditional cleanup sweep at the end:
`vastai show instances` → destroy any stragglers). API key only ever as `VAST_API_KEY` env.
