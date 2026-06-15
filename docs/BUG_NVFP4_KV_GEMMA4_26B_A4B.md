# BUG: Gemma-4-26B-A4B full-NVFP4 KV calibration reachability is still open

Status: OPEN (reader/kernel read path falsified as the root cause on 2026-06-15). NVFP4-specific at the
serving quality level, but **not yet proven uncalibratable**. 26B-A4B ships with **fp8 KV** (correct,
near-lossless) until a per-K/per-V NVFP4 calibration sweep either finds a green point or proves fp8 is the
honest floor; 12B/31B NVFP4 remain unaffected (GREEN, calibrated).

Note: the early sections below preserve the original tied-scale evidence and hypotheses. The
`LOCALIZATION (2026-06-15)` and `READ-CAPTURE VERDICT` sections supersede the earlier "reader bug" claim:
FlashInfer's NVFP4 paged reader faithfully matches dequant+SDPA for 26B. The active question is calibration
reachability, not a reader fix.

## Evidence (vast PRO 6000 / GB202 / sm_120, `google/gemma-4-26b-a4b-it`, ctx 8185 / prefix 4096, wikitext, 4088 scored tokens)

HF transformers eager bf16 is the dtype/kernel-independent ground truth.

| KV dtype | mean NLL (nats) | delta vs HF truth | verdict |
| --- | ---: | ---: | --- |
| **HF eager bf16 (TRUTH)** | **7.9923** | — | reference |
| vLLM bf16 | 7.9027 | -0.090 | correct |
| vLLM **fp8** | 7.7903 | **-0.202** | **correct (near-lossless) — SHIP PATH** |
| vLLM nvfp4 (k=v=0.05) | 7.3054 | -0.687 | broken |
| vLLM nvfp4 (k=v=0.07) | 5.7998 (x2, deterministic) | **-2.193** | broken |
| vLLM nvfp4 (k=v=0.10) | 7.3691 | -0.623 | broken |

## Diagnosis

1. **vLLM bf16 is correct** — matches HF truth to 0.09 nats. (The degenerate greedy chat smoke,
   `"the capital of Japan is the capital of Japan is..."`, is the raw-prompt-on-an-instruct-model
   artifact, NOT a serving bug — bf16 PPL proves the serving is sound.)
2. **nvfp4 scores BELOW truth at every scale** (-0.6 to -2.2 nats). Lower-than-truth NLL is physically
   impossible for a correct model, so the nvfp4 path is degrading attention into a repetition/
   high-frequency collapse: low *local* per-token NLL on a partly-predictable corpus, but globally
   broken generation (the 0.07 smoke loops `"Wait, I'm not sure."`). Deterministic (0.07 gave 5.7998
   twice). No global scale recovers truth → NOT a calibration problem.
3. **fp8 KV is correct** (-0.20 vs truth, normal quantization). Since fp8 uses the SAME paged-attention/
   MoE serving path and only differs in the KV dtype/dequant, the break is **NVFP4-SPECIFIC**, not MoE
   and not the general quantized-KV path. The MoE-pool hypothesis is weakened: MoE would break fp8 too.

## Scope / what's special about 26B-A4B (config analysis, 2026-06-15)

Fetched + compared the three text_configs. The attention path is NOT the differentiator:

| | heads | kv_heads | GQA ratio | head_dim | text rope (sliding/full) | layer_types | MoE |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 12B (nvfp4 OK) | 16 | 8 | 2 | 256 | 10000 / 1000000 | 5:1 global | dense (inter 15360) |
| 31B (nvfp4 OK) | 32 | 16 | 2 | 256 | 10000 / 1000000 | 5:1 global | dense |
| **26B-A4B (broken)** | **16** | **8** | **2** | **256** | **10000 / 1000000** | 25 sliding + 5 full | **MoE (inter 2112/4304)** |

12B and 26B-A4B have **identical** attention head geometry (16/8/256, GQA 2), identical text rope, same
head_dim_qk=512 VO-split path. RULED OUT: GQA-ratio, head geometry, rope (the `rope_theta=100` is in
`vision_config`, not the decoder — does not touch the KV cache). The ONLY architectural difference is
**MoE** (26B-A4B experts vs 12B dense MLP).

Since fp8 KV works on 26B through the same MoE+attention path and only nvfp4 breaks, this is NOT an
attention-kernel-geometry bug. **WHY nvfp4 specifically breaks is UNRESOLVED** — two candidate causes:

(A) **4-bit quantizability limit**: 26B-A4B's K/V activations have per-channel outliers that exceed what
nvfp4's per-16-block e4m3 scale can represent (clip/underflow), while fp8's wider per-element range copes.

(B) **An nvfp4-path kernel bug** (numerical pathology), with 4-bit precision NOT actually the limiter.

**The evidence currently LEANS (B), not (A)** — earlier writeups (and SOLUTIONS_STATUS) over-claimed (A);
this is the corrected reading:
- nvfp4 scores BELOW the bf16/HF truth at every scale (overconfident), with degenerate generation (a 0.07
  smoke loops "Wait, I'm not sure"). A precision/clipping limit makes predictions WORSE (higher NLL) and
  keeps them roughly coherent; below-truth + degenerate = the attention COLLAPSING (wrong-but-peaked
  output) = a kernel signature, not bits running out.
- It is NON-MONOTONIC in the global scale: _k=0.05→7.31, _k=0.07→**5.80** (worst), _k=0.10→7.37. A genuine
  clip/underflow limit moves monotonically with the scale; a sharp dip at one middle scale is a numerical
  pathology, not a precision ceiling.

DISCRIMINATING EXPERIMENT (task #55): capture the real 26B K/V, quantize->dequantize in numpy (the E2M1 +
per-16-block e4m3 simulator), measure round-trip L2/max-error vs 12B. **Large round-trip error -> (A)
quantizability. Small round-trip error but still-broken serving -> (B) the cache bytes are fine and the
kernel mishandles them -> kernel bug.** Until resolved, BOTH point to the same interim ship:
**26B-A4B uses fp8 KV** (vLLM) / bf16 (SGLang, where fp8 D512 is also blocked). If (B), a kernel fix could
restore full-nvfp4 26B on both stacks; if (A), 26B needs outlier-aware KV quant. fp8 is the defensible
permanent answer.

**Task #55 result (2026-06-15, Codex vast PRO 6000): points to (B).** Kernel-free HF eager
`past_key_values` capture at `ctx=2048`, tail `512` tokens, layers `0,1,2,12,24,last`, then NVFP4
quantize/dequantize simulation:

| model | tensor | n | mean best rel-L2 | max best rel-L2 |
| --- | --- | ---: | ---: | ---: |
| 12B | K | 6 | 0.093636 | 0.095294 |
| 12B | V | 6 | 0.092361 | 0.093938 |
| **26B-A4B** | **K** | **6** | **0.093285** | **0.095096** |
| **26B-A4B** | **V** | **6** | **0.092020** | **0.093281** |

Artifact: `results/20260615_vast_26b_kv_roundtrip/`. 26B-A4B's actual K/V tensors are not less
NVFP4-representable than 12B's in this discriminator. Treat the remaining full-NVFP4 serving failure as
a kernel/serving/feed bug unless a deeper sample falsifies this.

## Localization progress (2026-06-15, narrowing the kernel bug)

Ruled out, in order: (a) **quantizability** — round-trip rel-L2 identical to 12B (task #55, Codex 0178);
(b) **writer global-scale + stored SF bytes** — 884k fp8 SF bytes, 0 mismatch vs the NVFP4 recipe (Codex
0180); (c) **V-SF layout** — 512 layers require linear (got it), 256-layers-linear is proven good on 12B
(Claude 0179); (d) **SWA-crossing / long-ctx feed** — the ctx sweep below shows the break is present
WITHIN one 1024 window (Claude 0181).

ctx sweep (FIXSCALE k=v=0.1, bf16=per-ctx truth): ctx 512 (within window) nvfp4-bf16 = **-0.168**;
ctx 2048 = -0.712; ctx 8185 = -0.534. 12B at this config is **+0.024** (opposite sign). Chat smoke at
k=0.1 is COHERENT ("Tokyo"); at k=0.07 it degenerates — so it's a **scale-sensitive, systematically-
low-entropy READ bias**, present from the first cached read, not random corruption and not SWA-feed.

What remains: the **paged nvfp4 READ/dequant for 26B**. The only invariant tracking the break is **MoE**
(31B = same 512-VO-split read = green; 12B = same 16q/8kv head count = green). Since the read is
attention-only, leading hypothesis: the nvfp4 KV code path enables a fusion/custom-op/reader config that
misbehaves only with the MoE model (fp8 KV takes a different path, clean). Next probe: compare FlashInfer
nvfp4 attention output to a faithful dequant+SDPA reference on the same cached pages+Q at one layer.

## LOCALIZATION (2026-06-15): reader is faithful; what's left is calibration-reachability

Two captures on the e3 stack (vast PRO-6000), exact serving `wrapper.run()` I/O via
`docs/vast_anchor/sitecustomize.py`, ctx 512 scoring prefill, 26B vs 12B:

**(1) Reader faithfulness — `compare_fi_vs_ref.py` (SOLID).** Dequantized the SAME cached nvfp4
pages (validated roundtrip-probe math) + a faithful end-aligned-causal/SWA SDPA reference, vs the
FlashInfer output, every captured layer. Result: cosine **1.00000**, mean rel-err ~0.2%, max-abs
~0.1 on one element of 2M — identical residual for 26B and 12B, sliding AND global layers. **The
FlashInfer nvfp4 paged READER is mathematically correct for 26B. Not a kernel/reader bug.**
This overturns the earlier "paged nvfp4 READER math bug" hypothesis.

**(2) Pure quant perturbation — `compare_bf16_vs_nvfp4.py` (SOLID, but read carefully).** Layer-0
q is KV-independent, so it is IDENTICAL across a bf16 run and an nvfp4 run (cosine 1.000003,
max-abs 0.0000); layer-0 attention-output difference is then the PURE nvfp4 KV perturbation. It is
**~12.8% for 26B vs ~10.4% for 12B — essentially the same.** 26B's in-situ K/V is NOT meaningfully
less nvfp4-representable than 12B's (consistent with the round-trip rel-L2 == 12B from task #55).

**(3) Full-depth attention drift (CORRECTS an earlier over-claim).** An interim 8-layer read
suggested "26B amplifies the drift, 12B stays flat" -> MoE router amplification. **The full-depth
curve does NOT support that:**

| model | layers | seed L0 | final | amplification | q-traj final cos | q-traj final max-abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26B-A4B | 35 | 12.8% | 29.7% | 2.32x | 0.963 | 12.41 |
| 12B (k=v=0.1) | 56 | 10.4% | 28.7% | 2.75x | 0.964 | 7.06 |

Both trajectories drift to nearly the SAME attention error (~29%) and q-cosine (~0.96). 26B drifts
modestly FASTER per layer (29.7% in 35 layers vs 12B's 28.7% in 56; worst-case q max-abs 12.4 vs
7.1) — a weak MoE-sensitivity signal at most, NOT the clean 10x amplification the 8-layer slice
implied. **So differential attention-trajectory drift does NOT explain 26B's collapse.**

**CONFOUND / what is still genuinely OPEN.** This 12B "control" ran at k=v=0.1, which is 12B's
*degraded* config (12B is GREEN only at calibrated k=0.1, **v=0.06**: +0.024; uncalibrated k=v=0.1
is ~+0.42, itself off). So (3) compares two degraded configs and cannot isolate the real
differentiator, which is **calibration-reachability**: 12B HAS a green (k,v) sweet spot; does 26B?
The only 26B sweep on record is 3 points with k==v (0.05/0.07/0.1, all below-truth degenerate) —
a per-K/per-V 2D sweep has NOT been run. Until it is, "26B nvfp4 is unreachable" is unproven.

**What is settled vs open:**
- SETTLED: not a FlashInfer reader bug; not a quantizability/representability bug; per-layer seed
  perturbation == 12B; attention-trajectory drift ~== 12B at matched scale.
- OPEN: (a) does any (k,v) calibrate 26B near truth? (2D sweep, not yet run); (b) if not, is the
  residual sensitivity in the MoE FFN/router readout (the logits responding to ~29% attention
  drift more than 12B's dense FFN does)? Capture lm_head logits bf16-vs-nvfp4 to test.

## Ship decision (current after the 2D sweep)

- **26B-A4B**: ship **fp8 KV** (near-lossless, correct) as the safe long-context path TODAY. Full NVFP4
  K+V has a short-context calibration point but no claim-grade long-context point in the tested global K/V grid.
- **12B / 31B**: nvfp4 GREEN (calibrated) — dense decoders — unaffected.
- **DiffusionGemma**: 26B-A4B base → same question; fp8 KV safe today, truth-gate any nvfp4 claim.

## 2D calibration sweep verdict (2026-06-15, Codex Vast sm120)

Ran the per-K/per-V calibration sweep on a native sm120 RTX PRO 6000 WS box using the E3 vLLM wheel
`0.1.dev1+ge99078ddf.sm120a` (wheel sha256
`3d92d14d3c6f7f802eb381c6a68f84028978f70ed16055fc046407ca16b4036a`) and FlashInfer source overlay
`1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.

**Short context falsifier:** at `ctx=2048`, `prefix=1024`, the 25-point K/V grid found a near-parity full
NVFP4 point:

| row | NLL | delta vs vLLM bf16 | verdict |
| --- | ---: | ---: | --- |
| vLLM bf16 | 5.164656 | 0 | baseline |
| NVFP4 `k=0.10, v=0.05` | 5.173155 | **+0.008499** | short-context GREEN |
| NVFP4 `k=0.10, v=0.10` | 5.173155 | **+0.008499** | short-context GREEN |

Artifact: `results/vast_26b_2d_calib_20260615T142000Z/summary.md`.

**Long context gate:** the short-context best candidates were then re-run at `ctx=8185`, `prefix=4096`;
both collapsed to the old low-NLL failure:

| row | NLL | delta vs vLLM bf16 | verdict |
| --- | ---: | ---: | --- |
| vLLM bf16 | 7.933360 | 0 | baseline |
| NVFP4 `k=0.10, v=0.05` | 6.293518 | **-1.639843** | RED |
| NVFP4 `k=0.10, v=0.10` | 6.293518 | **-1.639843** | RED |

Artifact: `results/vast_26b_8185_best_20260615T142600Z/summary.md`.

**Full long-context grid:** a direct 25-point `ctx=8185` grid improved the best point but still did not find
a claim-grade full-NVFP4 calibration:

| row | NLL | delta vs vLLM bf16 | verdict |
| --- | ---: | ---: | --- |
| vLLM bf16 | 7.933360 | 0 | baseline |
| best NVFP4 `k=0.07, v=0.05` | 7.545217 | **-0.388143** | RED |
| tied best NVFP4 `k=0.07, v=0.10` | 7.545217 | **-0.388143** | RED |

Artifact: `results/vast_26b_8185_grid_20260615T145900Z/summary.md`.

**Updated decision:** full NVFP4 K+V is calibration-reachable at short context, but not claim-grade for
long context in the tested global-scale grid. For 26B-A4B long-context serving, **fp8 KV remains the honest
ship path**. Full NVFP4 K+V stays open as research, likely requiring context-/layer-aware calibration or
another source of the low-NLL bias beyond one global K scale and one global V scale.

## Next discriminator: layer-type calibration (2026-06-16)

The global-scale falsifier above leaves one directly testable implementation path before declaring the
long-context full-NVFP4 row unreachable: Gemma 4 mixes `sliding_attention` and `full_attention` layers, and
one global `(k_scale, v_scale)` may be fitting the wrong aggregate surface.

Codex added a backward-compatible vLLM calibration extension in `jethac/vllm@spark/hijinks-e3-vllm`
commit `1c9686c61`:

- existing JSON with top-level `k_scale` / `v_scale` still applies globally;
- optional `layer_type_scales` can override `sliding_attention` and `full_attention` separately;
- optional `layer_scales` can override individual layer indices, taking precedence over layer type.

The matching Vast packet is `docs/vast_anchor/run_26b_layer_type_calib_sweep.sh` on the hijinks
`autoresearch/vllm-26b-sm120` branch. It keeps the same `ctx=8185`, `prefix=4096`, Wikitext scoring setup
and emits the same `summary.tsv` / `best.tsv` artifact shape, but each NVFP4 row writes:

```json
{
  "layer_type_scales": {
    "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
    "full_attention": {"k_scale": 0.07, "v_scale": 0.05}
  }
}
```

The patched sm120a wheel build is green:

- GitHub Actions run: `27556321837`;
- release tag: `sm120a-wheels-1c9686c61-layercalib`;
- wheel: `vllm-0.1.dev1+g1c9686c61.sm120a-cp312-cp312-linux_x86_64.whl`;
- sha256: `256f4df758c463d3c9004b4778d72e078e777451090e1282fa1980490b091975`;
- summary: `results/vllm_sm120a_wheel_1c9686c61_layercalib_20260616T0025JST_summary.md`.

This is still a discriminator, not a blessed serving row: the claim-grade long-context status remains RED
until the layer-type sweep actually reaches near-parity against the vLLM bf16 baseline.

**Layer-type sweep verdict:** RED but informative. Ran the packet on Vast instance `41069113` with the
patched wheel, then destroyed the instance after pulling artifacts. The prior global-grid best was reproduced
at `sliding=(0.07,0.05)`, `full=(0.07,0.05)` with delta `-0.388143` nats/token. The best tested layer-type
point improved to `sliding=(0.10,0.08)`, `full=(0.07,0.05)`, NLL `7.785053609`, delta `-0.148306801` vs
vLLM bf16. This is still not claim-grade, but it proves the calibration surface is materially
layer-type-dependent and keeps finer layer/band calibration alive as the next research path. Artifact:
`results/vast_26b_layer_type_20260615T1545Z/summary.md`.

**Next discriminator: sliding-layer band calibration.** Added
`docs/vast_anchor/run_26b_layer_band_calib_sweep.sh`. It uses the same patched wheel and ctx/prefix scoring
setup, keeps the full-attention layers at the best tested `0.07/0.05`, and uses `layer_scales` overrides to
vary early/mid/late sliding bands over the known 26B-A4B layer map: early sliding `[0-4,6-10]`, mid sliding
`[12-16,18-22]`, late sliding `[24-28]`, full layers `[5,11,17,23,29]`. The default packet replays the
layer-type best and then tests which sliding band(s) actually need the higher `0.10/0.08` scale. This is the
next bounded reachability test before deeper per-layer search.

**Layer-band sweep verdict:** RED but improved again. Ran the packet on Vast instance `41072927` and destroyed
the instance after pulling artifacts. The `all_high` replay reproduced the layer-type best exactly
(`-0.148306801`). The best tested band row applies the higher `0.10/0.08` scale only to early+mid sliding
layers, leaving late sliding and all full layers at `0.07/0.05`: NLL `7.815396153`, delta `-0.117964257` vs
vLLM bf16. Still not claim-grade, but the bias is now demonstrably depth-sensitive. Next useful search is a
narrow per-layer/sub-band screen inside early+mid sliding layers, not a broad whole-model grid. Artifact:
`results/vast_26b_layer_band_20260615T1718Z/summary.md`.

**Next discriminator: early+mid five-layer sub-bands.** Added
`docs/vast_anchor/run_26b_subband_calib_sweep.sh`. It splits the winning early+mid sliding region into four
five-layer blocks: `e0=[0-4]`, `e1=[6-10]`, `m0=[12-16]`, `m1=[18-22]`, with late sliding `[24-28]` and full
layers `[5,11,17,23,29]` kept at the base `0.07/0.05`. The default packet replays the prior best
`all_four_high`, tests single/pair/drop-one block combinations, and probes nearby all-four scales
`0.10/0.07`, `0.10/0.09`, `0.09/0.08`, `0.11/0.08`. This is the narrow calibration search implied by the
band result.

## Cross-lane

Codex's SGLang 26B-A4B MoE red may be the SAME nvfp4-specific bug rather than (only) pool-sizing — he
should fp8-vs-nvfp4 his 26B against an HF-eager truth to check (mail 0171). Artifacts: vast
`/root/dx26/*` (hfref2.out, bf16/fp8/nv0{5,7,7b,1}.json + logs).

## e3 re-check (2026-06-15): STILL BROKEN on FlashInfer-main
Re-ran on the epoch-3 stack (vLLM v0.23.0 + FlashInfer main, e3 wheel ge99078ddf), ctx 8185 chunked,
HF truth 7.9923:
| | NLL | vs truth |
| --- | ---: | ---: |
| bf16 | 7.9334 | -0.06 (correct) |
| nvfp4 k=v=0.05 | 6.3877 | -1.60 |
| nvfp4 k=v=0.07 | 7.1592 | -0.83 |
| nvfp4 k=v=0.10 | 6.2935 | -1.70 (chat "Tokyo" coherent) |
The break PERSISTS unchanged on the newer FlashInfer — confirms a real kernel bug, not a version artifact
that main happened to fix. Proceeding to the read-vs-dequant+SDPA capture (mail 0182) on the e3 stack.

## READ-CAPTURE VERDICT (2026-06-15): the FlashInfer nvfp4 READER is NOT the bug
Captured the EXACT serving `BatchPrefillWithPagedKVCacheWrapper.run()` inputs+output for the ctx-512 scoring
prefill, layers 0-7, for 26B AND 12B (control), via `docs/vast_anchor/sitecustomize.py` (wraps run(), saves
q + paged nvfp4 split views k_data/v_data + fp8 block-scales + plan state). Then dequantized the SAME cached
pages (E2M1 LUT x per-16-block fp8 SF x global scale, the validated `nvfp4_writer_roundtrip_probe` math) and
ran a faithful end-aligned-causal + sliding-window SDPA reference (`docs/vast_anchor/compare_fi_vs_ref.py`).

Result — FlashInfer nvfp4 output vs dequant+SDPA reference, per layer:
| model | layers | cosine | mean-abs | ref mean-|.| | rel mean err |
| --- | --- | ---: | ---: | ---: | ---: |
| 26B-A4B | sliding 0-4,7 | 1.00000 | 0.00069-0.00079 | 0.30-0.38 | ~0.2% |
| 26B-A4B | global 5-6 | 1.00001 | 0.00103 | 0.49 | ~0.2% |
| 12B (control) | sliding | 1.00000 | 0.00069-0.00082 | 0.33-0.39 | ~0.2% |
| 12B (control) | global | 1.00002 | 0.00074-0.00081 | 0.35-0.38 | ~0.2% |

**The kernel faithfully reproduces dequant+SDPA over the cached pages for 26B, identical residual to 12B
(pure bf16 rounding, max-abs ~0.1 on one element of 2M, same in both models).** This OVERTURNS the leading
"paged nvfp4 READER math bug" hypothesis (the localization section above). The reader is correct; the bias is
**outside reader math** — it is in the cache CONTENTS (what the writer/quantization stores for 26B's specific
K/V distribution at this global scale) or downstream of attention. Per the goal decision tree this is the
"both match reference -> trace outside reader" branch. Next probe (running): capture a bf16 run and compare
layer-0 attention (q is KV-independent at layer 0, so identical across bf16/nvfp4 runs) between bf16-cache and
nvfp4-cache to measure pure per-layer quantization perturbation for 26B vs 12B.
