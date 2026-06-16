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

**Sub-band sweep verdict:** still RED, but now bracketed. Ran the packet on Vast instance `41077335` and
destroyed the instance after artifact pull. No single/pair/drop-one block subset beat the previous
early+mid-all-four row (`-0.117964257`). The decisive new point is scalar: all four early+mid blocks at
`0.11/0.08` overshoot to `+0.162258904`, while `0.10/0.08` is low by `-0.117964257`. The target is therefore
bracketed near `k≈0.104` at `v=0.08` for early+mid sliding, with late sliding and full layers left at
`0.07/0.05`. `0.10/0.09` and `0.09/0.08` collapse badly. One row (`0.10/0.07`) failed from an HF Hub
processor-list timeout during vLLM startup, not from model quality. Artifact:
`results/vast_26b_subband_20260615T1815Z/summary.md`.

**Next discriminator: narrow early+mid K refinement.** Added
`docs/vast_anchor/run_26b_subband_krefine_sweep.sh`, a wrapper over the sub-band packet. It fixes early+mid
`v=0.08`, late/full at `0.07/0.05`, and sweeps early+mid K through `0.102,0.103,0.104,0.105,0.106,0.108`,
with `0.100` and `0.110` replay rows. This directly tests the bracket implied by the previous run.

**K-refinement partial-stop verdict:** the smooth crossing hypothesis is falsified. The run was interrupted
after `k105_v08`, but the captured rows are enough to show the scalar surface is not locally smooth:
`k=0.100` replays the prior best at `-0.117964257`, `k=0.102` overshoots to `+0.203127729`, then adjacent
`k=0.103` and `k=0.104` collapse to `-1.686701135` and `-1.141697134`. `k105_v08` failed/incompleted and
`k106_v08` was killed mid-row during the stop. Artifact:
`results/vast_26b_krefine_20260615T1910Z_partial_stop/summary.md`.

Implication: do not treat the `0.10`→`0.11` bracket as a tunable one-dimensional calibration. If full NVFP4
26B continues, the next useful discriminator is targeted attribution of the collapse source
(per-layer/per-position logits or K/V contribution), not another broad scalar K grid. The current claim-grade
ship path for 26B-A4B remains fp8 KV.

**Next packet staged:** `docs/vast_anchor/run_26b_position_attribution.sh` runs a bf16 per-token logprob
baseline plus selected NVFP4 rows (`k=0.100`, `0.102`, `0.103`, `0.104` at the same early+mid/late/full
layer map) and emits `delta_report.tsv` plus top per-position delta JSON. It is intentionally an attribution
packet, not another calibration search: the expected answer is whether the low-NLL collapse is localized to
specific score-position bands/tokens or spread across the continuation.

**Attribution packet first attempt (2026-06-15 UTC):** first live attempt on Vast instance `41087840` produced
no quality row, but follow-up log review shows this was likely stopped too early rather than a proved startup
blocker. Prior successful bf16 rows sometimes spent 5-7 minutes at the same Gemma 4 encoder-cache profiling
line before continuing. Artifact: `results/vast_26b_position_attr_attempts_20260615T1902Z/summary.md`. The
packet was revised back to the normal matched-anchor launch mode; the next live run should allow at least 10
minutes after model load for the first bf16 row before judging it stuck.

**Short position-attribution discriminator (2026-06-15 UTC):** completed on Vast instance `41096054` after a
failed low-RAM attempt (`41093069`) showed FlashInfer `fused_moe_120` JIT can exit `137` on ~84 GiB host RAM.
The successful run used the same vLLM/FlashInfer stack with `MAX_JOBS=4` for JIT build throttling only.

Rows at `ctx=8185`, `prefix=4096`, 4088 scored tokens:

| row | mean NLL | delta vs bf16 | key result |
| --- | ---: | ---: | --- |
| `bf16` | `7.933360410` | `0` | baseline |
| `k100` | `7.815396153` | `-0.117964257` | prior best reproduced |
| `k103` | `6.246659276` | `-1.686701135` | collapse reproduced |

Bucket deltas show the `k103` collapse is **distributed**, not a single bad token/page/band: `-2.5787`
for relative positions `0-256`, `-2.2502` for `256-1024`, `-1.9097` for `1024-2048`, `-1.3114` for
`2048-3072`, and `-1.1895` through the tail. This sharpens the K-refinement verdict: the scalar
crossing path is not useful, and another narrow K sweep is unlikely to produce a claim-grade row.
If full NVFP4 continues, the next useful work is layer/activation attribution of the systematic low-NLL
shift or a different calibration model. Artifact:
`results/vast_26b_position_attr_short_20260615T1952Z/summary.md`.

**Block attribution discriminator (2026-06-15 UTC):** completed on Vast instance `41102885` (destroyed
after artifact pull) with the same `ctx=8185`, `prefix=4096`, 4088-token scoring setup and the
`g1c9686c61.sm120a` layer-aware wheel. This row tested whether the `k=0.103` collapse comes from one
early/mid sliding block by holding late sliding/full at `0.07/0.05`, holding non-tested early/mid blocks at
`0.100/0.080`, and hotting one five-layer block at a time to `K=0.103,V=0.080`.

| row | hot block | mean NLL | delta vs bf16 |
| --- | --- | ---: | ---: |
| `bf16` | none | `7.933360410` | `+0.000000000` |
| `base_k100` | none; early+mid all `0.100/0.080` | `7.815396153` | `-0.117964257` |
| `e0` | layers `0-4` | `6.125525339` | `-1.807835071` |
| `e1` | layers `6-10` | `7.391867353` | `-0.541493057` |
| `m0` | layers `12-16` | `7.749327096` | `-0.184033314` |
| `m1` | layers `18-22` | `7.833110515` | `-0.100249896` |
| `all_k103` | layers `0-4,6-10,12-16,18-22` | `6.246659276` | `-1.686701135` |

The collapse is now localized: hotting `e0` (`layers 0-4`) alone is even worse than hotting all four
early/mid blocks. Bucket deltas remain negative across the whole scored suffix (`e0` tail still
`-1.461658259`), so this is not one bad token or page. The next useful discriminator is a layer-level split
inside `e0` or an activation/logit capture around the first sliding stack. A broad scalar K sweep remains
the wrong search. Artifact: `results/vast_26b_block_attr_20260615T2110Z/summary.md`.

**e0 layer-attribution resume (2026-06-15 UTC):** completed the interrupted split of layers `0-4` on Vast
instance `41113137` (destroyed after artifact pull), merging its `l1-l4` rows with the prior stopped
baseline. Same `ctx=8185`, `prefix=4096`, 4088 scored-token setup and `g1c9686c61.sm120a` layer-aware wheel.

| row | hot layer(s) | mean NLL | delta vs bf16 |
| --- | --- | ---: | ---: |
| `bf16` | none | `7.933360410` | `+0.000000000` |
| `base_k100` | none | `7.815396153` | `-0.117964257` |
| `e0_all` | `0-4` | `6.125525339` | `-1.807835071` |
| `l0` | `0` | `6.853853891` | `-1.079506519` |
| `l1` | `1` | `6.806950847` | `-1.126409564` |
| `l2` | `2` | `7.131197863` | `-0.802162547` |
| `l3` | `3` | `7.444459589` | `-0.488900821` |
| `l4` | `4` | `7.384541935` | `-0.548818476` |

Verdict: the first sliding block collapse is not a single bad layer. Layers `0` and `1` are largest, but all
five layers move the score materially, and the all-hot `0-4` row is worse than any individual layer. Bucket
deltas remain negative across the scored suffix for every single-layer hot row. Next branch should be
activation/logit attribution around layers `0-4` or a different early-sliding calibration model, not another
broad scalar sweep. Artifact: `results/vast_26b_e0_layer_attr_resume_20260615T2330Z/summary.md`.

**Next packet staged: readout/top-logprob attribution.** Added
`docs/vast_anchor/vllm_toplogprob_attribution.py` and
`docs/vast_anchor/run_26b_toplogprob_attribution.sh`. This is the lowest-risk readout discriminator before
adding heavier hidden-state hooks: it uses vLLM `prompt_logprobs` top-k output to compare bf16 against selected
NVFP4 first-block rows at the same supplied-token positions. The default row set is `base_k100`, `e0_all`,
`l0`, and `l1`, with `prompt_logprobs=20`, dense top-k capture for the first 256 scored positions, and stride
16 sampling through the rest of the suffix. The artifact emits `summary.tsv` plus
`toplogprob_delta_report.tsv`, bucketed by scored-position range with target-NLL delta, top-k Jaccard overlap,
top-1 match rate, top-1 logprob delta, and top-k mass delta. This packet is staged only; it is not a result
until run on a Vast sm120 box.

**First live attempt note (2026-06-16):** no quality row produced. The initial run without
`--skip-mm-profiling` stalled at Gemma multimodal encoder-cache profiling; the packet now defaults
`SKIP_MM_PROFILING=1` and emits the vLLM proof line `Skipping memory profiling for multimodal encoder and
encoder cache`. The retry reached that proof line, but the Vast host stopped/exited before `bf16` completed
and could not be restarted to recover artifacts. Artifact:
`results/vast_26b_toplogprob_attempt_20260616T0029Z/summary.md`.

**Follow-up packet staged: hidden/readout capture.** The public `prompt_logprobs` top-k packet answers how the
visible distribution changes, but not whether the amplifier is already present in the final hidden states or
only appears after the lm_head/readout projection. Added the opt-in packet:

- `docs/vast_anchor/vllm_readout_capture_sitecustomize.py`
- `docs/vast_anchor/compare_readout_captures.py`
- `docs/vast_anchor/run_26b_readout_capture.sh`
- `docs/vast_anchor/launch_26b_readout_capture_live.sh`

It wraps `Gemma4ForCausalLM.compute_logits()` through `sitecustomize`, skips one-row sample/decode calls, and
saves sampled final hidden rows plus raw top-k logits for each prompt-logprob chunk. It can also wrap
`Gemma4DecoderLayer.forward()` for layers `0-4` and capture the same sampled rows at layer input, attention
output, MLP output, router logits, MoE output, and layer output. The comparators align bf16 vs selected NVFP4
rows by `(compute_logits call, local row)` and `(layer, layer call, local row)` and report hidden
cosine/rel-L2, hidden norm deltas, router top-1/top-k stability, top-1 match, top-k Jaccard, and logit max/LSE
deltas. This should be the next live Vast run after the hardened top-logprob packet if the first
distribution-level result still points at a readout collapse. No live result exists yet; current blocker
remains Vast allocation credit.

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

## Stop-point: top-logprob packet interrupted before quality data (2026-06-16)

A later Vast allocation succeeded and the top-logprob packet was relaunched on RTX PRO 6000 Blackwell
(`sm_120`) with the `g1c9686c61.sm120a` wheel. It reached Gemma 4 26B-A4B model load and the intended
`--skip-mm-profiling` path, then was stopped by operator request while compiling FlashInfer `fused_moe_120`
JIT objects.

No bf16 or NVFP4 row completed, no prompt-logprob JSON exists, and the empty `toplogprob_delta_report.tsv`
must not be treated as a red quality row. Artifact:
`results/vast_stop_26b_toplogprob_20260616T014002Z/summary.md`. The Vast instance was destroyed after pulling
the stop artifact.

## Top-logprob attribution result: broad distribution drift (2026-06-16)

Completed the hardened top-logprob packet on Vast RTX PRO 6000 Blackwell Server Edition (`sm_120`) with the
`g1c9686c61.sm120a` wheel and FlashInfer ref `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`. Same scoring setup:
`ctx=8185`, `prefix=4096`, `prompt_logprobs=20`, sampled positions `496`.

| row | mean NLL | delta vs bf16 | PPL | sampled positions |
| --- | ---: | ---: | ---: | ---: |
| bf16 | `7.933360410` | `+0.000000000` | `2788.782530` | `496` |
| NVFP4 `base_k100` | `7.815396153` | `-0.117964257` | `2478.468612` | `496` |

Capacity proof lines on the same settings:

| row | KV cache tokens | max concurrency @ 8192 |
| --- | ---: | ---: |
| bf16 / auto KV | `183,468` | `22.40x` |
| NVFP4 `base_k100` | `652,335` | `79.63x` |

The NVFP4 target-token NLL improvement is not a correctness signal; it is the same broad calibration bias seen
in the layer-band rows. Distribution metrics confirm broad drift:

| bucket | target delta | top-k Jaccard | top-1 match |
| --- | ---: | ---: | ---: |
| `0-256` | `-0.248220337` | `0.541004748` | `0.691406250` |
| `256-1024` | `-0.138702851` | `0.507458563` | `0.708333333` |
| `1024-2048` | `-0.070406208` | `0.507187984` | `0.546875000` |
| `2048-3072` | `-0.152492993` | `0.546192486` | `0.718750000` |
| `3072-end` | `-0.082599317` | `0.503665190` | `0.671875000` |

This argues against a single bad token/page and supports the already-staged hidden/readout capture packet as
the next discriminator: find whether the drift is already present in final hidden states, appears at lm_head
readout, or starts earlier around the first sliding MoE/router stack. Artifact:
`results/vast_26b_toplogprob_attr_20260616T021036Z/summary.md`.

## Readout/layer capture result: drift is pre-readout, router is not primary (2026-06-16)

Completed the readout/layer capture packet for bf16 vs NVFP4 `base_k100` on a Vast RTX PRO 6000 Blackwell
Max-Q Workstation Edition (`sm_120`) with the same `g1c9686c61.sm120a` wheel and FlashInfer ref
`1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`. This run captured 4 `compute_logits()` readout calls and 25
layer phase payloads per row for layers `0-4`.

Mean NLL reproduced the top-logprob packet:

| row | mean NLL | delta vs bf16 | readout calls | layer calls |
| --- | ---: | ---: | ---: | ---: |
| bf16 | `7.933360410` | `+0.000000000` | `4` | `25` |
| NVFP4 `base_k100` | `7.815396153` | `-0.117964257` | `4` | `25` |

Readout capture shows the drift is already in the final hidden state before lm_head:

| bucket | hidden cosine | hidden rel-L2 | logits top-1 match | logits top-k Jaccard |
| --- | ---: | ---: | ---: | ---: |
| `all` | `0.965832461` | `0.196514476` | `0.771634615` | `0.686271818` |
| `1024-end` | `0.951197733` | `0.288501285` | `0.591836735` | `0.511623651` |

This falsifies the "clean final hidden state, readout-only amplification" branch.

Layer capture caveat: raw `layer_capture_report.tsv` includes many zero/zero rows, and PyTorch reports
`cosine_similarity(0,0)=0`, so quote `layer_capture_nonzero_summary.tsv` for cosine. Nonzero summary:

| layer | phase | cosine | rel-L2 |
| ---: | --- | ---: | ---: |
| `0` | input | `1.000000461` | `0.000000000` |
| `0` | attention_output | `0.998060661` | `0.060645558` |
| `0` | moe_output | `0.992550026` | `0.085261208` |
| `1` | attention_output | `0.994595091` | `0.101807841` |
| `1` | moe_output | `0.993538929` | `0.107493976` |
| `2` | attention_output | `0.992789651` | `0.115578366` |
| `2` | moe_output | `0.992186355` | `0.115205889` |
| `3` | attention_output | `0.991944402` | `0.121874186` |
| `3` | moe_output | `0.990279242` | `0.126473372` |
| `4` | attention_output | `0.993190454` | `0.111771240` |
| `4` | moe_output | `0.986166848` | `0.141605817` |
| `4` | output | `0.997235950` | `0.062629446` |
| `4` | router_logits | `0.999563826` | `0.026655196` |

Layer-0 input is identical, and perturbation appears first at layer-0 attention output. MoE outputs show
larger rel-L2 perturbations through layers `0-4`, while router logits remain comparatively stable; this does
not currently look like a primary expert-routing flip. Since layer-4 output rel-L2 is still only `~0.063` but
final hidden rel-L2 is `~0.197`, the next discriminator should extend layer capture deeper or introduce a
mixed-K/FP8-K control to isolate K/V perturbation from downstream accumulation. Artifact:
`results/vast_26b_readout_capture_20260616T025500Z/summary.md`.

## Deep layer capture result: drift accumulates through the middle stack (2026-06-16)

Completed the deeper capture requested by the previous branch on a fresh Vast RTX PRO 6000 Blackwell Max-Q
Workstation Edition (`sm_120`) with the same `g1c9686c61.sm120a` wheel and FlashInfer ref
`1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`. The run used the Ubuntu 22 CUDA 13 container
`nvidia/cuda:13.0.1-devel-ubuntu22.04`, `ctx=8185`, `prefix=4096`, bf16 vs NVFP4 `base_k100`, and requested
layers `0,4,8,12,16,20,24,28,32,36,40`. The hook emitted layers `0,4,8,12,16,20,24,28`.

Mean NLL reproduced the prior result:

| row | mean NLL | delta vs bf16 | readout calls | layer calls |
| --- | ---: | ---: | ---: | ---: |
| bf16 | `7.933360410` | `+0.000000000` | `4` | `40` |
| NVFP4 `base_k100` | `7.815396153` | `-0.117964257` | `4` | `40` |

Nonzero layer summary (`all` bucket; raw report still includes zero/zero cosine rows):

| layer | input rel-L2 | attention rel-L2 | MoE rel-L2 | output rel-L2 | router rel-L2 | router top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.000000000` | `0.060645558` | `0.085261208` | `0.037239009` | `0.023324556` | `0.960937500` |
| `4` | `0.053168669` | `0.111771240` | `0.141605817` | `0.062629446` | `0.026655196` | `0.902343750` |
| `8` | `0.097610840` | `0.191567439` | `0.208625158` | `0.120631172` | `0.051521078` | `0.886718750` |
| `12` | `0.154590047` | `0.316604443` | `0.422641794` | `0.162862977` | `0.084291879` | `0.808593750` |
| `16` | `0.208417865` | `0.313635364` | `0.468606157` | `0.204482948` | `0.118675822` | `0.687500000` |
| `20` | `0.323810218` | `0.270760557` | `0.385326693` | `0.297251057` | `0.119151612` | `0.671875000` |
| `24` | `0.368327853` | `0.255562018` | `0.409718721` | `0.385122949` | `0.119406001` | `0.710937500` |
| `28` | `0.400962559` | `0.270923220` | `0.371406624` | `0.380914924` | `0.192313564` | `0.792968750` |

This closes the narrow "layer 4 to final hidden" gap: the drift accumulates through the middle/deep decoder
stack. It is not readout-only, and it is not a primary layer-0 router flip. Router logits are initially stable,
but by the middle stack router top-1 match is only `0.67-0.71` around layers `16-24`, so routing becomes a
downstream casualty of accumulated hidden drift.

The next useful discriminator is an FP8-K or mixed-K deep capture at the same layer taps. That should separate
K-driven attention drift from V/downstream MoE amplification before any more full-NVFP4 calibration tuning.
Artifact: `results/vast_26b_deep_layer_capture_20260616T034500Z/summary.md`.

## Active-KV mix attribution partial result (2026-06-16)

Staged and ran a new active-page FlashInfer capture packet:

- `docs/vast_anchor/active_kv_capture_sitecustomize.py`
- `docs/vast_anchor/compare_active_kv_mixed_ref.py`
- `docs/vast_anchor/run_26b_active_kv_mix_probe.sh`
- `docs/vast_anchor/launch_26b_active_kv_mix_probe_live.sh`

The packet captures only active request pages from FlashInfer prefill calls, allowing offline references with
bf16 K + NVFP4 V and NVFP4 K + bf16 V. Live run on Vast instance `41139265` reproduced the standard row:
bf16 NLL `7.933360410`; NVFP4 `base_k100` NLL `7.815396153` (`-0.117964257`); bf16 KV cache `183,455`
tokens; NVFP4 KV cache `652,291` tokens. The instance was destroyed after derived artifacts were pulled.

First comparator verdict is **partial**:

| call | q shape | KV tokens | NVFP4 K+V rel-L2 | bf16 K + NVFP4 V rel-L2 | NVFP4 K + bf16 V rel-L2 | dominant |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `5` | `(4096, 16, 512)` | `4096` | `0.117013252` | `0.102539937` | `0.072383685` | V |
| `6` | `(4096, 16, 512)` | `4096` | `0.113974661` | `0.098553602` | `0.070764879` | V |

For the captured global `D=512` calls, V quantization contributes more local attention-output drift than K
quantization. This does NOT close the full failure, because calls `0-4` and `7` (sliding/local shape) were
captured but skipped by the first comparator due to an unrecognized bf16 active-K/V layout. The scripts were
hardened after this run: default capture target is now `qo_len=4096`, large q tensors are saved by default,
and bf16 K/V layout detection tries tuple, packed `[2,P,T,H,D]`, packed `[P,2,T,H,D]`, and generic active
pairs. Next live rerun should produce the sliding-layer K-vs-V split.

Artifact: `results/vast_26b_active_kv_mix_20260616T053600Z/summary.md`.

**Active-KV mix rerun verdict (2026-06-16): sliding split completed.** Re-ran the hardened packet on Vast
instance `41143506` with the same `ctx=8185`, `prefix=4096`, bf16 vs NVFP4 `base_k100` setup, then destroyed
the instance after pulling derived artifacts. Mean NLL reproduced again: bf16 `7.933360410`, NVFP4
`7.815396153` (`-0.117964257`). Capacity proof lines: bf16 `183,556` tokens (`22.41x`), NVFP4 `652,652`
tokens (`79.67x`). All eight active-KV calls now produced mixed-reference rows:

| call | shape | window | all-NVFP4 rel-L2 | bf16K+NVFP4V rel-L2 | NVFP4K+bf16V rel-L2 | dominant |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `(4096,16,256)` | `1023` | `0.099016505` | `0.072167768` | `0.067827347` | mixed |
| `1` | `(4096,16,256)` | `1023` | `0.106627960` | `0.076131275` | `0.075108312` | mixed |
| `2` | `(4096,16,256)` | `1023` | `0.143606733` | `0.091123269` | `0.111562698` | mixed |
| `3` | `(4096,16,256)` | `1023` | `0.128243660` | `0.083340976` | `0.100394375` | mixed |
| `4` | `(4096,16,256)` | `1023` | `0.141596498` | `0.098104365` | `0.123338749` | K |
| `5` | `(4096,16,512)` | `-1` | `0.117013252` | `0.102539944` | `0.072383692` | V |
| `6` | `(4096,16,512)` | `-1` | `0.113974661` | `0.098553602` | `0.070764886` | V |
| `7` | `(4096,16,256)` | `1023` | `0.159155129` | `0.403632046` | `0.259386900` | V |

Interpretation: the first sliding calls are not a clean K-only failure. Calls `0-3` are mixed, call `4` is
K-dominant, global D512 calls `5-6` are V-dominant, and late sliding call `7` is strongly V-dominant with
nonlinear one-sided substitution behavior. This weakens "FP8-K alone fixes the whole row" as a complete
diagnosis, but it is still local attention-output attribution rather than a serving fix. Full 26B-A4B
NVFP4 K+V remains RED/open. Artifact:
`results/vast_26b_active_kv_mix_20260616T061800Z/summary.md`.

**Active-KV gain probe verdict (2026-06-16): simple output gain is not the full-NVFP4 fix.** Extended
`compare_active_kv_mixed_ref.py` to fit both a scalar output gain and a per-query-head output gain between
the all-NVFP4 local attention output and the bf16 K/V baseline. Re-ran the same active-KV packet on Vast
instance `41146434` with `TAR_ARTIFACT=0` so the raw 1.9 GB tensor tree was not compressed or retained. NLL
and capacity reproduced exactly: bf16 `7.933360410`, NVFP4 `7.815396153`; bf16 `183,556` tokens (`22.41x`),
NVFP4 `652,652` tokens (`79.67x`). Gain correction barely moved rel-L2:

| call | raw rel-L2 | scalar-gain rel-L2 | head-gain rel-L2 | scalar alpha | head alpha mean | dominant |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `0.099016505` | `0.098901258` | `0.098809141` | `0.995235630` | `0.993790388` | mixed |
| `1` | `0.106627960` | `0.106556877` | `0.106423980` | `0.996077763` | `0.994708300` | mixed |
| `2` | `0.143606733` | `0.143543869` | `0.143293198` | `0.995726836` | `0.994494438` | mixed |
| `3` | `0.128243660` | `0.128093007` | `0.127926136` | `0.993778952` | `0.994014800` | mixed |
| `4` | `0.141596498` | `0.141566997` | `0.140974324` | `0.997069158` | `0.996063054` | K |
| `5` | `0.117013252` | `0.116943320` | `0.116828023` | `1.004112546` | `1.002448916` | V |
| `6` | `0.113974661` | `0.113761235` | `0.113608030` | `1.007080891` | `1.006154180` | V |
| `7` | `0.159155129` | `0.158993903` | `0.158653305` | `0.992788567` | `0.990892291` | V |

Implication: the residual is directional/structural, not a uniform output-amplitude error. A read-side scalar
or per-head V/output gain is not a viable full-NVFP4 repair. This supports the mixed whole-layer fp8 ladder as
the practical serving route, while full NVFP4 remains a richer research problem. Artifact:
`results/vast_26b_active_kv_gain_20260616T064900Z/summary.md`.
