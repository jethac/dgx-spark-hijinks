# GOAL (Claude / vLLM lane): DiffusionGemma NVFP4 KV-cache on vLLM (DG-V rungs)

**One-liner:** bring full-NVFP4 K+V KV cache (3.556x, format-exact) to DiffusionGemma
26B-A4B on vLLM/Blackwell (sm_120 + sm_121), to parity with SGLang's DG-R5/R6 receipts.
Closes the gap that the engine with the *official* DiffusionGemma recipe serves it bf16-only.

## Why this exists
vLLM supports DiffusionGemma (official recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)
but in bf16 only -- no NVFP4 KV, no Blackwell, no FlashInfer. We routed ALL DiffusionGemma
work to Codex's SGLang lane and never applied the NVFP4-KV campaign to it on vLLM. DiffusionGemma
is the strongest 4-bit story we have (the decoder re-reads the full prefix every denoise step,
so 3.556x KV capacity compounds harder than anywhere in the AR ladder) -- so it should NOT be
missing from the more-visible vLLM path.

## Reference (do NOT re-derive -- match confirmed-good)
- vLLM AR NVFP4-KV impl (`spark/hijinks-e2-vllm`): NVFP4 writer, `config.py` per-layer
  routing, `flashinfer.py` VO-split. Proven across E2B->31B on Pro 6000 + Spark.
- SGLang DG-R5/R6 (Codex): proves the NVFP4 read path tolerates block-diffusion attention.
  Format is geometry-independent; these are the parity target.
- Upstream vLLM DiffusionGemma recipe: the model class + serving constraints
  (`--max-num-seqs 4`, `--gpu-memory-utilization 0.85`, entropy-bound sampling overrides,
  256-token block denoise loop).

## Gate 0 -- PROBE FIRST, decide before building (the one real unknown)
Does vLLM's DiffusionGemma attention backend expose a hook to the FlashInfer NVFP4 path
the way the AR models do?
- (a) Does the block-diffusion decoder route through the same paged-prefill/decode wrappers
  our VO-split patches?
- (b) Does the bidirectional-within-block mask survive the VO-split two-pass over V halves?

**If no clean hook -> STOP, write the blocker up, escalate. Do not force it.**
If yes -> proceed to Build.

## Build (only past Gate 0 -- wiring, not new kernels)
1. Wire vLLM's DiffusionGemma model class into the existing NVFP4-KV config routing
   (`kv_data_type=uint8`, D=512 VO-split, linear-V-SF knob).
2. DG-specific allocator accounting: the 256-token-block denoise loop re-reads the full
   prefix every step -- confirm the NVFP4 page budget (`9*head/16`) holds under that
   re-read pattern with `--max-num-seqs 4`.

## Per-rung green bar (zero-bug)
- **DG-V5** (= SGLang DG-R5): full-NVFP4 K+V serves coherent generations, **>=3.5x KV
  capacity** vs bf16 denominator, **double-run bitwise-deterministic**.
- **DG-V6** (= DG-R6): perf pair (NVFP4 vs bf16 throughput/latency at matched batch);
  quantify the compounding win from per-denoise-step prefix re-read.
- Both on **sm_120 (Colab Pro 6000) AND sm_121 (Spark)**; bf16 denominator captured each.

## Done
DG-V5 + DG-V6 green on both silicon, receipts banked to `results/`, the two vLLM
DiffusionGemma cells flip (gap -> green). DiffusionGemma is then cross-engine complete and
the FlashInfer surface is proven on the diffusion path in BOTH stacks -- clearing the last
DG-shaped item under the "surface stable across all Gemma variants before filing" PR gate.

## Gate 0 VERDICT (2026-06-12): GO -- lift smaller than assumed
Probe done. DiffusionGemma routes through machinery we already own and proved:
- **Model id** `google/diffusiongemma-26B-A4B-it`; arch `DiffusionGemmaForBlockDiffusion`,
  `model_type: diffusion_gemma`. Block-diffusion MoE (128 experts top-8) + vision,
  30 layers sliding/full alternating, `max-model-len 262144` (256k -> NVFP4 KV pays off hard).
- **It is NOT mainline-plain:** the recipe needs "a vLLM build with diffusion support in the
  Gemma docker image." Upstream vLLM main DOES have `vllm/model_executor/models/diffusion_gemma.py`;
  **our fork (e2-vllm @ e32459eea) predates it** -> port it in, don't author it.
- **diffusion_gemma.py delegates its text decoder to `Gemma4Model`** -- the exact backbone our
  NVFP4 writer + config routing + VO-split already cover. Attention goes through vLLM's standard
  `Attention` + `build_attn_metadata` + `kv_cache_dtype` (NOT a bespoke attention call).
- **The block-diffusion mask is a per-request `causal` flag** in attn_metadata: encoder phase=causal,
  decoder phase=bidirectional (`causal=False`). Standard non-causal prefill -- FlashInfer supports it;
  SGLang DG-R5 already proved the NVFP4 read path tolerates the bidirectional decoder phase.
- **`attention_k_eq_v` wrinkle is ALREADY HANDLED in our backbone:** DiffusionGemma full-attn layers
  have no v_proj (V=K pre-k_norm). Our `gemma4.py` (use_k_eq_v, lines ~431/462/570/626) loads K into
  both K and V qkv_proj slots, so V==K flows automatically and the NVFP4 V-writer sees a normal V.
- **head_dim 256 (16 heads / 8 KV) per config** -> VO-split likely NOT on the critical path here;
  whatever the gemma4 text_config sets, our head-dim-conditional VO-split engages or no-ops itself.

**Revised lift (wiring, confirmed):** (1) port `diffusion_gemma.py` + its config plumbing + the
block-diffusion scheduling (diffusion_states, per-req causal buf, denoise loop) from upstream main
onto our NVFP4-KV branch; (2) register the `DiffusionGemmaForBlockDiffusion` arch; (3) NVFP4 routing
is automatic (Gemma4Model underneath) -- confirm it engages for the new arch name; (4) empirically
confirm the bidirectional (causal=False) decoder phase reads NVFP4 correctly on vLLM (the one genuine
remaining unknown, de-risked by SGLang DG-R5). No new kernel work identified.

## CORRECTION (2026-06-12): this is a RECONCILIATION, not a port
DiffusionGemma-on-vLLM was NOT un-started. Branch `spark/hijinks-e2-dgemma` (local + origin)
already carries:
- the full model stack: `diffusion_gemma.py`, `config/diffusion.py`,
  `transformers_utils/configs/diffusion_gemma.py` (the DG-0 baseline, task #24);
- commit `6fe6d798f` "spark-hijinks epoch-2 patch set on the dgemma branch (DiffusionGemma stack)"
  -- the runner/scheduler/sampler block-diffusion denoise integration;
- commit `dfb427952` "DG-2: per-request causal grouping in the FlashInfer backend + gate lifts"
  -- config.py +25, **flashinfer.py +313**: the NVFP4 diffusion attention path, WIRED but never
  verified-and-banked (no receipts on the branch; task #27 still pending).

It never merged to the integrated NVFP4 line, and it diverged: dgemma is 23 commits off our
HEAD, and the integrated line moved **44 commits** since (AR-ladder completion, MTP, Triton
retirement, mm-prefix). Only `k_eq_v` reached mainline (via the MTP merge) -- which is why the
integrated branch showed DiffusionGemma as a gap.

**So DG-V = reconcile the e2-dgemma DiffusionGemma stack onto the current confirmed-good
integrated NVFP4 HEAD, then verify DG-V5/V6 + bank receipts. NOT a from-scratch port.**

**Crux conflict:** `flashinfer.py` three-way merge -- dgemma's DG-2 causal-grouping (+313) vs
HEAD's 44-commit evolution of the SAME file (VO-split refinements, MTP backend pin, mm-prefix
masking, the max_mma_kv dispatcher fix). Plus `config.py` routing. The model/config files are
clean adds (no conflict).

**Plan:** off a working branch from HEAD: (1) bring the clean model/config files over as-is;
(2) cherry-pick the two MEANINGFUL dgemma commits (6fe6d798f stack + dfb427952 DG-2), discarding
the noise commits (race-fix/cleanup/revert churn), resolving the flashinfer.py + config.py
three-way by hand against the current surface; (3) register the arch; (4) build the wheel;
(5) run DG-V5/V6 on sm_120 + sm_121 and bank. The DG-2 causal grouping is the asset to preserve
through the merge -- it's the diffusion-specific FlashInfer logic, already authored.

## EXECUTION STATUS (2026-06-12)
- Working branch `spark/hijinks-e2-dgv` created off integrated HEAD (e32459eea).
- **Relief found:** HEAD already has the gpu/ v2 runner infra (model_states/, attn_utils,
  sample/, buffer_utils) that diffusion_gemma.py imports -> NO runner subsystem to port.
  Same v2-runner base as upstream PR eb28452b1, so the CLEAN upstream PR is the preferred
  source for shared-file diffusion hunks (over the entangled dgemma branch).
- DONE: landed the 3 clean new files (diffusion_gemma.py 1359L + 2 config files) on e2-dgv.
- NEXT: (a) shared-file diffusion hunks from PR eb28452b1 -> registry, model/config.py routing,
  transformers config registration, model_runner denoise loop, model_states diffusion ModelState,
  scheduler/sampler/cudagraph hooks; (b) replay DG-2 flashinfer causal grouping (dfb427952,
  +313) onto the current flashinfer.py -- the crux, hand-resolved; (c) arch register; (d) build
  sm120a wheel; (e) DG-V5/V6 both silicon + bank.

## CODE INTEGRATION COMPLETE (2026-06-12) -- branch spark/hijinks-e2-dgv @ 52bfe5c34
Pushed to origin. Delta vs integrated HEAD: 53 files, +3500/-317. Three commits:
1. `291621ecb` cherry-pick upstream PR eb28452b1 (DiffusionGemma model + runner denoise
   integration + FA/Triton mixed-causal base) onto our NVFP4 HEAD. Auto-merged almost
   entirely (v2-runner match); only 2 trivial conflicts (docs table + the diffkv triton
   file kept from PR). py_compile clean. Arch `DiffusionGemmaForBlockDiffusion` registered.
2. `e8acd5236` **unify FlashInfer prefill grouping** -- the crux. Merged HEAD's
   FIPrefillMMGroup (image-span packed-mask) + DG-2's FIPrefillCausalGroup (per-request
   causal) into ONE `FIPrefillGroup` keyed by (is_mm, causal). Handles mm-only, causal-only,
   AND composed mm x causal (DiffusionGemma is multimodal -> needs both). Legacy scalar-causal
   no-mm path byte-identical behind `prefill_groups is None`. NVFP4 jit_args + head_dim_vo
   VO-split preserved per group. DG-2 guards + supports_non_causal=True preserved. (Subagent
   authored under precise spec; I reviewed the full diff -- dispatch gate, impl gather/run/
   scatter loop, VO-split-per-group, coherent rename all verified.)
3. `52bfe5c34` knob-gate the DiffusionGemma FLASHINFER allowance (DG-2 routing) -- replaces
   upstream's hard "unsupported" raise with the VLLM_FLASHINFER_VOSPLIT/VLLM_NVFP4_KV_VOSPLIT
   gated path.

**Verified statically:** py_compile clean on all changed Python; coherent rename (grep clean);
legacy AR-ladder path structurally untouched. **NOT yet verified:** anything numeric (no build/GPU).

## REMAINING (build + GPU -- needs marker coordination before touching a card)
1. Build sm120a + sm121a-arm64 wheels off e2-dgv.
2. **Composed mm x non-causal mask GPU cosine** -- the one flagged-untested unification branch
   (causal_base=False). Moot at DG shipping config (canvas 256 <= window 1024) but verify.
3. DG-2 4-request mixed-batch cosine harness re-run on the unified path (sanity vs the prior
   0.999998 baseline).
4. **AR-ladder regression**: re-run a Gemma 4 NVFP4 row (e.g. E4B) to confirm the unification
   didn't disturb the proven path (legacy + grouped both exercised).
5. DG-V5 (full-NVFP4 K+V coherent + >=3.5x + bitwise) + DG-V6 (perf pair) on sm_120 + sm_121.
6. Bank receipts to results/, flip the two vLLM DG cells green.

## VALIDATION PROGRESS (2026-06-12)
- **CI wheels building** off e2-dgv: sm120a (run 27404118729) + sm121a-arm64 (27404118751),
  triggered via push-branch add. ~2-5h compile. These are the clean deployable wheels +
  the proper base for all serve validation.
- **Overlay validated ABI-safe**: only non-Python drift base-wheel(6adc00f70)..e2-dgv is one
  cmake build-config file (vllm_flash_attn.cmake, build-time only); ZERO .cu/.cpp drift. So a
  Python-only overlay (wheel .so + e2-dgv .py at ~/dgv_overlay) is sound for bf16/legacy-path
  checks (the cmake bump only matters for the DG FA-diffkv kernel -> validated on Spark).
- **LIVE IMPORT + REGISTRATION SMOKE GREEN** (P520 WSL2, overlay, GPU-free): vllm imports e2-dgv
  with real _C + flashinfer source; unified `FIPrefillGroup` present + `supports_non_causal=True`;
  `DiffusionGemmaForBlockDiffusion` registers; diffusion configs import. Beyond py_compile --
  confirms the reconciliation loads in a real vLLM runtime (no import-time breakage). P520 GPU
  released (never occupied -- import-only).
- **DEFERRED to the e2-dgv CI wheel + proper harness** (the wheel env lacks pytest + the direct
  1B serve harness; ad-hoc reconstruction risks a false-green): the bf16 AR-ladder PPL regression
  (target: match banked g3-1b FLASHINFER bf16 2.3571850630239095 -> proves legacy path byte-id on
  silicon), the DG-2 4-request cosine, and (Spark, sm_121) DG-V5/V6 nvfp4 coherence.

## WHEEL GREEN + P520 LEGACY-PATH REGRESSION GREEN (2026-06-12)
- Both CI wheels built SUCCESS off e2-dgv @ 98cd3e59f: `sm120a-wheels-98cd3e59f`,
  `sm121a-arm64-wheels-98cd3e59f` (Latest). On-CI, no GPU-box build (per Jetha).
- **P520 (sm_120) bf16 legacy-path regression: GREEN, byte-identical.** Served Gemma 3 1B
  bf16 FLASHINFER on the e2-dgv overlay (VLLM_BUILD_CHECK -> dgv_overlay/vllm; backend proof
  = AttentionBackendEnum.FLASHINFER; ready 99s, no wedge). C1 ctx8191 x2:
  **mean_nll_nats = 2.3571850630239095 (both rows)** == banked baseline 2.3571850630239095
  to 16 figures. Coherent ("The capital of Japan is Tokyo."). => the unified FIPrefillGroup
  refactor does NOT disturb the legacy/AR-ladder serving path on real silicon. Crown-jewel
  safety gate PASSED. Artifacts: ~/g3_1b_retest/results/claude_p520_dgv_fi_bf16_*.

## REMAINING: Spark DG-V5/V6 nvfp4 coherence (sm_121, the headline)
Serve the sm121a-arm64-wheels-98cd3e59f wheel clean on Spark (no overlay -- divergent image
lineage; fresh container + pip install wheel). DiffusionGemma 26B-A4B FLASHINFER + nvfp4 KV +
VO-split knobs. Targets (SGLang DG-R5/R6 parity): coherent (Tokyo/2+2/DGX Spark), full-NVFP4
proof (kv uint8, mixed_kv False), VO-split proof (global head_dim 512 -> vo 256), ~3.56x
capacity vs bf16, double-run bitwise; DG-V6 perf pair. Harness staged:
scripts/run_vllm_dgemma_dgv_spark.sh.

## SPARK DG-V5 BLOCKED ON ARM64-WHEEL PORTABILITY (2026-06-12)
Tried to serve the e2-dgv arm64 wheel on Spark; hit a 3-layer env mismatch retrofitting the
existing r10 container, fully diagnosed:
1. lineage: existing Spark gemma4 image is 9759e3b06 (022 line, divergent) -> can't overlay e2-dgv
   .py (MoE compiled drift). Used the WHEEL instead (clean swap).
2. torch: wheel needs torch 2.12 (CI pin); both Spark images ship torch 2.11. `_C.abi3.so`
   undefined-symbol until I upgraded torch->2.12 in-container (then `_C` loaded).
3. glibc: wheel's `_C_stable_libtorch.abi3.so` needs GLIBC_2.38; both Spark images are
   Ubuntu 22.04 / glibc 2.35. HARD WALL.
ROOT CAUSE: the **sm121a-arm64 CI workflow lacks the glibc gate sm120a has**. sm120a builds on
`ubicloud-standard-30-ubuntu-2204` + fails if any .so needs > GLIBC_2.35; the arm64 workflow
builds on `ubicloud-standard-30-arm` (Ubuntu 24.04), no gate -> glibc-2.38 wheel.
FIX: mirror sm120a in the arm64 workflow -> build on a 22.04 arm runner + add the glibc-2.35
gate, rebuild. Then the glibc-2.35 + torch-2.12 wheel loads in the r10 container after an
in-container `pip install torch==2.12.0 cu130` (proven: `_C` loads once torch matches). Then
serve DG-V5 per scripts/run_vllm_dgemma_dgv_spark.sh. (Alternative: build a full Ubuntu-24.04
arm e2-dgv serving image so no glibc/torch retrofit is needed.) Either way it's a CI/Ubicloud
build, not a Spark build. NOTE: this arm64-wheel glibc bug affects ALL Spark wheel deploys,
not just DG-V -- worth fixing regardless.

## SPARK IMAGE DECISION REVISED (2026-06-12): match r10, don't rebuild from scratch
Jetha: "wouldn't it make more sense to build spark containers that are ubuntu 22.04 + torch 2.11."
YES -- better. The r10 image (22.04/glibc-2.35/torch-2.11/flashinfer-76af7982/transformers-5.11.0)
already serves the full Gemma 4 nvfp4 ladder green. So build the arm64 wheel to FIT r10, then
bake r11 = r10 + the e2-dgv wheel (swap only vLLM; reuse the validated stack). Implemented:
`build-sm121a-arm64-wheel.yml` retargeted to `ubicloud-standard-30-arm-ubuntu-2204` + torch 2.11.0
+ a GLIBC_2.35 gate. Rebuild: run 27409465227. The from-scratch 24.04/torch-2.12 image
(Dockerfile.spark + build-spark-image.yml) is PARKED dispatch-only (the 24.04 build failed anyway;
it's the heavier "full CI-repro" end-state). Wheel matrix: x64/Colab=22.04/torch-2.12 bare;
arm64/Spark=22.04/torch-2.11 baked into r11. Decision mailed to Codex (mail 0089).
NEXT: when the retargeted wheel lands, bake r11 on Spark (r10 + wheel layer-add, no compile),
serve DG-V5 via scripts/run_vllm_dgemma_dgv_spark.sh.

## (superseded) SPARK-READY IMAGE PATTERN (2026-06-12) -- the deterministic fix
Jetha: "adjust our work to build images that work as-is on spark." Implemented in the vLLM
repo (branch e2-dgv):
- `docker/Dockerfile.spark`: self-contained image -- Ubuntu 24.04 (glibc 2.39) + CUDA 13 cu130
  + torch 2.12 + the e2-dgv arm64 wheel + campaign FlashInfer source (JIT) + transformers 5.11.0
  + VO-split env defaults + GPU-free build-time provenance asserts. One consistent stack, so the
  glibc-2.38 wheel loads natively and it runs AS-IS on GB10 (host supplies only the driver).
- `.github/workflows/build-spark-image.yml`: builds on `ubicloud-standard-30-arm`, consumes an
  already-published arm64 wheel release (decoupled from compile), pushes to
  `ghcr.io/jethac/spark-vllm:e2-dgv-<sha>` via the built-in token. First run: 27408638063.
This kills the bare-wheel host-mismatch (lineage/torch/glibc) that blocked DG-V5. Spark just
`docker pull` + `docker run` -- no torch upgrade, no overlay, no glibc retrofit. Once the image
publishes, DG-V5/V6 serve via scripts/run_vllm_dgemma_dgv_spark.sh inside it.
Full end-state (later): move the rebuilt-C base build to Ubicloud too for end-to-end CI repro.

## DG-V5 SERVE: pipeline GREEN to the runtime; block-diffusion PROFILING HANGS (2026-06-12)
The whole serving pipeline now works end-to-end up to the serving runtime:
- Pivot validated ON HARDWARE: 22.04/torch-2.11 wheel (sm121a-arm64-wheels-3d6a0d507) loads in r10
  with ZERO retrofit (FIPrefillGroup/supports_non_causal/DiffusionGemma all True, no torch/glibc fight).
- **r11 baked + provenance-green**: jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11 (r10 + e2-dgv
  vLLM, swap-only). Dockerfile.r11 + docker/Dockerfile.r11 pushed.
- DiffusionGemma serve INITIALIZES correctly: nvfp4 KV engaged, FLASHINFER backend, **VO-split +
  VLLM_NVFP4_KV_LINEAR_V_SF=1 proof line present** ("NVFP4 V scale factors are linear"), 26B MoE
  loads (48.6 GiB). Two config gates surfaced + satisfied: enforce-eager (cudagraph OOM on 26B MoE),
  VLLM_NVFP4_KV_LINEAR_V_SF=1 (512-wide-head VO split).
- **BLOCKER: serve HANGS in the post-load memory-profiling dummy forward** (frozen right after
  "Using FlashInfer for top-p & top-k sampling"; no "GPU KV cache size" line; generation empty;
  container alive, no error). NOT fixed by --no-enable-prefix-caching --no-enable-chunked-prefill.
  This is a vLLM **block-diffusion serving-runtime** issue (the profiling forward vs the denoise /
  model_states integration), NOT our NVFP4-KV code (which is proven: P520 byte-identical + engages
  here correctly). Iteration cost ~6min/model-reload.

### NEXT (decisive bisect): serve **bf16** DiffusionGemma (--kv-cache-dtype auto) in r11.
- bf16 ALSO hangs -> block-diffusion serving-runtime general (upstream DG-in-our-base profiling
  path); investigate the profiling dummy-run vs the diffusion model_states/denoise setup.
- bf16 SERVES but nvfp4 hangs -> our nvfp4+VO-split+diffusion-attention interaction during profiling.
Either way it's a scoped serving-runtime follow-up; the NVFP4-KV contribution + the whole image/
wheel/r11 pipeline are done. (Note: upstream's recipe serves DG via `vllm serve` in the Gemma
docker image -> their newer vLLM base may profile DG correctly where our cherry-picked PR-on-older-
base has a gap.)

## Coordination
vLLM = my lane. No Spark/P520 GPU touch while another agent holds the marker. Mail Codex
the DG-V plan so SGLang DG-R5/R6 receipts are the agreed parity target.
