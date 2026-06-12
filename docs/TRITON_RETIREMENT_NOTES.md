# Triton retirement for bf16 Gemma on CC 12.x — implementation notes + morning Spark spec

Date: 2026-06-12 (overnight engineering lane, per OVERNIGHT_LADDER_PLAN_20260612.md
"Parallel engineering"). Author: agent (Claude), per the zero-bug amendment:

> **STATUS: NOT a support claim.** Code is implemented and statically validated
> (56/56 selection tests, no GPU). Serving validation on Spark is specified in
> §6 and PENDING. Nothing below counts as "Gemma serves on FlashInfer" until
> those rows are green.

## 1. What and where

Branch: `jethac/vllm@spark/hijinks-e2-triton-retire` (worktree
`B:\workshop\worktrees\vllm\spark-hijinks-triton-retire`), based on
`spark/hijinks-e2-vllm@7df3c67ec8` (upstream-main rebase + mm-prefix work).

- `6b96f69d1d` — knob + routing + honest head-512 selector + tests.
- `f5ed80e568` — mm-prefix guard (do not force FlashInfer for live multimodal
  spans without `VLLM_FLASHINFER_MM_PREFIX`).

New knob: **`VLLM_FLASHINFER_BF16_GEMMA=1`** (registered in `vllm/envs.py`).
Routes Gemma-family bf16-KV configs on CC 12.x to FLASHINFER:

- Routing: `_spark_route_gemma_bf16_to_flashinfer()` in
  `vllm/model_executor/models/config.py`, called from a new `Gemma3Config`
  (map entries `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`) and from
  `Gemma4Config` (before the heterogeneous-head early return, so uniform-head
  Gemma 4 variants route too). Engages only when ALL hold:
  knob set; user chose no backend; CUDA platform with CC major == 12;
  `cache_dtype` in (`auto`, `bfloat16`); and NOT (mm-prefix spans live
  without `VLLM_FLASHINFER_MM_PREFIX=1`).
- Backend side: the knob enables the exact FA2 two-pass VO split for
  head_size > 256 non-NVFP4 layers (`_vo_split_factor` in
  `vllm/v1/attention/backends/flashinfer.py`, same machinery as
  `VLLM_FLASHINFER_VOSPLIT`); the cudagraph `NEVER` gate for VO-split groups
  is extended to match (decodes route through the dynamically planned prefill
  wrapper).

**Decision: opt-in tonight.** Default-on for CC 12.x is a separate
flip-to-default proposal (§7) gated on the Spark serving rows. Rationale: the
E4B VO-split serving path was RED at real GQA geometry as recently as
2026-06-11 (`max_mma_kv: 0`, fixed in FlashInfer `76af7982`/r9), and the
zero-bug bar forbids defaulting a path with zero green serving rows.

## 2. Why upstream falls to Triton (unchanged understanding)

Upstream's FA4 unification (`Gemma4Config.verify_and_update_config`) handles
heterogeneous 256/512 heads by forcing FA4 — but FA4's TMEM gate
(`fa_utils.get_flash_attn_version`) excludes head_size > 128 (except 192) on
ALL Blackwell, and no FA4 resolves on CC 12.x anyway, so the `elif` forces
model-wide `TRITON_ATTN` (the vllm#38887 ~19 tok/s E4B complaint). Our routes
slot NEXT TO that logic (checked first, only when the backend is unset); the
upstream branches are untouched when the knob is off.

## 3. Selection truth table — CC 12.x, after this branch

Per-layer selection cells (model-level routing shown where it decides). "mm"
means multimodal spans live (`is_mm_prefix_lm`, i.e. NOT
`--language-model-only`).

| Model | KV dtype | Knobs | Chosen backend | Why |
|---|---|---|---|---|
| Gemma 4 (256/512) | bf16/auto | none | **TRITON_ATTN (model-wide force)** | upstream FA4-unavailable branch (unchanged baseline) |
| Gemma 4 (256/512) | bf16/auto | `BF16_GEMMA` | **FLASHINFER** (D512 via VO-split 2-pass) | new route in Gemma4Config |
| Gemma 4 uniform-head | bf16/auto | `BF16_GEMMA` | FLASHINFER | route sits before the heterogeneous early return |
| Gemma 4 (256/512) | bf16/auto | `FLASHINFER_VOSPLIT` | FLASHINFER (VO-split) | pre-existing campaign route, regression-tested |
| Gemma 4 mm | bf16/auto | `BF16_GEMMA` only | TRITON_ATTN | mm guard: FlashInfer would fail validation without `MM_PREFIX`; route stands down + logs |
| Gemma 4 mm | bf16/auto | `BF16_GEMMA` + `MM_PREFIX` | FLASHINFER (mm-prefix custom masks + VO-split) | both knobs |
| Gemma 4 (256/512) | fp8 | `BF16_GEMMA` | TRITON_ATTN | dtype scope: knob never touches quantized KV |
| Gemma 4 (256/512) | nvfp4 (full) | `NVFP4_KV_VOSPLIT` (+`LINEAR_V_SF`) | per-layer FLASHINFER | pre-existing route, unchanged |
| Gemma 4 mixed-KV (nvfp4+skip) | mixed | any | per-layer: local-256→FLASHINFER, global-512→TRITON (explicit pin, now also the honest-selector fallback) | pre-existing route, unchanged |
| Gemma 3 (uniform 256) | bf16/auto | none | upstream priority order (FLASH_ATTN if its FA2 validates, else FLASHINFER; mm text-spans case → TRITON since FLASH_ATTN/FLASHINFER reject mm-prefix on CC 12.x) | unchanged baseline |
| Gemma 3 | bf16/auto | `BF16_GEMMA` (+`MM_PREFIX` if mm) | **FLASHINFER** (plain FA2, no split needed) | new Gemma3Config route |
| Gemma 3/4 | bf16/auto | `BF16_GEMMA`, user passed `--attention-backend X` | X | explicit user choice always wins |
| any, CC != 12.x | any | `BF16_GEMMA` | unchanged | CC scope: knob never leaks off 12.x |

## 4. Selector-vs-kernel head-512 discrepancy — RESOLVED (scoped)

Banked bug (`results/upstream_draft_issue_flashinfer_head512_selector_overpromise_20260610TmanualJST.md`,
VLLM_GEMMA_RUNGS.md M6 note): `FlashInferBackend.get_supported_head_sizes()`
returns `[64, 128, 256, 512]`, so `validate_configuration` accepted head-512
while the FA2 kernel rejects `HEAD_DIM_VO > 256` at runtime
(`KernelTraits::IsInvalid()`, register-budget, dtype-independent — probed on
GB10). Consequence: automatic per-layer fallback never fired and the
mixed-KV global layers needed the explicit Triton pin in `gemma4.py`.

Resolution on this branch: **`FlashInferBackend.supports_combination` now
rejects head_size > 256 on CC major == 12 unless a VO-split knob makes the
FA2 launch actually valid** —

- non-NVFP4 KV: needs `VLLM_FLASHINFER_VOSPLIT=1` or
  `VLLM_FLASHINFER_BF16_GEMMA=1`;
- NVFP4 KV: needs (`VLLM_NVFP4_KV_VOSPLIT=1` or `VLLM_FLASHINFER_VOSPLIT=1`)
  AND `VLLM_NVFP4_KV_LINEAR_V_SF=1` (mirrors `_vo_split_factor`'s runtime
  contract exactly, incl. the divisibility check);
- plus a clear reason string, so a user forcing FLASHINFER (the #40677
  scenario) gets an actionable selection-time error instead of a runtime
  kernel crash.

Why scoped to CC 12.x and not resolved globally: on CC 12.x FA2 is the only
FlashInfer path, so selector truth == FA2 kernel truth there. On SM100 the
backend can route to TRTLLM kernels whose head-512 support we have NOT
probed; changing selection there could silently alter upstream SM100
behavior either direction. **The over-promise therefore still stands on
non-12.x CCs, deliberately and documented** — that residual plus the probe
ask is exactly the upstream draft issue's scope, unchanged.

The `gemma4.py` mixed-KV Triton pin is now redundant on CC 12.x (the honest
selector lands the same fallback automatically) but is KEPT as the
deterministic choice and for non-12.x CCs; its comment now records this.

## 5. Static validation (no GPU — P520 GPU owned by the ladder agent tonight)

`tests/v1/attention/test_sm12x_triton_retirement_selection.py` (in-repo, real
test file): **56 passed** under mocked `DeviceCapability` (12,0)/(12,1)/(9,0)/(10,0),
covering the §3 truth table (Gemma3Config/Gemma4Config routing incl. CC, dtype,
mm, explicit-backend, and pre-existing-route regression cells),
`FlashInferBackend.validate_configuration` honesty cells (head 256/512 ×
auto/bfloat16/fp8/nvfp4 × knob off/bf16/vosplit/nvfp4-set; Triton fallback
cell reachable; non-12.x untouched), `_vo_split_factor` knob semantics (incl.
"bf16 knob must NOT enable the NVFP4 split"), and envs.py knob registration.

Environment: WSL2, fresh isolated venv `~/e2_triton_retire_testenv` (vllm
0.22.1 wheel for dependency closure + its compiled `_C`; worktree sources
shadowed in via PYTHONPATH — verified the tracebacks/imports resolve to the
worktree). CPU-only; no GPU touched.

Environment incident, disclosed: an earlier attempt installed
`flashinfer-python` into the shared `~/sm120env`, which dragged torch
2.12.0+cu130 → 2.9.1 + cu12 NVIDIA wheels (namespace collision broke torch
import). **Fully restored and verified**: torch 2.12.0+cu130 imports w/ CUDA,
triton 3.7.0, nccl-cu13 2.29.7, cudnn-cu13 9.20.0.48, cusparselt-cu13 0.8.1,
nvshmem-cu13 3.4.5, cuda-runtime 13.0.96, nvjitlink 13.0.88, torchvision OK;
all cu12 packages removed. Additions that remain (nondestructive):
flashinfer-python 0.6.12, pytest, and flashinfer's extra dep packages. All
subsequent test work used the isolated venv.

What static tests CANNOT prove (morning scope): kernel health at real serving
GQA geometry (the E2 `max_mma_kv: 0` class), output coherence, throughput.

## 6. MORNING SPARK SPEC — serving validation (claim gate)

Image/build: claim rows on the **r9 baked image** (`sha256:8c37bdbc…`, vLLM
`9759e3b06` + FlashInfer `76af7982` — carries the `max_mma_kv=0` dispatcher
fix). The new knob is NOT in r9; for bf16 Gemma 4 the r9-era
`VLLM_FLASHINFER_VOSPLIT=1` route is selection-equivalent (same FLASHINFER
force + same VO-split machinery — §3), so:

- **Claim rows run with `VLLM_FLASHINFER_VOSPLIT=1` on r9** (baked image, no
  overlay) and validate the retirement PATH.
- **One Python-only overlay smoke** (explicitly labeled smoke, per the
  overlay rule) on `spark/hijinks-e2-triton-retire` validates the new
  knob's routing itself end-to-end (log proof: "forcing FLASHINFER … retiring
  the TRITON_ATTN fallback" + zero TRITON_ATTN lines). The flip-to-default
  image bake (§7) then makes the knob claim-grade.

Standard gates every row: marker WRITE-THEN-VERIFY; hf_model_access_probe on
exact names BEFORE the window; corpora md5 c1 `abb63f0e` / c2 `1686a33b` /
c3 `28dfeba9`; util 0.72, `--memory 100g`, one server at a time; EXT_PATH +
latch provenance. Zero-bug gates: C1 PPL twice per server (bitwise identical
or RED); smoke transcripts banked verbatim (incoherence = RED even if green
logs — remember the SGLang E4B r9fix row was exactly this failure);
bf16-vs-bf16 PPL band |delta| > 0.05 nats vs the Triton bf16 baseline = RED
pending investigation (this is a bf16↔bf16 backend swap: expect near-equality,
NOT the 0.5-nats quantization band).

Rows (text-only `--language-model-only`, -it checkpoints, ctx 8191, order as
listed; each FlashInfer row vs its Triton baseline):

1. **E4B bf16 + FlashInfer VO-split** — the vllm#38887 carrot AND the known
   risk row (E2 red was E4B GQA geometry; r9 carries the fix but E4B has
   never gone green on vLLM). bench_e3.py params identical to the banked
   baseline: decode 256-tok batch-1 median-of-3, TTFT 2083-tok prompt,
   concurrent x4, per-request nonces. Compare: **Triton baseline 19.03 tok/s
   decode / 0.317 s TTFT / x4 92.04 tok/s**
   (`results/claude_blockE23_20260611/`). Success: coherent transcripts +
   PPL gate + decode meaningfully > 19.03 (record the ratio; if ≤ baseline,
   retirement is performance-unjustified — that is a finding, not a fail).
   Plus C1 PPL pair vs an E4B Triton PPL cell (run the Triton cell in the
   same window if not already banked).
2. **12B bf16 + FlashInfer VO-split** — first 12B bf16-FlashInfer row; C1 PPL
   x2 + smoke + capacity log vs a same-window 12B bf16-Triton baseline
   (interleaves with the ladder's G4 12B block — coordinate, don't duplicate
   servers).
3. **31B bf16 + FlashInfer VO-split** — re-pin the anchor: banked
   **bf16-Triton C1 PPL 4.6532 (marked suspect) vs bf16-FlashInfer 4.6132**.
   The FlashInfer cell must reproduce 4.6132 bitwise (determinism finding);
   then ONE fresh Triton C1 cell to adjudicate the suspect 4.6532 (if the
   0.04-nat backend gap reproduces, that is the "Triton fallback quality
   deficit" exhibit for the retirement filing; if not, the suspect row dies).
4. **Gemma 3 cell (cheap, completes the family claim):** G3 4B-or-12B bf16
   with `--attention-backend FLASHINFER` (selection-equivalent to the knob's
   Gemma 3 route on r9) — smoke + C1 PPL x2 vs same-model default-backend
   row. Record which backend the default actually resolves on r9 (FLASH_ATTN
   vs Triton — pins the "what are we retiring for Gemma 3" cell of §3).
5. **Knob overlay smoke (labeled SMOKE):** e2-triton-retire overlay, E4B
   bf16, `VLLM_FLASHINFER_BF16_GEMMA=1` ONLY (no VOSPLIT knob) → routing log
   proof + one coherent generation. Optionally the mm guard: same overlay
   WITHOUT `--language-model-only` → must log "not routing" and serve on
   Triton.

Artifacts: one results dir per row, server logs with the selection log lines,
benchmark JSONs, transcripts, PPL pairs; ledger rows RED or GREEN per the
bar. Defer-if-tight order: 5's mm cell, then 4, then 2 (1 and 3 are the
deliverable's spine).

## 7. Flip-to-default proposal (post-validation, do not pre-empt)

If §6 rows 1–3 are green: make the route default-on for CC 12.x (knob becomes
opt-OUT, `VLLM_FLASHINFER_BF16_GEMMA=0` to disable), bake the e2 image from
`spark/hijinks-e2-triton-retire`, run one regression ladder pass on the baked
image, and fold the honest-selector + route into the upstream filing wave
(#38887 repro answered by row 1; #40677 answered by the selection-time reason
string; the draft over-promise issue updated with the CC 12.x resolution and
the remaining SM100 probe ask).

## 8. Integration + default flip (overnight)

Date: 2026-06-12 overnight (agent: Claude, campaign dgx-spark-hijinks,
zero-bug bar). Branch `spark/hijinks-e2-integrate`, fast-forwarded into the
canonical `spark/hijinks-e2-vllm` (the morning image builds from it).

### Merged heads

- Base: `spark/hijinks-e2-vllm @ 7df3c67ec8`.
- `spark/hijinks-e2-triton-retire @ f5ed80e568` — merged first (fast-forward).
- `spark/hijinks-e2-mtp @ 2d3411c331` — merge commit `6a7605cd9e`.
- Default flip commit: `20196b5946` (== new head of BOTH
  `spark/hijinks-e2-integrate` and `spark/hijinks-e2-vllm`, pushed to
  jethac/vllm).

### Conflict resolutions (one textual conflict, the predicted gemma4 seam)

`vllm/model_executor/models/gemma4.py`, one hunk in
`Gemma4Attention.__init__`: e2-triton-retire rewrote the mixed-KV
backend-pin comment in place (honest-selector history note); e2-mtp
extracted the pin into the shared `gemma4_global_attn_backend_override()`
(single seam used by target AND KV-sharing drafter) and moved the old
comment into its docstring. Resolution per the recorded rule: kept e2-mtp's
structure (shared function; helper call in both attention classes) and
folded e2-triton-retire's updated history paragraph into the shared
function's docstring, replacing the stale paragraph it superseded. The pin
CONDITION was changed by neither head and is unchanged; target and drafter
resolve through the one function, so they can never diverge per layer type
(divergence would crash spec decode — they KV-share).

### Default flip (Amendment 3, OVERNIGHT_LADDER_PLAN_20260612)

`VLLM_FLASHINFER_BF16_GEMMA` is now DEFAULT-ON; `=0` disables (escape
hatch); explicit `=1` keeps the exact pre-flip opt-in semantics. The
default is scoped so it cannot leak beyond knob-unset text-only bf16 Gemma
on CC 12.x: new `_vllm_flashinfer_bf16_gemma_explicit()` +
`_vllm_flashinfer_bf16_gemma_vo_split_enabled()` (flashinfer.py) gate the
runtime sites to CC 12.x and `supports_combination`'s default acceptance to
bf16/"auto" KV.

Truth-table delta vs §3 (everything not listed is UNCHANGED — mm carve-out,
fp8/nvfp4 routes, explicit --attention-backend, non-12.x CCs, all
knob-set rows):

| Cell (CC 12.x, knob UNSET) | Before | After |
|---|---|---|
| Gemma 4 bf16/auto text-only | TRITON_ATTN (model-wide force) | **FLASHINFER** (D512 via VO-split) |
| Gemma 3 bf16/auto text-only | upstream priority order | **FLASHINFER** |
| Selector: head 512, kv auto/bfloat16 | rejected (honest) | **valid** (default vouches for the VO split) |
| Selector: head 512, kv fp8 | rejected | rejected (default does NOT vouch; explicit =1 still does) |
| `_vo_split_factor(512, non-NVFP4)` on CC 12.x | 1 | **2** |
| `envs.VLLM_FLASHINFER_BF16_GEMMA` | False | **True** |
| Gemma 3/4 bf16 mm spans, no MM_PREFIX | Triton | Triton (carve-out stands; route logs and stands down) |
| any of the above with `=0` | n/a | pre-flip behavior exactly |

Composition note: mm spans + `VLLM_FLASHINFER_MM_PREFIX=1` now route to
FlashInfer with the knob UNSET (default-on composes with the carve-out's
"unless MM_PREFIX" wording; pre-flip this needed both knobs). Pinned by a
new test cell.

### Test results (WSL, isolated `~/e2_triton_retire_testenv`, CPU-only, no GPU)

- `tests/v1/attention/test_sm12x_triton_retirement_selection.py`: **71/71
  passed** (was 56; 8 cells deliberately flipped expectation — exactly the
  knob-unset text-only bf16 CC 12.x cells listed above — and 15 new cells
  pin the escape hatch and the default's scope; full list in commit
  `20196b5946`).
- `tests/models/test_gemma4_attn_backend_pin.py`: **9/9 passed** (MTP
  pin matrix, expectations untouched by the flip).
- `py_compile` clean on every merged/flipped file (envs.py, config.py,
  gemma4.py, gemma4_mtp.py, flashinfer.py, both test files).
- Worktree-shadowing verified (imports resolve to
  `B:\workshop\worktrees\vllm\spark-hijinks-e2-integrate`); the venv's
  vllm-0.22.1 compiled artifacts (`_C*.so`, `vllm_flash_attn` extensions,
  `_version.py`) overlaid untracked+gitignored into the worktree, same
  recipe as §5. Shared `~/sm120env` untouched.

### STANDING REVERT RULE (zero-bug bar)

**ANY RED bf16 row from tonight's ladder → revert the default before
morning**: single-commit `git revert 20196b5946` on
`spark/hijinks-e2-integrate`, fast-forward `spark/hijinks-e2-vllm`, push
both. The knob stays available (opt-in again); the RED row ships as the
reason. Per Amendment 3 this is not optional.

### Still pending (unchanged from §6)

Serving validation on Spark is the claim gate; nothing here upgrades the
"NOT a support claim" status banner. §6 rows + the knob overlay smoke now
double as the flipped-default validation.

## 9. Adjudication log

### 2026-06-12 — §8 default flip SCOPED to the Gemma 4 family only (urgent, zero-bug bar)

Adjudication: the Amendment 3 default flip (`20196b5946`) routed knob-unset
text-only bf16 **Gemma 3** off the upstream FLASH_ATTN default onto
FlashInfer on CC 12.x. New ground-truth evidence shows that was a
**regression on sm_120**, so the flip is now scoped to **Gemma 4 only**.

Evidence (`results/p520_gemma3_1b_serving_20260612/` + ledger, mail 0044):
on sm_120 (RTX 5060 Ti, WSL2) at Gemma 3 1B geometry (d256, SWA window
512) the FlashInfer backend is numerically wrong for EVERY KV dtype —
FI-bf16 is +0.221/+1.243/+1.380 nats (C1/C2/C3) off an
HF-reference/FLASH_ATTN pair that agree to <0.001; FI-fp8
+0.006/+0.159/+0.494; FI-nvfp4 +1.592/+2.436/+2.752 with deterministic
chat gibberish on a virgin JIT cache. sm_121 is corroborated fine
(Triton/FI within 0.04 nats across five sizes), but the default must not
regress sm_120.

Change (`jethac/vllm`, branch `spark/hijinks-e2-flip-scope @ 36c9bbc83c`,
fast-forwarded into `spark/hijinks-e2-vllm` — same head — both pushed):

- `_spark_route_gemma_bf16_to_flashinfer()` grew a `default_on` parameter:
  Gemma4Config passes True (Amendment 3 default stands; `=0` escape hatch
  unchanged); Gemma3Config passes False — knob-unset (and empty-string)
  Gemma 3 reverts to upstream default routing (FLASH_ATTN where
  supported).
- Explicit `VLLM_FLASHINFER_BF16_GEMMA=1` still opts Gemma 3 in for
  experiments; the route logs a warning that it is known numerically
  wrong on sm_120 d256/SWA-512.
- NOT touched: nvfp4/fp8 knob routes, the mm carve-out, all backend-side
  head>512 selector/VO-split machinery (head>256 is Gemma 4 geometry),
  the envs.py value semantics (only "0" disables).
- Test matrix 71 → 74: Gemma 3 knob-unset cell REVERTED to the pre-flip
  expectation (backend unset), parametrized over CC 12.0/12.1 and
  comment-marked with the sm_120 finding (the WHY cell); new pins for
  "empty string is not a Gemma 3 opt-in" and "empty string behaves like
  unset for Gemma 4". Validation (WSL `~/e2_triton_retire_testenv`,
  CPU-only, worktree `spark-hijinks-flip-scope` on PYTHONPATH, import
  provenance verified): selection suite **74/74**, MTP pin suite **9/9**;
  `py_compile` clean on all four touched files. No mm/audio suites exist
  on this branch (mm-retire/audio not merged into e2-vllm).

§8's truth-table delta row "Gemma 3 bf16/auto text-only: upstream →
FLASHINFER" is hereby VOID; the Gemma 4 rows stand. **Gemma 3 re-flip
gate:** FlashInfer d256/SWA root cause on sm_120 fixed upstream + a green
truth-referenced rerun (HF-reference or FLASH_ATTN pair) of the
`p520_gemma3_1b_serving` rows. Until then Gemma 3 stays on upstream
defaults; the §8 STANDING REVERT RULE continues to apply to the remaining
(Gemma 4) scope of `20196b5946`.

## 10. mm-prefix + audio merge into e2-vllm (campaign dgx-spark-hijinks)

### 2026-06-12 — multimodal mm-prefix masking + audio policy folded onto the canonical wheel line

TL;DR: `spark/hijinks-e2-mm-retire` (mm-prefix masking flip) and
`spark/hijinks-e2-audio` (tests-only) merged into `spark/hijinks-e2-vllm`
so the next sm120a wheel carries the FlashInfer mm-prefix custom-mask
path. New `e2-vllm` head **`e32459eea`** (pushed; `spark/hijinks-e2-mm-merge`
@ `e32459eea` is the same head, also pushed). RECOMMENDED by the P520
smoke agent (bf16 mm masking GREEN on sm_120: image-grounded, FI-route
byte-identical to Triton, text knob-on/off token-identical).

Merge topology (worktree `spark-hijinks-mm-merge`, base `512cca4e9`):
- `a1eefbb15` Merge e2-mm-retire — `envs.py`, `flashinfer.py`, and the
  selection test auto-merged clean; **`config.py` conflicted** and was
  resolved per the banked 3-way reference
  (`results/p520_mm_retirement_smokes_20260612/overlay/config_merged.py`).
- `91464e012` Merge e2-audio — clean (tests-only + the behavior-identical
  `mm_prefix_doc_ranges_for_request` extraction in `gpu_model_runner.py`).
- `e32459eea` Reconcile one stale Gemma 3 mm selection cell (below).

Conflict resolution (`vllm/model_executor/models/config.py`, 1 hunk):
e2-vllm advanced (`20196b594` → `512cca4e9`, incl. the §9 Gemma-3
scope-out) AFTER e2-mm-retire branched, and both rewrote
`_spark_route_gemma_bf16_to_flashinfer`'s mm-guard. Blobs matched the
bank exactly (base `b2468524`, ours/e2-vllm `03039477`, theirs/mm-retire
`b6886992`; resolved blob `c054aba8`). The single textual conflict was
one `logger.info()` message body; git had already auto-merged mm-retire's
`if/else` inverted-default restructure (MM_PREFIX default `"1"`, removal
of the old return-False) into e2-vllm's `default_on` Gemma3/4 split. The
resolution **COMBINES both**: keeps e2-vllm's `default_on` split (Gemma 3
`default_on=False`, scoped OUT of the bf16 TEXT flip per the §9 sm_120 1B
d256/SWA-512 bug; Gemma 4 `default_on=True`) AND takes mm-retire's
inverted mm default (the Amendment-4 FA2 custom-mask serving line). The
mm flip applies to CC 12.x Gemma mm (bf16/"auto" KV).

Selection-matrix cells whose expectation CHANGED with the merge:
- **Gemma 4 mm-prefix, MM_PREFIX unset: TRITON → FLASHINFER** (was the mm
  carve-out; now the FA2 custom-mask path by default).
- **Gemma 3 mm-prefix, MM_PREFIX unset: (new cell) → None** — NOT
  FLASHINFER. The mm-retire test asserted FLASHINFER, but that predated
  the §9 Gemma-3 scope-out: with `default_on=False` the bf16-knob-unset
  early-return fires BEFORE the mm-prefix branch, so knob-unset Gemma 3 —
  text or mm — leaves the backend unset. Stale test reconciled in
  `e32459eea` (`test_mm_prefix_lm_default_routes_flashinfer` →
  `test_mm_prefix_lm_default_leaves_backend_unset`, asserts None; added
  `test_mm_prefix_lm_explicit_opt_in_routes_flashinfer`: Gemma 3 mm routes
  FLASHINFER only on explicit bf16 `=1`).
- Newly-pinned escape-hatch cells (intent unchanged): Gemma 4 MM_PREFIX=0
  → TRITON; Gemma 4 BF16_GEMMA=0 → TRITON; Gemma 3 MM_PREFIX=0 → unset;
  Gemma 3 BF16_GEMMA=0 → unset; explicit MM_PREFIX=1 → FLASHINFER (now
  coincides with the Gemma 4 default).

`FlashInferBackend.supports_mm_prefix()` still CLAIMS Gemma 3 **and** 4 mm
on CC 12.x (capability ≠ routing-default; `TestFlashInferMMPrefixSupport`
unchanged) — Gemma 3 just isn't routed there by default.

Validation (WSL `~/e2_triton_retire_testenv`, CPU-only, worktree
`spark-hijinks-mm-merge` on PYTHONPATH; import provenance verified:
under-test Python from the worktree, compiled `_C`/`vllm_flash_attn` from
the build):
- selection matrix + mm-prefix policy
  (`test_sm12x_triton_retirement_selection.py`): **103/103**
- MTP pin (`tests/models/test_gemma4_attn_backend_pin.py`): **9/9**
- audio policy (`test_mm_prefix_audio_policy.py`): **18/18**
- Total **130 passed, 0 failed**; `py_compile` clean on all touched files
  (`config.py`, `envs.py`, `flashinfer.py`, `gpu_model_runner.py`, the
  three test files).

Wheel: a NEW sm120a wheel must be built from `e2-vllm @ e32459eea` so the
Colab notebook can serve multimodal with NVFP4 KV (the current wheel is
text-capable; mm via Triton cannot read nvfp4). Requested from Codex
(mail 0074).
