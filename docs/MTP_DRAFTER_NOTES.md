# MTP drafter support — recon, scoping, enablement (2026-06-12 overnight)

Directive (mail 0030 / OVERNIGHT_LADDER_PLAN_20260612.md Amendment 2): full
support for multi-token-prediction drafters on the vLLM lane. Zero-bug bar:
greedy spec decode must be OUTPUT-IDENTICAL to non-spec greedy at temp 0,
or the row is RED.

All vLLM file:line references are against the e2-vllm line
(`spark/hijinks-e2-vllm` @ 7df3c67ec8, worktree
`B:\workshop\worktrees\vllm\spark-hijinks-022-gemma4-mixed-kv`).

## 1. What exists for Gemma (recon)

### Gemma 4: NATIVE MTP drafters, officially released

Google ships "assistant" MTP drafter checkpoints for ALL Gemma 4 sizes
(Apache 2.0, lightweight ~4-layer drafters that KV-share with the target):

| target | drafter checkpoint | hf model_type |
|---|---|---|
| gemma-4-E2B-it | google/gemma-4-E2B-it-assistant | `gemma4_assistant` |
| gemma-4-E4B-it | google/gemma-4-E4B-it-assistant | `gemma4_assistant` |
| gemma-4-12B-it | google/gemma-4-12B-it-assistant (+ qat-q4_0-unquantized variant) | `gemma4_unified_assistant` |
| gemma-4-26B-A4B-it | google/gemma-4-26B-A4B-it-assistant | `gemma4_assistant` |
| gemma-4-31B-it | google/gemma-4-31B-it-assistant | `gemma4_assistant` |

Key properties (Google MTP docs ai.google.dev/gemma/docs/mtp/mtp, HF blog
huggingface.co/blog/gemma4):
- The drafter reuses the TARGET's KV cache (KV sharing, Gemma-3n style) —
  it never prefills and has no KV pool of its own.
- E2B/E4B assistants additionally use "centroids masking" (sparse lm_head:
  ~4K candidate tokens instead of the full ~262K vocab).
- Google claims up to 3x decode speedup, output-quality-identical.
- llama.cpp MTP merged 2026-06-07 (ggml-org/llama.cpp#23398) — relevant to
  the llama.cpp lane later; Unsloth reports 52 → 162 tok/s on B200 at 0.70
  acceptance.

### Gemma 4: community EAGLE-3 heads (secondary option)

- RedHatAI/gemma-4-31B-it-speculator.eagle3 and
  RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3 — vLLM-ready (speculators
  format, `method: eagle3`), accept-len ~2.1–3.9 at k=3–5.
- thoughtworks/Gemma-4-31B-Eagle3 — SGLang-fork-only, targets the BASE
  model; not useful for our -it rows.

### Gemma 3: nothing native, no released EAGLE heads

No MTP heads in Gemma 3 checkpoints, no official drafters, and no community
EAGLE/EAGLE-3 head for google/gemma-3-* surfaced. Spec decode for Gemma 3
therefore means plain small-model drafting: **google/gemma-3-270m-it
drafting for 1B/4B/12B/27B** via vLLM's `draft_model` method (same
tokenizer family; vocab-size equality is enforced at
vllm/config/speculative.py:1021-1036).

## 2. vLLM spec-decode infra map (file:line)

Upstream main (our base) already has FIRST-CLASS Gemma 4 MTP support — no
checkpoint invention needed:

- Method surface: `--speculative-config` with `method` ∈ ngram / medusa /
  draft_model / suffix / eagle / eagle3 / **mtp** / dflash …
  (vllm/config/speculative.py:34-68). `gemma4_mtp` is in MTPModelTypes
  (speculative.py:52).
- Auto-detection: drafter `model_type` ∈ {`gemma4_assistant`,
  `gemma4_unified_assistant`} → rewritten to `gemma4_mtp` /
  `Gemma4MTPModel`, `n_predict=1`, drafter's `num_kv_shared_layers` forced
  to 0 (cross-model sharing is wired later) — speculative.py:512-521;
  model_type→method=mtp detection at :713-716; `use_gemma4_mtp()` at
  :1054-1060. Registry: vllm/model_executor/models/registry.py:624.
- Model: vllm/model_executor/models/gemma4_mtp.py — Q-only attention
  (`Gemma4MTPAttention`, K/V read from the target's cache;
  `is_kv_shared_layer = True` at :221), pre/post projections between draft
  hidden (e.g. 256) and backbone hidden, own lm_head, optional
  `masked_embedding` centroids.
- Proposer: vllm/v1/spec_decode/gemma4.py (`Gemma4Proposer`):
  - all draft steps predict from the SAME position
    (`constant_draft_positions`, :47);
  - per-KV-group block tables for the sliding/full groups (:61-102),
    fed by gpu_model_runner.py:2479;
  - cross-model KV sharing setup: each draft layer maps to the last
    non-KV-shared target layer of its type (:280-340) — the drafter reads
    the TARGET's pages, whatever dtype they are;
  - per-spec attention groups so each head-dim variant gets its own
    metadata builder (:200-278); drafting metadata built via
    `build_for_drafting` (base impl = `build(fast_build=True)`,
    vllm/v1/attention/backend.py:644-664);
  - `_create_draft_vllm_config` carries a target-forced backend through to
    draft layers (:147-166); with our per-layer routes the backend is None
    and per-layer resolution applies to the drafter too.
  - Runner instantiation: gpu_model_runner.py:589.
- Rejection sampler: vllm/v1/sample/rejection_sampler.py:435-468 — greedy
  requests take `target_argmax` + `rejection_greedy_sample_kernel` (:715).
  At temp 0 every emitted token is exactly the target's argmax (accepted
  drafts match it by definition; first mismatch is replaced by it; bonus
  token is it). Output identity therefore reduces to: target verify logits
  (qo_len=k+1) must argmax-match non-spec decode logits (qo_len=1) over
  the same NVFP4 pages. That is precisely what the identity gate measures.

## 3. NVFP4 KV × spec decode interplay (file:line)

flashinfer.py = vllm/v1/attention/backends/flashinfer.py:

- **Verify step routes through the proven prefill path.** For FA2-NVFP4 on
  SM12x (`use_fa2_nvfp4_kv`, :1364-1392), TRTLLM decode is off
  (:1417-1421) so `_init_reorder_batch_threshold(1,
  supports_spec_as_decode=False)` (:1422-1423): any request with
  query_len > 1 — i.e. every spec-decode verify (qo_len=k+1) — classifies
  as PREFILL and runs the FA2 NVFP4 paged-prefill wrapper. With the VO
  split (D>256), `reorder_batch_threshold=0` (:1441-1457) sends literally
  everything (incl. drafter qo=1 steps) through the prefill wrapper; the
  decode pathway asserts unreachable (:2344-2349).
- **Geometry already probe-proven.** Multi-token qo against packed FP4
  pages is banked:
  - results/flashinfer_nvfp4_kv_probe_causal_20260609.json (causal qo>1,
    all_ok);
  - results/claude_geomprobe_20260611/ runs 3-4: NVFP4 VO-split at real
    31B and E4B global geometries, qo=16, NHD+HND, cosine ≥ 0.999998.
  Verify qo_len = k+1 (2..6 for k≤5) sits between the serving-proven qo=1
  decode-as-prefill and the probed qo=16.
- **Drafter KV pool: there is none (native MTP).** Gemma4 MTP drafter
  layers are KV-shared into the target's cache groups
  (spec_decode/gemma4.py:218-230 resolves their spec through
  `kv_sharing_target_layer_name` inside UniformTypeKVCacheSpecs). One
  cache, the target's, in the target's dtype. "bf16 drafter + nvfp4
  target" mixing is automatic: drafter WEIGHTS are bf16; the cache it
  reads is nvfp4.
- **draft_model method (Gemma 3) shares the cache dtype.** Drafter layers
  join the same KVCacheConfig (llm_base_proposer.py:1566-1624), so
  `--kv-cache-dtype nvfp4` quantizes the 270m drafter's cache too. There
  is no per-model cache-dtype knob today; not needed for correctness
  (drafter only influences proposals; identity is enforced by the target).
  Noted as a possible future tuning knob, not a gap.
- **q dtype:** FA2-NVFP4 keeps model-dtype (bf16) queries (:1396-1415) —
  no fp8-q in verify or drafting.
- **CUDA graphs:** VO-split groups report `AttentionCGSupport.NEVER`
  (:1567-1574) → piecewise-only; drafter also uses piecewise graphs
  (gpu_model_runner comment near :2490). Centroids lm_head has its own
  private CUDA graphs (spec_decode/gemma4.py:115-145) — orthogonal to
  attention.
- **Rejection / cache-write ordering:** the target writes KV for all k+1
  verify tokens; on rejection the scheduler rolls back
  num_computed_tokens and rejected slots are simply overwritten next
  step. The NVFP4 writer (csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu,
  reshape_and_cache_nvfp4 dispatch) writes per-token slots; the swizzled
  SF layout is a byte permutation that never shares bytes across tokens,
  and the linear V-SF mode (required for VO split) is per-token anyway.
  No ordering hazard found in code; the identity gate is the empirical
  proof.

## 4. Scoping decision

- **Gemma 4: enablement targets the NATIVE assistants.** vLLM support is
  already first-class on our base; our work is route compatibility
  (one real gap found and fixed, below) plus NVFP4 identity validation.
  RedHatAI EAGLE-3 heads are a banked secondary option for 26B/31B
  comparisons — not tonight's path.
- **Gemma 3: no native heads, no released drafters — said plainly.** "MTP
  drafter support" for Gemma 3 = (a) small-Gemma drafting
  (gemma-3-270m-it) via `draft_model` with NVFP4 KV on the target,
  validated by the same identity gate; (b) the architecture statement: a
  future Gemma 3-style MTP head needs nothing new from our backend —
  verify is the FA2-NVFP4 prefill wrapper (qo_len=k+1, proven), drafting
  is qo_len=1 reads, and if it KV-shares like Gemma 4's assistant it
  inherits the target's NVFP4 pages via the existing
  kv_sharing_target_layer_name plumbing.
- No checkpoints were invented; every model named above was verified to
  exist on HF Hub via web search tonight.

## 5. What was implemented

Branch `spark/hijinks-e2-mtp` (new worktree
`B:\workshop\worktrees\vllm\spark-hijinks-mtp`, cut from
`spark/hijinks-e2-vllm` @ 7df3c67ec8), head **2d3411c331**:

- **Gap (real crash):** under per-layer mixed KV
  (`--kv-cache-dtype X --kv-cache-dtype-skip-layers ...`) the campaign
  pinned the TARGET's D=512 global layers to TRITON_ATTN
  (models/gemma4.py) because of the banked selector-vs-kernel head-512
  bug, but `Gemma4MTPAttention` had no such pin — the drafter's global
  layers (which read the target's cache via KV sharing) would resolve to
  FlashInfer and hit the FA2 run-time trait guard at the first draft step.
- **Fix:** extracted the pin decision into
  `gemma4_global_attn_backend_override()` (models/gemma4.py) and applied
  it in BOTH `Gemma4Attention` and `Gemma4MTPAttention`, so target and
  drafter can never diverge per layer type again. Full-NVFP4
  (`VLLM_NVFP4_KV_VOSPLIT=1`) and all-dtype VO split
  (`VLLM_FLASHINFER_VOSPLIT=1`) behavior unchanged: no pin, FlashInfer
  VO-split serves D>256.
- **Tests:** tests/models/test_gemma4_attn_backend_pin.py — pin matrix
  (mixed-KV global D=512 → TRITON; sliding/uniform/full-NVFP4/auto/None →
  no pin; VOSPLIT env supersedes) plus a source-level regression guard
  that both attention classes use the shared helper.
- Diff is Python-only (no rebuild needed on an existing install).
- NOT claimed as "support" yet — per the zero-bug bar it is code +
  validation plan until the identity gate passes (Amendment 1).

For the bf16 and full-NVFP4 configs the recon conclusion is that NO new
glue is missing: bf16 Gemma 4 forces TRITON model-wide and the proposer
carries it to the drafter (spec_decode/gemma4.py:147-166); full-NVFP4 +
VOSPLIT resolves per-layer identically for target and drafter. The
identity ladder is the proof either way.

## 6. Validation — state and plan

P520 GPU was occupied all night by the ladder block (vLLM source build +
serving rows; 9.2 GB used at recon time), so per the contention rule the
identity ladder is banked READY-TO-RUN, not run:

- `scripts/mtp_identity_run.py` — offline runner: one config per
  invocation, 8-prompt greedy set (incl. a multi-page long prompt),
  temp 0, max_tokens 128, banks text + token_ids + tok/s + spec metrics
  JSON; `compare` mode emits a GREEN/RED identity verdict with the first
  divergence indexed.
- `scripts/p520_mtp_identity_ladder.sh` — preflight (GPU-free gate at
  1.5 GB, HF access probe on all 4 names, transformers
  `gemma4_assistant` config check, e2-mtp commit presence WARN) then four
  rows, each spec-off → spec-on → compare:

| row | target | drafter (method) | KV |
|---|---|---|---|
| A | gemma-3-1b-it | gemma-3-270m-it (draft_model) | bf16 |
| B | gemma-3-1b-it | gemma-3-270m-it (draft_model) | nvfp4 |
| C | gemma-4-E2B-it | E2B-it-assistant (native mtp) | bf16 |
| D | gemma-4-E2B-it | E2B-it-assistant (native mtp) | nvfp4 + `VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1` |

  Results land in `results/p520_mtp_<date>/` (runs, verdicts,
  ladder.log). Exit non-zero on any RED. Acceptance rate + speedup are
  captured per row (spec_metrics + tok/s ratio).
- Run it the moment the GPU frees (`bash scripts/p520_mtp_identity_ladder.sh`
  from WSL; env `~/sm120env`, vLLM install `~/vllm-hijinks`). Rows A–D
  work on the existing 022 install (the e2-mtp commit matters only for
  future mixed-KV MTP rows; preflight warns, does not block).

## 7. Morning Spark serving spec (vLLM lane)

One server at a time, r9-line image, util 0.72, marker protocol as in
0029. Per zero-bug bar: claim rows on a baked image containing
2d3411c331 (Python-only delta on the e2 line; cherry-pick + rebake or a
clearly-labeled overlay smoke first).

Row order per size: identity gate first, then speed.

1. **G4 12B-it + 12B-it-assistant** (gemma4_unified_assistant — same
   `gemma4_mtp` path, speculative.py:512):
   - bf16: `vllm serve google/gemma-4-12B-it --speculative-config
     '{"model": "google/gemma-4-12B-it-assistant",
     "num_speculative_tokens": 3}'` + spec-off twin; temp-0 identity over
     the same 8-prompt set via the OpenAI API (string equal), acceptance
     from `/metrics` (`vllm:spec_decode_*`), tok/s pair.
   - nvfp4: same + `--kv-cache-dtype nvfp4`, env
     `VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1`.
   - speed reference: the 19.03 tok/s Triton baseline protocol
     (results/claude_blockE23_20260611/) for the AFTER comparison.
2. **G4 26B-A4B-it + assistant**, same 3-row shape (MoE + MTP).
3. **G4 E4B-it + assistant** nvfp4 speed row (the existing E4B AFTER row,
   now with MTP on top).
4. Stretch: **31B-it + assistant** (VO-split flagship) and a RedHatAI
   eagle3 comparison row on 26B-A4B.
5. `num_speculative_tokens`: start k=3 (n_predict=1 reuses the head per
   step; speculative.py:765-780 allows any k, with the acceptance-decay
   warning at :717-726). Bank k=3 and k=5 if the window allows.

RED protocol: identity mismatch ships with both transcripts + the
divergence token index from the verdict JSON, per row, full repro
command line.
