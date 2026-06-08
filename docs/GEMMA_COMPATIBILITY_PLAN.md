# Gemma Compatibility Plan (3 → 3n → 4) on DGX Spark

The sequenced plan to get the whole Gemma family running with NVFP4 KV (and, where it
applies, native FP4) on GB10 / `sm_121`. It is a **ladder**, not a checklist: each rung
adds exactly **one** new complication on top of the last, so we are never debugging two
unknowns at once. This is the Gemma-specific campaign plan; `docs/COMPATIBILITY_BOARD.md`
is the live status snapshot, and the per-lane mechanics live in
`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md` and `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`.

> Provenance note: Gemma 4 is past the assistant's training cutoff. Its lineup and
> per-variant architecture below are **operator-provided** and must be confirmed by the
> rung −1 config audit before any rung is built on them.

## Principles
- **One new thing per rung.** If a step introduces two complications (e.g. SWA *and* a new
  architecture), insert an intermediate rung that isolates one of them.
- **Prove before you climb.** A rung is green only with a server-log-proven NVFP4-KV (or
  native-FP4) row + a capacity comparator + correct output. Not "it kind of loads."
- **Assign each model to its natural runtime; don't fill all cells.** Server dense/MoE →
  vLLM first, SGLang second. Mobile (E-variants) → llama.cpp + LiteRT-LM. 5 models × 3
  runtimes is 15 cells; we are deliberately not running most of them.

## The Gemma lineup (what each model brings)
| family | type | new complication | lane |
|---|---|---|---|
| **Gemma 3** (1/4/12/27B) | dense, SWA 5:1, **uniform head dim 256**, mature | SWA / hybrid local-global KV | vLLM/SGLang |
| **Gemma 3n** (E2B/E4B) | dense mobile, PLE, audio, MatFormer/elastic, KV-share | mobile memory machinery | llama.cpp/LiteRT |
| **Gemma 4 E2B/E4B** | dense mobile, PLE, audio (supersedes 3n) | (inherits 3n machinery) | llama.cpp/LiteRT |
| **Gemma 4 12B** | dense, **encoder-free**, newest | Gemma-4 arch support | vLLM/SGLang |
| **Gemma 4 31B** | dense | scale + D=512 *if present* | vLLM/SGLang |
| **Gemma 4 26B-A4B** | **MoE** (~4B active) | MoE weights + expert routing | vLLM/SGLang |

## Rung −1 — Config audit (first pass; re-confirmed every rung)
Audit the actual config of every variant we plan to climb: head dim (uniform vs dual, and
*where* `D=512` appears), SWA pattern, dense vs MoE, encoders (vision/audio vs
encoder-free), PLE. This is the cheap planning pass from config files — but it is *not* the
final word. **Every rung re-measures geometry from the running model** (per the evidence
gate), because config can be ambiguous and the running model is what actually dispatches.
Treat rung −1 as "plan the ladder"; treat the per-rung measurement as "confirm the rung."
It decides whether the top of the ladder is clean:
- **If `D=512` lives only in the 26B-A4B**, then 31B is uniform-head ("bigger 12B", no new
  category) and the 26B-A4B hits **D=512 + MoE together** — a double-step this lineup
  can't cleanly isolate. Plan a mitigation (e.g. prove the D=512 mixed-KV dodge on the
  26B-A4B's attention with MoE held at a known-good weight path first).
- **If 31B also has the dual head dim**, then 31B isolates D=512 (dense) and 26B-A4B adds
  only MoE on top — clean. Preferred outcome; confirm it.

## Main ladder — vLLM NVFP4-KV capacity
- **Rung 0 — Qwen (standard attention).** ✅ vLLM done (1.751× fp8 KV, clean,
  `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`). SGLang in progress
  (long-sequence quality bug; see its direction doc). Foundation for everything above.
- **Rung 1 — Gemma 3 27B.** *New: SWA / hybrid local-global mixed-KV.* Uniform head dim
  256, dense, mature ecosystem support, no D=512, no Gemma-4-newness risk — isolates the
  mixed-KV plumbing alone. **Also a real shippable capacity win** (first big Gemma-family
  NVFP4-KV row), not throwaway de-risk. vLLM mechanism: per-layer `kv_cache_dtype_skip_
  layers` / per-`AttentionSpec` dtype. SGLang mechanism: SWA subpool delegation.
- **Rung 2 — Gemma 4 12B (encoder-free).** *New: Gemma-4 architecture support.* SWA already
  proven on rung 1; encoder-free + dense = fewest remaining confounds. Main risk is
  ecosystem maturity (newest variant) → Objective A (does it load cleanly) is the work,
  not the KV path.
- **Rung 3 — Gemma 4 31B (dense).** *New: scale + D=512 if present (per audit).* The
  capacity hero — biggest dense KV, no MoE confound. If it carries the dual head dim, this
  is where the D=512 mixed-KV dodge gets proven on a dense model.
- **Rung 4 — Gemma 4 26B-A4B (MoE).** *New: MoE — NVFP4-MoE-weights + expert routing — on
  an attention/KV base already solved.* Most existing evidence (AEON image, ~48 tok/s
  NVFP4-weights), tackled **last** among server models because it stacks the most.

## Side track — mobile (llama.cpp + LiteRT-LM)
NVFP4-KV capacity is not the goal here; these are tiny on-device models. The complication
is PLE / audio / elastic, validated on the practical/GGUF rails.
- **Rung S0 — Gemma 3n (E2B/E4B).** *New: PLE / audio / elastic.* Mature standalone model;
  de-risks the machinery before the Gemma 4 successors. (Repo already has an E2B CPU path
  via LiteRT-LM.)
- **Rung S1 — Gemma 4 E2B / E4B.** The successors, same machinery, on the same rails.

## Per-rung evidence gate (what "green" means)
- **Measured attention geometry from the running model** (not just config JSON), captured
  every rung: per-layer `head_dim` (catches dual head dims / pins *where* `D=512` actually
  is), `num_attention_heads`, `num_kv_heads` (GQA factor), the SWA layer map (which layers
  local vs global + window size), KV page/block layout, and the resulting KV bytes/token.
  The running model is ground truth; config is a hint. For mixed KV this is load-bearing:
  the FP4-vs-fp8/bf16 layer classification must come from *measured* per-layer `head_dim`,
  not assumption — misclassify a layer and you mis-route its KV dtype.
- Server log proves the intended KV dtype per layer (not inference) — and for mixed KV,
  which layers are NVFP4 vs fp8/bf16.
- Matched capacity comparator (vs full-fp8 KV) at equal mem-fraction / page-size / graph
  mode.
- Correct output (raw sanity + a real quality comparator, not just "non-empty").
- Native-target / dispatch evidence where a native-FP4 claim is made.
- Explicit scope labels for what's untested on that rung.

## First moves
1. **Rung −1 config audit** of Gemma 3 27B + all Gemma 4 variants (settles the D=512
   location and rung 3-vs-4 cleanliness).
2. **Rung 1: Gemma 3 27B on vLLM** — the SWA-isolation rung and first big Gemma capacity
   win. Climb only after it's green.
