# Gemma Compatibility Plan (3 → 3n → 4) on DGX Spark

The sequenced plan to get the whole Gemma family running with NVFP4 KV (and, where it
applies, native FP4) on GB10 / `sm_121`. It is a **ladder**, not a checklist: each rung
adds exactly **one** new complication on top of the last, so we are never debugging two
unknowns at once. This is the Gemma-specific campaign plan; `docs/COMPATIBILITY_BOARD.md`
is the live status snapshot, and the per-lane mechanics live in
`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md` and `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`.

> Provenance note: Gemma 4 is past the assistant's training cutoff. Its lineup and
> per-variant encoder/modality architecture below are **operator-provided** and must be
> confirmed from the running model on each rung before any serving row is green.

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
| **Gemma 3** (1/4/12/27B) | dense, SWA 5:1, **uniform head dim 128 on audited 27B config**, mature | SWA / hybrid local-global KV | vLLM/SGLang |
| **Gemma 3n** (E2B/E4B) | dense mobile, PLE, audio, MatFormer/elastic, KV-share | mobile memory machinery | llama.cpp/LiteRT |
| **Gemma 4 E2B/E4B** | dense mobile, PLE, audio (supersedes 3n) | (inherits 3n machinery) | llama.cpp/LiteRT |
| **Gemma 4 31B** | dense, **encoder-based** multimodal (text+vision), D=512 | dense D=512 mixed-KV (run text-only) | vLLM/SGLang |
| **Gemma 4 26B-A4B** | **MoE** (~4B active), encoder-based multimodal (text+vision), D=512 | MoE on top of D=512 (run text-only) | vLLM/SGLang |
| **Gemma 4 12B** | dense, **encoder-free** multimodal (**text+vision+audio**), D=512 | multimodality fused into the decoder/KV — hardest | vLLM/SGLang |

## Rung -1 — Config audit (first pass; re-confirmed every rung)
Audit the actual config of every variant we plan to climb: head dim (uniform vs dual, and
*where* `D=512` appears), SWA pattern, dense vs MoE, encoders (vision/audio vs
encoder-free), PLE. This is the cheap planning pass from config files — but it is *not* the
final word. **Every rung re-measures geometry from the running model** (per the evidence
gate), because config can be ambiguous and the running model is what actually dispatches.
Treat rung -1 as "plan the ladder"; treat the per-rung measurement as "confirm the rung."
It decides whether the top of the ladder is clean:
- **If `D=512` lives only in the 26B-A4B**, then 31B is uniform-head ("bigger 12B", no new
  category) and the 26B-A4B hits **D=512 + MoE together** — a double-step this lineup
  can't cleanly isolate. Plan a mitigation (e.g. prove the D=512 mixed-KV dodge on the
  26B-A4B's attention with MoE held at a known-good weight path first).
- **If 31B also has the dual head dim**, then 31B isolates D=512 (dense) and 26B-A4B adds
  only MoE on top — clean. Preferred outcome; confirm it.

**2026-06-08 result:** done in `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` with machine-readable
artifact `results/gemma_rung_minus1_config_audit_20260608.json`. A stricter variant pass is
recorded in `results/gemma_rung_minus1_config_audit_strict_20260608.json`. The audit
settled the config-derived attention ladder and records the operator-provided modality
ordering:

- Gemma 3 27B normalizes to uniform `head_dim=128`, not `256`; it still cleanly isolates
  SWA/hybrid local-global KV because it has no `global_head_dim=512`.
- `D=512` appears in Gemma 4 12B, 31B, and 26B-A4B. Gemma 4 31B is dense and therefore
  isolates dense `D=512` before 26B-A4B adds MoE.
- Operator-provided architecture makes 12B the hardest server rung despite its smaller
  size: Gemma 4 12B is encoder-free multimodal, so multimodality is fused into the
  decoder/KV path instead of being quarantined by text-only serving.

## Main ladder — vLLM NVFP4-KV capacity
- **Rung 0 — Qwen (standard attention).** ✅ vLLM done (1.751× fp8 KV, clean,
  `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`). SGLang in progress
  (long-sequence quality bug; see its direction doc). Foundation for everything above.
> **Isolation lever (rungs 1–3):** Gemma 3 27B, Gemma 4 31B, and 26B-A4B are all
> **encoder-based**, so **serving text-only quarantines vision in the unfired encoder** and
> isolates the KV/attention path cleanly. Only the encoder-free 12B (rung 4) forces
> multimodality into the KV cache — which is exactly why it moved from a stepping stone to
> the final rung. Net climb: SWA → dense D=512 → MoE → multimodal-in-KV, one new thing each.

- **Rung 1 — Gemma 3 27B (text-only).** *New: SWA / hybrid local-global KV pool.* Uniform
  head dim 128, dense, encoder-based (vision quarantined by serving text-only), mature, no
  D=512. All layers FP4 — no mixed *dtype* needed yet, so this isolates the SWA KV plumbing
  alone. **Also a shippable capacity win** — first big Gemma-family NVFP4-KV row. vLLM:
  hybrid-SWA KV pool. SGLang: SWA subpool delegation.
- **Rung 2 — Gemma 4 31B (text-only).** *New: dense D=512 mixed-KV.* Encoder-based
  (text-only quarantines vision) + dense (no MoE confound) → the clean place to prove the
  per-layer mixed-KV dodge (NVFP4 local D=256 / bf16-or-fp8 global D=512). The capacity hero:
  biggest dense KV. This is where D=512 gets solved.
- **Rung 3 — Gemma 4 26B-A4B (text-only).** *New: MoE — NVFP4-MoE-weights + expert routing
  — on a D=512 attention/KV base already solved.* Encoder-based (text-only isolates KV).
  Most existing evidence (AEON image, ~48 tok/s NVFP4-weights). Adds exactly one thing: MoE.
- **Rung 4 — Gemma 4 12B — the destination.** *New: encoder-free multimodal **through the
  KV**.* Vision+audio are fused into the decoder, so unlike rungs 1–3 you can't quarantine
  them by serving text-only — multimodality is in the KV path itself. By here text + D=512 +
  MoE are all solved, so the one new thing is multimodal KV. **Highest payoff:** multimodal
  contexts are KV-heavy, exactly where FP4-KV capacity flexes hardest.

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

## Risk note — Gemma attention is outlier-sensitive (2026-06; affects what "green" means)
Checkpoints are NOT a blocker: NVFP4 weight releases exist for the server variants —
official NVIDIA `nvidia/Gemma-4-31B-IT-NVFP4` and `nvidia/Gemma-4-26B-A4B-NVFP4`, plus
`RedHatAI/gemma-4-31B-it-NVFP4` and community 12B quants. (And the KV-cache work needs no
NVFP4 checkpoint at all — KV dtype is a runtime knob; Rung 1 ran stock `gemma-3-27b-it`.)

But the *way* those releases are built is a warning for our KV work: **"NVFP4 Gemma 4" is
NVFP4 FFN/experts + BF16 attention.** NVIDIA and RedHat both deliberately keep self-attention
(q/k/v/o) in BF16 because Gemma's residual stream carries persistent per-channel activation
outliers far larger than ±6×block-scale, which 4-bit cannot represent. The KV cache *is*
4-bit attention state — so the same outlier sensitivity that forced attention weights to
stay BF16 may make NVFP4-KV **quality-lossier on Gemma than on Qwen**, even after the
kernel page-pairing bug is fixed. (Not certain — K/V are post-projection / post-QK-norm, and
NVFP4's two-level scaling targets outliers — but do not assume Qwen's PPL transfers.)

Consequence for the gate: on every Gemma rung, "correct output" must include a **PPL/quality
comparator measured on Gemma specifically** (vs fp8/bf16 KV), not a smoke pass and not
Qwen's number. If the Gemma KV PPL delta is large, the honest result may be "NVFP4-KV is a
capacity win on Qwen-class attention but lossy on Gemma's outlier-heavy attention" — which is
a finding, not a failure, and must be reported as such.

Speed context: NVFP4 *weights* already make Gemma 4 fast on the Spark (26B-A4B MoE ≈ 52 tok/s,
dense 31B ≈ 7–11 tok/s) — that's the existing weight-bandwidth win, not ours. Our
differentiated layer is NVFP4-KV *capacity* on top, so frame rung results as
"capacity added," not "speed."

## First moves
1. **Rung 1: Gemma 3 27B on vLLM** — the SWA-isolation rung and first big Gemma capacity
   win. Climb only after it's green.
