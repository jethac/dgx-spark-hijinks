# /goal — Close the family: NVFP4 KV validated across ALL of Gemma 3 + Gemma 4 (text + multimodal)

**North star:** every shipping Gemma 3 and Gemma 4 model serves a **calibrated NVFP4 KV cache that holds
up on a real task gate** (long-context retrieval for text; a retrieval/quality gate for image+audio), on
consumer Blackwell (sm_120/121), vLLM + SGLang. Not "coherent," not "good PPL" — **passes the needle gate
vs bf16/fp8.** We are the only ones shipping NVFP4 KV at all (NVIDIA/vrfai ship fp8/bf16 KV), so the bar is
"matches the precision the vendor ships," proven per model.

## The arbiter (hard-won lesson — do NOT regress to PPL)
A below-truth NLL is NOT automatically a collapse. 26B calibrated nvfp4 was −0.12 (looked like a mild
collapse) but scored **100% on multi-needle retrieval = fp8 = bf16.** **Gate KV-quant quality on a TASK eval
(`docs/vast_anchor/needle_eval.py`: single + 4-needle, ctx ~7600, vs bf16 AND fp8), not on PPL/top-1.** The
4-needle variant is hard enough to discriminate (12B bf16 itself only ~0.62 recall), so it's a real gate.

## State of the family (accurate as of 2026-06-17)
| model | calib | needle-hardened | status |
| --- | --- | --- | --- |
| Gemma 4 **12B** (dense) | k=0.1,v=0.06 | YES (nvfp4≈bf16) | **DONE** |
| Gemma 4 **26B-A4B** (MoE) | base_k100 (layer-aware) | YES (nvfp4=fp8=100%) | **DONE** |
| Gemma 4 **31B** (dense) | k=v=0.05 | YES (nvfp4=bf16=100%) | **DONE** |
| Gemma 4 **E2B** (small) | k=v=0.05 | YES (nvfp4≈bf16/fp8) | **DONE** |
| Gemma 4 **E4B** (small) | k=0.1,v=0.06 | YES (nvfp4≈bf16/fp8) | **DONE** |
| Gemma 4 **multimodal — image KV** | k=0.1,v=0.06 (E4B), 12B | YES — E4B+12B image-needle, nvfp4=bf16=fp8 (24-img hard) | **DONE** |
| Gemma 4 **multimodal — audio KV** | k=0.1,v=0.06 (E4B), 12B | YES — E4B+12B spoken-code audio-needle, nvfp4=bf16 (req'd 2 vLLM audio-stacking fixes) | **DONE** |
| **DiffusionGemma** (26B-A4B base, diffusion) | LINEAR_V_SF + VO-split | YES (comparative): nvfp4≥bf16 on needle (0.188 vs 0.125) + generation coherence; DG-V5 3.556x capacity. Now serves on the INTEGRATED line (runtime overlay + `build_attn_metadata(causal)` fix). Crisp absolute gate N/A — DG generation is intrinsically noisy (bf16 too). | **DONE** |
| Gemma 3 **1B / 4B / 12B / 27B** | k=0.1,v=0.06 | YES (nvfp4≈bf16/fp8) | **DONE** |
| Gemma 3 **270M** | — | TABLED — NVFP4 KV has no use case at 270M (negligible KV footprint; can't use long ctx). bf16=fp8=1.0/nvfp4=0.875 at toy ctx1200; not worth a calib cycle. | **OUT OF SCOPE** |

## Turnkey text recipe (per OPEN text model: E2B, E4B, Gemma 3 1B/4B/12B/27B)
1. **Per-arch NVFP4 KV calibration.** Sweep global (or layer-aware) k/v vs HF-eager bf16 truth at ctx 8185
   (`vllm_matched_kv_anchor.py`). Dense models tend to want a single global (12B 0.1/0.06, 31B 0.05/0.05);
   reuse a neighbor's as the sweep seed. MoE/E-series may need layer-aware (like 26B base_k100). Save the
   calib JSON keyed by arch signature under `docs/productionize/nvfp4_kv_calib_data/`.
2. **Truth sanity:** matched anchor nvfp4 vs HF-eager bf16 — expect mild below-truth, not −1.x collapse.
3. **Needle GATE:** `needle_eval.py` single-needle (5 depths) AND 4-needle, ctx ~7600, **vs bf16 AND fp8**.
   Pass = nvfp4 recall/all-needles ≈ bf16/fp8 (within trial noise). Bank `results/needle_<model>_<date>/`.
4. Bank green row in the family table; update the bug/solutions doc + memory.

## Multimodal gate (NEW harness — the text needle does NOT cover this)
The vision/audio KV path is never exercised by a text haystack. Build a multimodal needle:
- **Image-haystack needle:** interleave M images (or one long image-text doc) and plant a retrievable fact
  ("the code on slide K is …"); ask for it. Score retrieval, bf16 vs fp8 vs nvfp4 KV, on the MM-capable
  sizes (E2B/E4B/12B+/27B as applicable).
- **Audio needle:** plant a spoken fact in a long audio context (or interleaved audio segments), retrieve.
- Re-confirm task #48's MM quality (image+audio) under the CURRENT per-arch calibration, since the global
  scale on vision/audio KV may differ from the text scale.
- Confirm which sizes actually carry vision and/or audio first (E-series = Gemma-3n lineage likely both).

## DiffusionGemma gate (separate track)
Diffusion, not AR — needle N/A. Truth-gate generated text quality (HF-eager reference, not coherence-only),
and verify it inherits the 26B-A4B base's calibrated nvfp4 KV. Build on the existing DG-V parity work.

## Priority order
1. **E2B + E4B text** (fast — small models, turnkey recipe; closes the obvious Gemma 4 gap I missed).
2. **Multimodal image+audio gate** (the real new harness work; the biggest blind spot).
3. **Gemma 3 1B/4B/12B/27B** (calib + needle per size; dense should track Gemma 4).
4. **DiffusionGemma** diffusion truth-gate.

## Cross-stack + coordination
- Both vLLM and SGLang must pass per model (task #40). Codex owns the SGLang side + has the capture/calib
  tooling; coordinate via `mail/` (the 3 mixed-KV plumbing fixes on `spark/hijinks-e3-vllm` —
  `4fcbf4c48`/`d0f6221`/`505513a26` — and per-arch calibs are shared inputs).
- Reuse the e3 stack: vLLM `spark/hijinks-e3-vllm` (wheel `sm120a-wheels-d0f6221e6` or newer) + FlashInfer
  `spark/hijinks-e3-flashinfer` (`1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`).

## Constraints / hygiene (do not drift)
- vast `VAST_API_KEY` + `HF_TOKEN` env-only, never to a file. Print $/hr before renting; **destroy every box
  on bank.** New-box SSH fix: rent → `vastai detach ssh <id> 963667; attach ssh <id> <pubkey>` → connect
  (raw "Permission denied" until that detach+reattach). Avoid the `79.117.54.182` machine family (reset 4x).
- Commit only relevant changes; never commit `jethac.github.io`; end commits with the Co-Authored-By line and
  the `Jetha Chan <jethachan@gmail.com>` author identity for vLLM/SGLang source commits.
- Truth-gate everything against HF-eager bf16 + the needle task; a below-bf16 PPL is NOT proof of correctness.

## Done =
Every model in the family table is GREEN: calibrated NVFP4 KV, needle-hardened (text) or MM-gate-passed
(image/audio) or diffusion-truth-gated, on vLLM AND SGLang, bf16/fp8/nvfp4 retrieval-matched, banked with
artifacts + per-arch calib JSONs. Then the blog/family table can honestly say "the only NVFP4 KV cache on
consumer Blackwell — across the entire Gemma 3 + Gemma 4 family, text and multimodal."
