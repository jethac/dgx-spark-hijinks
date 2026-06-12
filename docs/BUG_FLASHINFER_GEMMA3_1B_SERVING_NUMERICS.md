# BUG: FlashInfer serving-path numerics wrong at Gemma 3 1B geometry (d256 / SWA-512 / 1 KV head)

Status (UPDATED 2026-06-12 by the 1B RIGOROUS re-test, § below): the **bf16 arm
is NOT a bug — defect (A) is REFUTED. It was an ENVIRONMENTAL / false-green
ARTIFACT.** A controlled 1B re-run (same `g6adc00f70` wheel, same FlashInfer
`7d5d477b` JIT, FlashInfer verified-engaged from the engine proof line, util 0.6,
serve-ready timeout 300 s as a wedge detector) gives **FI-bf16 − FLASH_ATTN =
−0.0006633 nats and NO engine wedge** — the earlier +0.221/wedge does NOT
reproduce. Both bf16 backends match HF eager truth (2.359755). This closes the
"width/depth-driven bf16 inflation" hypothesis the 270M test left open (1B is
also clean). Only ONE real sm_120 defect remains: **(B) the nvfp4 KV read-path
defect** (model/size-independent: 1B + 270M + 4B-mm all gibberish; 1B re-test
+1.587 nats / gibberish). The earlier "+0.22–1.38 nats bf16 inflation" and the
"wheel engine-wedge" are now attributed to methodology artifacts: the
`VLLM_ATTENTION_BACKEND` false-green fallback and the util-0.3 OOM / slot-mapping
JIT regime on the WDDM-shared card.

Bug RE-SCOPED: bf16/wedge arm CLOSED (no bug); nvfp4 read-path arm OPEN.

--- original status, now superseded ---
Status: OPEN, root cause unknown. Observed 2026-06-12 on sm_120; confirmed
sm_120-specific (Spark/sm_121 bisect clean). As of the 270M repro test (§ below)
the bug is understood to be TWO SEPARABLE sm_120 defects: (A) an FI-bf16 SWA-512
long-context inflation that tracks model width/depth (1B yes, 270M no) and also
manifests as the wheel engine-wedge; (B) an nvfp4 KV read-path defect that is
model/size-independent (1B + 270M + 4B-mm all gibberish). 270M is a fast minimal
repro for (B) but NOT (A).
Found by: P520 Gemma 3 1B serving verification (zero-bug diagnostics).
Severity: FI-bf16 quality silently wrong (+0.22 to +1.38 nats); FI-nvfp4
unusable (deterministic gibberish). Coherent short-prompt chat MASKS the bug.

## Environment

- P520: RTX 5060 Ti, CC 12.0 (sm_120), WSL2 Ubuntu 24.04, CUDA 13.0 (nvcc
  V13.0.88), torch 2.12.0+cu130.
- vLLM: jethac/vllm spark/hijinks-022-gemma4-mixed-kv @ 9759e3b06 (exact
  r9-image code), editable build, TORCH_CUDA_ARCH_LIST=12.0a, 42 sm_120a
  cubins confirmed, NVFP4 linear-latch diag PASS (head 128 and 256).
- FlashInfer: jethac/flashinfer spark/hijinks-022-fa2-d512 @ 7d5d477b,
  JIT-compiled on box (NOT the Spark AOT path).
- Model: google/gemma-3-1b-it - uniform head_dim 256, GQA 4q/1kv (the only
  Gemma with a single KV head), sliding window 512 (smaller than the 1024 of
  the larger Gemma 3 sizes), 5:1 sliding:global.

## Evidence (results/p520_gemma3_1b_serving_20260612/, ctx 8191, 8190 scored)

Truth references, agreeing to <0.001 nats on all corpora:
- HF transformers eager bf16: C1/C2/C3 = 2.35778 / 3.21392 / 1.42429
- vLLM FLASH_ATTN-backend bf16 serving row: matches HF on all three.

Deviations from truth (nats, C1/C2/C3):
| FI row | delta vs truth | chat smoke |
|---|---|---|
| bf16 | +0.221 / +1.243 / +1.380 | coherent ("Tokyo") |
| fp8_e4m3 | +0.006 / +0.159 / +0.494 | coherent |
| nvfp4 (+linear V-SF) | +1.592 / +2.436 / +2.752 | GIBBERISH, deterministic |

All FI rows internally deterministic. nvfp4 gibberish reproduced
byte-identical on a VIRGIN FlashInfer JIT cache (not stale kernels); writer
latch clean, so suspicion is on the read path. Pre-diagnostic JIT cache
preserved at WSL ~/.cache/flashinfer_prediag_070355 for forensics.

## Diagnostic structure (three tells)

1. Short-prompt chat coherent while long-ctx PPL inflated: chat prompts stay
   inside the 512-token sliding window; ctx-8191 scoring crosses it
   constantly. Points at sliding-window boundary handling in the FI serving
   path (paged KV at depth), which single-call kernel probes never
   reproduced (6/6 probes passed at 31B/E4B geometries).
2. fp8 closer to truth than bf16 on the SAME backend: dtype-conditional
   kernel templates / tile dispatch differ, so the defect is likely in a
   path-conditional spot, not shared mask math.
3. Novel geometry axes vs everything previously tested: window 512 (not
   1024) AND kv_heads=1. Either could be the unexercised path.

## What is and is not contaminated

- NOT contaminated: all sm_121 (GB10) results - Triton-vs-FlashInfer pairs
  within 0.04 nats across 5 sizes (12B-31B geometries) corroborate each
  other; G3-12B/27B FI rows checked against FLASH_ATTN/Triton baselines.
- Contaminated/blocked: any Gemma 3 1B FI claim; Gemma 3 bf16 retirement
  flip (scoped back to Gemma 4-only, see TRITON_RETIREMENT_SCORECARD
  adjudication log); Gemma 3 1B nvfp4 support claim (row banked RED).

## Bisect + investigation plan

1. Gemma 3 1B rerun on sm_121 Spark (cheap; next available window): same
   deviation -> geometry bug (window-512 / 1-kv-head path, platform-
   independent, simply never reached); clean -> sm_120-specific (JIT codegen
   or arch-conditional path). 
2. Logit-level diff probe FI vs FLASH_ATTN at exact 1B geometry on P520:
   find the first divergent layer/position; check position vs window
   boundary (expect divergence onset near token 512 if tell #1 is right).
3. Geometry ablation in the probe harness: window 512 vs 1024, kv_heads 1
   vs 2, d256 fixed - isolate the axis.
4. Once root-caused: fix on jethac/flashinfer branch + upstream filing with
   minimal repro (this doc is the filing draft skeleton).

## Bisect result 2026-06-12 (Spark / GB10, sm_121): PLATFORM / sm_120-specific

Bisect step 1 of the plan above, run on the Spark (GB10, sm_121) with the
baked r9 image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id 8c37bdbc4fdb), `google/gemma-3-1b-it`, C1 ctx 8191, each cell run TWICE
bitwise. Backend forced via `--attention-backend FLASH_ATTN|FLASHINFER`,
engaged backend verified from the `Using AttentionBackendEnum.<X> backend.`
proof lines (not the flag).

| row | backend (proof) | kv dtype | C1 ×2 (bitwise) | smoke |
|---|---|---|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.356493110435786 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.359283581557766 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 2.400746942552027 (IDENTICAL) | "Tokyo" COHERENT |

**FI-bf16 − FLASH_ATTN-bf16 on the Spark = +0.00279 nats** (P520 was +0.221).
That is well under the 0.01 threshold → FlashInfer MATCHES FLASH_ATTN at the
exact 1B geometry on sm_121. **Verdict: PLATFORM / sm_120-specific. The bug
does NOT reproduce on the Spark — the GEOMETRY hypothesis is REFUTED.** NVFP4 KV
is COHERENT and deterministic on the Spark (+0.044 vs truth) where the P520 gave
deterministic gibberish (+1.59). The Spark FLASH_ATTN truth row (2.35649) is
within 0.0014 nats of both the P520 FLASH_ATTN row (2.35785) and the HF eager
truth (2.35778), confirming the setup is sound.

Implication: the defect is scoped to sm_120 (the P520 editable / source-tree
FlashInfer build — JIT codegen or an arch-conditional path), NOT the
d256/SWA-512/1-kv-head geometry. The Spark/sm_121 Gemma 3 1B path (bf16 AND
NVFP4 KV) is fine. The Gemma 3 1B NVFP4 row banked RED on the P520 is
GREEN-class on the Spark. Bug stays OPEN but re-scoped to sm_120; investigation
steps 2-3 (P520 logit-diff probe + window/kv-head ablation) remain the way to
localize the sm_120 root cause. Artifacts:
results/claude_1b_bug_bisect_20260612/ (BISECT_SUMMARY.md, ppl JSONs ×2/row,
smoke transcripts, proof lines); Spark master copy + server logs + token dumps
at /home/jethac/spark_tmp/claude_1b_bug_bisect_20260612/.

## Baked-WHEEL disambiguation 2026-06-12 (P520 / sm_120): bug PERSISTS, harder form

Task #37. Re-ran the 1B bisect with the **sm120a RELEASE WHEEL**
(`vllm 0.1.dev1+g6adc00f70.sm120a`, from `spark/hijinks-e2-vllm @ 6adc00f70`)
instead of the editable build (`9759e3b06`), FlashInfer JIT-from-source
`7d5d477b` in BOTH (controlled variable). Provenance gates GREEN: EXT_PATH =
wheel `_C_stable_libtorch.abi3.so`, 42 sm_120a cubins, linear-V-SF latch
"writer wrote LINEAR V-SF" (hd128+hd256).

**The bug did NOT vanish — it WORSENED into an engine wedge.** Where the editable
build returned wrong-but-finite numbers (+0.221/+1.243/+1.380 nats), the wheel
**deadlocks the vLLM engine on the first window-crossing (>512-token) request**
at this geometry:
- 16-token prompt_logprobs → valid, server coherent ("Tokyo").
- 600-token (crosses SWA-512) COLD → server heartbeat logger FREEZES after a
  `Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel`
  warning; GPU 100% util / ~34 W (idle-spin, not matmul); `Running: 0 reqs`;
  curl times out at 120 s.
- WARMED 600-token (short req first) → RC=0 in 1 s, so the 512-crossing path is
  functionally fine once its slot-mapping shape is compiled.
- ctx-8191 prompt_logprobs → still wedges the engine even AFTER a 700-token
  warmup (socket ESTABLISHED to server, heartbeat logger silent >10 min). NO
  FLASH_ATTN/FLASHINFER/nvfp4 PPL rows could be produced.

Implication: the wedge is on the **FLASH_ATTN** backend's serving/slot-mapping
path (not only FlashInfer), tied to the SWA-512 window crossing on sm_120 — so
the defect is **NOT an editable-vs-wheel `_C` artifact**; it persists on the
release wheel and keeps suspicion on the sm_120 long-context / SWA-512 serving
path. Named confound: the wheel env excludes the optional accel-kernel pip
extras (tilelang/quack/tokenspeed/humming/cutlass-dsl/flashinfer-python|cubin —
omitted to keep FlashInfer source-tree JIT authoritative); a missing accel
kernel could aggravate the slot-mapping slow-JIT. Next localizer: the Colab G4
sm_120 cell at this geometry (off-WSL, bigger sm_120 card), and a P520 rerun
with the full accel-kernel pip set to rule the confound in/out.

Same nvfp4-on-sm_120 GIBBERISH also reproduced in the **Gemma 3 4B multimodal**
nvfp4-KV smoke (`results/p520_mm_retirement_smokes_20260612/`, cell
`g3_4b_nvfp4_fi`): coherent under bf16, gibberish under nvfp4 KV — confirming the
nvfp4 KV READ defect on sm_120 is model/size-independent and independent of the
mm-prefix mask path (the mask machinery ran; bf16 mm is clean).

Artifacts: results/p520_1b_wheel_disambig_20260612/DISAMBIG_SUMMARY.md
(server logs incl. `_WEDGED`, diag1/diag2 run logs, proof lines).

## 270M repro test 2026-06-12 (P520 / sm_120): SPLIT — nvfp4 REPRODUCES, FI-bf16 does NOT

Task #37 follow-up. Re-ran the bisect on `google/gemma-3-270m-it` to ask whether
the bug is 1B-geometry-specific or broad across small Gemma 3. Same **sm120a
RELEASE wheel `g6adc00f70`** + FlashInfer JIT `7d5d477b` as the task #37 1B
disambig. Backend engaged verified from the `Using AttentionBackendEnum.<X>
backend.` proof line (FI-bf16 and FI-nvfp4 both confirmed FLASHINFER, no
false-green). C1 ctx 8191, each row run TWICE bitwise, util 0.6 for the FI rows
(util 0.3 OOM'd on the WDDM-shared card — `No available memory for the cache
blocks`, environmental; FLASH_ATTN truth row ran at 0.3).

270M GEOMETRY: hidden 640, 18 layers, 4 attn heads, **head_dim 256,
num_key_value_heads 1, sliding_window 512** — shares ALL THREE suspect axes
(d256 / SWA-512 / 1-kv-head) with the 1B; only width (640 vs 1152) and depth (18
vs 26) differ. HF bf16 eager C1 ground truth = 2.912124 nats.

| row | backend (proof) | kv dtype | C1 ×2 (bitwise) | delta vs FLASH_ATTN | smoke |
|---|---|---|---|---:|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.911488 (IDENTICAL) | — (+0.00064 vs HF) | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.912821 (IDENTICAL) | **+0.00133** | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 11.034120 (IDENTICAL) | **+8.122** | **GIBBERISH, deterministic** |

**Verdict: SPLIT — the two sm_120 defects are SEPARABLE.**
- **FI-nvfp4 REPRODUCES**: deterministic gibberish + 11.034 nats (Cyrillic/JP
  salad, same failure class as the 1B's nvfp4 RED). Combined with the 1B and the
  4B-mm smoke, the **nvfp4 KV read-path defect is model/size-independent on
  sm_120**, and **270M IS a valid fast minimal repro for it** (~90 s to serve,
  ~10 s/PPL, 0.5 GB).
- **FI-bf16 does NOT reproduce**: +0.00133 nats vs FLASH_ATTN (the 1B was
  +0.221, ~165× larger) — effectively clean. So the **bf16 SWA-512 inflation arm
  is NOT a pure function of the d256/SWA-512/1-kv-head geometry** (270M has the
  geometry and stays clean); it scales with width/depth or a 1B-specific path.
  **270M is NOT a minimal repro for the bf16-inflation arm** — use the 1B for it.
- No engine WEDGE at 270M either: all C1 ctx-8191 (window-crossing) requests
  served fine on all three backends, where the 1B wheel run wedged. The wedge
  arm, like the bf16 inflation, did not reproduce at 270M.

Artifacts: results/p520_g3_270m_20260612/ (SUMMARY.md, per-row proof_lines,
chat_smoke.json, c1/c1b_ctx8191_ppl.json ×2, hf_bf16_reference_ppl.json,
util-0.3 OOM crash excerpt, status.txt, scripts).

## 1B rigorous re-test 2026-06-12 (P520 / sm_120): bf16 defect REFUTED — environmental/false-green ARTIFACT

Task #37 decisive re-run. The old 1B numbers were suspect on two counts: (i) one
earlier run used the `VLLM_ATTENTION_BACKEND` false-green trap that silently ran
FLASH_ATTN while claiming FlashInfer; (ii) the 270M agent established that
`--gpu-memory-utilization 0.3` OOMs the FlashInfer rows on this WDDM-shared 5060
Ti (a capacity ValueError mistakable for a wedge). Re-ran cleanly at util 0.6
with FlashInfer **verified-engaged from the engine proof line**, serve-ready
timeout 300 s as an explicit wedge detector.

Same **sm120a RELEASE wheel `g6adc00f70`** + FlashInfer source-JIT `7d5d477b`
(the exact stack of the task #37 wheel-disambig that reported the wedge, and of
the 270M test). Model google/gemma-3-1b-it. Corpus C1 md5 abb63f0e... (verified),
ctx 8191, 8190 scored, each row run TWICE bitwise. HF transformers bf16 **eager**
C1 ground truth on the identical token window = **2.359755277633667 nats**.

| row | backend (engine PROOF line) | kv dtype | C1 ×2 (bitwise) | delta vs FLASH_ATTN | wedge | smoke |
|---|---|---|---|---:|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN (FA v2) | bf16 | 2.3578483823599337 (IDENTICAL) | — (−0.00191 vs HF) | no | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.3571850630239095 (IDENTICAL) | **−0.0006633** | **no** | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 3.9452781399784085 (IDENTICAL) | **+1.5874298** | no | **GIBBERISH, deterministic** |

FI-bf16 forced with `--attention-backend flashinfer` AND
`VLLM_FLASHINFER_BF16_GEMMA=1`; engine proof line confirms
`Using AttentionBackendEnum.FLASHINFER backend.` with `attention_backend:
'flashinfer'` in non-default args — **no false-green FLASH_ATTN substitution**.

**Verdict: the bf16 1B FlashInfer defect is an ENVIRONMENTAL / FALSE-GREEN
ARTIFACT — it does NOT reproduce.**
- **bf16 inflation REFUTED:** FI-bf16 − FLASH_ATTN = −0.00066 nats (the old
  suspect was +0.221, ~330× larger and the *opposite sign*). Both bf16 backends
  match HF eager truth to <3e-3. This is consistent with the Spark sm_121 bisect
  (+0.00279, clean) and the 270M re-test (+0.00133, clean): bf16 FlashInfer on
  Gemma 3 1B is clean everywhere once FlashInfer is verified-engaged at a
  non-OOM util.
- **wedge REFUTED:** no engine wedge on any window-crossing (>512 tok) ctx-8191
  `prompt_logprobs` request; all three rows served, ready in 146–149 s, never
  approaching the 300 s timeout. The earlier wheel-disambig "wedge" (a freeze
  after a `_compute_slot_mapping_kernel` Triton JIT, at util 0.3) was the
  artifact — at util 0.6 the slot-mapping path compiles and serves fine.
- **nvfp4 read-path defect (B) REPRODUCES:** deterministic gibberish, +1.587
  nats vs truth — broad sm_120 defect, unchanged. (Matches the original 1B nvfp4
  +1.592 and corroborates 270M/4B-mm.)

**Reconciliation of all 1B arms:** the only real defect on Gemma 3 1B is the
nvfp4 KV read path (sm_120-specific, broad across sizes). The "bf16 inflation"
and "wedge" were methodology artifacts (false-green fallback / util-0.3 OOM &
slot-mapping JIT), now retired. **Gemma 3 retirement flip: NO 1B bf16 caveat
needed; the nvfp4-KV caveat stands.** Task #37 narrows to localizing defect (B)
only (use 270M as the fast minimal repro).

Artifacts: results/p520_g3_1b_retest_20260612/ (SUMMARY.md, per-row
proof_lines.txt / server.log / chat_smoke.json / c1{,b}_ctx8191_ppl.json ×2,
hf_bf16_reference_ppl.json, status.txt, scripts/ incl. run_row.sh &
hf_bf16_reference.py).

## Cross-references

- results/p520_g3_1b_retest_20260612/ (P520 sm_120 1B RIGOROUS re-test — bf16 defect REFUTED/artifact, nvfp4 reproduces)
- results/p520_g3_270m_20260612/ (P520 sm_120 270M repro — SPLIT verdict: nvfp4 reproduces, FI-bf16 clean)
- results/p520_1b_wheel_disambig_20260612/ (P520 sm_120 WHEEL disambig — wedge verdict)
- results/p520_mm_retirement_smokes_20260612/ (g3-4b nvfp4 mm gibberish corroboration)
- results/claude_1b_bug_bisect_20260612/ (Spark sm_121 bisect — PLATFORM verdict)
- results/p520_gemma3_1b_serving_20260612/ (full artifacts incl. token
  dumps on the P520 side: B:\workshop\wsl_sm120\results\gemma3_1b_serving_20260612\)
- docs/RESULTS_LEDGER.md row (2026-06-12); mail/0044, mail/0056.
- docs/TRITON_RETIREMENT_SCORECARD.md adjudication log (flip scoping).
