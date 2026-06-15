# /goal — Codex (SGLang lane), epoch-3

**North star (the ship bar, Jetha's words):** *we cannot call this campaign successful until Gemma 3 +
Gemma 4 (including 26B-A4B) + DiffusionGemma all ship with a working NVFP4 KV cache* — and SGLang must match
vLLM (task #40). vLLM + FlashInfer are now rebased to **epoch-3** (vLLM `v0.23.0`, FlashInfer `main`) and
serve-validated green. SGLang now needs to catch up to the same FlashInfer, and the shared 26B-A4B nvfp4
kernel bug needs closing.

State at handoff (mail 0185): vLLM `jethac/vllm@spark/hijinks-e3-vllm` + FlashInfer
`jethac/flashinfer@spark/hijinks-e3-flashinfer` (upstream/main `c15ac84c` + our 20 nvfp4-KV commits, clean
cherry-pick) DONE and validated. SGLang 12B was green (your deferred-sidecar, 0163); SGLang 31B greened
(Claude, calib `k=v=0.05`, mail 0173); SGLang 26B-A4B was bf16-only (fp8 hit the FlashInfer D512 crash because
the image's FlashInfer lacked the reject; nvfp4 broken).

## Priority 1 — BUILD THE SGLANG EPOCH-3 STACK (the missing third rebase)
The epoch-3 stack is THREE rebases: vLLM→v0.23.0 (done), FlashInfer→main (done), **SGLang→latest (yours)**.
Without this, SGLang is still on the epoch-2 FlashInfer and can't match the rebased vLLM.
1. **Pick the target SGLang upstream** first — SGLang moves on its own cadence, so identify the latest
   `sgl-project/sglang` release/commit that is API-compatible with FlashInfer `main` (the e3 FlashInfer pins
   `flashinfer-ai/flashinfer@c15ac84c`). That choice drives everything below.
2. **Rebase `jethac/sglang`** (our nvfp4-KV + Gemma-4 + sidecar-calibration patches) onto that SGLang, and
   **pin FlashInfer to `spark/hijinks-e3-flashinfer`** (the ONE shared ref — vLLM e3 uses it too). Same
   cherry-pick-onto-clean approach Claude used for vLLM (identity `Jetha Chan <jethachan@gmail.com>` for all
   our commits; defer/port any heavily-restructured shared file once rather than per-commit).
3. Rebuild the SGLang source-stack image off this e3 SGLang + e3 FlashInfer → push to ghcr; that baked image
   is the SGLang epoch-3 stack everything else here runs on.
2. Two e3 API-drift gotchas the *compile* won't catch (Claude hit these on vLLM; SGLang may too):
   - v0.23.0-era `validate_configuration` calls `FlashInferBackend.supports_combination(..., device_capability)`
     — make `device_capability` optional/threaded.
   - metadata fields (e.g. `mm_req_doc_ranges`) can get dropped in conflict resolution while references
     survive — grep for `AttributeError` on attention-metadata objects after cherry-pick.
3. Rebuild your Spark source-stack image off the e3 FlashInfer; re-confirm SGLang **12B** (sidecar calib) and
   **31B** (`k_global=v_global=0.05`) serve green on the rebased stack (matched anchor vs bf16).

## Priority 2 — 26B-A4B on the rebased stack (the ship gate)
- **fp8:** the e3 FlashInfer now carries the `FA2_REJECT_IF_KV_SMEM_INSUFFICIENT` clean-reject, so SGLang 26B
  fp8 should now either run or return an actionable message instead of the raw `NUM_MMA_KV=1` tvm crash.
  Re-test; if it still can't run fp8 D512, that path needs the in-loop-dequant kernel change (deep).
- **nvfp4:** SHARED kernel bug with vLLM. The read math is FlashInfer-common, so a FlashInfer fix benefits both
  stacks. Claude is driving the read-vs-dequant+SDPA capture on vLLM+vast (your 0182 request) — **you do the
  SGLang-side reader instrumentation** (you have `SGLANG_FP4_KV_TRACE_GLOBAL_SCALE` + the dense-cache trace).
  Converge the fix in `spark/hijinks-e3-flashinfer`. Re-scope your "26B MoE pool / negative token count" red:
  Claude's evidence says it's likely this nvfp4 read bug, not pool-sizing (confirm with fp8-vs-nvfp4-vs-HF-truth).

## Priority 3 — DiffusionGemma SGLang text-quality
- After the 26B nvfp4 kernel question lands (DiffusionGemma is the 26B-A4B base). Truth-gate it (HF-eager
  reference, not coherence-only) — same base, same nvfp4 risk as 26B-A4B-it.

## Constraints / hygiene
- Spark marker protocol unchanged; yield to `CLAUDE_WINDOW_OPEN`; password never in any file (key-based).
- Where possible run the kernel work on **vast.ai sm_120** (the fp8-D512 crash and the nvfp4 break both repro
  on any sm_120 5090/PRO-6000 — no Spark needed for the hunt); Spark only for the arm64 SGLang serving last mile.
- Commit only relevant changes; Co-Authored-By line. Coordinate via `mail/` (even=codex / odd=claude).

## Done =
SGLang rebased onto the e3 FlashInfer; 12B + 31B green on it; 26B-A4B has a viable quantized-KV path (fp8 via
the reject, or full-nvfp4 once the shared FlashInfer reader fix lands); DiffusionGemma truth-gated. SGLang AR
ladder (task #40) matches vLLM on the epoch-3 stack.
