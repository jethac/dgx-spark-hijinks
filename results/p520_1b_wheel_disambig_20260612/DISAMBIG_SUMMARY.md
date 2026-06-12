# Gemma 3 1B baked-wheel disambiguation (task #37) — P520 / sm_120

Date: 2026-06-12. Agent: p520-smokes. GPU: RTX 5060 Ti (CC 12.0), WSL2 Ubuntu.

## Goal

Re-run the Gemma 3 1B serving bisect with the **sm120a RELEASE WHEEL**
(`vllm 0.1.dev1+g6adc00f70.sm120a`, built from `spark/hijinks-e2-vllm @ 6adc00f70`)
instead of the prior **EDITABLE build** (`9759e3b06`). FlashInfer stays
**JIT-from-source @ 7d5d477b in BOTH** (the controlled variable). This swaps
only the vLLM `_C`/build path. Planned rows: FLASH_ATTN bf16 (truth),
FLASHINFER bf16, FLASHINFER nvfp4 (+LINEAR_V_SF), C1 ctx 8191, ×2 bitwise.

Provenance (Phase 1 gates, all GREEN): EXT_PATH = wheel
`_C_stable_libtorch.abi3.so`; 42 sm_120a cubins (matches source build);
NVFP4 linear-latch "writer wrote LINEAR V-SF" hd128+hd256. wheel md5
ad5e7fe06ef715550ee97b1c6763173a.

## RESULT: a NEW failure mode — the wheel WEDGES at long context, it does not
## merely shift numerics.

The disambiguation could NOT complete the PPL rows because the **release wheel
deadlocks the engine on the first window-crossing (>512-token) prompt at the
Gemma 3 1B geometry on sm_120**, where the prior editable build instead
RETURNED wrong-but-finite numbers (+0.221/+1.243/+1.380 nats).

Evidence (banked verbatim):

1. **Within-window works.** FLASH_ATTN bf16 server boots, serves, and the
   chat smoke is COHERENT ("The capital of Japan is **Tokyo**.").
   `diag1_probe_16.json`: a 16-token prompt_logprobs request returns valid.

2. **Crossing the 512 sliding window wedges (cold).** `diag1_run.log`:
   - probe 16 tokens  -> RC=0, 1s (valid)
   - probe 600 tokens -> **RC=28 (timeout) at 120s**  <- crosses SWA-512
   - probe 2000/6000  -> RC=22 instantly (server already wedged)
   Onset is at ~600 tokens = exactly the SWA-512 boundary (the bug doc's
   geometry axis). Server log shows
   `Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel`
   then the engine heartbeat logger FREEZES (no `loggers.py` line afterward),
   `Running: 0 reqs`, GPU 100% util at ~34 W (idle-spin signature, not real
   matmul which draws 100W+).

3. **Warmed-up 600-tok works → it is the in-inference JIT, not a numeric
   defect at 600.** `diag2_run.log` / `diag2_600tok_warmedup.json`: with a
   short warmup request first, a 600-token request returns RC=0 in 1s. So
   the 512-crossing path is functionally correct once its `_compute_slot_mapping_kernel`
   shape is compiled.

4. **The ctx-8191 PPL request still wedges the engine even after a 700-token
   warmup.** Full run (`flashattn_bf16_server_WEDGED.log`, status.txt):
   server SERVED in 97s, WARMUP_RC=0 (700-tok warmup absorbed ~270s of JIT),
   chat coherent — but the 8191-token `prompt_logprobs` request was ESTABLISHED
   to the server (ss showed the socket) yet the **APIServer heartbeat logger
   stopped emitting entirely** for >10 min, `Running: 0 reqs` frozen, GPU
   100%/34W. The engine event-loop wedges on the 8191 shape. Killed; no
   FLASH_ATTN/FLASHINFER/nvfp4 PPL rows could be produced.

## VERDICT (with the confound noted honestly)

- **The bug did NOT "vanish" with the wheel.** If anything it got WORSE: the
  editable build returned wrong numbers; the wheel **wedges the engine** at the
  same 1B SWA-512/d256/1-kv-head geometry on sm_120. The defect therefore is
  **NOT an editable-build artifact** — it persists (in a different, harder
  form) on the release wheel.
- **What this implicates:** the controlled variable is FlashInfer (JIT-from-
  source @7d5d477b in BOTH editable and wheel). The engine wedge appears on
  the **FLASH_ATTN** backend's serving path (slot-mapping/scheduler), not only
  FlashInfer, and is tied to the SWA-512 window crossing on sm_120. This keeps
  suspicion squarely on the **sm_120 long-context / SWA-512 serving path**
  rather than the vLLM-`_C` editable-vs-wheel axis. The Colab G4 cell (also
  sm_120, RTX PRO 6000) is the next split: it can run this geometry on a
  bigger sm_120 GPU off-WSL to separate "sm_120 codegen" from "WSL+P520".
- **Caveat:** the wheel excludes the optional accelerated-kernel pip extras
  (tilelang / quack-kernels / tokenspeed-mla / humming-kernels / cutlass-dsl
  pins / flashinfer-python|cubin — these were deliberately omitted to keep
  FlashInfer source-tree JIT authoritative per campaign discipline). It is
  POSSIBLE the `_compute_slot_mapping_kernel` slow-JIT / wedge is aggravated by
  a missing accel kernel that the editable env happened to have. This is a
  named open confound, not a clean win for either hypothesis.

## Bottom line for the campaign

- Gemma 3 1B on sm_120 (P520) remains **RED / unservable at long context** —
  now via an engine wedge, not just bad numerics. The retirement scoping
  (Gemma 3 default-OFF on the flip) is reinforced.
- sm_121 (Spark) is unaffected (prior bisect: FI matches FLASH_ATTN +0.0028
  nats, nvfp4 coherent) — the re-scope to sm_120-specific holds and HARDENS.
- Next localizer: the Colab G4 sm_120 cell at this exact geometry, and a P520
  run with the full accel-kernel pip set to rule the confound in/out.

Artifacts: results/ (server logs incl. `_WEDGED`, proof lines, chat smoke,
warmup), diag1_run.log, diag2_run.log, diag2_600tok_warmedup.json,
diag1_probe_16.json, status.txt.
