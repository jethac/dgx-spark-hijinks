# P520 Gemma 3 1B — FlashInfer bf16 serving-numerics RIGOROUS re-test

Date: 2026-06-12 JST. Campaign: dgx-spark-hijinks epoch2. Agent: p520-g3-1b-retest.
Decisive question (task #37): the earlier, less-careful 1B runs reported a bf16
FlashInfer defect (**+0.221 nats / engine wedge on window-crossing**). Is that
REAL and 1B-specific (so the Gemma 3 retirement flip needs a 1B caveat), or was
it an ENVIRONMENTAL / false-green ARTIFACT (so the flip is clean, no caveat)?

The old 1B numbers are suspect because (i) one earlier run used the false-green
`VLLM_ATTENTION_BACKEND` trap that silently fell back to FLASH_ATTN, and (ii) the
270M agent found `--gpu-memory-utilization 0.3` OOMs the FlashInfer rows on this
WDDM-shared card. So this is a clean re-run with FlashInfer **verified-engaged
from the engine proof line** and util 0.6.

## VERDICT: the bf16 1B FlashInfer defect is an ENVIRONMENTAL / FALSE-GREEN ARTIFACT — it does NOT reproduce

- **FI-bf16 vs FLASH_ATTN truth delta = -0.0006633 nats** (double-run bitwise,
  FlashInfer proven engaged). The old suspect was **+0.221**; this is ~330x
  smaller and the *opposite sign*. Effectively zero.
- **No engine wedge.** The server came ready in 149 s and served every C1
  ctx-8191 (window-crossing, >512 tok) `prompt_logprobs` request on FlashInfer
  with no hang. The earlier 1B "wedge" did not reproduce.
- Chat coherent ("The capital of Japan is Tokyo.") on both bf16 backends.

So: the bf16 serving-numerics defect on Gemma 3 1B **does not exist** -- it was an
artifact of the earlier methodology (false-green fallback and/or the util-0.3
OOM regime). This matches and extends the 270M clean result (270M FI-bf16 was
+0.00133). **The Gemma 3 retirement flip needs NO 1B bf16 caveat.**

- The **nvfp4 KV read-path defect (A) REPRODUCES** on 1B (gibberish, +1.587 nats)
  -- broad/environmental on this sm_120 card, exactly as on 270M/4B. The Gemma 3
  flip keeps its nvfp4-KV caveat (that is a separate, known, broad defect).

## Provenance

| item | value |
|---|---|
| Host | P520, RTX 5060 Ti 16 GB, CC 12.0 / sm_120, WSL2 Ubuntu, WDDM-shared |
| vLLM | wheel `0.1.dev1+g6adc00f70.sm120a` (`spark/hijinks-e2-vllm @ 6adc00f70`), EXT_PATH = wheel `_C_stable_libtorch.abi3.so` (same wheel as 270M test) |
| FlashInfer | `~/flashinfer` source-tree JIT (`PYTHONPATH=~/flashinfer`, `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`) |
| torch/CUDA | torch 2.12.0+cu130, CUDA 13.0 |
| Model | google/gemma-3-1b-it |
| Corpus | C1 `abb63f0e65247a25f870d3f2d57563ff` (md5 verified), ctx 8191, 8190 scored |
| Server | `--max-model-len 8192`, `--gpu-memory-utilization 0.6`, one server at a time; serve-ready timeout 300 s as the wedge detector |
| Backend engaged | verified from the `Using AttentionBackendEnum.<X> backend.` engine proof line, NOT the flag |
| GPU | CLAIM 06:20:50Z / RELEASE 06:36:12Z; verified free (WDDM-only, no compute apps) before & after |

HF transformers bf16 **eager** C1 ground truth (same ctx-8191 / 8190-token window,
positions 1..N-1): **2.359755277633667 nats**.

## Row table (C1 ctx-8191 PPL, double-run bitwise; backend actually engaged)

| row | backend (PROOF line) | kv dtype | C1 PPL x2 (bitwise) | delta vs truth | wedge | chat smoke |
|---|---|---|---|---:|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN (FA v2) | bf16 | 2.3578483823599337 / 2.3578483823599337 (IDENTICAL) | -0.00191 vs HF | no | COHERENT "...Tokyo." |
| FI bf16 (THE suspect) | **FLASHINFER** | bf16 | 2.3571850630239095 / 2.3571850630239095 (IDENTICAL) | **-0.00066** vs FLASH_ATTN | **no** | COHERENT "...Tokyo." |
| FI nvfp4 (+LINEAR_V_SF) | **FLASHINFER** | nvfp4 | 3.9452781399784085 / 3.9452781399784085 (IDENTICAL) | **+1.58743** vs FLASH_ATTN | no | **GIBBERISH, deterministic** |

All three rows internally deterministic (byte-identical across two independent PPL
runs). The FI-bf16 row was forced with `--attention-backend flashinfer` **and**
`VLLM_FLASHINFER_BF16_GEMMA=1`; the engine proof line confirms
`Using AttentionBackendEnum.FLASHINFER backend.` with `attention_backend:
'flashinfer'` in non-default args -- **no false-green FLASH_ATTN substitution**.
The nvfp4 row logged `VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors are
linear; FlashInfer in-kernel V-SF de-swizzle disabled.`

nvfp4 chat smoke verbatim (escaped): `* - \nWhat is the most radical, a. of</strong> icular?</strong> [arabic/telugu].\n\n`

## Head-to-head vs the 270M clean result (and vs the old suspect 1B)

| arm | old suspect 1B | 270M re-test | **1B rigorous re-test (this)** |
|---|---:|---:|---:|
| FI-bf16 - FLASH_ATTN | +0.221 + wedge | +0.00133, no wedge | **-0.00066, no wedge** |
| FI-nvfp4 - FLASH_ATTN | gibberish | +8.122 gibberish | **+1.587 gibberish** |

Both clean re-runs (270M and now 1B, both with FlashInfer verified-engaged and
util 0.6) agree: the bf16 inflation/wedge does NOT exist. The 270M SUMMARY had
left open that the bf16 arm "scales with model width/depth (1152/26L) or some
1B-specific path" -- **this 1B re-test closes that: the 1B is also clean, so there
is no width/depth-driven bf16 inflation at all.** The earlier +0.221/wedge is now
attributable to the methodology artifacts (false-green fallback and/or util-0.3
OOM regime), not to FlashInfer's bf16 path on Gemma 3 1B.

(Note on nvfp4 magnitude: 270M's nvfp4 gibberish PPL was +8.12 vs its truth;
1B's is +1.587 vs its truth. Both are deterministic gibberish with incoherent
chat -- the read-path is broken in both; the absolute nat gap differs by model,
which is expected for a corrupted-read regime and is not load-bearing. The
load-bearing fact is: nvfp4 chat is gibberish, bf16 chat is coherent.)

## Implication for the Gemma 3 retirement flip

- **bf16 serving numerics:** fully clean on Gemma 3 1B (and 270M). **No 1B
  caveat is needed** for the FLASH_ATTN<->FlashInfer bf16 flip; FlashInfer bf16 is
  within ~7e-4 nats of FLASH_ATTN and matches HF truth.
- **nvfp4 KV:** the broad sm_120 read-path defect still trips (gibberish) and is
  unrelated to the flip's bf16 decision; the existing nvfp4-KV caveat stands.

## Notes / methodology

- util 0.6 (not 0.3) used for all rows, as the 270M agent established 0.3 OOMs
  the FlashInfer rows on this WDDM-shared 5060 Ti. 0.6 was clean for all three.
- Serve-ready timeout set to 300 s specifically to catch the earlier "wedge on
  window-crossing"; no row approached it (all ready in 146-149 s).
- HF reference uses `attn_implementation="eager"` bf16, scoring the identical
  ctx-8191 token window (positions 1..N-1) that the vLLM harness scores, so the
  reference is directly comparable.

## Artifacts (this dir)

`results/`: per-row `*_server.log`, `*_proof_lines.txt`, `*_chat_smoke.json`,
`*_c1{,b}_ctx8191_ppl.json` (double-run); `hf_bf16_reference_ppl.json`.
`status.txt`; `scripts/` (`run_row.sh`, `hf_bf16_reference.py`,
`vllm_prompt_ppl_sweep.py`, `spark_hardware.py`, `openai_chat_smoke.py`).
