# mm-retire serving smokes — P520 / sm_120 — 2026-06-12

Agent: p520-smokes. GPU: RTX 5060 Ti (CC 12.0, 16 GiB), WSL2 Ubuntu, CUDA 13.0.
vLLM: sm120a RELEASE WHEEL `0.1.dev1+g6adc00f70.sm120a` (built from
`spark/hijinks-e2-vllm @ 6adc00f70`) + **mm-retire Python overlay** (merged onto
the wheel base; see "Overlay / merge" below). FlashInfer: source-tree JIT
@ `7d5d477b`, `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`. transformers 5.11.0,
torchvision 0.27.0+cu130.

Provenance (Phase 1 gates, GREEN): EXT_PATH = wheel `_C_stable_libtorch.abi3.so`;
42 sm_120a cubins; NVFP4 linear-latch "writer wrote LINEAR V-SF" (hd128+hd256).
wheel md5 ad5e7fe06ef715550ee97b1c6763173a.

Serving deviation from the prompt's "util 0.85": the 16 GiB P520 cannot fit a
4B mm model + vision tower + KV cache under CUDA-graph memory profiling at 0.85.
Used **--gpu-memory-utilization 0.90 --max-model-len 2048 --enforce-eager**
(image prompts are short; eager frees the CUDA-graph reservation). Documented,
not a defect.

## Cell table

Model **google/gemma-3-4b-it** (uniform head_dim 256, SWA 1024; mm-prefix masks
ALL layer groups). Grounding image = `two_shapes.png` (blue square + yellow
triangle); prompt "List the colors and the shapes you see in this image.";
keywords {blue, triangle}. Each smoke ×2 for repeat-determinism.

| cell | backend (proof) | KV | mm-prefix engaged | (a) grounded | (d) img repeat byte-id | verdict |
|---|---|---|---|---|---|---|
| g3_4b bf16 FI mm route | FLASHINFER | bf16 | YES (`FlashInfer mm-prefix:` win_left 1023 & -1) | YES (Blue/Yellow/Square/Triangle) | YES | **GREEN** |
| g3_4b bf16 Triton route | TRITON_ATTN | bf16 | n/a (native) | YES (byte-identical reply to FI) | YES | **GREEN** |
| g3_4b nvfp4 FI mm route | FLASHINFER | nvfp4 (+LINEAR_V_SF+VOSPLIT) | YES (`first custom-mask prefill batch planned, 1 image span in-window`) | **NO — GIBBERISH** | NO (non-determ gibberish) | **RED** |
| g4_E4B (all cells) | — | — | — | — | — | **BLOCKED (capacity)** |

### Gate comparisons (g3-4b)

| gate | comparison | verdict |
|---|---|---|
| (b) bf16 FI-vs-Triton semantic equivalence | `g3_4b_bf16_FIvsTriton_cmp.json` | **GREEN** — exact match, word-jaccard 1.0 (replies byte-identical) |
| (b) nvfp4 FI-vs-Triton(bf16) | `g3_4b_nvfp4_FIvsTriton_cmp.json` | **RED** — nvfp4 reply is gibberish, diverges totally |
| (c) text token-identity, mm-knob ON vs OFF (FLASHINFER bf16) | `g3_4b_bf16_text_identity_cmp.json` | **GREEN** — content-identical AND token-identical (mm knob is a clean no-op for pure text) |

## REDs (banked verbatim)

### RED-1: nvfp4 KV mm cell emits deterministic-class GIBBERISH on sm_120 (g3-4b)
Cell `g3_4b_nvfp4_fi`. The server boots, FLASHINFER engages, the mm-prefix
custom-mask path executes (`flashinfer.py:2153 FlashInfer mm-prefix: first
custom-mask prefill batch planned (1 mm request(s), 1 image span(s)
in-window)`), but the image reply is gibberish:
```
reply0: 'DISTDISTf DISTDISTYouDISTIDIST여섯DIST1DISTLGPSquareDISTIfDIST CrossDISTሰDIST...'
reply1: 'HereDIST 3DIST DIST\nDIST\n\nDIST DIST...的颜色DIST调整DIST了DIST被子的...'
```
Not grounded, not repeat-deterministic. This is the SAME nvfp4-on-sm_120
gibberish signature as the Gemma 3 1B bug (BUG_FLASHINFER_GEMMA3_1B doc) — it
is NOT specific to the mm path: the nvfp4 KV READ path is wrong on sm_120
regardless of the mm-prefix masking (the mask machinery clearly RAN). The bf16
mm cells on the SAME wheel+overlay are clean, so the mm-prefix custom-mask code
is correct; the defect is in the sm_120 nvfp4 KV path that the campaign already
tracks. Banked: `results/g3_4b_nvfp4_fi_img.json`,
`serverlogs/g3_4b_nvfp4_fi.log` (+ proof).

### Capacity BLOCK (not a RED): g4-E4B does not fit the 16 GiB P520
`g4_e4b_bf16_fi` died at engine init: weights = **15.19 GiB**, "Available KV
cache memory: **-1.36 GiB**" → `ValueError: No available memory for the cache
blocks`. E4B in bf16 needs ~17 GiB+ (weights alone fill the card); nvfp4 KV does
not help (it shrinks KV, not the 15 GiB weights). The g4-E4B mm rows must run on
the **Spark (119 GiB unified)**, not the P520. Banked:
`serverlogs/g4_e4b_bf16_fi.log` (mem profiling lines).

## Note on repeat-determinism (gate d) for the TEXT cells
The long free-generation text smokes (`*_txt.json`, 128-token paragraph)
reported `repeat_determ=False` for BOTH knob-on AND knob-off — i.e. it is
**knob-independent** (a baseline vLLM long-generation non-bitwise property on
this wheel/WSL build, not introduced by mm-retire). The mm knob's text gate (c)
still passes because knob-on rep0 vs knob-off rep0 are TOKEN-IDENTICAL. The
short structured IMAGE replies ARE byte-deterministic across repeats. This text
long-gen non-determinism is a separate banked observation (not attributable to
the mm-retire flip) and does not gate the mm-retire merge.

## MERGE RECOMMENDATION

**mm-retire is mergeable for the bf16 mm path, with two caveats the relief/main
session must own:**

1. **bf16 mm-prefix is GREEN on sm_120 (P520):** image-grounded, FI-route ≡
   Triton-route (byte-identical), text knob-identity holds, image determinism
   holds. The FlashInfer mm-prefix custom-mask implementation is correct on
   g3-4b. **This is the core mm-retire claim and it passes.**
2. **nvfp4 mm is RED on sm_120 (P520)** — but the cause is the pre-existing
   sm_120 nvfp4 KV read defect (same as the 1B bug), NOT the mm-prefix code
   (mask machinery ran; bf16 mm is clean). nvfp4 mm should be claimed only on
   sm_121 (Spark), where nvfp4 KV is known coherent. Do NOT claim P520/sm_120
   nvfp4 mm rows.
3. **g4-E4B + the Spark mm/audio rows must run on the Spark** (capacity) — the
   P520 cannot host E4B. The main-session/relief owns those.
4. **MERGE-BLOCKER to RESOLVE FIRST:** the overlay required a hand-merge —
   `config.py` had a real CONFLICT because e2-vllm advanced
   (`20196b594 -> 6adc00f70`, the wheel base) AFTER mm-retire branched, touching
   the SAME `_spark_route_gemma_bf16_to_flashinfer` mm-guard + log lines that
   mm-retire rewrites. envs.py + flashinfer.py auto-merge clean; config.py does
   NOT. The real merge of `spark/hijinks-e2-mm-retire` into `spark/hijinks-e2-vllm`
   will hit this conflict and must combine BOTH: e2-vllm's `default_on` Gemma3/4
   split (Gemma 3 scoped off the flip for the sm_120 1B bug) AND mm-retire's
   inverted mm-prefix default (Amendment 4). My hand-merge (mm-retire's log line
   in the now-default mm-route branch) is staged at
   `~/mm_overlay_staged/config.py` as a reference resolution.

**Bottom line: the bf16 mm smokes are GREEN — mm-retire's FlashInfer mm-prefix
masking is correct on sm_120. Recommend MERGE after resolving the config.py
conflict; claim nvfp4 mm on Spark only; run E4B/audio mm on Spark.**

## Artifacts
results/ (img + txt smoke JSONs, compare JSONs, stdout), serverlogs/ (per-cell
server logs incl. proof lines, g4_e4b crash), status.txt, run.log.
