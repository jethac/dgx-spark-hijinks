# P520 Gemma 3 270M — FlashInfer serving-bug repro test

Date: 2026-06-12 JST. Campaign: dgx-spark-hijinks epoch2. Agent: p520-g3-270m.
Question (task #37 follow-up): does the sm_120 FlashInfer-serving bug seen on
Gemma 3 1B reproduce on Gemma 3 270M? -> classifies the bug as 1B-geometry-
specific vs broad across small Gemma 3, and whether 270M is a fast minimal repro.

## Verdict: SPLIT — nvfp4 REPRODUCES, FI-bf16 does NOT

- **FI-nvfp4 REPRODUCES** the bug on 270M: deterministic gibberish chat +
  catastrophic PPL (11.034 nats, +8.12 vs truth). Same failure class as the 1B.
- **FI-bf16 does NOT reproduce** the 1B's bf16 inflation: delta vs FLASH_ATTN
  truth is only **+0.00133 nats** (the 1B was +0.221, ~165x larger). FI-bf16 is
  effectively clean on 270M.

Implication: the two P520/sm_120 defects are SEPARABLE.
- The **nvfp4 KV read-path defect** is model/size-independent on sm_120 (1B,
  270M, and the 4B-mm smoke all gibberish) — 270M is a valid, fast minimal
  repro for THIS one.
- The **FI-bf16 long-context inflation** (the +0.221 / SWA-512 numerics tell) is
  NOT reproduced at 270M despite identical geometry axes — so it is NOT simply a
  function of the d256/SWA-512/1-kv-head geometry; it scales with model
  width/depth (1152/26L inflates, 640/18L does not) or some 1B-specific path.
  270M is NOT a minimal repro for the bf16 inflation arm.

## Provenance

| item | value |
|---|---|
| Host | P520, RTX 5060 Ti 16 GB, CC 12.0 / sm_120, WSL2 Ubuntu, WDDM-shared |
| vLLM | wheel `0.1.dev1+g6adc00f70.sm120a` (`spark/hijinks-e2-vllm @ 6adc00f70`), EXT_PATH = wheel `_C_stable_libtorch.abi3.so` (same wheel as task #37 1B disambig) |
| FlashInfer | `7d5d477b` source-tree JIT (`PYTHONPATH=~/flashinfer`, `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`) |
| torch/CUDA | torch 2.12.0+cu130, CUDA 13.0 |
| Model | google/gemma-3-270m-it @ ac82b4e8 |
| Corpus | C1 `abb63f0e65247a25f870d3f2d57563ff` (md5 verified), ctx 8191, 8190 scored |
| Server | `--max-model-len 8192`, one server at a time. util 0.6 for FI rows (util 0.3 per the goal OOM'd on the WDDM-shared card: `ValueError: No available memory for the cache blocks` — environmental, not the bug; FLASH_ATTN truth row ran at 0.3) |
| Backend engaged | verified from the `Using AttentionBackendEnum.<X> backend.` proof line, NOT the flag |

HF transformers bf16 eager C1 ground truth (same token window): **2.912124 nats**.

## Row table (C1 ctx-8191 PPL, double-run bitwise; smoke; backend actually engaged)

| row | backend (PROOF line) | kv dtype | C1 PPL x2 (bitwise) | delta vs truth | chat smoke |
|---|---|---|---|---:|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.911488 / 2.911488 (IDENTICAL) | +0.00064 vs HF | COHERENT "...Tokyo." |
| FI bf16 (suspect) | **FLASHINFER** | bf16 | 2.912821 / 2.912821 (IDENTICAL) | **+0.00133** vs FLASH_ATTN | COHERENT "...Tokyo." |
| FI nvfp4 (+LINEAR_V_SF) | **FLASHINFER** | nvfp4 | 11.034120 / 11.034120 (IDENTICAL) | **+8.122** vs FLASH_ATTN | **GIBBERISH, deterministic** |

All three rows internally deterministic (byte-identical across two independent
PPL runs). FI-bf16 and FI-nvfp4 proof lines confirm
`Using AttentionBackendEnum.FLASHINFER backend.` (no false-green FLASH_ATTN
substitution). nvfp4 row logged `VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale
factors are linear; FlashInfer in-kernel V-SF de-swizzle disabled.` and KV
size 1,724,431 tokens.

nvfp4 chat smoke verbatim (escaped):
```
\なぜ\nкъ методи:\n\nОрганизум фактът ти сиристо: 25 представря проблеми$:\n\n$[]$$? оказа: оказа: документ?"\n\nМолястя се на поставяние:
```

## Geometry comparison (270M vs 1B)

| axis | Gemma 3 270M | Gemma 3 1B | shared? |
|---|---|---|---|
| head_dim | 256 | 256 | YES (suspect axis) |
| num_key_value_heads | 1 | 1 | YES (suspect axis — only Gemmas with 1 KV head) |
| sliding_window | 512 | 512 | YES (suspect axis) |
| num_attention_heads | 4 | 4 | YES |
| hidden_size | 640 | 1152 | no |
| num_hidden_layers | 18 | 26 | no |
| VO-split | none (uniform d256) | none | YES |

270M shares ALL THREE suspect axes (d256 / SWA-512 / 1-kv-head) with the 1B.
Only width and depth differ. So the geometry is NOT sufficient to explain the
bf16 inflation arm (270M has it and stays clean), but the nvfp4 read defect
trips regardless of width/depth.

## Is 270M a fast minimal repro?

- For the **nvfp4 gibberish / read-path defect**: YES. Fast (~90 s to serve,
  ~10 s/PPL), tiny (0.5 GB), reproduces deterministically. Use 270M to localize
  the sm_120 NVFP4 KV read path.
- For the **FI-bf16 SWA-512 inflation**: NO. 270M does not exhibit it; must use
  the 1B (or larger) for that arm.

## Notes / incidents

- util 0.3 (per goal) OOM'd both FI rows on the WDDM-shared 5060 Ti at engine
  KV-block allocation; re-ran the two FI rows at util 0.6 (clean). FLASH_ATTN
  truth row completed at 0.3. Not the wedge — a plain capacity ValueError.
- No engine WEDGE observed at 270M (the 1B wheel run wedged on window-crossing
  ctx-8191 prompt_logprobs; 270M served all C1 ctx-8191 PPL requests fine on all
  three backends). The wedge arm, like the bf16 inflation, did NOT reproduce.
- First driver attempt failed because the copied ppl sweep imported
  `spark_hardware` (sibling module not copied); fixed and re-run.

## Artifacts (this dir)

`results/`: per-row server proof_lines.txt, chat_smoke.json,
c1/c1b_ctx8191_ppl.json; `hf_bf16_reference_ppl.json`;
`claude_p520_fi_bf16_crash_excerpt.txt` (util-0.3 OOM). `status.txt`, `scripts/`.
