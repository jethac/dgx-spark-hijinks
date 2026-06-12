# Gemma 3 1B FlashInfer-numerics bisect — Spark (GB10, sm_121), 2026-06-12

Bisect for `docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md`. The bug was
observed on the P520 (RTX 5060 Ti, sm_120): FlashInfer-backend serving at the
Gemma 3 1B geometry (uniform head_dim 256 / sliding-window 512 / 1 KV head)
deviated +0.221 nats (C1) from a FLASH_ATTN/HF truth pair, and NVFP4 KV was
deterministic gibberish. Gemma 3 1B had NEVER been served on sm_121 until now.
This run decides GEOMETRY (reproduces on Spark) vs sm_120-PLATFORM (clean on
Spark).

## Verdict: **PLATFORM / sm_120-SPECIFIC.** The bug does NOT reproduce on the Spark.

FI-bf16 minus FLASH_ATTN-bf16 on the Spark = **+0.00279 nats** (well under the
0.01 threshold) — FlashInfer MATCHES FLASH_ATTN at the exact 1B geometry on
sm_121. On the P520 the same comparison was **+0.22061 nats**. NVFP4 KV is
**coherent** on the Spark (Tokyo, deterministic) where the P520 produced
deterministic gibberish. The defect is sm_120-specific (JIT-codegen /
arch-conditional path on the editable P520 build); the Spark uses baked
sm_121a kernels and is clean.

## Provenance

| item | value |
|---|---|
| Host | Spark / GB10 (sm_121, 119 GiB unified), ssh jethac@100.113.98.11 |
| Image | `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9` (id 8c37bdbc4fdb) |
| vLLM | `v0.1.dev1+g9759e3b06`; EXT_PATH `/opt/jethac-vllm/vllm/_C_stable_libtorch.abi3.so` |
| FlashInfer | `/opt/jethac-flashinfer/flashinfer` (baked AOT, NOT JIT-on-box) |
| transformers | 5.7.0 (r9; loaded `google/gemma-3-1b-it` without fallback to r10) |
| Model | google/gemma-3-1b-it (downloaded this window — was NOT in Spark HF cache, ~2GB) |
| Corpus | C1 `c1_ppl_corpus.md` md5 `abb63f0e65247a25f870d3f2d57563ff` (verified) |
| PPL | `vllm_prompt_ppl_sweep.py --ctx 8191`, no `--add-special-tokens`, `--dump-token-logprobs <dir>` |
| Server flags | `--attention-backend FLASH_ATTN|FLASHINFER`, `--gpu-memory-utilization 0.3`, `--max-model-len 8192`; docker `--memory 100g --memory-swap 100g`, one server at a time |
| Backend force | `--attention-backend` CLI flag; engaged backend VERIFIED from server proof lines (`Using AttentionBackendEnum.<X> backend.`), not trusted from the flag |

## Rows (C1 PPL mean NLL nats/token, ctx 8191, 8190 scored; each run TWICE)

| row | backend engaged (proof) | kv dtype | C1 run a | C1 run b | bitwise | smoke "The capital of Japan is" | GPU KV cache |
|---|---|---|---|---|---|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN (SEL=1) | auto/bf16 | 2.356493110435786 | 2.356493110435786 | IDENTICAL | "The capital of Japan is **Tokyo**." COHERENT | 2,813,248 tok |
| FLASHINFER bf16 (suspect) | FLASHINFER (SEL=1) | auto/bf16 | 2.359283581557766 | 2.359283581557766 | IDENTICAL | "The capital of Japan is **Tokyo**." COHERENT | 2,162,726 tok |
| FLASHINFER nvfp4 | FLASHINFER (SEL=1) | nvfp4 (+LINEAR_V_SF=1) | 2.400746942552027 | 2.400746942552027 | IDENTICAL | "The capital of Japan is **Tokyo**." COHERENT | 7,446,234 tok |

All three rows internally deterministic (bitwise-identical double-runs). nvfp4
row logged the latch line `VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors
are linear; FlashInfer in-kernel V-SF de-swizzle disabled.`

## Deltas (Spark)

| comparison | Spark | P520 (from bug doc) |
|---|---:|---:|
| **FI-bf16 − FLASH_ATTN-bf16** | **+0.00279** | **+0.22061** |
| FI-nvfp4 − FLASH_ATTN-bf16 | +0.04425 | +1.592 (vs HF; gibberish) |
| FLASH_ATTN-bf16 cross-check vs P520 FLASH_ATTN (2.35785) | −0.00136 | — |
| FLASH_ATTN-bf16 cross-check vs HF truth (2.35778) | −0.00129 | — |

The FLASH_ATTN-bf16 truth reference on the Spark (2.35649) sits within 0.0014
nats of both the P520 FLASH_ATTN row and the HF eager truth — confirming a
correct backend is platform-independent and the harness/setup is sound (no
red-flag offset).

## What this implies for next steps

- **PLATFORM verdict ⇒ the Spark / sm_121 Gemma 3 1B path is fine.** FlashInfer
  bf16 and NVFP4 KV both serve correctly at the d256/SWA-512/1-kv-head geometry
  on baked sm_121a kernels. The Gemma 3 1B NVFP4 row that was banked RED on the
  P520 is GREEN-class on the Spark (coherent, +0.044 vs truth).
- The bug remains OPEN but is now **scoped to sm_120 (P520 editable build)**:
  JIT codegen or an arch-conditional FlashInfer path on the source-tree build,
  NOT the geometry. The "geometry bug" hypothesis is REFUTED.
- Gemma 3 retirement: the geometry is no longer the blocker on sm_121. The
  P520/sm_120 path stays flagged until the JIT-codegen root cause is found
  (bug-doc investigation steps 2-3: P520 logit-diff probe + window/kv-head
  ablation remain the way to localize the sm_120 defect).
- SGLang implication (for Codex): the risk is sm_120-only, not a shared
  small-model FlashInfer geometry defect — SGLang's sm_121 FlashInfer Gemma-3
  small-model rows are NOT at risk from this bug. (If it had been geometry,
  SGLang's FlashInfer Gemma-3 small-model paths would have shared the risk on
  any platform.)

## Artifacts (this dir)

- `status.txt` (run 2; run 1 PPL was broken by a `--dump-token-logprobs` arg
  signature mismatch — fixed, re-run clean; `status_run1_pplbroken.txt` kept on Spark)
- `run_1b_bisect.sh` (the runner)
- `results/claude_{fa_bf16,fi_bf16,fi_nvfp4}_c1{a,b}_ctx8191_ppl.json` (PPL ×2 each)
- `results/claude_{...}_smoke_tokyo.json` (verbatim chat transcripts)
- `results/claude_{...}_proof_lines.txt` (backend engaged, kv dtype, EXT_PATH, KV cache size, latch line)
- Spark host master copy + server logs + token dumps: `/home/jethac/spark_tmp/claude_1b_bug_bisect_20260612/`
