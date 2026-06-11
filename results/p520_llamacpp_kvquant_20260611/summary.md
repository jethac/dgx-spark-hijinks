# llama.cpp KV-cache-quant comparison (campaign task #25/#26, cross-implementation arm)

**Date:** 2026-06-11 (JST) - **Host:** local WSL2 Ubuntu, RTX 5060 Ti 16GB (sm_120 / CC 12.0), driver 595.79, CUDA 13.0

## THE QUESTION: does quantized KV lower perplexity vs f16 in llama.cpp too?

**YES - direction replicates.** On llama.cpp's fully independent implementation (q8_0/q4_0 GGML
block quants, its own CUDA FA kernels, no shared code with vLLM/SGLang/FlashInfer), quantized
KV cache produced **equal-or-lower perplexity than f16 on every row, on both corpora**, with a
monotonic trend: more KV quantization -> lower PPL. The cumulative running PPL was below the
f16 baseline at every one of the 72 chunks for both q8_0 and q4_0 (q4_0 < q8_0 < f16 throughout).

Magnitude caveat: in log-space the effect here is **~0.004-0.021 nats**, i.e. the same *sign*
as the vLLM/SGLang/FlashInfer anomaly but 1-2 orders of magnitude smaller than the 0.14-0.33 nats
seen on our stack family. So: the "quantized KV beats bf16" direction is **general across
implementations**, but the *size* of our effect is not explained by llama.cpp's behavior -
the magnitude remains specific to our stack family.

## Main table - wikitext-2 raw test, ctx 4096, 72 chunks, seed 42, -fa on, only KV type varies

| Row | cache-type-k | cache-type-v | Final PPL | +/- SE | dPPL vs f16 | d ln(PPL) (nats) | max VRAM (MiB) |
|---|---|---|---|---|---|---|---|
| A (baseline) | f16 | f16 | **49.8046** | 0.60455 | - | - | 10054 |
| B | q8_0 | q8_0 | **49.6085** | 0.60166 | -0.1961 | -0.0039 | 9996 |
| C | q4_0 | q4_0 | **48.7920** | 0.58602 | -1.0126 | -0.0206 | 10002 |
| D (mixed) | q8_0 | q4_0 | **48.8381** | 0.58854 | -0.9665 | -0.0196 | 10066 |

## Replication - second corpus (local_english_50k.txt), ctx 4096, 3 chunks, same seed/flags

| Row | k / v | Final PPL | +/- SE | dPPL vs f16 | per-chunk (cumulative) |
|---|---|---|---|---|---|
| A | f16 / f16 | **12.8026** | 0.67749 | - | 8.4039, 10.4065, 12.8026 |
| B | q8_0 / q8_0 | **12.7756** | 0.67597 | -0.0270 | 8.3244, 10.3666, 12.7756 |
| C | q4_0 / q4_0 | **12.7376** | 0.67192 | -0.0650 | 8.4874, 10.3914, 12.7376 |

Same ordering: q4_0 < q8_0 < f16. (Chunk-1 of row C is above A - the effect is not uniform
per-chunk on this tiny corpus - but the 3-chunk aggregate again favors quantized.)

Per-chunk running PPL for all wikitext rows is in ppl_wikitext2_{A,B,C,D}.log
(e.g. chunk 1: A 60.2191 / B 59.7397 / C 58.2766 / D 58.6710; chunk 36: A 48.0512 /
B 47.8820 / C 47.0431; chunk 72 = finals above).

## Speed/capacity context - llama-bench, pp512 + tg128, -fa 1, ngl 99

| type_k | type_v | pp512 t/s | tg128 t/s |
|---|---|---|---|
| f16 | f16 | 2147.63 +/- 231.45 | 52.32 +/- 3.23 |
| q4_0 | q4_0 | 1468.50 +/- 201.56 | 62.97 +/- 39.79 |
| f16 | q4_0 | 207.56 +/- 69.92 | 8.31 +/- 1.00 |
| q4_0 | f16 | 532.99 +/- 533.03 | 8.63 +/- 0.94 |

Note the cliff on *mixed* K/V types: no fast FA path, ~6-10x slowdown. This is also why the
mixed row D perplexity run took 74 min vs ~2.5 min for the homogeneous rows (D used
k=q8_0/v=q4_0 - same fallback behavior).

VRAM was nearly flat across rows because at ctx 4096 the E4B KV cache is small relative to the
5GB model; the win would scale with context length.

## Provenance

- **llama.cpp:** tag **b9596** (>= required b8967 sm_120 NVFP4 MMQ build), commit 18ef86e,
  built with -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a (120a accepted), CUDA 13.0,
  targets llama-cli / llama-perplexity / llama-bench. See build_info.txt.
- **Model:** ggml-org/gemma-4-E4B-it-GGUF / gemma-4-E4B-it-Q4_K_M.gguf (4.95 GiB, 7.52B
  params, gemma4 E4B), sha256 90ce98129eb3...13e9f in model_provenance.txt. No NVFP4 GGUF
  was published in the ggml-org or unsloth gemma-4-E4B repos at run time (recorded; Q4_K_M used).
- **Corpus 1:** wikitext-2-raw wiki.test.raw (1,290,590 bytes, sha256 173c87a5..., from
  huggingface.co/datasets/ggml-org/ci/wikitext-2-raw-v1.zip - the llama.cpp standard source).
- **Corpus 2:** local_english_50k.txt = first 50 KiB of de-markdowned llama.cpp README+docs
  at tag b9596 (deterministic; identical bytes across rows). sha256 in corpus_provenance.txt.
- **Command (only k/v types vary):**
  llama-perplexity -m <model> -f <corpus> -c 4096 --seed 42 -ngl 99 -fa on --cache-type-k X --cache-type-v Y
  Flash attention pinned ON for **all** rows including the f16 baseline (quantized V cache
  requires FA; pinning it keeps the cache type the only variable).

## Caveats / notes

- Weights are themselves Q4_K_M; interplay between weight quant and KV quant is untested here
  (a bf16-weights rerun would need ~13.6 GiB + KV + compute - too tight on 16GB at ctx 4096;
  the Q8_0 GGUF would be a feasible middle ground).
- Absolute wikitext PPL (~49) is high because gemma-4-E4B-it is instruction-tuned;
  comparison is strictly internal so this does not affect the A/B/C/D contrast.
- q8_0's wikitext delta (-0.196 PPL) is within 1 marginal SE, but the running estimate stayed
  below f16 at all 72 checkpoints on identical data, and the second corpus agrees in sign;
  q4_0's delta (-1.01 PPL) is consistent and larger.

## Files

ppl_wikitext2_{A,B,C,D}.log, ppl_local50k_{A,B,C}.log (full llama-perplexity output +
vram_max_mib), bench_pp512_tg128.md, build_info.txt, model_provenance.txt,
corpus_provenance.txt, runlogs/ (build/assets/ppl/bench stage logs + pipeline status).
Scripts: B:\workshop\wsl_sm120\1{0,1,2,3,4,5,7,8,9}_*.sh, 20_collect.sh.
