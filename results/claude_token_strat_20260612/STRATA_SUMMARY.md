# Token-stratification capture + strata analysis — 2026-06-12 (task #25, anomaly arm 3)

Window: `R=/home/jethac/spark_tmp/claude_token_strat_20260612`, r9 baked image
(`jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`, id `8c37bdbc4fdb`),
`google/gemma-4-31B-it --language-model-only`, util 0.72, ctx 8191, sequential
servers bf16_vosplit -> fp8 -> nvfp4, corpora C1/C2/C3 (md5s verified in
`status.txt`). Per-token logprob dumps in `token_dumps/` (9 files, schema
`vllm-prompt-token-logprobs/v1`).

## Determinism gate: CLEAN

All NINE cells reproduced the banked corpus-sweep means BITWISE
(`REPRO=EXACT` in `status.txt`, total wall 2016 s). The window is NOT poisoned.
Offline cross-check: all six recomputed mean deltas match the corpus-sweep
deltas to float round-trip (|diff| <= 1e-15).

## Hypothesis verdicts (H definitions in docs/WINDOW_PACKET_TOKEN_STRAT.md)

| pair | mean delta (nats) | winner | evidence |
|---|---:|---|---|
| fp8 vs bf16, C1 | -0.1392 | **H-hard** (benign direction) | `>=8`-nat band carries +128% of total delta (improvement); low bands slightly worsen; top-10 tokens only -0.1% of total (not H-tail); no position growth (not H-late) |
| fp8 vs bf16, C2 | -0.1643 | **H-hard** | `>=8` band +77% of total; mild late drift (-0.13 -> -0.24 mean delta by last bucket) but band structure dominates |
| fp8 vs bf16, C3 | -0.0184 | **H-hard** (small aggregate) | `>=8` band +457% of (small) total; top-10 = 39.8% of total — tail-influenced but band-driven; signs nearly balanced 49.9/48.8% |
| nvfp4 vs bf16, C1 | -0.3318 | **H-hard** | `>=8` band mean delta -1.351, +104% of total; top-10 only 4.3%; position trend DECAYS (anti-H-late) |
| **nvfp4 vs bf16, C2 (the prose inversion)** | **+0.2526** | **H-broad, secondary H-late; explicitly NOT H-tail** | see below |
| nvfp4 vs bf16, C3 | -0.0468 | **H-hard** | `>=8` band -1.003 mean, +352% of (small) total; low bands worsen (+0.17..+0.54) |

## Priority readout: nvfp4/C2 (+0.2526 nats, Pride & Prejudice prose)

The inversion is structurally DIFFERENT from every other pair:

| baseline-NLL band | count | mean delta | share of total |
|---|---|---|---|
| [0,0.5) | 2477 | +0.2432 | +29.1% |
| [0.5,1) | 318 | +0.8630 | +13.3% |
| [1,2) | 442 | +0.6654 | +14.2% |
| [2,4) | 737 | +0.6491 | +23.1% |
| [4,8) | 1469 | +0.5449 | +38.7% |
| >=8 | 2747 | -0.1289 | -17.1% |

| positions | mean delta | | positions | mean delta |
|---|---|---|---|---|
| 0-1022 | +0.0389 | | 4092-5114 | +0.2307 |
| 1023-2045 | +0.0731 | | 5115-6137 | +0.4954 |
| 2046-3068 | +0.0919 | | 6138-7160 | +0.4037 |
| 3069-4091 | +0.3504 | | 7161-8189 | +0.3357 |

- Worsening is DIFFUSE: 56.4% of tokens worsen, every band below 8 nats worsens;
  the top-40 |delta| tokens are NOT a catastrophic tail (top-10 signed = +5.6%
  of total). H-tail is dead for this pair — no "destroyed attention read" signature.
- Clear positional growth: first ~3k positions are nearly flat (+0.04..+0.09),
  then the per-token penalty roughly quadruples and stays high -> KV-accumulation
  component (H-late) on top of the broad effect.
- The HARDEST band still IMPROVES (-0.13), same regularization direction as
  every other pair; the inversion lives in the bulk easy/mid-surprisal prose
  tokens. Top worsened tokens are high-frequency entity/noun continuations
  (`iness` in "easiness" +18.5, `friend` +17.4, ` George` +16.9, ` Mr` +14.0,
  `field` in "Netherfield" +13.3) — long-range entity recall, not noise spikes.

Interpretation: NVFP4-KV on natural prose degrades the model's bulk
copy/recall precision progressively with context depth (broad + late), while
still flattening (improving) already-uncertain predictions everywhere. This is
consistent with quantization noise washing out long-range low-entropy reads,
NOT with a bug that occasionally destroys specific reads. Follow-on per packet:
H-tail replay through the FlashInfer probe harness is NOT warranted; a
depth-stratified C2 rerun (e.g. ctx 2047 vs 8191) would directly test the
H-late component.

## Cross-cutting observation (both comparators, all corpora)

Mean |delta| per token is 0.62-1.14 nats — individual token NLLs churn HARD
under KV quantization in both directions; the small aggregate means are
residuals of near-cancelling distributions (e.g. nvfp4/C1: mean |d| 1.143,
mean d -0.332). "Quality delta ~0.1 nats" claims must therefore never be read
as "tokens move ~0.1 nats"; they are population residuals. fp8 shows the same
shape at ~60% of nvfp4's churn magnitude.

## Artifacts

- `status.txt` — REPRO=EXACT x9, corpus md5s, image id.
- `claude_{bf16_vosplit,fp8,nvfp4}_{c1,c2,c3}_ctx8191_ppl.json` + stdout/stderr.
- `claude_*_proof_lines.txt`, `claude_*_server.log` — backend/EXT_PATH proofs.
- `token_dumps/*_tokens.json` — 9 per-token dumps.
- `strata_{fp8,nvfp4}_vs_bf16_{c1,c2,c3}.{json,md}` — six strata reports
  (`scripts/anomaly_token_strata_analyze.py`, self-test green).
