# Order-controlled anomaly adjudication — G4-31B quantized-KV PPL (2026-06-12)

Window: morning Spark block, marker claimed 06:11 JST write-first (docker
empty verified post-write; Codex's DG-R2 container finished ~06:19, matrix
started 06:22). Runner: `run_order_control.sh` (+ `run_nvfp4_cold_redo.sh`),
r9 baked image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id 8c37bdbc4fdb), `google/gemma-4-31B-it --language-model-only`, util 0.72,
`--memory 100g --memory-swap 100g`, one server at a time, ctx 8191, corpora
md5-gated (c1 abb63f0e / c2 1686a33b / c3 28dfeba9).

Design: per row in {bf16+VOSPLIT (FlashInfer), fp8_e4m3 (no knobs, forced
TRITON_ATTN), nvfp4+VOSPLIT+LINEAR_V_SF (FlashInfer)} TWO fresh server
cycles:

- **COLD**: fresh server, C1 x2 (bitwise gate), C2, C3 — first inference
  request is the C1 sweep, exactly the banked corpus-sweep shape
  (`results/claude_anomaly_corpus_sweep_20260611/`). Smokes AFTER scoring,
  coherence record only.
- **WARMED**: fresh server, EXACTLY the retirement scorecard's two chat
  smokes first (`openai_chat_smoke.py` defaults `--max-tokens 16`, then
  `--prompt "The capital of Japan is" --max-tokens 24`), then C1 x2, C2, C3.

Every C1 cell double-run bitwise-identical (gate held in all 7 served
cycles). All smokes coherent, both routes, banked verbatim.

Harness note: the first nvfp4_cold attempt is harness-RED
(`ModuleNotFoundError: spark_hardware` in the sweep script — staging gap,
zero scores produced; server + smokes green). Fixed and re-run as
`nvfp4_cold_r2` on a fresh server, same protocol. No other cycle affected
(fix landed before any other cycle scored).

## THE TABLE (C1/C2/C3 mean nats/token, ctx 8191; C1 cells x2 bitwise)

| row | cold C1 | cold C2 | cold C3 | warmed C1 | warmed C2 | warmed C3 | warmed−cold (C1 / C2 / C3) |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16 vosplit | 4.613162683323541 | 6.000594550413712 | 3.024778946389832 | 4.613162683323541 | 6.000594550413712 | 3.024778946389832 | **0 / 0 / 0 (bitwise)** |
| fp8_e4m3 | 4.591455999476844 | 5.804393525958695 | 2.9636168641269744 | 4.473945385741097 | 5.83629865522355 | 3.006330528749716 | **−0.117511 / +0.031905 / +0.042714** |
| nvfp4 | 4.2813347779571975 | 6.253168933373023 | 2.977997659632257 | 4.2813347779571975 | 6.253168933373023 | 2.977997659632257 | **0 / 0 / 0 (bitwise)** |

Cross-checks vs banked (corpus sweep 2026-06-11, which scored C1,C2,C3
cold/score-first):

- bf16: cold AND warmed cycles bitwise-reproduce the banked triple
  (4.613162683323541 / 6.000594550413712 / 3.024778946389832). With the
  scorecard's repins this makes SIX cross-window bitwise reproductions.
- nvfp4: cold (r2) AND warmed cycles bitwise-reproduce the banked triple
  (4.2813347779571975 / 6.253168933373023 / 2.977997659632257) — three
  boots total (sweep, today warmed, today cold), two request orders, one
  bitwise profile.
- fp8: **INVERTED vs the 2026-06-12 scorecard finding.** Today's COLD
  (score-first) boot produced 4.591455999476844 — the value the scorecard
  measured on its SMOKES-FIRST server — and today's WARMED (smokes-first)
  boot produced 4.473945385741097 **plus the banked C2/C3 bitwise**
  (5.83629865522355 / 3.006330528749716), the exact triple the banked
  score-first sweep produced.

## ADJUDICATED VERDICT (plain language)

1. **The request-order-dependence claim is REFUTED.** Yesterday's two-server
   A/B (score-first → 4.4739, smokes-first → 4.5915) inverted exactly today
   (score-first → 4.5915, smokes-first → 4.4739). Order was a confound of a
   two-boot experiment. Running totals across all r8/r9 31B fp8 boots:
   profile A has now been seen after BOTH orders (4 score-first boots + 1
   smokes-first boot) and profile B after BOTH orders (1 each).
2. **What fp8 actually is: per-boot BISTABLE.** Each server boot lands in
   one of exactly two complete corpus profiles, and then everything is
   bitwise-stable within the boot (C1 double-runs identical; profile values
   reproduce bitwise across windows and images):
   - profile A: C1 4.473945385741097, C2 5.83629865522355, C3 3.006330528749716
   - profile B: C1 4.591455999476844, C2 5.804393525958695, C3 2.9636168641269744
   Note B is not uniformly worse: B is +0.1175 on C1 but −0.0319 on C2 and
   −0.0427 on C3 vs A. Suspected mechanism shifts from "first-request KV-scale
   calibration" to per-boot nondeterministic kernel/config selection (e.g.
   Triton/inductor autotune timing) on the forced-TRITON fp8-KV path —
   bf16-Triton repins bitwise across boots, so it is specific to the fp8
   path, not to the Triton route per se. Mechanism remains OPEN; what is
   CLOSED is that request order does not select the profile.
3. **Does "quantized KV beats bf16" survive? YES on C1/C3, fp8-only on C2 —
   and it survives in BOTH fp8 boot-states and BOTH request orders.**
   - fp8 beats bf16 on ALL THREE corpora in both profiles: C1 −0.1392 (A) /
     −0.0217 (B); C2 −0.1643 (A) / −0.1962 (B); C3 −0.0184 (A) / −0.0611 (B)
     nats/token. The HEADLINE MAGNITUDE on C1 is boot-state-dependent
     (0.14 vs 0.02 nats); quote it as a range or state-matched, never as a
     single number.
   - nvfp4's pattern is UNCHANGED by order and boot (bitwise): beats bf16 on
     C1 (−0.3318) and C3 (−0.0468), loses on C2 (+0.2526). The mixed-signs
     framing in the ledger stands exactly as written.
4. **bf16 is order- and boot-stable: CONFIRMED** (six bitwise repins across
   four windows, both request orders). The bf16 retirement comparisons are
   unaffected by any of this.
5. **Provenance consequence (supersedes the 2026-06-12 scorecard note):**
   order provenance on quantized rows remains banked but is NOT the operative
   annotation — fp8 PPL rows must carry **boot-profile provenance** (A/B,
   identifiable from C1: 4.4739→A, 4.5915→B). Cross-window fp8 comparisons
   are valid only profile-matched. The overnight-ladder fp8 cells (g312b
   2.032127, g426b 3.257858) are single-boot values with unknown profile
   multiplicity at those sizes — a same-size two-boot repin is the cheap
   follow-up if those numbers feed a claim. nvfp4 rows need no such caveat
   on current evidence (one profile, three boots, both orders, 31B).

## Blog-claim language that survives

"At 31B with fp8 KV, scored perplexity on our three corpora is consistently
BETTER than bf16 — across both request orders and both of the two
boot-profiles we can reproduce bitwise. NVFP4 KV beats bf16 on two of three
corpora and loses on natural prose, identically in every boot and order. The
earlier 'fp8 quality depends on whether you chat before you score' framing
is wrong and is retired: the dependence is per-server-boot, not
per-request-order."

## Artifacts

Per-cycle in `results/`: server logs, proof-line files (route evidence:
bf16/nvfp4 rows FLASHINFER-only + "FA2 VO split" + linear V-SF latch lines,
zero TRITON_ATTN dispatch; fp8 rows forced TRITON_ATTN x2, zero FlashInfer
dispatch), C1 a/b + C2 + C3 PPL JSONs with stdout/stderr, verbatim smoke
JSONs (sparkok + tokyo per cycle), nvfp4_cold harness-RED stderr kept.
Status ledger: `status.txt`. KV capacities (per boot, informational): bf16
61,168-62,013; fp8 123,389-124,255; nvfp4 215,628-220,425 tokens.
