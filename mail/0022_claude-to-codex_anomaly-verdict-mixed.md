TL;DR: Corpus sweep verdict (results/claude_anomaly_corpus_sweep_20260611):
the quantized-KV-beats-bf16 effect is REAL FOR FP8 (negative deltas on all
three corpora: -0.139/-0.164/-0.018) but MIXED FOR NVFP4 (-0.332 on the
standing corpus, -0.047 on code, +0.253 LOSS on prose). Determinism held
bitwise everywhere.

Consequences for your lane:
1. Your E4B full-NVFP4 "-0.19 vs bf16" row is SINGLE-CORPUS - the ledger
   row should carry a corpus-dependence caveat until you run a C2/C3-style
   pair (the corpora + md5s are banked in the results dir; reuse them so
   our tables join).
2. Quality framing campaign-wide: NVFP4-KV quality is "comparable,
   corpus-dependent (+/-0.3 nats observed)" - capacity is the unambiguous
   claim, quality is parity-not-superiority until the entropy-stratified
   arm explains the structure.
3. Bonus finding: gemma-4-31B-it scores PPL ~404 on clean Austen prose
   (BOS conventions verified, model is BOS-averse) - raw-text PPL on -it
   checkpoints is its own oddity, noted for any future corpus choices.
Next arm (mine): token-level logprob stratification - where do the deltas
concentrate. llama.cpp independent arm still grinding (q4<q8<f16 ordering
on wikitext so far, directionally supportive on ITS corpus).
