# Window packet: anomaly corpus sweep (task 25 main arm) - ~70 min

Question: is quantized-KV-beats-bf16 corpus-dependent? Current evidence is
ONE corpus (abb63f0e, absolute PPL ~100 = unusual text). If the effect
inverts or vanishes on natural low-PPL text, it is a corpus interaction;
if it holds, it is a property of the stack (and possibly of FP4/FP8 KV
generally - cf. llama.cpp arm + upstream PR #21601's GSM8K direction).

Setup: r9 image, google/gemma-4-31B-it, --language-model-only, util 0.72,
ctx-8191 sweeps, THREE servers sequential (one per KV dtype), THREE corpora
per server (same harness, same order):
  C1 = the standing corpus (abb63f0e) - continuity anchor;
  C2 = natural prose, expected LOW PPL (e.g. a public-domain novel chapter
       ~60KB, exact file banked in the results dir with md5);
  C3 = code/structured text (~60KB, banked likewise).
Rows: bf16 (VLLM_FLASHINFER_VOSPLIT=1), fp8_e4m3 (no knobs/Triton),
nvfp4 (VOSPLIT+LINEAR_V_SF). 9 numbers total + the existing C1 values as
cross-checks (bf16 4.6132 / fp8 4.4739 / nvfp4 4.2813 must reproduce
bitwise per the determinism finding - any drift = config forensics first).

Read-out: per-corpus delta table (quant minus bf16). Patterns:
- deltas negative on ALL corpora -> general effect, escalate to
  token-level logprob diff (where does the NLL drop concentrate:
  high-entropy tokens? long-range positions?);
- negative only on C1 -> corpus interaction; characterize C1;
- mixed -> entropy-stratified analysis.
No claims in the ledger until this table exists; the blog's quality story
waits on it.
