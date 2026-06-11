# Colab sm_120 session 2: FAMILY-WIDE — all 8 rows match GB10 (2026-06-11)

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition, capability (12,0),
torch 2.12.0+cu130, flashinfer 0.6.13 @ fb7d62ea (source-tree mode),
notebook version DINGO (CUDA-13 apt bootstrap; stock Colab ships 12.8).

| probe | sm_120 | GB10 ref | match |
|---|---|---|---|
| A1 vosplit batch4 | PASS NHD 0.9999986 / HND 0.9999985 | 0.9999986/0.9999985 | yes |
| A2 vosplit qo_len=1 | PASS 0.9999986/0.9999986 | 0.9999985/0.9999987 | yes |
| A3 vosplit signed | PASS 0.9999984/0.9999983 | 0.9999984/0.9999983 | yes |
| A4 d128 control | PASS 0.9999995-0.9999999 | same band | yes |
| geom e4b bf16 (group 4) | PASS | PASS | yes |
| geom 31b bf16 (group 2) | PASS | PASS | yes |
| fp8 (512,256) | RED: trait guard, VERBATIM config match (NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1, prefill.cuh:3215) | same | yes - verified genuine, not env |
| bf16 d512 symmetric | RED: trait guard (the D=512 wall) | same | yes |

Consequences:
- Every dispatcher finding is CC 12.x FAMILY-WIDE: the upstream issues
  (selector overpromise, fp8 1-byte term, D=512 wall, out-width fix they
  motivate) are now two-platform reports - including on vllm#40677's exact
  hardware class.
- The VO-split kernel route is validated on consumer Blackwell: the eventual
  public notebook's AFTER path has working kernels on its target silicon.
- Session-1 lesson held: expected-RED rows were re-verified by error STRING,
  not just status (session 1's REDs had matched for the wrong reason).
Raw per-probe JSONs: user's Drive results_colab_sm120/ (fp8 record verified
inline above; full sync on request).
