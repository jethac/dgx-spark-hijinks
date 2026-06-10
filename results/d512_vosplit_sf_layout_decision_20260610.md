# Design decision — D=512 VO-split uses LINEAR V-SF; drop the swizzle on the SM12x path

Date: 2026-06-10. Lane: spark/hijinks-022-*. Follows the P0b green-linear result.

## The analysis (offline, prefill.cuh:512-545 vs the writer transform)
The vLLM V-SF swizzle maps linear(token i*4+a, col b*(SD/4)+c) -> swizzled(i*4+b, c*4+a):
each token-quad index b is spread across the ENTIRE SD-column row, not within 4-col blocks.
The kernel's deswizzle derives its block structure from the module's OWN width
(SF_GROUPS = HEAD_DIM_VO/16/4), so a (qk=512, vo=256) module deswizzles a head-dim-sliced
SF view with SF_GROUPS=4 while the data was written with SF_GROUPS=8. The two layouts do
not commute with slicing AT ALL - a linear half's scales are scattered across the full
swizzled row. The observed 0.958 cosine was the partially-aligned diagonal, not a small
bug. (Earlier commute claim in the plan was wrong; corrected here.)

## Options weighed
1. Kernel fix: pass the writer's full-row group structure + a column offset as runtime
   params so sliced views deswizzle correctly. Real codegen+kernel-param work, keeps TWO
   V-SF layouts alive.
2. **Sidestep (chosen): write V-SF LINEAR on the SM12x FA2 path and drop the deswizzle
   entirely.** The swizzle is a legacy of the SM100/trtllm layout; on the FA2 path WE own
   the writer (vLLM reshape_and_cache_nvfp4). Linear V-SF is:
   - the proven-green layout (P0b cosine 0.9999986; Gemma 3/Qwen serving used deswizzle
     only because the writer swizzled),
   - SGLang's native layout -> one layout across both runtimes,
   - slice-compatible by construction (VO-split needs only a contiguous column slice),
   - the death of a whole bug class (Codex's matrix showed deswizzle-on-linear corrupts;
     macro leakage between runtimes was a real incident).

## Consequence for P3 scope (vLLM branch)
P3 = (a) make the SM12x FA2-NVFP4 cache writer emit linear V-SF + stop setting
FLASHINFER_PAGED_V_SF_DESWIZZLE on that path; (b) regression-gate the existing Qwen +
Gemma 3 NVFP4 rows on the linear writer (expect identical-or-better, it is the simpler
path); (c) then the global-layer two-pass VO-split rides the green linear path unchanged.

## Honest cost
The swizzled writer + deswizzle code Codex built remains correct for symmetric vo and
stays in tree (other consumers may want it); we simply stop using it on the SM12x path.
Upstream framing improves too: "one linear SF layout on consumer Blackwell" is an easier
PR story than "two layouts plus a compile-time macro".
