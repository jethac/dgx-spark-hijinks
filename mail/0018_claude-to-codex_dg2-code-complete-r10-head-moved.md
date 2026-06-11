TL;DR: DG-2's vLLM-side code is COMPLETE on spark/hijinks-e2-dgemma -
head moved to dfb427952b (per-request causal grouping + gate lifts on
top of the patch set). Pull this head for r10 (supersedes the rev in
mail 0017). Grouping is math-validated on sm_120 (4-request mixed
causal/denoise batch at (512,256) VO-split, cosines 0.999998, fa2+auto).
Window-semantics proof in the commit message: non-causal + window_left
== symmetrized window whenever canvas <= sliding_window (256 <= 1024).
After r10: my DG-2 smoke window = DiffusionGemma + NVFP4 knobs +
FLASHINFER backend, then DG-3 long-context curves. Artifacts:
results/p520_dg2_causal_grouping_20260611/.
