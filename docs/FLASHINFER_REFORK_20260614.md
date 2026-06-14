# FlashInfer proper re-fork onto current upstream (2026-06-14)

**Problem:** the campaign flashinfer fork (`spark/hijinks-022-fa2-d512`, HEAD 3fa0775c) had NO shared
git history with upstream `flashinfer-ai/flashinfer` — it was a tree-copy + re-committed upstream
history (the `#PR` commits had different SHAs than real upstream), so `git rebase upstream/main`
tried to replay 2362 commits from the initial commit. Not a real fork.

**Fix:** identified our work as exactly **19 Jetha-authored commits** on base `a2870343` (a copy of
upstream #3371). Rebased the right way: `git rebase --onto upstream/main a2870343`. One conflict
(`flashinfer/gemm/gemm_base.py` — upstream added `is_sm120` SM120-only dispatch that intentionally
excludes SM121; resolved to our `is_sm12x` to keep SM121/GB10 b12x dispatch, the whole point of our
"Enable SM121 FP4 dispatch" commit). The other 18 commits replayed clean.

**Result:** `origin/spark/hijinks-022-refork` (HEAD 6663729d). `upstream/main` is now a true
ancestor; 19 clean campaign commits on top. nvfp4 attention + FA2DetermineCtaTileQ smem fix intact.

**Verified on vast PRO6000 (sm_120):** built from a raw clone (JIT-compiles on the newer upstream
base — no API-drift break), and the 12B nvfp4 anchor is **bit-identical** to the old fork:
single (nbt8192) **8.7031**, chunked (nbt4096) **8.467**. So the re-fork is correct AND the
+0.42-vs-+0.19 long-ctx bug **persists identically** → it is a real bug in the nvfp4 attention code
(ours/#3097), not a stale-base artifact. Old branch `spark/hijinks-022-fa2-d512` on origin is
preserved unchanged.

**Adopt:** point the campaign builds (CI wheel workflow, Spark image, box setup) at
`spark/hijinks-022-refork` once a wheel is rebuilt + the AR ladder re-validated on it.
