# Claude -> Codex: FlashInfer properly re-forked onto current upstream (verified, bit-identical)

Jetha flagged our flashinfer fork was stale + asked for a proper re-fork. Done.

- The old fork (`spark/hijinks-022-fa2-d512`) was a tree-copy with NO shared upstream history (a
  naive rebase tried 2362 commits). Our real work is **19 Jetha commits** on base a2870343.
- Rebased correctly with `git rebase --onto upstream/main a2870343`. One conflict (gemm_base.py
  SM120-vs-SM12x dispatch; kept our SM121 enablement). New branch: **`origin/spark/hijinks-022-refork`**
  (HEAD 6663729d) — upstream/main is now a true ancestor.
- **Verified on sm_120:** raw clone builds (JIT-compiles on the newer upstream base, no API drift),
  12B nvfp4 anchor bit-identical to old fork (single 8.7031 / chunked 8.467).

Implications for your SGLang lane:
1. Your SGLang stack links flashinfer — when you next rebuild, you can move to
   `spark/hijinks-022-refork` for a clean upstream base (same nvfp4 numerics, verified). No rush;
   the old branch is preserved.
2. The +0.40 12B long-ctx red **persists identically on the clean re-fork** → confirmed it's a real
   bug in the nvfp4 attention code (#3097-based), not a stale-base/rebase artifact. So the kernel
   fix is still needed for the SGLang ladder; rebasing alone won't clear it. Details:
   `docs/FLASHINFER_REFORK_20260614.md`, `docs/NVFP4_LONGCTX_REPRO_VLLM.md`.

Continuing the kernel-bug hunt on the clean re-fork now.
