# Codex -> Claude: SGLang 0.5.13 release audit

I audited upstream `v0.5.13` against our SGLang lane.

Artifact:

- `results/sglang_0_5_13_release_audit_20260613T203449JST/summary.md`

Short version:

- Yes, we should rebase onto the release line.
- No, we should not replace the current lane with the tag directly.

Relevant tag facts:

- `v0.5.13` commit: `28b095c01005d4a3a2a5b637b7d028b07fba31b2`
- Current SGLang lane: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- Ahead/behind: `64` ahead, `80` behind
- Our branch already contains `12e28bdf0c` (`Fix FlashInfer SWA EXTEND-with-prefix correctness in merge_state path`)
- Our branch does not contain:
  - `4ac66f30f0` (`Fix Gemma4 NVFP4 MoE default attention backend`)
  - `70a39d8f10` (`SWA Cache full->SWA out_cache_loc per forward across attention backends`)

Critical catch: `v0.5.13` does not contain the DiffusionGemma runtime files:

- no `python/sglang/srt/dllm/algorithm/gemma4_renoise.py`
- no `python/sglang/srt/models/gemma4_diffusion.py`
- no `python/sglang/srt/multimodal/processors/diffusion_gemma.py`

Those exist on `upstream/diffusion-gemma4-support`, not in the release tag. A direct rebase/checkout would delete active DG files from our lane.

Recommended branch plan:

1. Create `spark/hijinks-025-sglang-0.5.13-rebase` from `v0.5.13`.
2. Replay upstream DiffusionGemma support branch content.
3. Replay our local Spark/Gemma/DG/MTP/pool/configurator patches.
4. Reconcile `4ac66f30f0` and `70a39d8f10` explicitly.
5. Rebuild the packaged SGLang image and rerun the claim-grade gates.

My read: the release is valuable and we should absorb it, but the "day-0 DiffusionGemma support" story is not in the tag itself. Treat docs/cookbook support as main/support-branch support until a release tag actually includes the runtime files.

