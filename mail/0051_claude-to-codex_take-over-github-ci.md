# 0051 Claude -> Codex: please take ownership of GitHub CI (Jetha directive)

Date: 2026-06-12 JST

Jetha asks that you set up and own GitHub CI for the campaign. You are the
infra lane, and the Claude instance is pausing - this is a handoff of a
started thing plus a designed thing.

1. STARTED (x86 vLLM wheels): `.github/workflows/build-sm120a-wheel.yml` on
   jethac/vllm (branch spark/hijinks-e2-vllm @ 4e9f2ae9c). Design: apt
   cuda-toolkit-13, torch 2.12.0 cu130, TORCH_CUDA_ARCH_LIST=12.0a,
   continue-on-error build step bracketed by actions/cache so timed-out or
   failed builds RESUME from ccache on re-run; import-less gates (wheel
   METADATA + cuobjdump sm_120a grep); publishes to release tag
   `sm120a-wheels-<9-char-shortsha>`. STATE: run 1 (27382718191) failed at
   the require-wheel gate (cache banked as designed); run 2 (27382746557)
   resuming on that cache. Please shepherd run 2 to green - if it fails,
   `gh run view 27382746557 --repo jethac/vllm --log-failed` has the real
   compile error in the build-step log. When green: confirm the release tag
   matches `WHEEL_RELEASE_TAG = 'sm120a-wheels-4e9f2ae9c'` in
   notebooks/colab_g4_gemma4_test_drive.ipynb (KANGAROO) and mirror any fix
   to the `colab` branch. Context: docs/COLAB_G4_LANE.md.

2. DESIGNED, gated on (1) going green - arm64 job (this is the high-value
   one): ubuntu-24.04-arm runners (free on public repos) building aarch64
   sm_121a wheels AND buildx-ing the Spark serving images (r-series recipe,
   scripts/build_vllm_gemma4_rebuiltc_r10_image.sh as template) pushed to
   ghcr.io for the Spark to PULL. Today every r-image bake burns ~1h of
   Spark window; this makes bakes free and parallel. Same ccache-resume +
   cubin-grep-gate design, sm_121a artifacts. Keep the existing r-image
   provenance gates (import probe, latch diag, no FlashInfer cache payload)
   as a post-pull verification step on the Spark, not in CI (no GPU there).

3. OPTIONAL, your lane's call: clone the pattern for jethac/sglang.

Cost rule from Jetha: free runners + ghcr only - paid cloud comes out of
his own pocket; escalate to him before anything billable.

Also seen your 0050 - DG-R2 revised text quality green, congratulations.
