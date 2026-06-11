# Colab sm_120 session 1: blocked by toolchain, not kernels (2026-06-11)

All 8 probe rows failed in 1-4 s with the identical pre-JIT error:
`Failed to get device capability: SM 12.x requires CUDA >= 12.9.`
(SUMMARY_sm120.json from the user's Drive, retrieved 2026-06-11.)

Finding: stock Colab Pro G4 runtimes ship CUDA < 12.9; FlashInfer's
capability gate refuses SM 12.x below that floor, so NO sm_120 JIT work is
possible on an unmodified Colab runtime. The two expected-RED rows "matched"
for the wrong reason (same env error, not the trait guard) - no kernel
conclusion of any kind from this session.

Consequences:
1. The probe matrix needs a CUDA >= 12.9 toolchain bootstrap cell (torch
   cu130 build + matching nvcc) with a runtime restart - notebook update
   queued pending the user's `torch.version.cuda` reading.
2. This is a REQUIRED ingredient for the eventual public BEFORE/AFTER
   notebook: stock Colab cannot run any of this without the bootstrap.
   That cell is now on the critical path of the Colab deliverable.
3. Family-wide kernel verdict: still unanswered; session 2 attempt after
   the bootstrap lands.
