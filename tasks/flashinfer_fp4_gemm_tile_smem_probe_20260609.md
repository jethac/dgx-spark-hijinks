# FlashInfer FP4 GEMM Tile/SMEM Probe, 2026-06-09

Status: queued live GB10 packet.

Purpose: separate three claims that have been getting blurred:

1. SM121 auto-dispatch includes the `b12x` FP4 GEMM path.
2. JIT/AOT artifacts are compiled for a GB10-compatible target such as `sm_121a`.
3. The selected FP4 GEMM tile actually fits the CC-12.x per-block shared-memory ceiling
   and performs acceptably on the live device.

The first two are not enough. TensorRT-LLM #11368 documents the failure class: routing an
SM12x device into a Blackwell FP4 path can still select tiles that exceed the CC-12.x
per-block shared-memory limit. The campaign already proved that the FlashInfer SM121
gate can enable `b12x`; this packet proves or falsifies whether the selected model-shaped
FP4 GEMM path is usable as a performance primitive.

## Preconditions

- Use a source-built or source-overlay FlashInfer stack, not stale `flashinfer-cubin` or
  `flashinfer-jit-cache` wheels.
- Record the FlashInfer commit, CUDA version, Torch version, and container/image tag.
- Clear or isolate the FlashInfer JIT cache before the first run when the goal is compile
  evidence.
- Keep this separate from NVFP4 KV. This packet is weight/GEMM evidence only.

## Command

Run a short single-artifact backend comparison first:

```bash
RUN_ID=flashinfer_fp4_gemm_tile_smem_$(date +%Y%m%dT%H%MJST)

python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase tile-smem-probe \
  --run-id ${RUN_ID} \
  --container ${CONTAINER_TAG} \
  --preset dense_decode \
  --preset moe_expert \
  --backend auto \
  --backend b12x \
  --backend cutlass \
  --backend cudnn \
  --iterations 30 \
  --warmup 5 \
  --output results/${RUN_ID}.json \
  2>&1 | tee results/${RUN_ID}.log
```

If a backend fails, preserve the JSON and log. A launch/configuration failure is the point
of the probe, not a reason to discard the run.

## Required Artifact Fields

The JSON must include:

- `device.compute_capability`
- `device.multi_processor_count`
- `device.shared_memory_per_block`
- `device.shared_memory_per_block_optin`
- `device.shared_memory_per_multiprocessor`
- `heuristic`
- one row per `(shape, backend)` with either:
  - `ok=true`, finite output, cosine versus BF16 `torch.mm`, mean latency, approximate TFLOP/s
  - or `ok=false` with the CUDA/configuration error text

## Acceptance

Green for "tile usable":

- device is GB10 / compute capability `12.1`;
- `shared_memory_per_block_optin` or equivalent live property is recorded;
- `auto` selects or reaches the intended SM12x FP4 path without stale binary packages;
- model-shaped dense and MoE proxy cases run without invalid-configuration/shared-memory
  failures;
- output is finite with a sane cosine versus BF16 reference;
- latency is compared against `cutlass`/`cudnn` on the same image and shapes.

Red but useful:

- `b12x` compiles for `sm_121a` but fails launch/configuration on model-shaped cases;
- `auto` reaches `b12x` but the chosen tile is slower across MoE-shaped cases;
- logs show a shared-memory/tile selection problem even though dispatch and JIT target
  evidence are correct.

## Reporting Rule

Do not cite this as an end-to-end serving speedup. Even a green packet is only a kernel
primitive. A serving claim still needs model logs or profiler evidence showing the same
FP4 GEMM path is on the critical path, plus matched before/after throughput.
