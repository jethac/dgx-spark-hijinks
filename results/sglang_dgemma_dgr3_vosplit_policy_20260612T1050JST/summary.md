# SGLang DiffusionGemma DG-R3 VO-Split Policy

Date: 2026-06-12 JST

Scope: source-policy enablement and off-box build/static validation only. No
Spark run, no model weights, no server, and no serving-quality claim.

## Result

GREEN for the DG-R3 opt-in policy gate.

The stock DiffusionGemma cookbook path remains Triton. The experimental
FlashInfer D=512 VO-split path is allowed only when both conditions are true:

- `--attention-backend flashinfer`
- `SGLANG_FLASHINFER_VOSPLIT=1`

In all DiffusionGemma modes, CUDA graphs stay disabled and chunked prefill stays
disabled.

## Commits

- Runtime policy commit: `jethac/sglang@f36ecf495bc71bf6c8a7f30952dc2adcf8a91899`
- Static-audit marker fix / branch head:
  `jethac/sglang@dec4c040a8ede4561c1f26cccc599286643b49fd`
- Hijinks branch records this row after the submodule update.

## Ubicloud Runs

Static audit:

- Workflow: `hijinks-dgemma-static-audit`
- Run: <https://github.com/jethac/sglang/actions/runs/27389284256>
- Commit: `dec4c040a8ede4561c1f26cccc599286643b49fd`
- Result: success
- Proof lines: runner `ubicloud-persistent-sglang-x64`, machine `vmzrp4jj`,
  CUDA toolkit `13.0 / V13.0.88`, `ok: true`, and policy text:
  `stock Gemma4Renoise stays Triton; FlashInfer requires
  SGLANG_FLASHINFER_VOSPLIT=1 plus --attention-backend flashinfer`.

Wheel build:

- Workflow: `hijinks-sglang-wheel-build`
- Run: <https://github.com/jethac/sglang/actions/runs/27389235780>
- Commit: `f36ecf495bc71bf6c8a7f30952dc2adcf8a91899`
- Result: success
- Wheel:
  `artifacts/sglang-wheel-f36ecf495bc71bf6c8a7f30952dc2adcf8a91899/sglang-0.0.0.dev0+gf36ecf495b-cp312-cp312-linux_x86_64.whl`
- Size: `11646076` bytes

## Interpretation

This unblocks a controlled DG-R3 Spark experiment without destabilizing the
DG-R1/DG-R2 stock-runtime evidence. A default `Gemma4Renoise` launch still
forces Triton. A DG-R3 launch must opt into FlashInfer VO-split explicitly, then
prove at runtime that D=512 global layers route through VO-split and that BF16
text-only output remains coherent.

The next GPU gate is still a serving gate, not a documentation claim:

- BF16/no-KV-quant DiffusionGemma
- `SGLANG_FLASHINFER_VOSPLIT=1`
- `--attention-backend flashinfer`
- graphs disabled, chunked prefill disabled
- compare against the stock Triton DG-R2 revised text-only prompt set
- stop on incoherence, empty output, or routing evidence missing
