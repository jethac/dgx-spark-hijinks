# vLLM Qwen/DFlash SM121a Patch Verification

Date: 2026-06-08 JST

Branch: `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a`

Commit: `0667185d5adaec32ff8cc8289a4d7716f6cdf966`

## Scope

This verifies the local source patch only. It is not a GB10 serving proof.

Changes:

- `vllm/platforms/cuda.py`: retry `_C_stable_libtorch` import with lazy/global dlopen flags after an initial `ImportError`.
- `vllm/config/compilation.py`: align speculative-decode CUDA graph capture sizes for every non-`NONE` CUDA graph mode, not only FULL decode graphs.
- `tests/compile/test_config.py`: add a pure `PIECEWISE` regression case.

## Commands

```powershell
python -m pytest tests\compile\test_config.py -q -k "piecewise_cudagraph_sizes_align_for_spec_decode"
```

Result:

```text
ImportError while loading conftest 'B:\workshop\dgx-spark-hijinks\third_party\vllm\tests\conftest.py'.
tests\conftest.py:7: in <module>
    from tblib import pickling_support
E   ModuleNotFoundError: No module named 'tblib'
```

The local Windows workspace is not a vLLM dev/test environment.

```powershell
python -m py_compile vllm\platforms\cuda.py vllm\config\compilation.py tests\compile\test_config.py
```

Result: passed.

```powershell
git diff --check
```

Result: passed, with line-ending warnings only.

## Remaining Proof

- Reproduce AEON Qwen3.6 NVFP4+DFlash serving on GB10.
- Run the same row on the `jethac/vllm` fork with the matching FlashInfer fork.
- Record CUDA graph mode, backend selection logs, DFlash acceptance, throughput, TTFT, and error/soak behavior.
- Run fp8-vs-NVFP4 KV capacity/quality/throughput comparisons only after the server log proves FA2 NVFP4 KV selection.
