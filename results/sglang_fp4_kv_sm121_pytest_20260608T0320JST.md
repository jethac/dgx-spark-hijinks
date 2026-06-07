# SGLang FP4 KV SM121 Targeted Pytest

Date: 2026-06-08 JST

Target branch:

- repo: `https://github.com/jethac/sglang.git`
- branch: `spark/hijinks-018-fp4-e2m1-kv-sm121`
- commit tested: `eefe8aded`
- parent gate commit: `67c7967a1`
- reference: `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`

Intent:

- Verify the Python-level `fp4_e2m1` KV compatibility gates on Linux `aarch64` without compiling SGLang kernels or running a GPU serving workload.
- Replace the previous syntax-only check and the failed ARM64 CPU Docker route with an actual targeted pytest result.

Environment:

- checkout: `/root/spark-validation/sglang-fp4-kv-sm121`
- venv: `/root/spark-validation/venvs/sglang-kv4-pytest`
- dependency reuse: benchmark venv site-packages were added read-only through `PYTHONPATH` for existing Torch/Transformers-era dependencies
- scratch venv additions: `pytest`, `orjson`, `IPython`
- GPU workload: none intended; `CUDA_VISIBLE_DEVICES=0` was set only because `sglang.test.test_utils` indexes the first character of that variable during import

Command:

```bash
cd /root/spark-validation/sglang-fp4-kv-sm121
git fetch origin spark/hijinks-018-fp4-e2m1-kv-sm121
git checkout eefe8aded
. /root/spark-validation/venvs/sglang-kv4-pytest/bin/activate
PYTHONPATH=python:/home/jethac/gemma4-evals/.venv/lib/python3.12/site-packages \
CUDA_VISIBLE_DEVICES=0 \
NVIDIA_VISIBLE_DEVICES=void \
python -m pytest test/registered/unit/server_args/test_server_args.py -k KV4Compatibility -q
```

Result:

```text
...                                                                      [100%]
3 passed, 56 deselected, 3 warnings in 3.44s
```

Interpretation:

- The SGLang SM12x `fp4_e2m1` compatibility gate tests now pass on Linux `aarch64`.
- The first pytest run against `67c7967a1` failed because the added unit-test helper populated internal `_str` fields that `_handle_kv4_compatibility()` recomputes from the public backend fields. The follow-up `eefe8aded` fixes the test setup to use `prefill_attention_backend` and `decode_attention_backend`.
- This is still not an SGLang NVFP4 serving proof. Native FP4 KV pools, FlashInfer backend wrapper plumbing, quality checks, capacity comparison, and end-to-end `--kv-cache-dtype fp4_e2m1` serving remain pending.
