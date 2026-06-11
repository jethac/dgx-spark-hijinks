# SGLang DiffusionGemma Uniform Config Fix

Date: 2026-06-11 22:07 JST

## Finding

While validating the rebased upstream DiffusionGemma runtime, the
`DiffusionGemmaForBlockDiffusion` branch in `DllmConfig.from_server_args()` set
`block_size` and `mask_id` but did not assign `is_uniform` before constructing
`DllmConfig`.

Impact: the cookbook route for `--dllm-algorithm Gemma4Renoise` could fail with
an unbound local or miss the uniform-state scheduler flag.

## Fix

SGLang branch:

- `jethac/sglang:spark/hijinks-024-diffusiongemma-upstream-rebase`
- fix commit: `0705924c1d`

Patch:

- In `python/sglang/srt/dllm/config.py`, the
  `DiffusionGemmaForBlockDiffusion` branch now reads its table entry and sets
  `is_uniform = params["is_uniform"]`.

## Validation

Environment:

- WSL provisioned env: `~/sglang_env`
- source path:
  `/mnt/b/workshop/worktrees/dgx-spark-hijinks/spark-hijinks-022-gemma4-mixed-kv/third_party/sglang/python`
- deps: `orjson 3.11.9`

Checks:

- `python -m py_compile third_party\sglang\python\sglang\srt\dllm\config.py`:
  pass.
- WSL monkey-patched `ModelConfig.from_server_args()` to return a minimal HF
  config with `architectures=["DiffusionGemmaForBlockDiffusion"]`,
  `canvas_length=256`, and `pad_token_id=0`.

Observed output:

```text
algorithm Gemma4Renoise
block_size 256
mask_id 0
max_running_requests 1
is_uniform True
```

## Limitations

This is a config-path validation only. It does not claim model weight load,
forward pass, coherence, or serving.

The full model-registry import probe remains unreliable in this local WSL
session because one earlier `git status` process is stuck in uninterruptible I/O
against the B-backed mount, and a registry scan imports many files from the same
path. The prior static AST scan still proves there is only one
`DiffusionGemmaForBlockDiffusion` `EntryClass`.
