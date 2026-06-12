# Split-dtype removal (descoped from the FA2 d512 + NVFP4-KV headline)

Campaign `dgx-spark-hijinks`. FlashInfer fork `jethac/flashinfer`, branch
`spark/hijinks-022-fa2-d512`. Jetha descoped the split-dtype
(`k_data_type != v_data_type`) work from the headline (memory:
`split-dtype-descoped`); the branch should match the clean PR. This note records
the removal.

New FlashInfer head: **`3fa0775c`** (was `11ebef4c`).

## What was removed

The split-dtype shim lived entirely in two Python commits, both touching only
`flashinfer/decode.py` + `flashinfer/prefill.py`:

- `8d85fff9` — `plan()`: accept `k_data_type`/`v_data_type` kwargs (SGLang
  API-drift shim). Equal dtypes collapse to `kv_data_type`; unequal raised
  `NotImplementedError`.
- `f99323bd` — `plan()`: accept the validated `(fp8_e4m3, uint8-packed-NVFP4)`
  mixed-KV pair (mapped to `kv_data_type=uint8`); other unequal pairs still
  raised.

Removal commit `3fa0775c` (`plan(): remove descoped split-dtype ...`) deletes, in
all four wrapper `plan()`s — `BatchPrefillWithPagedKVCacheWrapper`,
`BatchPrefillWithRaggedKVCacheWrapper`, `BatchDecodeWithPagedKVCacheWrapper`, and
the underlying decode `plan()`:

- the `k_data_type` / `v_data_type` keyword parameters (4 signatures), and
- the `if k_data_type is not None or v_data_type is not None:` blocks — the
  `_kd`/`_vd` canonicalization, the equal-collapse, the `(float8_e4m3fn, uint8)`
  mixed-pair special case, and the `NotImplementedError`.

It is a **pure removal**: `-99` lines, **0 added** (`git diff` shows no `+`
content lines). No C++/CUDA, JIT, jinja, or csrc files were touched.

## Caller-compatibility decision: reverted the params entirely (no shim kept)

Decision rule from the task: if any in-scope caller (esp. full-NVFP4 vLLM) passes
the `k_data_type`/`v_data_type` param **names**, keep them accepted-but-collapsed;
otherwise revert the params and (if needed) update callers.

Audit of the worktrees:

- `B:\workshop\worktrees\vllm\*` (all `spark-hijinks-*` vLLM worktrees) — every
  `wrapper.plan(...)` / `self.plan(...)` call site passes `kv_data_type=...`
  (the single-dtype param). **Zero** call sites pass `k_data_type=` or
  `v_data_type=`.
- `B:\workshop\worktrees\sglang\*` — same: probes/refs pass
  `kv_data_type=torch.uint8` for the full-NVFP4 path; **zero** `k_data_type=` /
  `v_data_type=` keyword call sites.

Because **no caller passes the split param names**, the params were **reverted
entirely** (option b with an empty caller-update set). No coordinated
vLLM/SGLang change was required.

## Full-NVFP4 is intact

The full-NVFP4 path (`k == v == uint8`) never depended on the split-dtype block.
Callers pass `kv_data_type=torch.uint8` directly; with the block gone, that flows
straight into the unchanged sequence
`q_data_type = canonicalize_torch_dtype(...)` → `kv_data_type =
canonicalize_torch_dtype(torch.uint8)` — byte-identical to pre-shim behavior.

The FP4 JIT **module-keying guards are untouched** and still key on the single
`kv_data_type`:

- `flashinfer/jit/attention/modules.py`:
  `require_fp4_kv_cache = dtype_map_kv[dtype_kv] == "__nv_fp4x2_e2m1"` and the
  scale-factor-tensor requirement — unchanged.
- `csrc/batch_prefill_customize_config.jinja`: the `require_fp4_kv_cache`
  `#error` + `static_assert(std::is_same_v<DTypeKV, __nv_fp4x2_e2m1>)` —
  unchanged.

`git diff 3fa0775c~1..3fa0775c -- flashinfer/jit/ csrc/ include/` is empty.

## Validation

- `python -m py_compile flashinfer/decode.py flashinfer/prefill.py` — OK.
- AST check: all four `plan()` signatures parse and contain **no**
  `k_data_type`/`v_data_type`; grep confirms zero `k_data_type`/`v_data_type`/
  `_kd`/`_vd` references remain in `flashinfer/`.
- Pure-removal diff confirmed (no added content lines).
- Full `import flashinfer` and the FP4 JIT tests
  (`tests/jit/test_attention_utils.py`:
  `test_batch_prefill_nvfp4_swa_paged_params_declares_sf_strides`,
  `test_batch_prefill_nvfp4_requires_sf_tensors`) require `tvm_ffi`, which is not
  installed in the CPU edit/test env, so they could not run here. The change is
  Python-surface-only and does not affect those guards; a **GPU JIT smoke is
  deferred to the next wheel's validation** (batched with the Gemma 3 flip). The
  P520 was not contended for this.
