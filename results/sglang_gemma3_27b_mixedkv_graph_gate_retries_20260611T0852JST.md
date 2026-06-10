# SGLang Gemma 3 27B mixed-KV CUDA graph gate retries

Date: 2026-06-11 JST

Scope: follow-up to `results/sglang_gemma3_27b_mixedkv_graph_gate_20260611T071237JST.md`.
These retries test whether `jethac/sglang@d048bfedb` clears the fp8 graph-capture
failure from the original row.

## Code State

- Parent repo: `b325d36865d6f3a680846c6827bc7b104aa3fbdb`
- SGLang: `d048bfedb5bcb52db9d1b6042037fce258e92744`
- FlashInfer: `fb7d62ea45f19cb61f19057a93519c17b6e257f3`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Model: `google/gemma-3-27b-it`
- Graphs: enabled (`DISABLE_GRAPHS=0`)
- Docker memory cgroup: `100g`

## Retry 1

Run: `sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST`

Artifacts:

- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST_launch_env.txt`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST_fp8_server.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST_fp8_install.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST_fp8_container_inspect.json`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry1_20260611T083410JST_runner.log`

Result: red before readiness, but the original `k_data_type` / `v_data_type`
TypeError is gone.

New failure:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/work/third_party/flashinfer/flashinfer/data/csrc/batch_prefill_customize_config.jinja'
```

Interpretation: FlashInfer JIT graph capture now gets far enough to generate a
batch-prefill module, but the editable source overlay does not expose the package-data
`data/csrc` layout expected by FlashInfer's JIT loader.

## Retry 2

Run: `sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST`

One isolated-checkout source-overlay hygiene fix was applied before launch:

```text
third_party/flashinfer/flashinfer/data/csrc -> ../../csrc
```

Artifacts:

- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST_launch_env.txt`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST_fp8_server.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST_fp8_install.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST_fp8_container_inspect.json`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry2_pkgdata_20260611T084259JST_runner.log`

Result: red before readiness. The Jinja template miss is fixed, and JIT reaches
`ninja`, but generated kernels cannot include FlashInfer headers:

```text
fatal error: flashinfer/attention/prefill.cuh: No such file or directory
fatal error: flashinfer/attention/mask.cuh: No such file or directory
fatal error: flashinfer/page.cuh: No such file or directory
```

Interpretation: same root class as retry 1. The editable source overlay is missing
the packaged `flashinfer/data/include` header layout expected by JIT. This is a
binary/source provenance and package-data layout problem, not a Gemma quality result.

## Stop Decision

Stop live graph-gate retries here. Continuing by adding ad hoc source-overlay links one
at a time would produce a fragile row and risks hiding stale-binary/package-data issues.

Next safe options:

1. Build/use a FlashInfer image or wheel where `flashinfer/data/csrc` and
   `flashinfer/data/include` are populated from the same `fb7d62ea` source tree, then
   print native binary/package md5 proof lines before graph capture.
2. Add a deliberate overlay-preflight script that verifies every FlashInfer JIT path
   before serving:
   - `flashinfer/data/csrc/batch_prefill_customize_config.jinja`
   - `flashinfer/data/include/flashinfer/page.cuh`
   - `flashinfer/data/include/flashinfer/attention/prefill.cuh`
   - `flashinfer/data/include/flashinfer/attention/mask.cuh`
3. Rerun the same fp8 graph gate only after that provenance gate is green.

Status: graph gate still red, but narrowed. The SGLang decode API-drift patch cleared
the first bug; the remaining blocker is FlashInfer editable-overlay package data.
