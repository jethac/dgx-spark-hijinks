# SGLang Gemma 4 VO-split validation on FlashInfer 8d85fff9

Date: 2026-06-11 JST

Scope: SGLang lane, weight-free Gemma 4 VO-split validation packet Blocks A/B/C. This rerun targets `jethac/flashinfer@8d85fff9`, which adds prefill and standard decode `plan()` compatibility for `k_data_type`/`v_data_type` kwargs when K/V dtypes are equal.

## Code State

- Parent repo: `jethac/dgx-spark-hijinks@025e90c` plus the harness fallback committed with this result.
- SGLang source overlay: `jethac/sglang@d048bfedb`
- FlashInfer source overlay: `jethac/flashinfer@8d85fff9`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Feature flag under test: `SGLANG_FLASHINFER_VOSPLIT=1`
- `FLASHINFER_PAGED_V_SF_DESWIZZLE` absent throughout.

## Block A: Source/Import And Cache Hygiene

Run id: `sglang_gemma4_vosplit_blockA_8d85fff9_20260611T105526JST`

Artifacts:

- `results/sglang_gemma4_vosplit_blockA_8d85fff9_20260611T105526JST/flashinfer_commit.txt`
- `results/sglang_gemma4_vosplit_blockA_8d85fff9_20260611T105526JST/import_probe.log`

Verdict: green.

Proof:

```text
flashinfer_commit: 8d85fff9
flashinfer /tmp/flashinfer-python-path/flashinfer/__init__.py
sglang /work/third_party/sglang/python/sglang/__init__.py
flashinfer_csrc /flashinfer-src/csrc
flashinfer_include /flashinfer-src/include
flashinfer_cutlass [PosixPath('/flashinfer-src/3rdparty/cutlass/include'), PosixPath('/flashinfer-src/3rdparty/cutlass/tools/util/include')]
flashinfer_cccl [PosixPath('/flashinfer-src/3rdparty/cccl/cub'), PosixPath('/flashinfer-src/3rdparty/cccl/libcudacxx/include'), PosixPath('/flashinfer-src/3rdparty/cccl/thrust')]
flashinfer_spdlog /flashinfer-src/3rdparty/spdlog/include
```

Note: an earlier Block A attempt tried to run `git -C /flashinfer-src` from inside the bind-mounted submodule and failed because the parent `.git/modules` path is not mounted into the container. The commit proof is now recorded host-side in `flashinfer_commit.txt`.

## Block B: Head-256 Writer/Reader Regression

Run id: `sglang_fp4_kv_writer_roundtrip_vosplit_regression_8d85fff9_20260611T105550JST`

Artifacts:

- `results/sglang_fp4_kv_writer_roundtrip_vosplit_regression_8d85fff9_20260611T105550JST/container.stdout`
- `results/sglang_fp4_kv_writer_roundtrip_vosplit_regression_8d85fff9_20260611T105550JST/run.log`
- `results/sglang_fp4_kv_writer_roundtrip_vosplit_regression_8d85fff9_20260611T105550JST/output.json`

Verdict: green.

Output cosines:

- global: `0.9999911785`
- SWA/window: `0.9997446537`
- decode-as-prefill: `0.9999914169`

This confirms the `8d85fff9` equal-dtype ABI shim unblocks the existing head-256 full-NVFP4 K+V writer/reader regression.

## Block C: Head-512 VO-split Writer/Roundtrip

Final run id: `sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_torchref2_20260611T110500JST`

Artifacts:

- `results/sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_torchref2_20260611T110500JST/container.stdout`
- `results/sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_torchref2_20260611T110500JST/run.log`
- `results/sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_torchref2_20260611T110500JST/output.json`

Verdict: green.

Output cosines versus the dequantized-pool torch attention reference:

- global: `0.9999923110`
- SWA/window: `0.9991498590`
- decode-as-prefill: `0.9999921918`

SWA residual note: the SWA/window cosine is lower than the global and decode-as-prefill
rows, but the follow-up mask diagnostic shows it is not a hidden data/scale-factor pairing
failure. Replaying the same seed against several torch window conventions reproduced the
original `0.9991498589` only for the `left_only` convention used by the harness; causal
variants were much worse (`0.9848601818` / `0.9838094115`) and same-length local masking
collapsed (`0.0074732359`). The residual is therefore the bounded SWA/windowed paged-reader
numerical difference against the dequantized attention reference, not evidence of a wrong
window convention or packed-KV layout bug. The absolute error stayed small:
`max_abs=0.006103515625`, `mean_abs=0.0007735347608104348`. Artifact:
`results/sglang_vosplit_swa_mask_diag_20260611T113630JST/container.stdout`.

The FlashInfer prefill debug output proves the SGLang reader requested the expected asymmetric module:

```text
dtype_kv=__nv_fp4x2_e2m1
head_dim_qk=512
head_dim_vo=256
fp4_kv=1
layout=0
page_size=1
num_qo_heads=32
num_kv_heads=16
```

Two earlier Block C attempts were harness-reference false reds:

- `sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_20260611T105828JST`: FlashInfer `single_prefill_with_kv_cache` reference rejected D=512 with `Invalid configuration`.
- `sglang_fp4_kv_writer_roundtrip_head512_vosplit_8d85fff9_torchref_20260611T110151JST`: the fallback missed the second known reference failure string, `Unsupported max_mma_kv: 0`.

The final harness falls back to `torch_dequant_attention` for those reference-only failures. The paged VO-split reader itself ran in all attempts and emitted the expected `head_dim_qk=512;head_dim_vo=256` module lines.

## Status

Green through Block C. This validates SGLang's real `MHATokenToKVPoolFP4.set_kv_buffer()` writer plus FlashInfer FA2 paged reader for:

- head-256 full NVFP4 K+V with linear scale factors;
- head-512 QK / 256 VO two-pass full NVFP4 K+V with linear scale factors;
- SWA/windowed and decode-as-prefill variants.

This is still a weight-free writer/reader gate, not a Gemma 4 serving claim. The next SGLang Gemma 4 step is wrapper-plan dry smoke / serving prep, with the same provenance proof lines and the `8d85fff9` FlashInfer baseline.

## Graph-Gate Decision

The Gemma 3 mixed-KV CUDA graph gate cannot safely collapse to a single `kv_data_type`.
SGLang's graph planner sets `k_data_type=torch.float8_e4m3fn` and
`v_data_type=torch.uint8` only when `is_fp8_k_nvfp4_v` is active; that describes true
per-tensor dtypes for the mixed FP8-K + NVFP4-V pool, not just a pool-container layout.
Under FlashInfer `8d85fff9`, equal dtypes are now unblocked, but this mixed graph-capture
site remains a genuine unequal-dtype consumer. Split-dtype module keying still belongs on
the FlashInfer queue before this graph gate can be green.
