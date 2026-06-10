# Draft upstream issue: FlashInfer module cache key ignores effective compile flags

Target repo: `flashinfer-ai/flashinfer`

## Title

FlashInfer JIT/AOT module cache identity ignores effective compile flags, allowing stale
modules to shadow mode-specific builds

## Summary

FlashInfer's generated module identity does not appear to include the effective CUDA
compile flags or mode knobs that change kernel behavior. `FLASHINFER_EXTRA_CUDAFLAGS`
can change the compiled kernel, but the generated module name/URI and AOT lookup stay
the same. As a result, a stale JIT module or AOT module compiled under one flag set can
be reused when the caller expects another flag set.

This matters for any behavior selected by compile flags. In this campaign it surfaced
while validating NVFP4 paged attention V scale-factor layout modes:

```text
-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1
```

vLLM's swizzled V-scale layout and SGLang's linear V-scale layout require different
reader behavior. If those modes share the same module cache identity, one runtime can
silently run a kernel compiled for the other layout.

## Evidence

Campaign analysis:

- `results/jit_cache_mode_unsoundness_analysis_20260611.md`

Code-level findings from that analysis:

- `FLASHINFER_EXTRA_CUDAFLAGS` enters the compile command.
- The generated module URI/name does not include an effective flag hash.
- `FLASHINFER_JIT_DIR/<module_name>/<module_name>.so` is keyed by module name.
- AOT lookup returns `FLASHINFER_AOT_DIR/<module_name>.so` whenever present, before a
  flag-specific rebuild can occur.

Observed campaign consequence:

- The r7 rebuilt vLLM image initially produced a green Block C linear-V-SF regression row
  even though later forensics showed its `_C_stable_libtorch.abi3.so` writer still wrote
  swizzled V scale factors under the linear-SF latch.
- The latch diagnostic convicted that binary directly:

```text
artifact: results/claude_roundtrip_20260611/diag_linear_latch_head128.json
verdict: writer wrote SWIZZLED V-SF despite env=1 (latch/dispatch ignored)
v_dequant_as_linear cosine: 0.945627...
v_dequant_as_swizzled cosine: 0.995489...
```

- A clean r8 rebuild from the corrected source then passed the same latch diagnostic:

```text
artifact: results/vllm_gemma4_rebuiltc_image_r8_clean_latch_diag_20260611.json
verdict: writer wrote LINEAR V-SF
```

Cache listings captured for the images:

- `results/jethac-vllm-aeon-gemma4_ad2337814-rebuiltc-fb7d62ea-sm121a_flashinfer_module_dirs_20260611.txt`
- `results/jethac-vllm-aeon-gemma4_e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8_clean_flashinfer_module_dirs_20260611.txt`

Note: the simple local r7 listing in this branch did not itself expose an AOT `.so`
payload. The unsoundness claim does not depend on that listing alone; it is the
combination of the cache-key code path, the defective-r7/clean-r8 latch diagnostic, and
the real-world green-on-defective-binary accident that makes this worth filing.

## Expected Result

Generated module identity should include every input that can change compiled kernel
semantics. Changing effective CUDA flags or semantic mode knobs should not reuse or be
shadowed by a module compiled under different settings.

At minimum:

- A JIT module compiled with `FLASHINFER_PAGED_V_SF_DESWIZZLE=1` should not share the same
  module key with one compiled without it.
- AOT modules should not silently shadow a requested JIT build when env flags imply a
  different semantic mode.
- Debug output should make the selected AOT/JIT module path and effective flags visible.

## Minimal Repro Shape

The concrete campaign trigger is a paged-prefill/readback test with NVFP4 KV and V scale
factor layout differences:

```text
layout: NHD
page_size: 16 for vLLM, 1 for SGLang
KV dtype: packed NVFP4
V-SF mode A: swizzled, read with FLASHINFER_PAGED_V_SF_DESWIZZLE=1
V-SF mode B: linear, read without deswizzle
```

Build/run the same generated module name once with the deswizzle flag and once without.
If the module cache is not cleared, the second run can reuse the first build's semantics.

The campaign latch diagnostic used for this:

- `scripts/nvfp4_linear_latch_diag.py`

## Candidate Fix Direction

Preferred:

- Include an effective compile-flag hash, or explicit semantic JIT args, in the module
  key/URI.
- Promote V-SF layout mode from an ambient compile flag into an explicit generated-module
  parameter so the module name records the behavior.

Also useful:

- Refuse AOT shadowing when `FLASHINFER_EXTRA_CUDAFLAGS` or other effective JIT flags are
  present unless the AOT artifact declares a matching flag hash.
- Emit the selected module path, AOT/JIT source, and effective flags in debug logs.

## Workarounds Used By The Campaign

- Ship images with no prebuilt FlashInfer AOT/cubin module cache unless the payload is
  documented.
- Clear FlashInfer JIT/AOT caches before mode-sensitive validation.
- Add a build-verification latch diagnostic that directly checks whether the binary honors
  the requested V-SF layout.
- Treat V-SF layout as a future module-identity knob in downstream wrapper work.

## Attached Campaign Artifacts

- `results/jit_cache_mode_unsoundness_analysis_20260611.md`
- `results/claude_roundtrip_20260611/diag_linear_latch_head128.json`
- `results/vllm_gemma4_rebuiltc_image_r8_clean_latch_diag_20260611.json`
- `results/vllm_gemma4_rebuiltc_image_r8_verification_20260611.md`
- `results/jethac-vllm-aeon-gemma4_ad2337814-rebuiltc-fb7d62ea-sm121a_flashinfer_module_dirs_20260611.txt`
- `results/jethac-vllm-aeon-gemma4_e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8_clean_flashinfer_module_dirs_20260611.txt`
