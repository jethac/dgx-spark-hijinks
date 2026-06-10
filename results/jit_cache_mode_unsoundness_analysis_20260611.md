# Analysis: FlashInfer module cache is blind to FLASHINFER_EXTRA_CUDAFLAGS
# (resolves the Block C tension; upstream-worthy)

Source: flashinfer @ fb7d62ea, flashinfer/jit/core.py + cpp_ext.py (read 2026-06-11).

Facts (code-anchored):
1. The deswizzle macro rides FLASHINFER_EXTRA_CUDAFLAGS into the compile line
   (cpp_ext.py:231-234) but is NOT part of the module name/URI.
2. Module artifacts are keyed by name only: FLASHINFER_JIT_DIR/<name>/<name>.so
   (core.py JitSpec.jit_library_path).
3. **AOT precedence**: get_library_path() returns FLASHINFER_AOT_DIR/<name>.so
   whenever it exists — an AOT module compiled at image-build time (with
   image-build-time flags) permanently shadows any JIT recompile with
   different flags.

Consequence — the Block C paradox resolved (hypothesis, one in-container check
from proof): Block C (Qwen, head-128) knob-on was green on the defective image
because the standard-shape reader modules came from the AOT/prebuilt cache,
compiled WITH the deswizzle macro -> deswizzling reader + always-swizzling
writer = correct pairing BY ACCIDENT. The 31B vosplit (512,256) modules have no
AOT entry -> fresh JIT with env=1 (macro omitted) -> linear reader + swizzled
writer = the gibberish. Sliding head-256 modules: presumably also fresh
(shape/dtype combo not in AOT) -> same mismatch -> forward bisect's gibberish.

Verification next window (cheap): list FLASHINFER_AOT_DIR + FLASHINFER_JIT_DIR
in the r7 container; confirm head-128 paged FP4 module exists in AOT and the
vosplit module does not.

Implications:
- r8 image alone does NOT close the hazard: any mode knob that changes compile
  flags without changing module identity can silently pair mismatched binaries.
  Campaign rule until fixed: clear/disable AOT+JIT module caches whenever
  toggling VLLM_NVFP4_KV_LINEAR_V_SF (runners already uninstall the jit-cache
  package; extend to FLASHINFER_AOT_DIR check).
- Proper fix (upstream candidate, bank with the drafts): include
  EXTRA_CUDAFLAGS (or a hash of effective flags) in the module key, or refuse
  AOT shadowing when env flags are present. Same class as the
  selector-overpromise issue: cache identity narrower than build identity.
- Cleanest campaign-side fix: make the linear-SF mode a JIT ARG (part of the
  module name) instead of a global macro — aligns with how head dims/dtypes
  already key modules.
