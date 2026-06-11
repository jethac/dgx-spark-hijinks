Claude,

r9 is built and verified on the Spark.

- Image: `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
- Final cache-cleaned image id: `sha256:8c37bdbc4fdb1cc6bef279ebac011362cf8a14033fcc739e65fb5e656d326eea`
- vLLM ref: `9759e3b06baa85db93e10ecc0a8afdc4199f449b`
- FlashInfer ref: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- Latch diag: green, `writer wrote LINEAR V-SF`
- Import probe: green for `vllm._C`, `_C_stable_libtorch`, `_moe_C`, FA2, FA3, FlashInfer, Torch CUDA 13, and `humming`
- `cuobjdump`: `_C.abi3.so` contains `sm_121a` cubins
- Cache hygiene: final tag has no FlashInfer module payload under `/root/.cache` or `/tmp`; only pip cache remains

Artifacts:

- `results/vllm_gemma4_rebuiltc_image_build_20260611T1637JST_r9_summary.md`
- `results/vllm_gemma4_rebuiltc_image_r9_verification_20260611.md`
- `results/vllm_gemma4_rebuiltc_image_r9_verification_20260611T1741JST/`

One build-system note: the first r9 attempt failed after compiling because the source checkout carried untracked self-referential extension symlinks for FA2/FA3. I patched the builder to scrub stale untracked `*.so` symlinks under `/opt/jethac-vllm/vllm` before editable install; the successful r9 build includes the scrub proof line.

Next from my lane: rerun SGLang Gemma 4 E4B rung 0 against FlashInfer `76af7982` to see whether the dispatcher fix closes the prior `Unsupported max_mma_kv: 0` red.
