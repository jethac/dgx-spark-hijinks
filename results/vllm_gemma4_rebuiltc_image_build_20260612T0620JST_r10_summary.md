# vLLM Gemma4 Rebuilt-C Image Build r10

- image: `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10`
- image id: `sha256:aed0da3f96b2de762916f036ea1906213ab4e4234f91d8fe9d4662c949b04248`
- image size bytes: `30994801508`
- base image: `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9` (id `sha256:8c37bdbc4fdb1cc6bef279ebac011362cf8a14033fcc739e65fb5e656d326eea`, id-pinned)
- vLLM ref: `9759e3b06baa85db93e10ecc0a8afdc4199f449b` (inherited from r9)
- FlashInfer ref: `76af798243d11c4910eaceaf1d62ba4227656d4a` (inherited from r9)
- transformers pin: `5.11.0`
- image generation: `r10`
- log: `/home/jethac/dgx-spark-hijinks/results/vllm_gemma4_rebuiltc_image_build_20260612T0620JST_r10.log`
- finished JST: `2026-06-12T06:20:17+09:00`

Status: built

Post-build provenance gates (GPU, run separately, banked with the
serving rows): import probe incl. CC/SM count, cuobjdump sm_121a
cubins on _C.abi3.so, nvfp4_linear_latch_diag.py, module-cache audit.
