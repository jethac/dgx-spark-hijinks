# SGLang Gemma 4 E4B bf16 Multimodal Natural-Image Smoke

Status: RED for the original natural-image keyword gate, useful as request-path evidence.

- Runtime: DGX Spark / GB10, packaged SGLang image `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Model: `google/gemma-4-E4B-it`
- KV: bf16 / auto
- Backend: FlashInfer, `SGLANG_FLASHINFER_VOSPLIT=1`, CUDA graphs disabled
- Asset: SGLang `man_ironing_on_back_of_suv.png`

All three probes returned HTTP 200 and were deterministic:

- text: `TOKYO` / `TOKYO`
- image: `A man is washing a yellow taxi cab on the side of a city street.` twice
- audio: correct Quilter transcript twice

The row is red for the image keyword gate because the image output missed the expected `iron`/`car`/`suv`/`vehicle`/`person` keyword set despite visibly consuming the image. This motivated the stricter deterministic shape-card gate used in the later bf16 and full-NVFP4 rows.

The wrapper exited nonzero after the client because the shell-embedded summary writer was malformed; the raw `multimodal_probe.json` and server logs are complete.
