# SGLang Gemma 4 Multimodal NVFP4 Validation Packet

Purpose: close the SGLang text-only evidence gap for Gemma 4 NVFP4 KV on DGX Spark. Claude's vast.ai reference-sim lane measures modality-specific KV ranges; this packet turns those findings into SGLang serving evidence with real multimodal requests and cache reuse.

Status update, 2026-06-14: the first package-level gate is complete. The baked
mm-prefix carrier
`ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
serves E4B full-NVFP4 text/image/audio with FlashInfer image-prefix masking
active and no old Triton-only/bidirectional fallback warning; see
`results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_baked_20260614T072000JST/STOP_SUMMARY.md`.
The remaining work is broader matched quality/capacity after the shared 12B
long-context FlashInfer/numerics red is fixed, not another global-scale
calibration pass.

## Scope

- Runtime: SGLang on DGX Spark / GB10 (`sm_121`)
- Branch: `epoch2`
- Packaged image baseline: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-mm-prefix-f920e2d-arm64`
- Current packaged digest: `sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Diagnostic SF-saturation overlays are historical only unless a new failure
  specifically needs them; the current text red is not a global-scale root cause.
- Models, in order:
  - `google/gemma-4-E4B-it` first, because it is already text scoped-green on full NVFP4.
  - `google/gemma-4-12B-it` only after E4B establishes the multimodal request path.
- Rows:
  - bf16 / auto KV baseline
  - full NVFP4 K+V from the baked mm-prefix package image
  - future claim-grade matched rows after the shared text-quality red is fixed

## Why This Exists

Current SGLang Gemma 4 evidence is text-only. Gemma 4 serving includes image and audio prefix tokens, and those tokens can have different K/V dynamic range from text. A text-only `cached_tokens > 0` row does not prove:

- multimodal prefix writes use correct KV and scale-factor storage,
- repeated multimodal prompts reuse cache safely,
- text-path calibration or numerics choices cover vision/audio KV,
- NVFP4 block scale factors avoid saturation for vision/audio tokens.

## Assets

Audio assets are already banked:

- `results/p520_audio_mm_20260612/assets/speech_librispeech_1272-128104-0000.wav`
- Reference transcript: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`
- `results/p520_audio_mm_20260612/assets/tone_control.wav`

Image asset policy:

- Prefer a banked local image passed to `scripts/sglang_gemma4_multimodal_chat_probe.py --image-path`.
- Use a deterministic shape/color card for strict bf16-vs-NVFP4 gates. The natural SGLang yellow-cab image is useful as a request-path smoke, but small Gemma 4 checkpoints can miss the fine-grained action label under bf16.
- If no local image is banked yet, use the SGLang test image URL only as a bootstrap smoke and bank a copy in the result directory before making any claim.
- Default bootstrap URL: `https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png`

## Stop-On-Red Order

1. **bf16 E4B multimodal smoke**
   - Launch E4B bf16/auto KV.
   - Run `scripts/sglang_gemma4_multimodal_chat_probe.py` with text+image+audio, `--repeat 2`.
   - Gate: HTTP 200 for all probes, non-empty text, expected keyword hit for image/audio, same request repeated without server crash.
   - If red, stop; this is not an NVFP4 bug yet.

2. **full NVFP4 E4B multimodal smoke from the baked mm-prefix package**
   - Launch E4B full NVFP4 K+V.
   - Enable:
     - `SGLANG_FLASHINFER_VOSPLIT=1`
     - `SGLANG_FP4_KV_MIXED_KV=0`
     - `FLASHINFER_PREFILL_DEBUG_ONCE=1`
   - Run the same client payload twice.
   - Gate: same functional gates as bf16, plus server log proves the mm-prefix
     custom mask, no old fallback warning, and FP4 prefill module selection.
   - The current E4B baked row is green for this scope; rerun only after the
     packaged image changes or when extending to a matched claim row.

3. **12B multimodal after E4B**
   - Repeat bf16 then full NVFP4 on 12B.
   - Carry the known text caveat: 12B full NVFP4 is still red by `+0.402969`
     nats/token on the matched long-context row, and mail 0138 classifies that
     red as a shared FlashInfer/numerics issue. A 12B multimodal smoke can only
     be scoped as request-path evidence until the text quality bug is fixed.

4. **Claim-grade rerun**
   - Once text full-NVFP4 quality is fixed, rerun bf16 vs full NVFP4 with identical multimodal prompts and record:
     - first/second repeated responses,
     - usage/cache fields if exposed,
     - server logs proving radix/cache reuse or cached-token accounting,
     - SF saturation summaries,
     - exact image/audio assets and checksums.

## Client Command

Run after the server is ready:

```bash
python3 scripts/sglang_gemma4_multimodal_chat_probe.py \
  --url http://127.0.0.1:30000 \
    --model google/gemma-4-E4B-it \
    --repeat 2 \
    --image-path results/<run_id>/assets/man_ironing_on_back_of_suv.png \
    --image-prompt "What unusual activity is the man doing on the back of the yellow cab? Answer in one short sentence." \
    --image-keywords "iron,ironing" \
    --audio-url /hijinks/results/p520_audio_mm_20260612/assets/speech_librispeech_1272-128104-0000.wav \
    --output results/<run_id>/multimodal_probe.json
```

For the strict shape/color card gate, use `--keyword-mode all` and phrase keywords:

```bash
python3 scripts/sglang_gemma4_multimodal_chat_probe.py \
  --url http://127.0.0.1:30000 \
  --model google/gemma-4-E4B-it \
  --repeat 2 \
  --image-path results/<run_id>/assets/red_square_blue_triangle.png \
  --image-prompt "Identify the two colored shapes. Answer with a short phrase." \
  --image-keywords "red square,blue triangle" \
  --audio-url /hijinks/results/p520_audio_mm_20260612/assets/speech_librispeech_1272-128104-0000.wav \
  --keyword-mode all \
  --output results/<run_id>/multimodal_probe.json
```

If no image is banked yet, use `--image-url` for the first smoke only and copy the image into `results/<run_id>/assets/` before treating the row as evidence.

## Server Notes

Use one server at a time and keep the GB10 memory rules:

- Docker `--memory 100g --memory-swap 100g`
- No concurrent comparators
- Check `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` and `docker ps` before launch
- Keep >=15-20 GiB OS headroom

For diagnostic source-overlay rows, label the row explicitly as source-overlay.
Packaged-image claim rows need a rebuilt image carrying the same SGLang change,
not a mounted source tree. Do not revive the retracted global-scale multiplier
path as a claim fix.

## Evidence Requirements

Each row must produce:

- `preflight.log` with image digest, parent commit, SGLang commit, model, KV dtype, graph settings, and memory.
- raw request/response JSON from `sglang_gemma4_multimodal_chat_probe.py`.
- server log with routing lines and, for NVFP4, proof of FP4 prefill module
  selection plus the Gemma 4 FlashInfer image-prefix custom-mask line.
- `STOP_SUMMARY.md` stating whether the row is:
  - functional multimodal request-path evidence,
  - cache/reuse evidence,
  - quality evidence,
  - or only a red diagnostic.

Do not claim broad SGLang Gemma 4 NVFP4 multimodal support until both text and multimodal default serving behavior with cache reuse are green or explicitly scoped.
