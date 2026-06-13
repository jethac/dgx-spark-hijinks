# SGLang Gemma 4 Multimodal NVFP4 Validation Packet

Purpose: close the SGLang text-only evidence gap for Gemma 4 NVFP4 KV on DGX Spark. Claude's vast.ai reference-sim lane measures modality-specific KV ranges; this packet turns those findings into SGLang serving evidence with real multimodal requests and cache reuse.

## Scope

- Runtime: SGLang on DGX Spark / GB10 (`sm_121`)
- Branch: `epoch2`
- Packaged image baseline: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-0513-42ce5dad-arm64`
- Current packaged digest: `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Diagnostic source overlay for SF saturation trace: `jethac/sglang@58a39849fc`
- Models, in order:
  - `google/gemma-4-E4B-it` first, because it is already text scoped-green on full NVFP4.
  - `google/gemma-4-12B-it` only after E4B establishes the multimodal request path.
- Rows:
  - bf16 / auto KV baseline
  - full NVFP4 K+V with `SGLANG_FP4_KV_TRACE_SF_SATURATION=1`
  - optional full NVFP4 K+V with calibrated/global-scale fix once the text row is repaired

## Why This Exists

Current SGLang Gemma 4 evidence is text-only. Gemma 4 serving includes image and audio prefix tokens, and those tokens can have different K/V dynamic range from text. A text-only `cached_tokens > 0` row does not prove:

- multimodal prefix writes use correct KV and scale-factor storage,
- repeated multimodal prompts reuse cache safely,
- global-scale calibration based on text covers vision/audio KV,
- NVFP4 block scale factors avoid saturation for vision/audio tokens.

## Assets

Audio assets are already banked:

- `results/p520_audio_mm_20260612/assets/speech_librispeech_1272-128104-0000.wav`
- Reference transcript: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`
- `results/p520_audio_mm_20260612/assets/tone_control.wav`

Image asset policy:

- Prefer a banked local image passed to `scripts/sglang_gemma4_multimodal_chat_probe.py --image-path`.
- If no local image is banked yet, use the SGLang test image URL only as a bootstrap smoke and bank a copy in the result directory before making any claim.
- Default bootstrap URL: `https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png`

## Stop-On-Red Order

1. **bf16 E4B multimodal smoke**
   - Launch E4B bf16/auto KV.
   - Run `scripts/sglang_gemma4_multimodal_chat_probe.py` with text+image+audio, `--repeat 2`.
   - Gate: HTTP 200 for all probes, non-empty text, expected keyword hit for image/audio, same request repeated without server crash.
   - If red, stop; this is not an NVFP4 bug yet.

2. **full NVFP4 E4B multimodal smoke + SF saturation trace**
   - Launch E4B full NVFP4 K+V.
   - Enable:
     - `SGLANG_FP4_KV_TRACE_SF_SATURATION=1`
     - `SGLANG_FP4_KV_TRACE_GLOBAL_SCALE=1`
     - `SGLANG_FP4_KV_TRACE_LAYERS=0,5` initially to limit log size; widen if red.
   - Run the same client payload twice.
   - Gate: same functional gates as bf16, plus server log contains `FP4 KV SF saturation trace` for the exercised layers.
   - Compare block-scale saturation stats between text-only and multimodal requests. A vision/audio-specific saturation increase is a calibration bug candidate.

3. **12B multimodal after E4B**
   - Repeat bf16 then full NVFP4 on 12B.
   - Carry the known text caveat: 12B full NVFP4 is still red by `+0.3436` nats/token even with a scale multiplier, so a multimodal smoke can only be scoped as request-path evidence until the text quality bug is fixed.

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
  --audio-url /hijinks/results/p520_audio_mm_20260612/assets/speech_librispeech_1272-128104-0000.wav \
  --output results/<run_id>/multimodal_probe.json
```

If no image is banked yet, use `--image-url` for the first smoke only and copy the image into `results/<run_id>/assets/` before treating the row as evidence.

## Server Notes

Use one server at a time and keep the GB10 memory rules:

- Docker `--memory 100g --memory-swap 100g`
- No concurrent comparators
- Check `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` and `docker ps` before launch
- Keep >=15-20 GiB OS headroom

For diagnostic source-overlay rows that need `58a39849fc`, label the row explicitly as source-overlay. Packaged-image claim rows need a rebuilt image carrying the same SGLang change, not a mounted source tree.

## Evidence Requirements

Each row must produce:

- `preflight.log` with image digest, parent commit, SGLang commit, model, KV dtype, graph settings, and memory.
- raw request/response JSON from `sglang_gemma4_multimodal_chat_probe.py`.
- server log with routing lines and, for NVFP4, `FP4 KV SF saturation trace`.
- `STOP_SUMMARY.md` stating whether the row is:
  - functional multimodal request-path evidence,
  - cache/reuse evidence,
  - quality evidence,
  - or only a red diagnostic.

Do not claim broad SGLang Gemma 4 NVFP4 multimodal support until both text and multimodal default serving behavior with cache reuse are green or explicitly scoped.
