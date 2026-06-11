# Audio multimodality under the Triton retirement (Amendment 5)

Status: recon + static policy tests COMPLETE (2026-06-12); P520 serving
cells queued LAST in the GPU queue (after small-size ladder, MTP identity
ladder, image mm smokes). Assets banked. Spark 12B-audio cell specced for
the morning block.

Scope: Gemma 4 E2B / E4B / 12B carry audio encoders; the mm-prefix
retirement flip (`spark/hijinks-e2-mm-retire` @ 976626448d) was recon'd
against VISION spans only. This note closes the audio gap.

## Recon Q1: how gemma4_mm handles AUDIO (vLLM @ 20196b5946)

- Encoder: `Gemma4AudioModel` (USM-architecture conformer) loaded via
  `AutoModel.from_config(config.audio_config)` —
  `vllm/model_executor/models/gemma4_mm.py:1057`; projection
  `Gemma4MultimodalEmbedder` (`embed_audio`, output_proj_dims 1536) at
  `gemma4_mm.py:1063`. The Unified variant is encoder-free (raw frame
  features straight into `embed_audio`,
  `gemma4_unified.py:285,417`).
- Token positions: the processor replaces the audio placeholder with
  `audio_token_id` repeated `num_tokens` times; `num_tokens` replicates
  the encoder's mel-framing + conv subsampling arithmetic, capped at
  `audio_seq_length` (`gemma4_mm.py:349-369` `_compute_audio_num_tokens`,
  `:371-396` `get_audio_repl`). Encoder outputs are masked to real frames
  and scattered into `inputs_embeds` at those positions
  (`gemma4_mm.py:1468-1492` `_process_audio_input`).
- `mm_req_doc_ranges`: built in
  `vllm/v1/worker/gpu_model_runner.py` (the ~2298-2338 block at our base
  head); audio features are EXPLICITLY SKIPPED:
  `if mm_feature.modality == "audio": continue`
  (`gpu_model_runner.py:2311-2312` at 20196b5946). **Audio spans
  therefore produce NO bidirectional ranges, unlike images/videos.**
  Provenance: the skip was added upstream by the Gemma4 model author in
  PR #44429 (`a248b45d05`, Gemma4 Unified support) and reaffirmed in the
  PR #42175 refactor (`6deb05e0e4`) — a deliberate policy, not an
  accident of plumbing.

## Recon Q2: THE policy verdict

**AUDIO != VISION. Vision spans (image AND video) are bidirectional on
sliding layers; audio soft tokens are strictly causal on ALL layers.**

Authoritative sources (HF transformers 5.10.2 reference implementation):

- `transformers/models/gemma4/modeling_gemma4.py:2106-2115`
  (`get_block_sequence_ids_for_mask`):
  `is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)`;
  every other token (text=0, AUDIO=3) maps to block `-1` = "keep
  complete causality". Type-id assignment:
  `transformers/processing_utils.py:900-928`
  (`create_mm_token_type_ids`: image=1, video=2, audio=3).
- `transformers/models/gemma4/configuration_gemma4.py:89-92`:
  `use_bidirectional_attention='vision'` -> "vision tokens attend
  bidirectionally while text tokens use causal attention". Audio is
  never mentioned as bidirectional anywhere in config or modeling code.
- The audio TOWER is bidirectional internally (USM encoder:
  `modeling_gemma4.py:1988` `create_bidirectional_mask` with a chunked
  local window), but that is upstream of the LM; its soft tokens are
  decoded causally.
- vLLM upstream implements the same: the `gpu_model_runner.py` audio
  skip above; `decode`/`prefill` mask paths consume only
  `mm_req_doc_ranges`.

Consequence for our FlashInfer custom-mask path (7df3c67ec8 design):

- `FlashInferMetadataBuilder._mm_prefix_prefill_spans`
  (`vllm/v1/attention/backends/flashinfer.py:1891-1930`) reads ONLY
  `common_attn_metadata.mm_req_doc_ranges`, which excludes audio at the
  source. Audio-only mm requests classify all-plain -> the legacy
  scalar-causal path runs UNCHANGED (the byte-identical regression
  guarantee from MM_PREFIX_MASK_NOTES). Mixed audio+image requests carry
  only the image spans. The static 'vision' layer-group gate
  (`flashinfer.py:1603-1634`) needs NO audio case: it gates WHERE vision
  spans apply, and audio never produces a span.

**VERDICT: NO FIX NEEDED — the builder matches the audio truth by
construction.** Pinned statically so a regression at either level is
caught:

- Branch `spark/hijinks-e2-audio` @ `7e326fd037` (pushed to
  jethac/vllm; base 20196b5946):
  - `tests/v1/attention/test_mm_prefix_audio_policy.py` (NEW, 18 cells):
    range-source policy per modality (image/video ranged, audio NEVER —
    with/without window guard, mixed features, is_embed splits),
    FlashInfer builder span filter (audio-only -> None/legacy path,
    mixed batches, in-context + degenerate spans), routing invariance
    (audio_config presence does not change `is_mm_prefix_lm`).
  - `gpu_model_runner.py`: behavior-identical extraction of the
    range-building loop body into `mm_prefix_doc_ranges_for_request()`
    so the modality policy is unit-testable.
  - Static validation (WSL `~/e2_triton_retire_testenv`, CPU-only,
    PYTHONPATH shadowing verified): 18/18 new; selection truth-table
    71/71 unchanged at this head; py_compile clean.
  - MERGE NOTE: must land on e2-vllm alongside `spark/hijinks-e2-mm-retire`
    (the mm agent owns that merge gate; tests-only semantics, no
    conflict surface — the flip commit touches envs.py/config.py/
    flashinfer.py/selection tests, this branch touches
    gpu_model_runner.py + a new test file).

Footnote (VISION lane, not audio; for the record): the HF reference
applies the vision block overlay to ALL layer types
(`transformers/masking_utils.py:995-997` and `:1214-1216` — both
`create_causal_mask` and `create_sliding_window_causal_mask` accept
`block_sequence_ids`), while vLLM upstream restricts vision bidi to
sliding layers (PR #40534, `gemma4_mm.py:1618-1655` clearing hook) and
our builder mirrors vLLM. This discrepancy predates the campaign, is
vision-scoped, and does not affect the audio verdict (audio is causal in
BOTH implementations). Flagged for a possible upstream question; not
relitigated tonight.

## Recon Q3: --language-model-only and chunked-mm for audio

- `--language-model-only`: `ModelConfig.is_mm_prefix_lm` returns False
  (`vllm/config/model.py:1270-1282`) and every modality limit is 0
  (`vllm/config/multimodal.py:315-316`), so audio inputs are REJECTED
  and no ranges are built. Audio cells must serve WITHOUT
  `--language-model-only` (audio tower loads whenever the checkpoint has
  `audio_config`, `gemma4_mm.py:1054-1075`).
- Chunked mm: `cuda.py:259-270` forces `--disable_chunked_mm_input` for
  ALL mm-prefix models, modality-blind — audio placeholder spans never
  straddle a prefill-chunk boundary either. For audio this is a
  scheduling nicety, not a masking-correctness dependence (audio is
  causal); the harness passes the flag explicitly anyway.

## Assets (banked, results/p520_audio_mm_20260612/assets/)

| file | source | content | md5 |
| --- | --- | --- | --- |
| `speech_librispeech_1272-128104-0000.wav` | LibriSpeech (Panayotov et al. 2015, CC BY 4.0) dev-clean utterance 1272-128104-0000, via HF `hf-internal-testing/librispeech_asr_dummy` config=clean split=validation row=0 | 5.855 s, 16 kHz mono PCM16; reference transcript (from the dataset, verbatim in `speech_transcript.txt`): "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL" | `2317b0eb294e2363c818ecc4289b6ffb` |
| `tone_control.wav` | synthesized, no RNG (`wsl_sm120/make_audio_mm_assets.py`, copy in assets dir) | 4.0 s, 16 kHz mono PCM16: 1 s 440 Hz sine, 1 s silence, 1 s 880 Hz sine, 1 s silence, amplitude 0.3 | `ead0918e31771b4e1c6b4a9d688efbfb` |

Generator + manifest with md5s: `assets_manifest.json`.

## P520 cells (Amendment 5; LAST in tonight's GPU queue)

Harness: `wsl_sm120/audio_mm_smoke.py` + `wsl_sm120/run_audio_mm_cells.sh`
(pattern of the gemma3 serving rows + image smoke protocol; copies in
`results/p520_audio_mm_20260612/harness/`). Install: the Amendment-4
second install (`~/vllm_e2_env` + `~/vllm-e2` @ mm-retire head);
FlashInfer source tree @ 7d5d477b on PYTHONPATH. One server at a time,
claim only after 3 consecutive free `nvidia-smi` checks 2 min apart.

Per model (E2B-it, E4B-it), three rows:

| row | route | knobs |
| --- | --- | --- |
| `triton_bf16` | Triton comparator | `VLLM_FLASHINFER_MM_PREFIX=0`, `--attention-backend TRITON_ATTN` |
| `fi_bf16` | FlashInfer mm (flip default) | none (defaults at mm-retire head), `--attention-backend FLASHINFER` |
| `fi_nvfp4` | FlashInfer mm + NVFP4 KV | `--kv-cache-dtype nvfp4`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`, `VLLM_NVFP4_KV_VOSPLIT=1` |

No `triton_nvfp4` cell exists: Triton cannot read quantized KV
(scorecard I2); the nvfp4 comparator is `triton_bf16`, semantic gate.

Gates per row (zero-bug bar): speech smoke transcript-grounded
("quilter" keyword + banked verbatim), tone-control smoke banked
verbatim and sane (no hallucinated speech), text-only smoke coherent,
x2 repeats byte-identical per smoke, R5 proof lines (FI rows: zero
Triton attention dispatch; Triton row: zero FlashInfer attention
dispatch), FI-vs-Triton semantic equivalence adjudicated from the banked
transcripts. ANY RED: bank verbatim, name the row, the mm flip does not
merge (or reverts) — audio row named as the reason.

## Cell results

BLOCKED at 2026-06-12 02:26 JST (precise state, poll log in
`results/p520_audio_mm_20260612/poll_log_20260612.txt`):

- GPU occupied on all 3 protocol polls (02:21 / 02:23 / 02:26, ~9.26 GiB
  resident — small-size ladder server up); two queue positions (MTP
  identity ladder, image mm smokes) still ahead of the audio cells.
- The Amendment-4 e2 install was still building (`~/vllm-e2` editable
  compile in flight, `~/vllm_e2_build_20260612.log`).

Everything is staged for the next free window — runner:
`bash /mnt/b/workshop/wsl_sm120/run_audio_mm_cells.sh`
(port 8078, distinct from ladder 8000 / image smokes 8077; assets
copied + md5-logged from the banked set; per-row proof lines, speech +
tone + text smokes, x2 byte-identity gates; results auto-copied to
`results/p520_audio_mm_20260612/cells/`). The runner requires no code
checkout switch: the audio branch is tests-only, so the mm-retire head
in `~/vllm-e2` is the correct serving code as-is.

| cell | route | verdict |
| --- | --- | --- |
| e2b_triton_bf16 | Triton comparator | PENDING (blocked, see above) |
| e2b_fi_bf16 | FlashInfer mm default | PENDING |
| e2b_fi_nvfp4 | FlashInfer + NVFP4 KV | PENDING |
| e4b_triton_bf16 | Triton comparator | PENDING |
| e4b_fi_bf16 | FlashInfer mm default | PENDING |
| e4b_fi_nvfp4 | FlashInfer + NVFP4 KV | PENDING |

## SPARK 12B-audio cell spec (morning block)

After the mm-retire + audio branches merge to e2-vllm and the post-merge
image is built (TRITON_RETIREMENT_NOTES 6/8/9 build spec):

- Model: `google/gemma-4-12B-it` (audio_config present), r-image with
  e2-vllm post-merge head, `--attention-backend FLASHINFER`, util 0.72,
  NO `--language-model-only`, `--disable-chunked-mm-input` (auto-forced,
  pass explicitly).
- Rows: `fi_bf16` and `fi_nvfp4` (+ `VLLM_NVFP4_KV_LINEAR_V_SF=1
  VLLM_NVFP4_KV_VOSPLIT=1`; 12B globals are D512 — VO split required),
  paired with one `triton_bf16` comparator
  (`VLLM_FLASHINFER_MM_PREFIX=0`).
- Same assets (copy `results/p520_audio_mm_20260612/assets/` onto the
  Spark host; md5-verify against `assets_manifest.json` before the
  window), same smokes (`audio_mm_smoke.py`, speech expect "quilter",
  tone control verbatim, text smoke), same gates incl. x2 byte-identity
  and R5 proof lines.
- Marker protocol: WRITE-THEN-VERIFY; pre-flight
  `hf_model_access_probe.py google/gemma-4-12B-it` before claiming.
- Expected mechanics: audio-only requests must show NO "FlashInfer
  mm-prefix" custom-mask plan lines (audio produces no spans — that
  absence is itself a policy proof line to bank); image+audio mixed
  requests show mask plans driven by the image spans only.
