<!-- source: docs/AEON_PRIOR_ART_PORT_MAP.md -->

# AEON Prior-Art Port Map

Date: 2026-06-08 JST

Purpose: track what from `B:/workshop/CODEX_REPORT_AEON7.md` has been ported, what still needs a counterpart in SGLang or llama.cpp, and what should not be blindly copied across runtimes.

AEON-7's public work is primarily vLLM work. Treat it as high-value Spark evidence, not as proof that SGLang or llama.cpp are fixed.

Primary sources checked:

- `B:/workshop/CODEX_REPORT_AEON7.md`
- `https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash/tree/main/patches`
- `https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- `https://github.com/AEON-7/vllm-dflash`
- current forks: `jethac/vllm@db4b210c1`, `jethac/sglang@98ad46961`, `jethac/flashinfer@e152cf4d`

## Rules

- Direct vLLM patches stay in `jethac/vllm`; do not port them to SGLang or llama.cpp unless the same failure is reproduced there.
- SGLang and llama.cpp need counterpart experiments, not string-for-string vLLM patch copies.
- AEON Gemma proves NVFP4 weights plus ordinary KV plus Triton target attention plus DFlash, not FA2 NVFP4 KV.
- A runtime is not broadly blessed from a Gemma-only or Qwen-only row.

## Audit Command

Use the local audit to verify that the submodules still match this map:

```bash
python3 scripts/aeon_prior_art_audit.py \
  --output results/aeon_prior_art_audit_YYYYMMDD.json
```

Current validation artifact: `results/aeon_prior_art_audit_20260608.json`.

Use the counterpart evidence audit to keep source-marker checks separate from live proof:

```bash
python3 scripts/counterpart_evidence_audit.py \
  --output results/counterpart_evidence_audit_YYYYMMDD.json
```

Current validation artifact: `results/counterpart_evidence_audit_20260608.json`. It now accepts the AEON and derived `jethac/vllm` Qwen3.6+DFlash rows as claim evidence, while the SGLang and llama.cpp counterpart rows remain missing or partial. AEON source coverage is not the same thing as SGLang/llama.cpp serving evidence.

The command contracts for those seven rows live in `tasks/counterpart_evidence_tasks.jsonl` and are validated by:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260608.json \
  --output results/counterpart_task_matrix_20260608.json
```

Current validation artifact: `results/counterpart_task_matrix_20260608.json`.

## Literal AEON Qwen Patch Inventory

| AEON patch | purpose | current vLLM status | SGLang counterpart | llama.cpp counterpart |
|---|---|---|---|---|
| `register_qwen3_5_text.py` | register text-only `Qwen3_5*ForCausalLM` classes so Qwen3.5/3.6 does not fall into the multimodal path | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/model_executor/models/registry.py`, `tests/model_executor/test_qwen3_5_registry.py` | no direct port; SGLang has its own model registry and must be tested with Qwen3.6 | no direct port; Qwen3.6 GGUF conversion/loading is a separate experiment |
| `patch_cuda_optional_import.py` | make `_C_stable_libtorch` import lazy when unused SM100 MXFP4 symbols are missing on SM12x | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/platforms/cuda.py` | reject direct port; SGLang does not import vLLM's extension | reject direct port |
| `patch_kv_cache_utils.py` | avoid `block_size=None` crashes for hybrid linear-attention/Mamba groups | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/v1/engine/core.py`, `third_party/vllm/vllm/v1/worker/gpu_model_runner.py`, `third_party/vllm/vllm/model_executor/layers/mamba/abstract.py` | candidate only if Qwen3.6/SGLang reproduces a `block_size=None` failure; current SGLang already has hybrid/spec paths to audit | reject direct port |
| `patch_mrope_text_fallback.py` | text-only M-RoPE fallback for Qwen3.6 when the model class lacks `SupportsMRoPE` | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/v1/worker/gpu_model_runner.py` | candidate only if SGLang Qwen3.6 text-only load shows an M-RoPE mismatch | no direct port; GGUF RoPE metadata must be verified instead |
| `patch_cudagraph_align.py` | align speculative-decode CUDA graph capture sizes for non-`NONE` graph modes, not only `FULL` | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/config/compilation.py`, `tests/compile/test_config.py` | do not copy literally; SGLang has its own speculative graph machinery and must be tested with DFlash/EAGLE on GB10 | no direct port; llama.cpp graph/spec-decode behavior is separate |
| `strip_language_model_prefix.py` | checkpoint-conversion helper for one Qwen source model layout | not a vLLM source patch; current AEON v2 weights do not require this as a fork change | maybe useful as a one-off conversion check if SGLang loads a text-only Qwen checkpoint with multimodal prefixes | no direct source patch; any GGUF converter must handle names during conversion |

## AEON Gemma Lessons

| lesson | vLLM status | SGLang action | llama.cpp action |
|---|---|---|---|
| NVFP4 weights are the proven Gemma speed lever | locally reproduced through AEON image; about `48-54 tok/s` short/medium decode and `98 tok/s` long-prefill | test Gemma 4 NVFP4 weights with ordinary KV before FP4 KV; current SGLang Gemma path is not blessed | current GGUF Q4_0 path is practical; native NVFP4/MXFP4 GGUF remains a separate experiment |
| Gemma 4 target attention is Triton because local/global head dimensions differ | documented in vLLM recipe and local logs | do not make Gemma FP4 KV the first SGLang goal; prove ordinary KV serving first | no FA2 KV implication |
| Routers, vision tower, and vision embeddings must not be blindly NVFP4-quantized | use AEON weights unless re-quantizing; run `scripts/nvfp4_checkpoint_audit.py` before treating a checkpoint as safe | if building SGLang-loadable Gemma NVFP4 checkpoints, preserve BF16 routers and vision tensors and attach the checkpoint audit artifact | if making GGUFs from AEON or our own quantization, record whether routers/vision stayed high precision before conversion |
| EOS/control-token cleanup matters | AEON weights/config sidestep this for the reproduced row | include deterministic output sanity in Gemma SGLang row | include chat smoke and template checks for Gemma GGUF |

## DFlash And Speculative Decode

AEON's DFlash result is the main single-stream multiplier, but the port story is runtime-specific.

| runtime | current evidence | decision |
|---|---|---|
| vLLM | AEON Gemma DFlash, AEON Qwen36 DFlash, and derived `jethac/vllm` Qwen36 DFlash rows are locally reproduced; Qwen36 requires `chat_template_kwargs={"enable_thinking": false}` for normal OpenAI content output | replace AEON-binary dependencies with clean fork packaging and collect native-target evidence before deciding what belongs upstream |
| SGLang | current `jethac/sglang@98ad46961` tree contains DFlash surfaces including `python/sglang/srt/arg_groups/speculative_hook.py`, `python/sglang/srt/models/dflash.py`, Qwen model `set_dflash_layers_to_capture` hooks, and metrics for accepted drafts | candidate, not proven; add a SGLang Qwen/Gemma DFlash smoke only after ordinary Qwen/Gemma serving is stable |
| llama.cpp | no `third_party/llama.cpp` submodule exists because no llama.cpp code change has been needed yet; current proof is practical GGUF serving and native loglikelihood harness work | do not port DFlash literally; evaluate llama.cpp-native speculative/draft-model support only when a GGUF drafter/model pair exists |

## Still Needed Counterparts

1. SGLang Gemma NVFP4-weight serving with ordinary KV.
2. SGLang FP4 KV after-row with quality checks; the autosafe source-overlay row proves `1.779x` fp8 capacity but still fails standardized output quality.
3. SGLang DFlash or EAGLE row on a Qwen-class model if ordinary serving is stable.
4. Clean `jethac/vllm` packaging for Qwen3.6 NVFP4+DFlash using the precompiled FA2/FA3 skip knob, followed by in-container native-target/JIT audit.
5. llama.cpp larger Qwen3/Qwen3.6 GGUF rows.
6. llama.cpp native NVFP4/MXFP4 GGUF tensor-core proof, separate from Q4_0/Q4_K serving.
7. llama.cpp live native loglikelihood task proof before paper-comparable GGUF accuracy claims.

These rows are mechanically tracked by `scripts/counterpart_evidence_audit.py`; the vLLM row is claim-ready for serving evidence, while clean packaging and native-target proof remain tracked in the runtime docs.

## Explicit Non-Ports

Do not spend time porting these directly:

- vLLM `_C_stable_libtorch` lazy import into SGLang or llama.cpp.
- vLLM registry changes into SGLang or llama.cpp.
- vLLM M-RoPE fallback into llama.cpp.
- vLLM CUDA graph capture-size code into SGLang without a reproduced SGLang graph failure.
- AEON Gemma FA2 NVFP4-KV claims; AEON Gemma is not an FA2 NVFP4-KV result.
- vLLM env knobs such as `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4`, and `VLLM_TEST_FORCE_FP8_MARLIN` as if they were cross-runtime fixes.

## Next Proof Order

1. Build a clean `jethac/vllm` Qwen36 image with `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1`, replace the AEON FA2 binary dependency with an ABI-matched FA2 build, then rerun the no-think serving row and in-container binary/JIT target audit.
2. Run `scripts/nvfp4_checkpoint_audit.py` on any NVFP4 Qwen/Gemma checkpoint before using it as speed evidence or conversion input.
3. Run the Qwen speed lane for vLLM/SGLang/llama.cpp with `scripts/qwen_speed_lane.py`.
4. Attempt SGLang Gemma NVFP4-weight serving with ordinary KV.
5. Only then decide whether a SGLang DFlash or FP4-KV code change is justified.
6. Create `third_party/llama.cpp` only when a real llama.cpp source change is required.

<!-- source: docs/AUDIO_MM_NOTES.md -->

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

<!-- source: docs/BASELINE_RESULTS.md -->

# Baseline Results

Status: first compact before row.

This file indexes before/after benchmark artifacts. Raw JSON remains the source of truth.

## 2026-06-08: AEON vLLM Gemma 4 26B NVFP4+DFlash Row

Target:

- image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`
- vLLM: `0.20.1`
- PyTorch: `2.11.0+cu130`
- FlashInfer: `0.6.8.post1`

Artifacts:

- summary: `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`
- chat smoke: `results/aeon_gemma26_dflash_20260608T0436JST_chat_smoke.json`
- first compact benchmark: `results/aeon_gemma26_dflash_20260608T0436JST_openai_benchmark.json`
- warmed compact benchmark: `results/aeon_gemma26_dflash_20260608T0436JST_warm2_openai_benchmark.json`
- server log: `results/aeon_gemma26_dflash_20260608T0436JST_server.log`
- key log lines: `results/aeon_gemma26_dflash_20260608T0436JST_key_log_lines.txt`
- container versions: `results/aeon_gemma26_dflash_20260608T0436JST_container_versions.json`
- `spark_doctor`: `results/spark_doctor_aeon_gemma26_dflash_20260608T0436JST.md`

Warmed result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.098 | 1.434 | 47.91 |
| `medium_decode` | 40 | 192 | 0.087 | 3.669 | 53.60 |
| `long_prefill` | 2270 | 64 | 0.118 | 0.768 | 98.38 |

Interpretation:

- This is the first local vLLM Gemma 4 26B row that materially beats the earlier BF16/unquantized vLLM row at about 24 tok/s.
- The server log proves `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, target `TRITON_ATTN`, drafter `FLASH_ATTN`, CUDA graph capture, and DFlash model load.
- This is not proof that the local `jethac/vllm` or `jethac/flashinfer` forks improved throughput; the run used AEON's container and checkpoint.
- The container reports device capability `[12, 1]`, but its PyTorch arch list contains `sm_120` and not explicit `sm_121`; keep this as packaging evidence to inspect.
- The server log warns about differing NVFP4 global scales across fused parallel layers, so accuracy still needs a separate check.

## 2026-06-07: vLLM Gemma 4 E4B W4A16 Before Row

Server observed on `thinkstationpgx-00b4`:

```text
/usr/local/bin/vllm serve google/gemma-4-E4B-it-qat-w4a16-ct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --host 0.0.0.0 \
  --port 8000
```

Artifacts:

- environment: `results/spark_doctor_before_vllm_gemma4_e4b_w4a16_20260607T1126Z.md`
- runtime probe: `results/runtime_probe_vllm_gemma4_e4b_w4a16_root_20260607T1136Z.json`
- benchmark: `results/vllm_gemma4_e4b_w4a16_before_compact_20260607T1126Z.json`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 24 | 64 | 0.032 | 1.271 | 51.65 |
| `medium_decode` | 36 | 192 | 0.032 | 3.779 | 51.25 |
| `long_prefill` | 2266 | 64 | 0.470 | 1.753 | 49.89 |

Interpretation:

- This is a valid first before row for an already-running vLLM server.
- It is not a cold-start benchmark.
- Runtime probe evidence shows the server was running as root from `/vllm-workspace` with `VLLM_USAGE_SOURCE=production-docker-image`.
- Runtime probe evidence shows `TORCH_CUDA_ARCH_LIST=8.7 8.9 9.0 10.0+PTX 12.0 12.1`.
- Runtime probe evidence found loaded vLLM extension paths including `_C.abi3.so`, `_C_stable_libtorch.abi3.so`, `_moe_C.abi3.so`, `_vllm_fa2_C.abi3.so`, and `_vllm_fa3_C.abi3.so`.
- It is not a full blessed-stack result because exact per-request kernel selection is still under investigation.
- It is useful for future before/after comparisons against the same model, prompts, settings, and server API.

## 2026-06-07: SGLang 26.05 Container Exploratory Row

Server tested on `thinkstationpgx-00b4`:

```text
nvcr.io/nvidia/sglang:26.05-py3
python3 -m sglang.launch_server Qwen/Qwen2.5-1.5B-Instruct
```

Key settings:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- dtype: `bfloat16`
- KV cache dtype: `torch.bfloat16`
- attention backend: `flashinfer`
- CUDA graphs: enabled
- port: `30000`
- vLLM on port `8000` was left running

Artifacts:

- smoke: `results/sglang_20260607T115213Z_chat_smoke.json`
- versions: `results/sglang_20260607T115213Z_python_versions.txt`
- CUDA object audit: `results/sglang_20260607T115213Z_cuda_so_audit_sglang.json`
- benchmark, 0.20 memory fraction: `results/sglang_bench_20260607T120315Z_openai_benchmark.json`
- long-prefill retry, 0.40 memory fraction: `results/sglang_bench_longprefill_20260607T120614Z_openai_benchmark.json`

Result summary:

| case | memory fraction | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `short_decode` | 0.20 | 44 | 64 | 0.647 | 1.705 | 60.50 | pass |
| `medium_decode` | 0.20 | 56 | 192 | 0.036 | 3.218 | 60.34 | pass |
| `long_prefill` | 0.20 | 1 | 1 | n/a | 0.035 | n/a | fail; insufficient/incorrect token budget |
| `long_prefill` | 0.40 | 2369 | 64 | 0.683 | 1.763 | 59.23 | pass |

Interpretation:

- NVIDIA's SGLang 26.05 ARM64 container can serve an OpenAI-compatible request on GB10.
- The container is a better first SGLang path than bare-metal pip.
- The first long-prefill failure is a real tuning/evidence point: `mem_fraction_static=0.20` can produce a too-small effective token budget for the benchmark while coexisting with the live vLLM service.
- The passing long-prefill retry used `mem_fraction_static=0.40`.
- This row is exploratory because it uses Qwen rather than Gemma and because sm121-specific kernel dispatch remains unresolved.
- The CUDA object audit found no explicit `sm_121` SASS in audited SGLang/FlashInfer objects.
- The SGLang log labeled the GB10 path as `SM120 (Blackwell)`, so this still needs upstream dispatch/packaging scrutiny.

## 2026-06-08: SGLang Qwen2.5 1.5B fp8-vs-fp4 KV Probe

Target:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- dtype: `bfloat16`
- memory fraction: `0.40`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`

Artifacts:

- summary: `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`
- BF16/auto benchmark: `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_openai_benchmark.json`
- BF16/auto server log: `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_server.log`
- fp8 smoke: `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_chat_smoke.json`
- fp8 benchmark: `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_openai_benchmark.json`
- fp8 server log: `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_server.log`
- fp4 FlashInfer startup failure: `results/sglang_qwen25_1_5b_fp4kv_20260608T0336JST_startup.log`
- fp4 Triton startup failure: `results/sglang_qwen25_1_5b_fp4kv_triton_20260608T0338JST_startup.log`
- patched fp4 FlashInfer startup failure: `results/sglang_qwen25_1_5b_fp4kv_patched_flashinfer_20260608T0349JST_startup.log`
- patched fp4 Triton no-graphs benchmark: `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nographs_20260608T0404JST_openai_benchmark.json`

BF16/auto result summary:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 44 | 64 | 0.042 | 58.89 |
| `medium_decode` | 56 | 192 | 0.035 | 58.59 |
| `long_prefill` | 2369 | 64 | 0.136 | 57.73 |

fp8 result summary:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 44 | 64 | 0.043 | 59.09 |
| `medium_decode` | 56 | 192 | 0.035 | 58.43 |
| `long_prefill` | 2369 | 64 | 0.036 | 58.22 |

Interpretation:

- fp8 KV is now the concrete SGLang Qwen comparator row for issue #20.
- BF16/auto KV is the matched memory-fraction comparator row: it allocated `1,557,709` KV tokens and ran at roughly the same decode speed as fp8.
- fp8 KV selected FlashInfer attention, enabled CUDA graphs, and allocated a `3,113,713` token KV pool.
- stock `fp4_e2m1` with FlashInfer attention failed at SGLang's `KV4Compatibility` gate.
- stock `fp4_e2m1` with Triton attention allocated a larger `5,534,509` token KV pool, about `1.78x` the fp8 row, then failed on missing `KVFP4QuantizeUtil`.
- patched `jethac/sglang@98ad46961` overlay cleared those SGLang blockers. FlashInfer attention then failed inside FlashInfer FP4 E2M1 decode JIT, while Triton attention served only with both CUDA graph modes disabled.
- The patched no-graphs Triton FP4 KV row allocated `5,541,103` KV tokens but decoded at only `0.276 tok/s` on `short_decode` with visibly repetitive output. This is a capacity/debug proof, not a blessed FP4 KV serving result.

## 2026-06-08: SGLang Qwen FP4 KV Autosafe Capacity Row

Artifacts:

- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_row_manifest.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_row_manifest.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_raw_2plus2.json`

Result:

- fp8 comparator: `3,101,822` KV tokens, no-graph policy matched to FP4, decode `56.73`, `56.81`, and `57.10 tok/s`, raw `2+2` returns `4`.
- FP4 KV: `5,519,481` KV tokens, auto-safe no-graph policy, `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`, `1.779x` fp8 capacity.
- FP4 quality fails: raw `2+2` is malformed and benchmark text degenerates.

Interpretation:

- This proves the FP4 KV capacity path in the clean SGLang source-overlay stack.
- It does not prove SGLang FP4 KV serving quality, graph safety, or speed. Keep the counterpart row partial until quality passes.

## 2026-06-08: vLLM Qwen NVFP4-KV Capacity Row

Target:

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- runtime ref: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d`
- model: AEON Qwen3.6 35B-A3B NVFP4 weights, no DFlash
- attention backend argument: `flashinfer`
- max model length: `262144`
- memory utilization: `0.85`

Artifacts:

- summary: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`
- fp8 comparator manifest: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_row_manifest.json`
- fp8 comparator benchmark: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_openai_benchmark.json`
- fp8 comparator server log: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_server.log`
- NVFP4-KV manifest: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_row_manifest.json`
- NVFP4-KV benchmark: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_openai_benchmark.json`
- NVFP4-KV server log: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_server.log`

Result:

- fp8 KV: `6,364,935` KV tokens, `24.28x` max concurrency at 262k context.
- NVFP4 KV: `11,146,226` KV tokens, `42.52x` max concurrency at 262k context.
- capacity/concurrency ratio: `1.751x`.
- decode: fp8 `43.001`, `42.512`, `42.684 tok/s`; NVFP4 KV `43.014`, `42.615`, `42.898 tok/s`.
- backend proof: the NVFP4 server log says `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`

Interpretation:

- This is the first end-to-end vLLM NVFP4-KV serving capacity proof on the Qwen lane.
- It proves capacity/concurrency and normal content with Qwen thinking disabled; it does not prove a decode-speed uplift.
- It does not prove Gemma NVFP4-KV. Gemma remains blocked by heterogeneous attention dimensions and the FlashInfer FA2 global `D=512` failure.
- It does not prove native FP4 weight/MoE MMA. The server still selects `MARLIN` NvFp4 MoE for weights.

## 2026-06-07: SGLang Gemma 4 E2B Blocker

Target:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- model: `google/gemma-4-E2B-it-qat-w4a16-ct`
- dtype: `bfloat16`
- memory fraction: `0.40`

Artifacts:

- default launch: `results/sglang_gemma4_e2b_w4a16_20260607T121536Z_server.log`
- language-only retry: `results/sglang_gemma4_e2b_w4a16_language_only_20260607T121751Z_server.log`

Result:

- Default launch exited before health while constructing the Gemma4 audio tower.
- The concrete exception was `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`.
- A retry with `--language-only` exited before health during server argument validation.
- The concrete retry exception was `ValueError: requires at least one encoder urls to be set via --encoder-urls`.

Interpretation:

- This is a SGLang Gemma4 model-path blocker, not a proof of a GB10 kernel failure.
- It keeps SGLang marked as functional for some supported models but not yet a Gemma-class blessed path.

## 2026-06-07: FlashInfer SM121 Source/JIT Validation

Target:

- fork: `jethac/flashinfer`
- branch: `spark/hijinks-004-sm121-flashinfer`
- commit: `a42c8f07`
- host: `thinkstationpgx-00b4`

Artifact:

- `results/flashinfer_sm121_source_jit_20260607T1250Z.json`

Result:

- Installed vLLM container baseline: FlashInfer `0.6.8.post1`, CUDA `13.0`, real SM121 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.
- Installed SGLang container baseline: FlashInfer `0.6.10+cf494fca.nv26.5.cu132.50619265`, CUDA `13.2`, real SM121 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.
- Patched source: real SM121 NVFP4 `mm_fp4` heuristic returned `["b12x", "cutlass", "cudnn"]`.
- Source/JIT path built FP4 quantization under `/root/.cache/flashinfer/0.6.13/121a/cached_ops/fp4_quantization_120f`.
- Tiny forced-`b12x` NVFP4 GEMM produced finite BF16 output with cosine similarity `0.9882067441940308` against BF16 `torch.mm`.

Interpretation:

- This proves the FlashInfer fork makes the high-impact SM121 dispatch behavior better and that the `b12x` path can execute on GB10 when the source/JIT package set is consistent.
- This does not yet prove a serving-speed improvement. The deployable before/after needs a clean vLLM or SGLang image/wheel set with matching FlashInfer Python, JIT-cache/cubin, CUTLASS DSL, and CUDA targets.

## 2026-06-07: FlashInfer NVFP4 `mm_fp4` Auto Microbenchmark

Artifact:

- `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`

Script:

- `scripts/flashinfer_mm_fp4_microbench.py`

Result summary:

| case | installed auto heuristic | installed mean ms | patched auto heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x128x128` | `cudnn`, `cutlass` | 0.0727 | `b12x`, `cutlass`, `cudnn` | 0.0769 | +5.9% |
| `16x256x256` | `cudnn`, `cutlass` | 0.0654 | `b12x`, `cutlass`, `cudnn` | 0.0661 | +1.0% |
| `64x512x512` | `cudnn`, `cutlass` | 0.0651 | `b12x`, `cutlass`, `cudnn` | 0.0757 | +16.3% |

Interpretation:

- The patch did what it was supposed to do at dispatch level: SM121 auto-dispatch now includes `b12x`.
- On these three small dense NVFP4 `mm_fp4` cases, `b12x` was not faster than the installed container path.
- This does not rule out wins for model-shaped GEMMs, MoE `b12x` kernels, underfilled decode paths, or full serving stacks.
- Do not claim an end-to-end throughput improvement from this FlashInfer patch until a clean image/wheel set and model-level before/after rows prove it.

## 2026-06-07: FlashInfer NVFP4 `mm_fp4` Model-Shaped Proxy Microbenchmarks

Artifacts:

- installed dense-decode proxy: `results/flashinfer_mm_fp4_sglang_installed_dense_decode_20260607T161500Z.json`
- installed MoE proxy: `results/flashinfer_mm_fp4_sglang_installed_moe_expert_20260607T161500Z.json`
- patched dense-decode proxy: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_dense_decode.json`
- patched MoE proxy: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_moe_expert.json`

Target:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- installed FlashInfer: `0.6.10+cf494fca.nv26.05.cu132.50619265`
- patched source: `jethac/flashinfer@a42c8f07`
- patched source/JIT FlashInfer version: `0.6.13`
- GPU: `NVIDIA GB10`, compute capability `12.1`

Result summary, dense-decode proxy:

| case | installed heuristic | installed mean ms | patched heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x4096x4096` | `cudnn`, `cutlass` | 0.0738 | `b12x`, `cutlass`, `cudnn` | 0.0893 | +21.1% |
| `4x4096x4096` | `cudnn`, `cutlass` | 0.0704 | `b12x`, `cutlass`, `cudnn` | 0.0677 | -3.9% |
| `16x4096x4096` | `cudnn`, `cutlass` | 0.0692 | `b12x`, `cutlass`, `cudnn` | 0.0620 | -10.4% |
| `1x8192x4096` | `cudnn`, `cutlass` | 0.0707 | `b12x`, `cutlass`, `cudnn` | 0.0857 | +21.3% |
| `4x8192x4096` | `cudnn`, `cutlass` | 0.0700 | `b12x`, `cutlass`, `cudnn` | 0.0786 | +12.4% |
| `16x8192x4096` | `cudnn`, `cutlass` | 0.0709 | `b12x`, `cutlass`, `cudnn` | 0.0741 | +4.5% |

Result summary, MoE-shaped proxy:

| case | installed heuristic | installed mean ms | patched heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x14336x4096` | `cudnn`, `cutlass` | 0.1443 | `b12x`, `cutlass`, `cudnn` | 0.1543 | +6.9% |
| `4x14336x4096` | `cudnn`, `cutlass` | 0.1382 | `b12x`, `cutlass`, `cudnn` | 0.1510 | +9.3% |
| `16x14336x4096` | `cudnn`, `cutlass` | 0.1413 | `b12x`, `cutlass`, `cudnn` | 0.1535 | +8.6% |
| `1x4096x14336` | `cudnn`, `cutlass` | 0.1401 | `b12x`, `cutlass`, `cudnn` | 0.1688 | +20.5% |
| `4x4096x14336` | `cudnn`, `cutlass` | 0.1397 | `b12x`, `cutlass`, `cudnn` | 0.1551 | +11.0% |
| `16x4096x14336` | `cudnn`, `cutlass` | 0.1390 | `b12x`, `cutlass`, `cudnn` | 0.1546 | +11.2% |

Interpretation:

- The patched source/JIT path selected `b12x` on real GB10 and produced finite outputs with cosine similarity around `0.991` against BF16 `torch.mm`.
- The patched source/JIT container compiled FlashInfer FP4 GEMM under an SM121a-targeted path during this run.
- The model-shaped proxy result is not a speedup. Dense-decode proxies were mixed, and all MoE-shaped proxy cases were slower than the installed SGLang container path.
- This makes the FlashInfer predicate patch a correctness/enablement fix. The remaining performance question must be answered in fused serving paths, NVFP4 KV, model-specific quantization plumbing, CUDA graph behavior, or clean package builds.

## 2026-06-07: vLLM SM12x NVFP4 KV Routing And Deswizzle Probe

Artifact:

- `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`

Environment:

- host: `thinkstationpgx-00b4`
- GPU: `NVIDIA GB10`
- Torch CUDA capability: `[12, 1]`
- vLLM platform capability: `[12, 1]`
- vLLM capability-family check: `is_capability_family_120: true`
- Torch: `2.11.0+cu130`
- CUDA: `13.0`
- installed vLLM dependency context: `0.22.1`
- fork source revision: `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24`

Result:

- SM12x NVFP4 KV prefill wrapper backend: `fa2`
- SM12x NVFP4 KV decode wrapper backend: `fa2`
- SM100-style NVFP4 fallback case remains `trtllm-gen`
- non-NVFP4 case remains `auto`
- vLLM FlashInfer JIT flag helper enables `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- probe result: `all_ok: true`

Interpretation:

- This proves the vLLM fork's routing predicate and vLLM-specific V-scale-factor deswizzle flag helper behave as intended on real GB10/SM121.
- `family 120` is intentional here: FA2 NVFP4 KV is the SM12x consumer-Blackwell family path. This is different from native FP4 MMA work, where `sm_121a` remains required for Spark.
- This does not prove the full vLLM fork installs cleanly, that FlashInfer FA2 NVFP4 KV kernels build/run, or that serving correctness/capacity/performance improves.
- Remaining proof must use the layout/correctness harness, including NHD and HND cosine checks, plus an end-to-end serve.

## 2026-06-08: FlashInfer FA2 NVFP4 KV Runtime Correctness Probe

Artifact:

- `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json`

Environment:

- source: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- import path: `/root/spark-validation/flashinfer-fa2-nvfp4-kv-sm121/flashinfer/__init__.py`
- source root supplied to JIT: `/root/spark-validation/flashinfer-fa2-nvfp4-kv-sm121`
- env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`
- Torch: `2.11.0+cu130`
- CUDA: `13.0`

Shape:

- `batch_size=2`
- `kv_len=64`
- `qo_len=16`
- `page_size=16`
- `num_kv_heads=2`
- `num_qo_heads=4`
- `head_dim=128`
- `dtype=bfloat16`
- backend: FlashInfer FA2 paged KV

Result:

| operation | layout | cosine | max abs error | passed |
|---|---:|---:|---:|---|
| decode | NHD | 0.9999995232 | 0.0078125 | true |
| prefill | NHD | 0.9999998808 | 0.0 | true |
| decode | HND | 0.9999994636 | 0.0078125 | true |
| prefill | HND | 0.9999998808 | 0.0 | true |

Interpretation:

- This is the first GB10 runtime proof that the patched FlashInfer FA2 NVFP4 paged-KV path builds, runs, and reads vLLM-style swizzled V scale factors correctly when the in-kernel de-swizzle macro is enabled.
- It is stronger than the vLLM routing probe because it executes FlashInfer kernels and compares against dequantized reference attention for both NHD and HND layouts.
- It is still not an end-to-end vLLM serving proof. It does not prove clean wheel packaging, vLLM metadata integration, fp8-vs-NVFP4 KV capacity, output quality, CUDA graph replay, or serving throughput.

## 2026-06-08: Gemma 4 26B-Shaped FlashInfer NVFP4 KV Probe

Artifacts:

- `results/flashinfer_nvfp4_kv_probe_gemma4_26b_sliding_1024_20260608T0340JST.json`
- `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json`

Config source:

- cached `google/gemma-4-26B-A4B-it` config under the benchmark host's Hugging Face cache
- text attention heads: `num_attention_heads=16`
- sliding/local KV heads: `num_key_value_heads=8`
- global/full KV heads: `num_global_key_value_heads=2`
- sliding/local `head_dim=256`
- global/full `global_head_dim=512`
- page size tested: `16`

Sliding/local result:

- shape: `batch_size=2`, `kv_len=1024`, `qo_len=128`, `num_qo_heads=16`, `num_kv_heads=8`, `head_dim=256`
- outcome: NHD decode, NHD prefill, HND decode, and HND prefill all passed.
- minimum cosine: `0.9999961853`
- maximum absolute error: `0.015625`

Global/full result:

- shape: `batch_size=2`, `kv_len=128`, `qo_len=16`, `num_qo_heads=16`, `num_kv_heads=2`, `head_dim=512`
- outcome: all NHD/HND decode/prefill operations failed before numerical comparison.
- failure class: FlashInfer FA2 paged KV invalid configuration from `include/flashinfer/attention/prefill.cuh:3215`
- representative message: `Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 ...`

Interpretation:

- This narrows the vLLM NVFP4-KV blocker. The patched FlashInfer FA2 path is correct for Gemma 4 26B sliding/local attention geometry, including vLLM-style V-scale-factor de-swizzle.
- Gemma 4 26B also has global/full attention layers with `global_head_dim=512`; that geometry currently fails in the standalone FlashInfer probe.
- Do not start or bless a Gemma 4 26B vLLM `--kv-cache-dtype nvfp4` serving row until the `D=512` global path is fixed, routed to a proven fallback, or shown irrelevant for the specific model path.

## 2026-06-07: vLLM Gemma 4 26B A4B Compact MoE Serving Check

Target:

- image: `vllm/vllm-openai:latest-cu130`
- vLLM: `0.20.0`
- PyTorch: `2.11.0+cu130`
- FlashInfer: `0.6.8.post1`
- model: `google/gemma-4-26B-A4B-it`
- served model: `gemma4-26b-a4b-it`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- benchmark: `results/vllm_gemma4_26b_a4b_bf16_compact_20260607T131917Z.json`
- server log: `results/vllm_gemma4_26b_a4b_bf16_20260607T131917Z_server.log`
- run info: `results/vllm_gemma4_26b_a4b_bf16_20260607T131917Z_run_info.txt`
- default-setting failure: `results/vllm_gemma4_26b_a4b_bf16_default_fail_20260607T131837Z_server.log`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 1.228 | 3.832 | 24.58 |
| `medium_decode` | 40 | 192 | 0.137 | 8.039 | 24.30 |
| `long_prefill` | 2270 | 64 | 0.551 | 3.223 | 23.95 |

Startup observations:

- The default launch failed before readiness because Gemma 4 disabled chunked multimodal input while vLLM's default `max_num_batched_tokens=2048` was below Gemma's `max_tokens_per_mm_item=2496`.
- Retrying with `--max-num-batched-tokens 4096` reached readiness.
- BF16 checkpoint loading took about 6m11s for 48.07 GiB of safetensors; CUDA graph capture took another 11s.
- The server log selected `TRITON_ATTN` for attention and `TRITON Unquantized MoE` from `['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']`.

Interpretation:

- This is a useful compact MoE serving row: Gemma 4 26B A4B serves successfully on GB10 through vLLM and sustains about 24 tok/s decode on this three-case OpenAI-compatible harness.
- This row does not exercise the FlashInfer NVFP4 `mm_fp4` dispatch fix. The vLLM path is BF16/unquantized MoE and explicitly chose Triton.
- The `--max-num-batched-tokens 4096` requirement should be part of future Gemma 4 vLLM recipes for this image/model combination.

## 2026-06-07: vLLM Gemma 4 12B Unified Source/Precompiled Probe

Target:

- base image: `vllm/vllm-openai:latest-cu130`
- vLLM source commit: `da1daf40bf18e5eaae04f26a80a537c8168a8bc2`
- install mode: editable source install using `VLLM_USE_PRECOMPILED=1`, `VLLM_MAIN_CUDA_VERSION=13.0`, and matching precompiled wheel metadata
- Transformers: main snapshot installed from `git+https://github.com/huggingface/transformers.git@effde20942e3f82a1b97449f60b3a48c5ff96145`
- model: `google/gemma-4-12B-it`
- served model: `gemma4-12b-it`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- launcher: `scripts/run_vllm_gemma4_12b_unified_probe.sh`
- import probe: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_import_probe.txt`
- chat smoke: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_openai_chat_smoke.json`
- compact benchmark: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_compact_benchmark.json`
- runtime probe: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_runtime_probe.json`
- server log: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_server.log`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.268 | 8.448 | 7.82 |
| `medium_decode` | 40 | 192 | 0.261 | 25.318 | 7.66 |
| `long_prefill` | 2270 | 64 | 0.976 | 9.252 | 7.73 |

Startup observations:

- The import probe recorded GB10 compute capability `[12, 1]`, PyTorch `2.11.0+cu130`, vLLM `0.1.dev1+gda1daf40b`, Transformers `5.10.0.dev0`, and `has_gemma4_unified: true`.
- Released/current local images checked before this run did not expose `Gemma4UnifiedForConditionalGeneration` through the registry: `vllm/vllm-openai:latest-cu130`, `vllm/vllm-openai:cu130-nightly-aarch64`, and the earlier `gemma4-vllm:v0.22.1-pip` image.
- Upstream vLLM main contains the architecture, and wheel metadata existed for `da1daf40`; the current main commit tested during this audit did not have matching precompiled wheel metadata yet.
- Stale `flashinfer-jit-cache` from the base image had to be removed. Without cleanup, FlashInfer reported a JIT-cache package version mismatch; after `pip uninstall` plus deleting the remaining package directory and dist-info, the server reached health.
- The server resolved `Gemma4UnifiedForConditionalGeneration`, loaded 22.28 GiB of checkpoints, and created a 336,566-token GPU KV cache for 8,192-token requests.
- vLLM forced `TRITON_ATTN` because Gemma 4 has heterogeneous head dimensions. The log also reported Triton JIT compilation during inference for `_compute_slot_mapping_kernel` and `kernel_unified_attention`.

Interpretation:

- This overturns the older local result that 12B was simply not usable in vLLM on Spark: the architecture path can run on GB10 when vLLM and Transformers are new enough and the FlashInfer JIT-cache package set is consistent.
- This is not a blessed clean stack. It required source overlay, a specific precompiled-wheel commit, Transformers main, and manual stale-package cleanup inside the container.
- The measured decode speed is much slower than the 26B A4B vLLM row and the llama.cpp 26B Q4_0 row. Treat it as a compatibility proof and a packaging target, not a performance win.
- The next proof is a clean release/nightly container that starts the same model without source surgery, then a quantized/MTP row that explains whether the SM120 reference results transfer to GB10 `sm_121`.

## 2026-06-07: vLLM Gemma 4 26B A4B QAT-Unquantized Probe

Target:

- image: `vllm/vllm-openai:latest-cu130`
- model: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
- served model: `gemma4-26b-a4b-it-qat-q4_0-unquantized`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- short benchmark: `results/vllm_gemma4_26b_a4b_qat_unquantized_short_20260607T133040Z.json`
- server log: `results/vllm_gemma4_26b_a4b_qat_unquantized_20260607T133040Z_server.log`
- run info: `results/vllm_gemma4_26b_a4b_qat_unquantized_20260607T133040Z_run_info.txt`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 1.248 | 3.857 | 24.53 |

Interpretation:

- The QAT-unquantized snapshot loads and serves in vLLM with the same corrected batching setting.
- It is not a direct quantized/NVFP4 serving row in this image. The engine config reported `quantization=None`, `dtype=torch.bfloat16`, and the same `TRITON Unquantized MoE` backend.
- If the campaign needs to prove NVFP4 end-to-end impact, this snapshot name is not enough. The run must prove the actual quantization/backend path from logs or profiler evidence.

## 2026-06-07: llama.cpp Gemma 4 26B Q4_0 Practical Serving Row

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- build: `308f61c31 (9536)`
- model: `/home/jethac/gemma4-vllm/models/gemma-4-26B_q4_0-it.gguf`
- alias: `gemma4-26b-q4_0-gguf`
- settings: `--ctx-size 8192 --gpu-layers all --reasoning off`

Artifacts:

- smoke: `results/llamacpp_gemma4_26b_q4_0_chat_smoke_20260607T135911Z.json`
- serving benchmark: `results/llamacpp_gemma4_26b_q4_0_compact_20260607T135911Z.json`
- `llama-bench`: `results/llamacpp_gemma4_26b_q4_0_bench_20260607T135911Z.txt`
- server log: `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_server.log`
- `spark_doctor`: `results/spark_doctor_llamacpp_gemma4_26b_q4_0_20260607T135911Z.md`
- logprobs probe: `results/gguf_logprobs_probe_llamacpp_b9536_reasoning_off_20260607T135911Z.json`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.107 | 0.939 | 76.94 |
| `medium_decode` | 40 | 192 | 0.106 | 2.633 | 75.97 |

`llama-bench`:

| test | throughput |
|---|---:|
| `pp512` | 3021.76 +/- 34.41 tok/s |
| `tg128` | 77.35 +/- 0.13 tok/s |

Interpretation:

- llama.cpp is now blessed as a practical single-Spark serving path for this GGUF model.
- `--reasoning off` is required for normal OpenAI chat `message.content` output on this Gemma 4 server path.
- Server logs confirm CUDA on `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`.
- This row is Q4_0 GGUF, not NVFP4/MXFP4 GGUF. The measured win is practical 4-bit bandwidth reduction plus mature CUDA graph/quantized-serving kernels on `sm_121`; it does not prove native `sm_121a` FP4 tensor-core MMA dispatch.
- GGUF lm-eval accuracy remains blocked. The same server still exposes logprobs under `choices[0].logprobs.content`, not the `tokens` and `token_logprobs` shape expected by the existing lm-eval adapter.

## 2026-06-08: llama.cpp Qwen2.5 1.5B Q4_K_M Practical Serving Row

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- build: `308f61c31 (9536)`
- model repo: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- model file: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- local model: `/home/jethac/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- alias: `qwen25-1.5b-q4_k_m-gguf`
- settings: `--ctx-size 8192 -ngl 999`

Artifacts:

- run info: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_run_info.txt`
- smoke: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_chat_smoke.json`
- serving benchmark: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_openai_benchmark.json`
- `llama-bench`: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_llama_bench.txt`
- server log: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_server.log`
- build-target audit: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_build_target_audit.json`
- runtime probe: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_runtime_probe.json`
- `spark_doctor`: `results/spark_doctor_llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST.md`
- logprobs probe: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_gguf_logprobs_probe.json`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 44 | 64 | 0.032 | 0.397 | 175.19 |
| `medium_decode` | 56 | 192 | 0.015 | 1.113 | 174.86 |
| `long_prefill` | 2369 | 64 | 0.214 | 0.598 | 166.66 |

`llama-bench`:

| test | throughput |
|---|---:|
| `pp512` | 12505.79 +/- 615.87 tok/s |
| `tg128` | 178.10 +/- 0.95 tok/s |

Interpretation:

- llama.cpp is now also proven as a practical Qwen GGUF serving path on GB10.
- Server logs confirm CUDA on `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`.
- This row is Q4_K_M GGUF, not NVFP4/MXFP4 GGUF. It does not prove native `sm_121a` FP4 tensor-core MMA dispatch.
- The logprobs probe still fails the lm-eval compatibility check for the same schema reason as the Gemma llama.cpp row.

## 2026-06-08: llama.cpp Native Loglikelihood Live Probe

Artifacts:

- `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_probe.json`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_task.json`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_server.log`
- `results/llamacpp_native_loglikelihood_20260608T1331JST_build_target_audit.json`

Result:

- live `llama-server` loaded the Qwen2.5 1.5B Q4_K_M GGUF on CUDA, with log evidence `NVIDIA GB10`, `CUDA : ARCHS = 1210`, and `USE_GRAPHS = 1`.
- native `/tokenize` plus `/completion` scoring at `n_probs=512` found likely continuation tokens.
- the `zebra_unlikely` continuation was not present in the returned top-512 probabilities.
- task summary: `target_found=2`, `target_missing=1`, `ok=false`.

Interpretation:

- This is a live negative result for the current GGUF accuracy adapter path.
- llama.cpp practical serving remains blessed, but paper-comparable lm-eval-style GGUF accuracy remains blocked until the native API can score arbitrary supplied continuation tokens.

## 2026-06-07: LiteRT-LM Gemma 4 E2B CPU/GPU Smoke

Target:

- venv: `/home/jethac/spark-validation/litert-lm-20260607T140617Z/venv`
- package: `litert-lm==0.13.1`
- API package: `litert-lm-api==0.13.1`
- model: `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm`
- host: Linux `aarch64`, Python 3.12.3

Artifacts:

- import probe: `results/litert_lm_20260607T140617Z_import_probe.json`
- run info: `results/litert_lm_20260607T140617Z_run_info.txt`
- CPU chat smoke: `results/litert_lm_cpu_e2b_smoke_no_max_telemetry.json`
- CPU bad-KV smoke: `results/litert_lm_cpu_e2b_smoke4_telemetry.json`
- GPU chat smoke after group fix: `results/litert_lm_gpu_e2b_smoke_after_groups_telemetry.json`
- CPU benchmark: `results/litert_lm_cpu_e2b_bench_256p64d_telemetry.json`
- GPU benchmark: `results/litert_lm_gpu_e2b_bench_256p64d_telemetry.json`

Result summary:

| path | prefill tokens | decode tokens | result |
|---|---:|---:|---|
| CPU `run` chat | n/a | n/a | returned `spark-ok`, exit 0 |
| CPU `benchmark` | 256 | 64 | prefill `125.77` tok/s, decode `43.57` tok/s, init `0.3235` s, TTFT `2.0584` s |
| GPU `benchmark` | 256 | 64 | prefill `1773.07` tok/s, decode `43.70` tok/s, init `2.5860` s, TTFT `0.1673` s |
| GPU `run` chat | n/a | n/a | printed `spark-ok`, then exited `returncode=-11` |
| CPU `run --max-num-tokens 512` | n/a | n/a | failed with `DYNAMIC_UPDATE_SLICE` / `Failed to allocate tensors`; CLI still returned 0 |

Operational observations:

- `litert-lm run` reads all non-TTY stdin before loading the model. The telemetry wrapper originally left stdin open, which made LiteRT-LM appear to hang in `anon_pipe_read`.
- The telemetry wrapper now passes `stdin=subprocess.DEVNULL` for benchmark commands.
- Hugging Face cache subdirectories under `/home/jethac/.cache/huggingface` were root-owned from earlier work and caused permission-denied download errors. Ownership was restored to `jethac`.
- `jethac` was added to `render` and `video` so LiteRT GPU can open `/dev/dri` nodes. This removed the device permission errors but did not fix the GPU chat `SIGSEGV`.
- GPU logs still show `Failed to load OpenCL library with dlopen: libOpenCL.so`; LiteRT falls back to the ICD loader.

Interpretation:

- LiteRT-LM is viable on Spark as a lightweight Gemma E2B CPU path and as a benchmarkable GPU path.
- It is not yet blessed for GPU chat/serving because the `run` command can generate text and still crash at process exit.
- The GPU benchmark has a clear prefill advantage over CPU on this small row, but decode is effectively tied. This should be treated as a complement to llama.cpp/vLLM, not as the main path for extracting GB10 tensor throughput.

<!-- source: docs/BENCHMARKING_REPORT.md -->

# Benchmarking Report

Date: 2026-06-07

This report summarizes the PGX Workstation benchmarking work performed so far and the state of the generated benchmark artifacts at the point monitoring was stopped.

## Scope

The work monitored and summarized the initial personal Gemma 4 benchmark run described by `BENCHMARK_PLAN.md` on `thinkstationpgx-00b4`, with results synchronized into:

- Remote generated report: `/home/jethac/gemma4-evals/20260606_BENCHMARKING.md`
- Local generated report: `B:\workshop\20260606_BENCHMARKING.md`
- This narrative report: `B:\workshop\BENCHMARKING_REPORT.md`

The personal benchmark run was not terminated. Monitoring was stopped per request; the last observed remote process was still running.

Later targeted Spark probes were run outside the original personal campaign to make the compatibility story more actionable. Those results are indexed in `docs/BASELINE_RESULTS.md` and summarized below.

## Campaign Setup

The benchmark plan was revised to run host-native on the PGX workstation rather than through Docker. The active toolchain and sources recorded in the generated report were:

- vLLM `0.22.1`
- stock llama.cpp `b9536`
- llama.cpp MTP checkout from PR `23398`
- lm-eval-harness via the local Python environment

The manifest contained:

- 152 accuracy rows
- 248 MTP rows

The personal benchmark run stages were:

1. smoke safetensors
2. full accuracy
3. safetensors throughput
4. GGUF throughput
5. MTP speed
6. final report generation

During execution, timeout handling was adjusted for larger models:

- Safetensors smoke probes were raised to 600 seconds normally and 2400 seconds for large probes.
- MTP speed runs were given 900 second default timeout and 2400 second large-model timeout.
- `run_mtp_speed.py` was updated to accept those timeout arguments and record `timeout_s`.
- `bash -n run_campaign.sh` and Python compilation of `run_mtp_speed.py` passed after the edits.

## Last Synced Artifact State

The latest local generated benchmark snapshot was:

- File: `B:\workshop\20260606_BENCHMARKING.md`
- Last write time: 2026-06-07 18:40:15
- Size: 19,517 bytes

At that snapshot:

| artifact | status |
|---|---:|
| smoke rows | 152 complete |
| smoke ok | 21 |
| smoke eval_failed | 11 |
| smoke loader_failed | 120 |
| full eval task records | 70 |
| full eval ok records | 65 |
| full eval failed records | 5 |
| throughput JSONL rows observed | 2 |
| MTP JSONL rows observed | 2 |

The generated report showed the throughput and MTP sections as partial/early data only. The full throughput and MTP stages had not yet run in the active personal benchmark process by the last synced snapshot.

## Last Live Observation

The last live status sample before stopping showed:

- Campaign PID: `310239`
- Stage: `full_accuracy`
- Active model row: `unsloth-26b-a4b-baseline-bf16`
- Active task: `hellaswag`, 10-shot
- Backend: vLLM
- Active command: `lm_eval --model vllm ... --tasks hellaswag --num_fewshot 10`
- GPU utilization: 96%
- Active vLLM engine present and compute-bound
- Full result rows still at 70, meaning the active HellaSwag row had not yet written its result

The remote benchmark was not stopped.

## Accuracy Progress

The full-accuracy stage had progressed through E2B, E4B, 12B, and part of the 26B-A4B rows. The following rows were complete across all selected tasks unless noted otherwise.

| row | backend | status | notable result |
|---|---|---|---|
| `google-e2b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.350229`, ARC Challenge `0.331058` |
| `unsloth-e2b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.350229`, ARC Challenge `0.329352` |
| `unsloth-e2b-qat-q4-0-unquantized-bf16` | HF | partial | zero-shot completed; ARC Challenge failed with `returncode=-9` |
| `google-e2b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.463653`, ARC Challenge `0.438567` |
| `unsloth-e2b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.463155`, ARC Challenge `0.438567` |
| `google-e4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.467935`, ARC Challenge `0.391638` |
| `unsloth-e4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.467935`, ARC Challenge `0.392491` |
| `unsloth-e4b-qat-q4-0-unquantized-bf16` | HF | failed | zero-shot group failed with `returncode=-9` |
| `google-e4b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.485162`, WinoGrande `0.532755` |
| `unsloth-e4b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.485162`, WinoGrande `0.539858` |
| `google-12b-baseline-bf16` | HF | partial | zero-shot completed; ARC Challenge failed with `returncode=-9` |
| `google-26b-a4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.494822`, ARC Challenge `0.414676` |
| `unsloth-26b-a4b-baseline-bf16` | vLLM | in progress | zero-shot, ARC Challenge, and WinoGrande completed; HellaSwag running |

Selected 26B-A4B results observed before stopping:

| row | BoolQ | PIQA | ARC Easy | ARC Challenge | WinoGrande | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|
| `google-26b-a4b-baseline-bf16` | 0.690826 | 0.576170 | 0.358165 | 0.414676 | 0.546172 | 0.494822 |
| `unsloth-26b-a4b-baseline-bf16` | 0.688991 | 0.577258 | 0.359007 | 0.414676 | 0.548540 | running |

## Failure Modes

The main failure modes observed were:

- GGUF lm-eval compatibility: llama.cpp `b9536` `/v1/completions` logprobs did not match what lm-eval's GGUF adapter requires for loglikelihood scoring. These rows were treated as compatibility failures for paper-comparable accuracy.
- HF fallback memory/process failures: several HF-backed QAT/unquantized or larger rows exited with `returncode=-9`.
- Some vLLM load probes for unsupported 12B/QAT/mobile rows failed during smoke.

The campaign correctly recorded these as benchmark outcomes rather than silently skipping them.

## Runtime Notes

HellaSwag dominated wall time. Observed full HellaSwag durations included:

| row | elapsed seconds |
|---|---:|
| `google-e2b-baseline-bf16` | 5472 |
| `unsloth-e2b-baseline-bf16` | 5436 |
| `google-e4b-baseline-bf16` | 7890 |
| `unsloth-e4b-baseline-bf16` | 7860 |
| `google-e4b-qat-w4a16-w4a16` | 8401 |
| `unsloth-e4b-qat-w4a16-w4a16` | 8416 |
| `google-26b-a4b-baseline-bf16` | 9017 |

The active `unsloth-26b-a4b-baseline-bf16` HellaSwag row had not completed at the last observation.

## Throughput And MTP

Only early throughput/MTP rows were present in the synced snapshot:

| area | observed row |
|---|---|
| safetensors throughput | `google-e2b-baseline-bf16`, vLLM, prompt `506.98 tok/s`, generation `130.17 tok/s` |
| GGUF throughput | `google-e2b-qat-q4_0...gguf`, llama.cpp, prompt `3923.63 tok/s`, generation `122.11 tok/s` |
| MTP speed | `mtp-12b-ud-iq2_m-q8_0...`, prompt `1042.40 tok/s`, generation `36.40 tok/s` |

The active personal benchmark run had not reached the full throughput or MTP stages by the last synced report.

## Targeted Follow-Up Benchmarks

After the initial personal campaign snapshot, targeted compact probes were added to answer specific Spark/GB10 questions.

| target | result | interpretation |
|---|---|---|
| vLLM Gemma 4 E4B W4A16 | compact OpenAI harness around 50-52 tok/s decode | first before row for an already-running vLLM server |
| SGLang 26.05 Qwen BF16 | Qwen smoke passed; short/medium/long-prefill decode around 59-60 tok/s | SGLang works on GB10 for at least one supported BF16 model, but this is not Gemma or NVFP4 |
| SGLang Gemma 4 E2B | failed before health; default path crashed in Gemma4 audio tower, `--language-only` required encoder URLs | SGLang Gemma4 model glue blocker, not a proven `sm_121` kernel failure |
| FlashInfer SM121 `mm_fp4` source/JIT | patched auto-dispatch includes `b12x`; finite outputs on GB10 | dispatch enablement, not a speedup claim |
| FlashInfer model-shaped proxies | dense-decode proxies mixed; MoE-shaped proxies slower after the patch | the one-line `b12x` gate is not enough to make Spark faster |
| vLLM Gemma 4 26B A4B | serves in `vllm/vllm-openai:latest-cu130` at about 24 tok/s after `--max-num-batched-tokens 4096` | useful BF16/unquantized MoE serving baseline, not NVFP4 |
| vLLM AEON Gemma 4 26B A4B NVFP4+DFlash | serves in `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`; warmed compact row is 47.91, 53.60, and 98.38 tok/s across short/medium/long-prefill cases | first local vLLM Gemma 26B row materially above the BF16 baseline; gain comes from AEON NVFP4 checkpoint/container plus DFlash, not from our fork |
| vLLM Gemma 4 12B | source/precompiled vLLM commit `da1daf40` plus Transformers main serves at about 7.7 tok/s | proves `gemma4_unified` can run on GB10, but not a clean release container or performance win |
| llama.cpp Gemma 4 26B Q4_0 | OpenAI-compatible serving around 76 tok/s decode with `--reasoning off`; `llama-bench` tg128 around 77 tok/s | practical GGUF serving path is blessed; lm-eval GGUF accuracy remains blocked |
| LiteRT-LM Gemma 4 E2B | CPU chat works; GPU benchmark works; GPU chat prints output then exits `-11` | optional side-runtime evidence, not a main Spark performance path |
| SGLang Qwen2.5 1.5B BF16/auto vs fp8 | matched `mem_fraction_static=0.40` rows run at about 58-59 tok/s; fp8 KV pool is 3.11M tokens vs 1.56M BF16/auto | fp8 is a capacity win with decode-speed parity on this small Qwen row |
| SGLang Qwen2.5 1.5B patched FP4 KV | patched overlay serves only with both CUDA graph modes disabled; FP4 KV pool is 5.54M tokens but short decode is 0.276 tok/s with repetitive output | FP4 KV capacity path is real, but SGLang FP4 KV is not a usable speed path yet |
| llama.cpp Qwen2.5 1.5B Q4_K_M | OpenAI-compatible serving around 167-175 tok/s decode; `llama-bench` tg128 around 178 tok/s | practical Qwen GGUF serving is proven; lm-eval GGUF accuracy remains blocked by logprobs schema |
| vLLM AEON Qwen3.6 35B-A3B NVFP4+DFlash | `v2` image now starts and serves locally; with `chat_template_kwargs={"enable_thinking": false}`, compact serving passes at about `50-56 tok/s`; `jethac/vllm@6804e1b` derived image also passes at `47.22`, `58.88`, and `61.62 tok/s` after dependency alignment plus AEON FA2 binary restoration | AEON and fork-derived rows are real Qwen speed results, but not native `sm_121a` target proof; the fork row proves runtime parity through an AEON-derived image, not a clean upstreamable install |

The current headline has changed: local end-to-end vLLM Gemma 26B NVFP4 serving and Qwen3.6 NVFP4+DFlash serving are now banked, and the prepared `jethac/vllm` Qwen fork can serve the AEON row after dependency alignment. The open question is no longer whether a Spark-class GB10 can run fast NVFP4 Gemma/Qwen vLLM paths; it is how much of that path belongs upstream, how to remove AEON-binary dependencies, and whether native `sm_121a` target proof changes throughput or capacity.

## SM120 Reference Work

Two hikarioyama reference repos are now tracked as prior art:

- `hikarioyama/vllm-nvfp4-kv-sm120` at `f6156ee3b22b24885a52c02bdafb34a9c201fe86`
- `hikarioyama/sglang-nvfp4-kv-sm120` at `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`

They are relevant because they implement NVFP4 KV paths through vLLM/SGLang plus FlashInfer FA2 changes on SM120 RTX Blackwell systems. They are not GB10 `sm_121` validation. The repo policy is to build on them through clean `jethac` forks and worktrees, not vendor overlay trees into production images.

The vLLM reference changes the measurement priority. Its headline is not weight-GEMM speed: it reports roughly 1.78x fp8 KV pool and much higher maximum concurrency at matched utilization, while decode stays near fp8 parity. For Spark, the first useful proof should therefore measure KV pool tokens, maximum concurrency, hidden scratch allocations, quality, and long-context behavior. Decode tok/s is still recorded, but a flat decode result can be acceptable if capacity and quality improve.

## Open State

At the point monitoring stopped:

- Smoke was complete.
- Full accuracy was still running.
- The active row was `unsloth-26b-a4b-baseline-bf16` HellaSwag.
- Throughput stages were still pending in the active personal run.
- MTP stage was still pending in the active personal run.
- The final generated report was a current snapshot, not completion of that personal run.

## Files

- `B:\workshop\20260606_BENCHMARKING.md`: generated benchmark snapshot
- `B:\workshop\BENCHMARKING_REPORT.md`: this narrative status report
- Remote artifacts remain under `/home/jethac/gemma4-evals/results` and `/home/jethac/gemma4-evals/logs`.

<!-- source: docs/BENCHMARK_PROTOCOL.md -->

# Before/After GB10 Benchmark Protocol

Status: draft.

Goal: prove that this campaign makes the GB10 perform closer to what the silicon is capable of.

Compatibility is not enough. A 900k JPY machine should not merely start a model. It should run the best available stack for the workload, with measured speed, memory behavior, and output quality.

Tracked by:

- https://github.com/jethac/dgx-spark-hijinks/issues/19

## Rules

- Every improvement claim needs a before row and an after row.
- Before and after rows must use the same machine, model, prompt set, quantization, context length, batch/concurrency, and output length.
- Before and after performance rows must also match CUDA compute capability and `multi_processor_count`; GB10 `sm_121` can appear in different SM-count bins, and correctness compatibility does not make throughput comparable.
- See `docs/SM_COUNT_AWARENESS.md` for the current observed 48-SM reference evidence and fork audit.
- Capture `spark_doctor` before each run.
- Capture CUDA build/JIT target evidence and shared-object evidence when testing kernel changes.
- Capture runtime process evidence with `scripts/runtime_process_probe.py` for serving baselines.
- Annotate failures with `scripts/failure_annotator.py` so killed processes, API schema mismatches, runtime exceptions, and configuration errors are not lumped together.
- Wrap HF fallback and other fragile local commands with `scripts/run_with_telemetry.py` so return codes include RSS, swap, `free`, `nvidia-smi`, and kernel OOM evidence.
- Separate cold start, first-token, warm decode, and long-context behavior.
- Keep backend families separate: vLLM, SGLang, llama.cpp/Ollama, LiteRT-LM, and HF fallback.
- Do not compare paper accuracy across backends unless the scoring path is validated.

## Required Columns

| field | why |
|---|---|
| run id | joins logs/results |
| phase | before or after |
| backend | vLLM, SGLang, llama.cpp, LiteRT-LM, HF |
| repo commit/container | reproducibility |
| model id/revision | model control |
| GPU name / compute capability / SM count | performance rows are only comparable within the same device bin |
| hardware comparison key | quick grouping key such as `NVIDIA_GB10:sm_121:sms_48` |
| quantization | performance/quality control |
| KV cache dtype | fp8 vs NVFP4 impact |
| KV pool tokens | capacity/concurrency impact |
| maximum concurrency | long-context capacity impact |
| attention backend | kernel path |
| CUDA graph mode | performance control |
| prompt tokens | throughput denominator |
| generated tokens | throughput denominator |
| TTFT | interactive feel |
| decode tok/s | steady-state generation |
| prefill tok/s | prompt-processing speed |
| memory state | unified-memory pressure and hidden scratch allocation detection |
| quality check | avoids fast garbage |
| `spark_doctor` path | environment evidence |
| build target audit path | intended CUDA target evidence from build/JIT logs |
| `.so`/JIT audit path | compiled kernel evidence |

## Baseline Suite

Keep this short enough to run repeatedly on one unit.

## Required Model Lanes

Runtime claims must cover both model families unless the claim is explicitly model-specific:

- Qwen speed/capacity: required for throughput, speculative decode, NVFP4-weight, and fp8-vs-NVFP4 KV claims. Qwen is the cleaner first lane for SM121a runtime mechanics because it avoids Gemma 4's heterogeneous attention-shape complications.
- Gemma compatibility/performance: required for the campaign's original workload and for proving the stack handles the harder Gemma 4 model-family path.

Do not generalize a Gemma-only row to Qwen, or a Qwen-only row to Gemma. A runtime can be marked `partial` with one family, but it should not be called broadly blessed until both lanes have at least one artifact-backed speed row and a basic quality sanity check.

1. Environment
   - `spark_doctor`
   - CUDA build/JIT target audit for any source-built or JIT-built backend
   - CUDA `.so` audit for the active backend when compiled objects are available
   - container target audit for image/container artifacts when the only available evidence is environment labels, `TORCH_CUDA_ARCH_LIST`, package versions, or `torch.cuda.get_arch_list()`

Example build/JIT target audit:

```bash
python3 scripts/cuda_build_target_audit.py \
  --log results/RUN_ID_build.log \
  --output results/RUN_ID_build_target_audit.json
```

Family/PTX evidence such as `12.0+PTX` or a package arch list ending at `sm_120` should be recorded, but it is not native `sm_121` or `sm_121a` proof.

2. Serving smoke
   - one short deterministic OpenAI-compatible chat request
   - one medium generation request
   - use `scripts/openai_serving_benchmark.py` when the backend exposes an OpenAI-compatible API
   - use `scripts/spark_smoke_suite.py` before and after larger stack changes so vLLM, SGLang, llama.cpp, HF fallback, MTP/spec decode, and NVFP4 evidence are tracked together
   - for Gemma 4 26B with `vllm/vllm-openai:latest-cu130`, include `--max-num-batched-tokens 4096` on the server; the default 2048 is below Gemma's multimodal item budget and fails before readiness

3. Throughput
   - short prompt / short output
   - long prompt / short output
   - short prompt / long output
   - for Qwen rows, prefer `scripts/qwen_speed_lane.py` with a JSONL row file so vLLM, SGLang, and llama.cpp evidence share the same manifest shape
   - run `scripts/serving_manifest_audit.py` before treating a row as claim-ready; dry-run manifests are planning evidence only, and a live row still needs metadata, hardware evidence, backend markers, and accepted build-target evidence

Example:

```bash
python3 scripts/openai_serving_benchmark.py \
  --url http://127.0.0.1:8000 \
  --backend vllm \
  --phase before \
  --run-id vllm-gemma4-e4b-w4a16-before-001 \
  --output results/vllm_gemma4_e4b_w4a16_before_001.json
```

4. Quality sanity
   - deterministic prompt expected to produce stable text
   - fp8/bf16 reference for any NVFP4 test
   - no NaN/inf/empty/zero-output path

5. Kernel microbenchmarks
   - use `scripts/flashinfer_mm_fp4_microbench.py` for FlashInfer NVFP4 `mm_fp4` dispatch checks
   - use `--preset dense_decode` and `--preset moe_expert` before making performance claims about the SM121 `b12x` path
   - treat kernel microbenchmarks as diagnostic evidence, not serving throughput
   - follow with model-shaped and serving before/after rows before claiming user-visible speedups

6. NVFP4 KV capacity checks
   - compare fp8 versus NVFP4 KV at the same model, prompts, memory utilization, graph settings, and concurrency
   - record vLLM/SGLang startup KV pool tokens and maximum concurrency directly from server logs
   - capture memory telemetry before, during, and after serving so hidden scale-factor scratch allocations do not masquerade as capacity wins
   - treat decode speed as secondary to capacity for the first proof; a flat tok/s result can still be a win if KV pool and concurrency improve without quality loss
   - for vLLM, run the reference layout harness for the target `H_q/H_kv/D/page` before serving

7. Optional long checks
   - HellaSwag and other long lm-eval tasks run as separate campaigns
   - RULER/needle-style checks for long-context KV changes

## Initial Before State

The first imported personal benchmark run is a partial before-state artifact, not a complete protocol run.

Known before-state evidence:

- vLLM `0.22.1`
- PyTorch `2.11.0+cu130`
- FlashInfer `0.6.11.post2`
- GB10 reports `sm_121`
- inspected vLLM/FlashInfer objects had no explicit `sm_121` SASS
- vLLM safetensors rows worked for several E2B/E4B/26B-A4B rows
- GGUF lm-eval accuracy path was blocked by logprobs/API mismatch
- HF fallback had `returncode=-9` failures
- HellaSwag dominated wall time

The next step is to turn this into a compact repeatable before/after suite.

First compact before row:

- `docs/BASELINE_RESULTS.md`
- `results/vllm_gemma4_e4b_w4a16_before_compact_20260607T1126Z.json`
- `results/spark_doctor_before_vllm_gemma4_e4b_w4a16_20260607T1126Z.md`
- `results/runtime_probe_vllm_gemma4_e4b_w4a16_root_20260607T1136Z.json`

<!-- source: docs/BLESSED_STACK.md -->

# Blessed Stack

This is the current known-good / known-bad stack record. It is intentionally conservative.

## Hardware

- System class: DGX Spark-class GB10 workstation
- GPU observed in benchmark logs: `NVIDIA GB10`
- CUDA compute capability target: `sm_121`
- Available hardware count: one Spark-class machine

## Current Benchmark Stack

From the initial personal Gemma 4 benchmark run:

- vLLM: `0.22.1`
- llama.cpp: `b9536`
- llama.cpp MTP checkout: PR `23398`
- lm-eval-harness: local Python environment
- PyTorch in benchmark venv: `2.11.0+cu130`
- FlashInfer in benchmark venv: `0.6.11.post2`
- SGLang: NVIDIA 26.05 container smoke passed; not installed/blessed in the first benchmark venv
- LiteRT-LM: optional side-runtime evidence from a clean `litert-lm==0.13.1` venv evaluated outside the first benchmark venv

## Known Good So Far

- vLLM safetensors rows ran for E2B, E4B, and 26B-A4B with sustained GPU utilization.
- NVIDIA SGLang `26.05-py3` served `Qwen/Qwen2.5-1.5B-Instruct` on GB10 through the OpenAI-compatible API.
- stock llama.cpp CUDA throughput worked for at least the early GGUF throughput row.
- llama.cpp `b9536` serves Gemma 4 26B Q4_0 through the OpenAI-compatible API with `--reasoning off`; compact decode was about 76 tok/s.
- llama.cpp `b9536` serves Qwen2.5 1.5B Q4_K_M GGUF through the OpenAI-compatible API; compact decode was about 167-175 tok/s.
- Optional LiteRT-LM `0.13.1` CPU chat serves Gemma 4 E2B and returns `spark-ok`.
- Optional LiteRT-LM `0.13.1` GPU benchmark runs for Gemma 4 E2B and shows high prefill throughput on the tiny benchmark row.
- llama.cpp MTP executed at least one speed row.

## Known Bad Or Not Yet Blessed

- AEON's vLLM `0.20.1` Gemma 4 26B NVFP4+DFlash image is now locally proven for practical serving on GB10; keep it as the fastest measured vLLM Gemma 26B path while accuracy and clean fork packaging remain pending.
- vLLM `0.22.1` is not blessed for Gemma 4 12B `gemma4_unified` on Spark.
- NVIDIA SGLang `26.05-py3` is not currently a working Gemma 4 E2B path in our test: default Gemma4 multimodal setup crashes in the audio tower, and `--language-only` is not a valid standalone workaround.
- HF fallback is not a transparent substitute for vLLM; several rows died with `returncode=-9`.
- GGUF accuracy through the tested lm-eval/llama.cpp path is blocked by logprobs/API compatibility.
- `--kv-cache-dtype nvfp4` is not generally blessed on Spark yet. vLLM Qwen has a capacity-proven FlashInfer FA2 NVFP4-KV row, but Gemma remains blocked by the global `D=512` FlashInfer shape and SGLang remains blocked by FP4-KV quality corruption.
- Qwen speed/capacity is now a required benchmark lane. Small SGLang Qwen BF16/auto and fp8 rows are locally proven at about 58-59 tok/s; AEON Qwen3.6 NVFP4+DFlash is locally reproduced at about 50-56 tok/s when Qwen thinking is disabled with `chat_template_kwargs`; clean `jethac/vllm@a919d635d` + `jethac/flash-attention@7d53245` packaging serves the same Qwen row at `61.07`, `56.97`, and `60.10 tok/s` with separate `sm_121a` FA2 cubin proof; a no-DFlash vLLM Qwen NVFP4-KV row records `1.751x` fp8 KV pool/concurrency through FlashInfer FA2 with decode-speed parity.
- `hikarioyama/vllm-nvfp4-kv-sm120` and `hikarioyama/sglang-nvfp4-kv-sm120` are audited SM120 reference implementations and should be used as prior art for our forks. They are not GB10 `sm_121` blessed stacks until fp8-vs-NVFP4 quality, capacity, and speed are reproduced on Spark-class hardware.
- Multi-Spark recipes are not validated because we currently have only one unit.
- The inspected vLLM/FlashInfer extension set has no explicit `sm_121` SASS. General vLLM extensions include `sm_120`, while several attention/MLA extensions are `sm_80`, `sm_90a`, or `sm_100` only. Treat this as a validation requirement, not an automatic failure.
- SGLang NVFP4 KV is capacity-proven but not blessed. The current fork calibrates native FP4 KV before capture, then disables CUDA graph and piecewise graph capture by default because graph-enabled decode corrupts Qwen output; the matched row expands KV capacity by about `1.78x` versus fp8, but compact benchmark content still fails quality. Prompt reconciliation proves OpenAI prompt IDs match local Qwen rendering, so the remaining bug is endpoint/path-specific FP4 KV numerics or metadata rather than chat-template tokenization. Track `hikarioyama/sglang-nvfp4-kv-sm120` as a design reference, but do not bless SGLang FP4 KV performance or quality until a clean fp4-vs-fp8 row passes.
- The tested NVIDIA SGLang container still logs `SM120 (Blackwell) detected` and audited SGLang/FlashInfer objects contain no explicit `sm_121` SASS. Treat the BF16 smoke as functional evidence, not proof of fully Spark-native kernel coverage.
- FlashInfer source/JIT validation at `jethac/flashinfer@a42c8f07` proves one important lower-level fix: installed vLLM/SGLang containers exclude `b12x` from SM121 NVFP4 `mm_fp4` auto-dispatch, while the patched source selects `b12x` and can run tiny and model-shaped NVFP4 GEMMs on GB10. Current microbenchmarks do not show a speedup. This is not yet a blessed serving stack because it required an ephemeral source install and removal of stale FlashInfer JIT/cubin packages.
- FlashInfer FA2 NVFP4 paged-KV standalone correctness now passes on GB10 with `jethac/flashinfer@e152cf4d` and vLLM-style V-scale-factor de-swizzle enabled. This is kernel-level evidence, not a blessed vLLM/SGLang serving stack.
- FlashInfer is not the whole Spark fix. The remaining work spans packaging, vLLM/SGLang integration, Gemma 4 12B support, NVFP4 KV serving quality/capacity, llama.cpp/lm-eval accuracy, optional LiteRT GPU stability, and short before/after benchmark proof.
- Optional LiteRT-LM GPU chat is not blessed: after fixing `/dev/dri` group access it prints `spark-ok` but exits with `returncode=-11`.
- llama.cpp serving is blessed for the tested Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M GGUF paths, but GGUF lm-eval accuracy is still blocked by API/logprobs schema compatibility. These rows do not prove native NVFP4/MXFP4 `sm_121a` tensor-core dispatch.

## Candidate Next Stack

To be tested:

- NVIDIA/vLLM NGC container validated for DGX Spark, if available for the target date.
- vLLM native FP4 weight/MoE proof for the passing Qwen3.6 NVFP4+DFlash row. Clean FA2 packaging is now proven, but the server still selects Marlin weight-only FP4 and warns that the weight path is not native FP4 compute.
- AEON Gemma and Qwen3.6 NVFP4+DFlash are now measured locally, both the AEON-derived and clean-FA2 `jethac/vllm` Qwen rows pass, and the Qwen NVFP4-KV row proves the expected capacity gain. The remaining vLLM gaps are Gemma NVFP4-KV, native FP4 weight/MoE evidence, accuracy, and upstreamable packaging beyond this AEON-checkpoint recipe.
- vLLM build with native `Gemma4UnifiedForConditionalGeneration`.
- SGLang Gemma model-path fix or documented go/no-go, then NVFP4/fp8 quality comparison on Spark.
- SGLang Qwen fp8-vs-`fp4_e2m1` KV quality fix after the capacity row; the capacity delta is recorded and prompt serialization is cleared, but the FP4 quality check still fails.
- vLLM NVFP4 KV fork probe derived from the hikarioyama SM120 implementation, reduced to a single-Spark GB10 test before any TP=2 or long-context claims.
- Optional LiteRT-LM GPU chat fix or documented CPU-only/complement role.
- llama.cpp commit with an API schema that can satisfy lm-eval loglikelihood scoring, or a patched adapter.
- llama.cpp commit/build recipe for practical serving even if lm-eval accuracy remains separate.
- larger llama.cpp Qwen3/Qwen3.6 GGUF serving and native FP4/MXFP4 experiments.

<!-- source: docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md -->

# BUG: FlashInfer serving-path numerics wrong at Gemma 3 1B geometry (d256 / SWA-512 / 1 KV head)

Status (UPDATED 2026-06-12 by the 1B RIGOROUS re-test, § below): the **bf16 arm
is NOT a bug — defect (A) is REFUTED. It was an ENVIRONMENTAL / false-green
ARTIFACT.** A controlled 1B re-run (same `g6adc00f70` wheel, same FlashInfer
`7d5d477b` JIT, FlashInfer verified-engaged from the engine proof line, util 0.6,
serve-ready timeout 300 s as a wedge detector) gives **FI-bf16 − FLASH_ATTN =
−0.0006633 nats and NO engine wedge** — the earlier +0.221/wedge does NOT
reproduce. Both bf16 backends match HF eager truth (2.359755). This closes the
"width/depth-driven bf16 inflation" hypothesis the 270M test left open (1B is
also clean). Only ONE real sm_120 defect remains: **(B) the nvfp4 KV read-path
defect** (model/size-independent: 1B + 270M + 4B-mm all gibberish; 1B re-test
+1.587 nats / gibberish). The earlier "+0.22–1.38 nats bf16 inflation" and the
"wheel engine-wedge" are now attributed to methodology artifacts: the
`VLLM_ATTENTION_BACKEND` false-green fallback and the util-0.3 OOM / slot-mapping
JIT regime on the WDDM-shared card.

Bug RE-SCOPED: bf16/wedge arm CLOSED (no bug); nvfp4 read-path arm OPEN.

--- original status, now superseded ---
Status: OPEN, root cause unknown. Observed 2026-06-12 on sm_120; confirmed
sm_120-specific (Spark/sm_121 bisect clean). As of the 270M repro test (§ below)
the bug is understood to be TWO SEPARABLE sm_120 defects: (A) an FI-bf16 SWA-512
long-context inflation that tracks model width/depth (1B yes, 270M no) and also
manifests as the wheel engine-wedge; (B) an nvfp4 KV read-path defect that is
model/size-independent (1B + 270M + 4B-mm all gibberish). 270M is a fast minimal
repro for (B) but NOT (A).
Found by: P520 Gemma 3 1B serving verification (zero-bug diagnostics).
Severity: FI-bf16 quality silently wrong (+0.22 to +1.38 nats); FI-nvfp4
unusable (deterministic gibberish). Coherent short-prompt chat MASKS the bug.

## Environment

- P520: RTX 5060 Ti, CC 12.0 (sm_120), WSL2 Ubuntu 24.04, CUDA 13.0 (nvcc
  V13.0.88), torch 2.12.0+cu130.
- vLLM: jethac/vllm spark/hijinks-022-gemma4-mixed-kv @ 9759e3b06 (exact
  r9-image code), editable build, TORCH_CUDA_ARCH_LIST=12.0a, 42 sm_120a
  cubins confirmed, NVFP4 linear-latch diag PASS (head 128 and 256).
- FlashInfer: jethac/flashinfer spark/hijinks-022-fa2-d512 @ 7d5d477b,
  JIT-compiled on box (NOT the Spark AOT path).
- Model: google/gemma-3-1b-it - uniform head_dim 256, GQA 4q/1kv (the only
  Gemma with a single KV head), sliding window 512 (smaller than the 1024 of
  the larger Gemma 3 sizes), 5:1 sliding:global.

## Evidence (results/p520_gemma3_1b_serving_20260612/, ctx 8191, 8190 scored)

Truth references, agreeing to <0.001 nats on all corpora:
- HF transformers eager bf16: C1/C2/C3 = 2.35778 / 3.21392 / 1.42429
- vLLM FLASH_ATTN-backend bf16 serving row: matches HF on all three.

Deviations from truth (nats, C1/C2/C3):
| FI row | delta vs truth | chat smoke |
|---|---|---|
| bf16 | +0.221 / +1.243 / +1.380 | coherent ("Tokyo") |
| fp8_e4m3 | +0.006 / +0.159 / +0.494 | coherent |
| nvfp4 (+linear V-SF) | +1.592 / +2.436 / +2.752 | GIBBERISH, deterministic |

All FI rows internally deterministic. nvfp4 gibberish reproduced
byte-identical on a VIRGIN FlashInfer JIT cache (not stale kernels); writer
latch clean, so suspicion is on the read path. Pre-diagnostic JIT cache
preserved at WSL ~/.cache/flashinfer_prediag_070355 for forensics.

## Diagnostic structure (three tells)

1. Short-prompt chat coherent while long-ctx PPL inflated: chat prompts stay
   inside the 512-token sliding window; ctx-8191 scoring crosses it
   constantly. Points at sliding-window boundary handling in the FI serving
   path (paged KV at depth), which single-call kernel probes never
   reproduced (6/6 probes passed at 31B/E4B geometries).
2. fp8 closer to truth than bf16 on the SAME backend: dtype-conditional
   kernel templates / tile dispatch differ, so the defect is likely in a
   path-conditional spot, not shared mask math.
3. Novel geometry axes vs everything previously tested: window 512 (not
   1024) AND kv_heads=1. Either could be the unexercised path.

## What is and is not contaminated

- NOT contaminated: all sm_121 (GB10) results - Triton-vs-FlashInfer pairs
  within 0.04 nats across 5 sizes (12B-31B geometries) corroborate each
  other; G3-12B/27B FI rows checked against FLASH_ATTN/Triton baselines.
- Contaminated/blocked: any Gemma 3 1B FI claim; Gemma 3 bf16 retirement
  flip (scoped back to Gemma 4-only, see TRITON_RETIREMENT_SCORECARD
  adjudication log); Gemma 3 1B nvfp4 support claim (row banked RED).

## Bisect + investigation plan

1. Gemma 3 1B rerun on sm_121 Spark (cheap; next available window): same
   deviation -> geometry bug (window-512 / 1-kv-head path, platform-
   independent, simply never reached); clean -> sm_120-specific (JIT codegen
   or arch-conditional path). 
2. Logit-level diff probe FI vs FLASH_ATTN at exact 1B geometry on P520:
   find the first divergent layer/position; check position vs window
   boundary (expect divergence onset near token 512 if tell #1 is right).
3. Geometry ablation in the probe harness: window 512 vs 1024, kv_heads 1
   vs 2, d256 fixed - isolate the axis.
4. Once root-caused: fix on jethac/flashinfer branch + upstream filing with
   minimal repro (this doc is the filing draft skeleton).

## Bisect result 2026-06-12 (Spark / GB10, sm_121): PLATFORM / sm_120-specific

Bisect step 1 of the plan above, run on the Spark (GB10, sm_121) with the
baked r9 image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id 8c37bdbc4fdb), `google/gemma-3-1b-it`, C1 ctx 8191, each cell run TWICE
bitwise. Backend forced via `--attention-backend FLASH_ATTN|FLASHINFER`,
engaged backend verified from the `Using AttentionBackendEnum.<X> backend.`
proof lines (not the flag).

| row | backend (proof) | kv dtype | C1 ×2 (bitwise) | smoke |
|---|---|---|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.356493110435786 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.359283581557766 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 2.400746942552027 (IDENTICAL) | "Tokyo" COHERENT |

**FI-bf16 − FLASH_ATTN-bf16 on the Spark = +0.00279 nats** (P520 was +0.221).
That is well under the 0.01 threshold → FlashInfer MATCHES FLASH_ATTN at the
exact 1B geometry on sm_121. **Verdict: PLATFORM / sm_120-specific. The bug
does NOT reproduce on the Spark — the GEOMETRY hypothesis is REFUTED.** NVFP4 KV
is COHERENT and deterministic on the Spark (+0.044 vs truth) where the P520 gave
deterministic gibberish (+1.59). The Spark FLASH_ATTN truth row (2.35649) is
within 0.0014 nats of both the P520 FLASH_ATTN row (2.35785) and the HF eager
truth (2.35778), confirming the setup is sound.

Implication: the defect is scoped to sm_120 (the P520 editable / source-tree
FlashInfer build — JIT codegen or an arch-conditional path), NOT the
d256/SWA-512/1-kv-head geometry. The Spark/sm_121 Gemma 3 1B path (bf16 AND
NVFP4 KV) is fine. The Gemma 3 1B NVFP4 row banked RED on the P520 is
GREEN-class on the Spark. Bug stays OPEN but re-scoped to sm_120; investigation
steps 2-3 (P520 logit-diff probe + window/kv-head ablation) remain the way to
localize the sm_120 root cause. Artifacts:
results/claude_1b_bug_bisect_20260612/ (BISECT_SUMMARY.md, ppl JSONs ×2/row,
smoke transcripts, proof lines); Spark master copy + server logs + token dumps
at /home/jethac/spark_tmp/claude_1b_bug_bisect_20260612/.

## Baked-WHEEL disambiguation 2026-06-12 (P520 / sm_120): bug PERSISTS, harder form

Task #37. Re-ran the 1B bisect with the **sm120a RELEASE WHEEL**
(`vllm 0.1.dev1+g6adc00f70.sm120a`, from `spark/hijinks-e2-vllm @ 6adc00f70`)
instead of the editable build (`9759e3b06`), FlashInfer JIT-from-source
`7d5d477b` in BOTH (controlled variable). Provenance gates GREEN: EXT_PATH =
wheel `_C_stable_libtorch.abi3.so`, 42 sm_120a cubins, linear-V-SF latch
"writer wrote LINEAR V-SF" (hd128+hd256).

**The bug did NOT vanish — it WORSENED into an engine wedge.** Where the editable
build returned wrong-but-finite numbers (+0.221/+1.243/+1.380 nats), the wheel
**deadlocks the vLLM engine on the first window-crossing (>512-token) request**
at this geometry:
- 16-token prompt_logprobs → valid, server coherent ("Tokyo").
- 600-token (crosses SWA-512) COLD → server heartbeat logger FREEZES after a
  `Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel`
  warning; GPU 100% util / ~34 W (idle-spin, not matmul); `Running: 0 reqs`;
  curl times out at 120 s.
- WARMED 600-token (short req first) → RC=0 in 1 s, so the 512-crossing path is
  functionally fine once its slot-mapping shape is compiled.
- ctx-8191 prompt_logprobs → still wedges the engine even AFTER a 700-token
  warmup (socket ESTABLISHED to server, heartbeat logger silent >10 min). NO
  FLASH_ATTN/FLASHINFER/nvfp4 PPL rows could be produced.

Implication: the wedge is on the **FLASH_ATTN** backend's serving/slot-mapping
path (not only FlashInfer), tied to the SWA-512 window crossing on sm_120 — so
the defect is **NOT an editable-vs-wheel `_C` artifact**; it persists on the
release wheel and keeps suspicion on the sm_120 long-context / SWA-512 serving
path. Named confound: the wheel env excludes the optional accel-kernel pip
extras (tilelang/quack/tokenspeed/humming/cutlass-dsl/flashinfer-python|cubin —
omitted to keep FlashInfer source-tree JIT authoritative); a missing accel
kernel could aggravate the slot-mapping slow-JIT. Next localizer: the Colab G4
sm_120 cell at this geometry (off-WSL, bigger sm_120 card), and a P520 rerun
with the full accel-kernel pip set to rule the confound in/out.

Same nvfp4-on-sm_120 GIBBERISH also reproduced in the **Gemma 3 4B multimodal**
nvfp4-KV smoke (`results/p520_mm_retirement_smokes_20260612/`, cell
`g3_4b_nvfp4_fi`): coherent under bf16, gibberish under nvfp4 KV — confirming the
nvfp4 KV READ defect on sm_120 is model/size-independent and independent of the
mm-prefix mask path (the mask machinery ran; bf16 mm is clean).

Artifacts: results/p520_1b_wheel_disambig_20260612/DISAMBIG_SUMMARY.md
(server logs incl. `_WEDGED`, diag1/diag2 run logs, proof lines).

## 270M repro test 2026-06-12 (P520 / sm_120): SPLIT — nvfp4 REPRODUCES, FI-bf16 does NOT

Task #37 follow-up. Re-ran the bisect on `google/gemma-3-270m-it` to ask whether
the bug is 1B-geometry-specific or broad across small Gemma 3. Same **sm120a
RELEASE wheel `g6adc00f70`** + FlashInfer JIT `7d5d477b` as the task #37 1B
disambig. Backend engaged verified from the `Using AttentionBackendEnum.<X>
backend.` proof line (FI-bf16 and FI-nvfp4 both confirmed FLASHINFER, no
false-green). C1 ctx 8191, each row run TWICE bitwise, util 0.6 for the FI rows
(util 0.3 OOM'd on the WDDM-shared card — `No available memory for the cache
blocks`, environmental; FLASH_ATTN truth row ran at 0.3).

270M GEOMETRY: hidden 640, 18 layers, 4 attn heads, **head_dim 256,
num_key_value_heads 1, sliding_window 512** — shares ALL THREE suspect axes
(d256 / SWA-512 / 1-kv-head) with the 1B; only width (640 vs 1152) and depth (18
vs 26) differ. HF bf16 eager C1 ground truth = 2.912124 nats.

| row | backend (proof) | kv dtype | C1 ×2 (bitwise) | delta vs FLASH_ATTN | smoke |
|---|---|---|---|---:|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.911488 (IDENTICAL) | — (+0.00064 vs HF) | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.912821 (IDENTICAL) | **+0.00133** | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 11.034120 (IDENTICAL) | **+8.122** | **GIBBERISH, deterministic** |

**Verdict: SPLIT — the two sm_120 defects are SEPARABLE.**
- **FI-nvfp4 REPRODUCES**: deterministic gibberish + 11.034 nats (Cyrillic/JP
  salad, same failure class as the 1B's nvfp4 RED). Combined with the 1B and the
  4B-mm smoke, the **nvfp4 KV read-path defect is model/size-independent on
  sm_120**, and **270M IS a valid fast minimal repro for it** (~90 s to serve,
  ~10 s/PPL, 0.5 GB).
- **FI-bf16 does NOT reproduce**: +0.00133 nats vs FLASH_ATTN (the 1B was
  +0.221, ~165× larger) — effectively clean. So the **bf16 SWA-512 inflation arm
  is NOT a pure function of the d256/SWA-512/1-kv-head geometry** (270M has the
  geometry and stays clean); it scales with width/depth or a 1B-specific path.
  **270M is NOT a minimal repro for the bf16-inflation arm** — use the 1B for it.
- No engine WEDGE at 270M either: all C1 ctx-8191 (window-crossing) requests
  served fine on all three backends, where the 1B wheel run wedged. The wedge
  arm, like the bf16 inflation, did not reproduce at 270M.

Artifacts: results/p520_g3_270m_20260612/ (SUMMARY.md, per-row proof_lines,
chat_smoke.json, c1/c1b_ctx8191_ppl.json ×2, hf_bf16_reference_ppl.json,
util-0.3 OOM crash excerpt, status.txt, scripts).

## 1B rigorous re-test 2026-06-12 (P520 / sm_120): bf16 defect REFUTED — environmental/false-green ARTIFACT

Task #37 decisive re-run. The old 1B numbers were suspect on two counts: (i) one
earlier run used the `VLLM_ATTENTION_BACKEND` false-green trap that silently ran
FLASH_ATTN while claiming FlashInfer; (ii) the 270M agent established that
`--gpu-memory-utilization 0.3` OOMs the FlashInfer rows on this WDDM-shared 5060
Ti (a capacity ValueError mistakable for a wedge). Re-ran cleanly at util 0.6
with FlashInfer **verified-engaged from the engine proof line**, serve-ready
timeout 300 s as an explicit wedge detector.

Same **sm120a RELEASE wheel `g6adc00f70`** + FlashInfer source-JIT `7d5d477b`
(the exact stack of the task #37 wheel-disambig that reported the wedge, and of
the 270M test). Model google/gemma-3-1b-it. Corpus C1 md5 abb63f0e... (verified),
ctx 8191, 8190 scored, each row run TWICE bitwise. HF transformers bf16 **eager**
C1 ground truth on the identical token window = **2.359755277633667 nats**.

| row | backend (engine PROOF line) | kv dtype | C1 ×2 (bitwise) | delta vs FLASH_ATTN | wedge | smoke |
|---|---|---|---|---:|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN (FA v2) | bf16 | 2.3578483823599337 (IDENTICAL) | — (−0.00191 vs HF) | no | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.3571850630239095 (IDENTICAL) | **−0.0006633** | **no** | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 3.9452781399784085 (IDENTICAL) | **+1.5874298** | no | **GIBBERISH, deterministic** |

FI-bf16 forced with `--attention-backend flashinfer` AND
`VLLM_FLASHINFER_BF16_GEMMA=1`; engine proof line confirms
`Using AttentionBackendEnum.FLASHINFER backend.` with `attention_backend:
'flashinfer'` in non-default args — **no false-green FLASH_ATTN substitution**.

**Verdict: the bf16 1B FlashInfer defect is an ENVIRONMENTAL / FALSE-GREEN
ARTIFACT — it does NOT reproduce.**
- **bf16 inflation REFUTED:** FI-bf16 − FLASH_ATTN = −0.00066 nats (the old
  suspect was +0.221, ~330× larger and the *opposite sign*). Both bf16 backends
  match HF eager truth to <3e-3. This is consistent with the Spark sm_121 bisect
  (+0.00279, clean) and the 270M re-test (+0.00133, clean): bf16 FlashInfer on
  Gemma 3 1B is clean everywhere once FlashInfer is verified-engaged at a
  non-OOM util.
- **wedge REFUTED:** no engine wedge on any window-crossing (>512 tok) ctx-8191
  `prompt_logprobs` request; all three rows served, ready in 146–149 s, never
  approaching the 300 s timeout. The earlier wheel-disambig "wedge" (a freeze
  after a `_compute_slot_mapping_kernel` Triton JIT, at util 0.3) was the
  artifact — at util 0.6 the slot-mapping path compiles and serves fine.
- **nvfp4 read-path defect (B) REPRODUCES:** deterministic gibberish, +1.587
  nats vs truth — broad sm_120 defect, unchanged. (Matches the original 1B nvfp4
  +1.592 and corroborates 270M/4B-mm.)

**Reconciliation of all 1B arms:** the only real defect on Gemma 3 1B is the
nvfp4 KV read path (sm_120-specific, broad across sizes). The "bf16 inflation"
and "wedge" were methodology artifacts (false-green fallback / util-0.3 OOM &
slot-mapping JIT), now retired. **Gemma 3 retirement flip: NO 1B bf16 caveat
needed; the nvfp4-KV caveat stands.** Task #37 narrows to localizing defect (B)
only (use 270M as the fast minimal repro).

Artifacts: results/p520_g3_1b_retest_20260612/ (SUMMARY.md, per-row
proof_lines.txt / server.log / chat_smoke.json / c1{,b}_ctx8191_ppl.json ×2,
hf_bf16_reference_ppl.json, status.txt, scripts/ incl. run_row.sh &
hf_bf16_reference.py).

## Cross-references

- results/p520_g3_1b_retest_20260612/ (P520 sm_120 1B RIGOROUS re-test — bf16 defect REFUTED/artifact, nvfp4 reproduces)
- results/p520_g3_270m_20260612/ (P520 sm_120 270M repro — SPLIT verdict: nvfp4 reproduces, FI-bf16 clean)
- results/p520_1b_wheel_disambig_20260612/ (P520 sm_120 WHEEL disambig — wedge verdict)
- results/p520_mm_retirement_smokes_20260612/ (g3-4b nvfp4 mm gibberish corroboration)
- results/claude_1b_bug_bisect_20260612/ (Spark sm_121 bisect — PLATFORM verdict)
- results/p520_gemma3_1b_serving_20260612/ (full artifacts incl. token
  dumps on the P520 side: B:\workshop\wsl_sm120\results\gemma3_1b_serving_20260612\)
- docs/RESULTS_LEDGER.md row (2026-06-12); mail/0044, mail/0056.
- docs/TRITON_RETIREMENT_SCORECARD.md adjudication log (flip scoping).

<!-- source: docs/CAMPAIGN_LOG.md -->

# Campaign Log

## 2026-06-12

- Task #38 (long-context RETRIEVAL: NVFP4 KV vs fp8 vs bf16 on Gemma 4) —
  harness READY, awaiting a Spark window. Backs Jetha's public promise
  ("i've measured perplexity, not retrieval — needle-in-a-haystack test is
  next"). Motivated by the per-token stratification finding that NVFP4's prose
  error grows mildly with position (H-late), the signature that could nip deep
  retrieval.
  - Harness: `scripts/vllm_needle_retrieval.py` (schema
    `vllm-needle-retrieval/v1`) — needle-in-a-haystack + RULER-style multi-needle
    probe through the vLLM OpenAI chat endpoint. Deterministic filler, controllable
    DEPTH (context fraction) and CONTEXT LENGTH, temperature-0 exact-code scoring,
    per-(length, depth) accuracy grid, fp8 boot-profile note field,
    `spark_hardware` provenance, `--self-test` (21/21 green offline).
  - Runner: `scripts/run_needle_retrieval_spark.sh` — STAGED, not executed.
    Three sequential servers (bf16 VOSPLIT / fp8 / nvfp4 VOSPLIT+LINEAR_V_SF) on
    the r10 baked image, `google/gemma-4-E4B-it` first, `gemma-4-31B-it` as a
    documented stretch (`RUN_31B=1`). Marker protocol + memory guardrails
    (`--memory 100g`) + per-row double-run determinism gate, matching the
    overnight/corpus-sweep runners.
  - Plan + hypothesis + decisive cells: `docs/RETRIEVAL_EVAL_PLAN.md`. Null =
    "retrieval holds, strong claim"; positive = "the real depth limit of the
    capacity win". Self-test banked at
    `results/claude_needle_retrieval_selftest_20260612/self_test.json`.

## 2026-06-07

- Created this public record repository.
- Imported the first Gemma 4 benchmark reports.
- Added the initial diagnosis and solution plan.
- Added first-pass tools:
  - `scripts/spark_doctor.py`
  - `scripts/gguf_logprobs_probe.py`
  - `scripts/openai_chat_smoke.py`
- Added SGLang as a first-class runtime workstream after reviewing `hikarioyama/sglang-nvfp4-kv-sm120`.
- Added practical llama.cpp serving as a first-class runtime workstream and LiteRT-LM as optional side-runtime coverage.
- Added the upstream fork/submodule/worktree policy for patched libraries.
- Added before/after GB10 benchmark protocol for measuring campaign impact.
- Added NVFP4 dependency map from the FlashInfer/vLLM/SGLang subagent investigation.
- Added a remediation matrix clarifying that the FlashInfer SM121 `mm_fp4` patch is only one workstream; remaining work includes packaging, vLLM/SGLang integration, Gemma 4 12B, NVFP4 KV, llama.cpp/lm-eval, optional LiteRT-LM, HF fallback, observability, benchmark proof, and upstream coordination.
- Added a compact OpenAI-compatible serving benchmark harness and captured the first vLLM before row.
- Added root-readable runtime process probe evidence for the live vLLM server.
- Added `scripts/spark_smoke_suite.py` as the compact before/after orchestrator. Core tracks are vLLM, SGLang, llama.cpp, HF fallback telemetry, MTP/spec decode, and NVFP4; LiteRT-LM is opt-in.
- Reproduced the llama.cpp GGUF logprobs incompatibility with a temporary `llama-server` on port `18081`; server was stopped after the probe.
- Captured runtime availability matrix showing vLLM/FlashInfer/PyTorch present, SGLang and LiteRT-LM absent, Docker available, and llama.cpp built but not on `PATH`.
- Started tracking work through GitHub Issues.
- Ran `spark_doctor` on `thinkstationpgx-00b4` using `/home/jethac/gemma4-evals/.venv/bin/python`.
  - GPU: `NVIDIA GB10`
  - compute capability: `12.1` / `sm_121`
  - host: `aarch64`
  - driver: `580.159.03`
  - CUDA runtime reported by `nvidia-smi`: `13.0`
  - `nvcc`: CUDA `13.0`, found through `/usr/local/cuda`
  - `cuobjdump`: CUDA `13.0`, found through `/usr/local/cuda`
  - PyTorch: `2.11.0+cu130`
  - vLLM: `0.22.1`
  - FlashInfer: `0.6.11.post2`
  - PyTorch arch list: `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`; no explicit `sm_121`
  - snapshot: `results/spark_doctor_20260607T110833Z.md`
- Ran `cuda_so_audit` against vLLM and FlashInfer package roots in the benchmark venv.
  - inspected objects: 14
  - objects with explicit `sm_121`: 0
  - objects with `sm_120`: 3
  - vLLM FA2 extension: `sm_80`
  - vLLM FA3 extension: `sm_90a`
  - vLLM FlashMLA extensions: `sm_100`, `sm_90a`
  - snapshot: `results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json`
- Proved llama.cpp `b9536` as a practical Gemma 4 26B Q4_0 serving path with `--reasoning off`.
- Evaluated LiteRT-LM `0.13.1` on Linux `aarch64`: CPU chat works for Gemma 4 E2B, CPU/GPU benchmark commands work, GPU chat still exits `returncode=-11` after producing `spark-ok`.
- Ran SGLang 26.05 model-shaped FlashInfer `mm_fp4` proxy microbenchmarks against installed and patched source/JIT FlashInfer.
  - patched source selected `b12x`, `cutlass`, `cudnn` and built an SM121a-targeted FP4 GEMM path.
  - patched dense-decode proxies were mixed; patched MoE-shaped proxies were slower on all tested shapes.
  - conclusion: the FlashInfer SM121 predicate patch remains dispatch enablement, not a proven speedup.
- Served Gemma 4 12B through a vLLM source/precompiled probe at upstream commit `da1daf40` plus Transformers main after removing stale FlashInfer JIT-cache files.
  - the run proved `Gemma4UnifiedForConditionalGeneration` can start on GB10, but compact decode was only about 7.7 tok/s and vLLM forced `TRITON_ATTN`.
  - conclusion: this is a compatibility and packaging proof, not a clean blessed container or performance win.
- Audited hikarioyama's vLLM and SGLang NVFP4-KV SM120 reference repos.
  - vLLM reference HEAD: `f6156ee3b22b24885a52c02bdafb34a9c201fe86`.
  - SGLang reference HEAD: `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`.
  - conclusion: build on them as prior art through `jethac` forks, but do not vendor overlays or call them Spark validation until GB10 `sm_121` fp8-vs-NVFP4 proof exists.
  - priority shift: vLLM NVFP4 KV is a capacity/concurrency lane first. Measure KV pool tokens, maximum concurrency, quality, and hidden allocations before chasing decode tok/s.
- Tightened the smoke suite:
  - `run_with_telemetry.py` now captures `pre_memory` before launching the child command.
  - `spark_smoke_suite.py` wraps MTP/spec-decode commands with telemetry and supports `--mtp-model`.
- Created `jethac/vllm` and `jethac/sglang` forks, added them as submodules, and pushed issue-named worktree branches for NVFP4 KV work.
  - vLLM branch: `spark/hijinks-007-nvfp4-kv-sm121` at `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`.
  - SGLang branch: `spark/hijinks-018-fp4-e2m1-kv-sm121` at `02be2e71899491b7aaf2849dce6431f61fc190b6`.
- Created the FlashInfer FA2 NVFP4 KV branch `spark/hijinks-007-fa2-nvfp4-kv-sm121` from `a42c8f0751c70a2f69596f063170e284710c94ac`, so the KV lane inherits the earlier SM121 `mm_fp4` dispatch and `121a` JIT-cache work.
- Recorded the NVFP4 KV porting map from the vLLM and SGLang subagent audits:
  - build on hikarioyama's SM120 work as prior art, but re-derive minimal upstream-shaped patches in `jethac` forks.
  - keep FlashInfer kernel/page/stride changes in FlashInfer, vLLM routing/tensor plumbing in vLLM, and SGLang memory-pool/calibration/backend-wrapper changes in SGLang.
  - first port scope is symmetric NVFP4 B2 only; no hidden V scale-factor scratch cache and no mixed K/V experiments until GB10 proof exists.
- Pushed FlashInfer FA2 NVFP4 KV patch branch `spark/hijinks-007-fa2-nvfp4-kv-sm121` at `e152cf4da4ab2a9d093b7d9d4b499198b0211c61`.
  - added explicit scale-factor stride plumbing for `maybe_k_cache_sf` and `maybe_v_cache_sf`.
  - added independent K/V page strides and V offset helpers in `paged_kv_t`.
  - updated FA2 prefill and persistent attention to consume explicit scale-factor strides.
  - added compile-time gated vLLM B2 V scale-factor de-swizzle while preserving symmetric-linear V scale-factor behavior by default for SGLang/reference compatibility.
  - verification: Python syntax compile and `git diff --check` passed.
  - limitation: targeted pytest collection is blocked in this Windows workspace by missing `tvm_ffi`; no clean GB10 build, harness proof, or serving proof yet.
- Pushed vLLM SM12x NVFP4 KV routing patch branch `spark/hijinks-007-nvfp4-kv-sm121` at `2c1405dd129d873d268b8baea78c5739cd384951`.
  - routes SM12x `--kv-cache-dtype nvfp4` through FlashInfer FA2 instead of `trtllm-gen`.
  - keeps SM100 NVFP4 on the existing TRTLLM path.
  - uses model dtype query/output on the SM12x FA2 path instead of the TRTLLM FP8-query/FP8-output workaround.
  - adds a one-time runtime log for FA2 NVFP4 KV selection.
  - adds wrapper-routing regression coverage and updates vLLM's attention backend design doc.
  - verification: Python syntax compile and staged `git diff --check` passed.
  - limitations: local `ruff` is unavailable and pytest collection is blocked by missing `tblib`; no clean GB10 vLLM plus FlashInfer build or serving proof yet.
- Ran a GB10 source-file routing probe for `jethac/vllm@2c1405dd129d873d268b8baea78c5739cd384951`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T165144Z.json`
  - host/GPU: `thinkstationpgx-00b4`, `NVIDIA GB10`, capability `[12, 1]`
  - installed dependency context: vLLM `0.22.1`, Torch `2.11.0+cu130`, CUDA `13.0`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`; SM100-style NVFP4 still selects `trtllm-gen`; non-NVFP4 still selects `auto`
  - limitation: this loads the forked `flashinfer.py` source file against installed compiled dependencies. It does not install the full fork, build FlashInfer kernels, start a server, or prove correctness/capacity/performance.
- Pushed vLLM V-scale-factor deswizzle follow-up at `8916796bc50926fd61e606718b194a71e2e31a24`.
  - reason: vLLM's native NVFP4 cache writer stores V scale factors in a swizzled layout for the old SM100 TRTLLM path, so the SM12x FA2 path must enable FlashInfer's in-kernel V-SF deswizzle variant.
  - scope: still keyed only on `kv_cache_dtype == "nvfp4"` and the SM12x consumer-Blackwell family helper; fp8/auto routing remains unchanged.
  - family note: vLLM reporting GB10 as capability family `120` is correct for FA2 NVFP4 KV routing, but native FP4/MXFP4 MMA work still needs Spark-appropriate `sm_121a` or validated compatible targets.
- Ran the scripted GB10 routing/deswizzle probe for `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`, SM100-style NVFP4 still selects `trtllm-gen`, non-NVFP4 still selects `auto`, and the deswizzle helper sets `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`.
  - limitation: this is still routing/JIT-flag evidence only. It does not replace hikari-style NHD/HND cosine checks or an end-to-end serving proof.
- Pushed SGLang SM12x FP4 KV gate patch branch `spark/hijinks-018-fp4-e2m1-kv-sm121` at `67c7967a1c1b6145a8c9d26a7b941258735ebd8d`.
  - allows FlashInfer MHA in `fp4_e2m1` KV compatibility gates only when SGLang's `is_sm120_supported()` helper is true.
  - allows `NVFP4KVQuantizeUtil.quantize()` on SM120-family devices and routes SM100/SM120 through `flashinfer.nvfp4_kv_quantize`.
  - adds server-args unit coverage for SM12x FlashInfer MHA KV4 gates.
  - verification: Python syntax compile and `git diff --check` passed.
  - limitations: local `ruff` is unavailable and pytest collection is blocked on Windows by missing POSIX `resource`; native FP4 KV memory-pool/backend wrapper work and GB10 serving proof are still pending.
- Ran a Linux/aarch64 verification pass on the GB10 host for `jethac/sglang@67c7967a1913960055e64c49c26c5f622c1f1ff1`.
  - result: `results/sglang_fp4_kv_sm121_pgx_verify_20260608T0205JST.md`
  - outcome: Linux/aarch64 branch fetch and detached worktree checkout passed; `python3 -m py_compile` passed for the touched SGLang files.
  - limitation: targeted `KV4Compatibility` pytest was not run because the host has no `python` shim and `python3` does not have `pytest` installed.
- Added SM-count awareness to diagnostics and benchmark harnesses.
  - new helper: `scripts/spark_hardware.py`
  - updated scripts: `spark_doctor.py`, `run_with_telemetry.py`, `openai_serving_benchmark.py`, `spark_smoke_suite.py`
  - new evidence: `results/spark_doctor_smcount_20260607T172142Z.md`
  - current GB10 comparison key: `NVIDIA_GB10:sm_121:sms_48`
  - audit doc: `docs/SM_COUNT_AWARENESS.md`
  - conclusion: current fork patches do not hardcode a 48-SM performance heuristic; future performance rows must still match `multi_processor_count`.
- Added and ran a standalone FlashInfer FA2 NVFP4 paged-KV correctness probe.
  - new script: `scripts/flashinfer_nvfp4_kv_probe.py`
  - result: `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json`
  - source: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
  - env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
  - hardware key: `NVIDIA_GB10:sm_121:sms_48`
  - outcome: NHD decode, NHD prefill, HND decode, and HND prefill all passed with cosine >= `0.99999946`.
  - limitation: this proves the patched FlashInfer FA2 kernel path and vLLM-style V-scale-factor de-swizzle, not clean vLLM/SGLang serving, KV capacity, quality, or throughput.
- Tightened llama.cpp wording.
  - Gemma 4 26B Q4_0 remains the blessed practical GGUF serving path.
  - The Q4_0 result does not prove native NVFP4/MXFP4 `sm_121a` tensor-core MMA dispatch; that needs a separate native-FP4 GGUF experiment.
- Tried a CPU-only Linux `aarch64` Docker verification route for the SGLang FP4 KV branch to avoid spending GPU time on Python-level gate tests.
  - result: `results/sglang_fp4_kv_sm121_cpu_docker_verify_20260608T0243JST.md`
  - target: `jethac/sglang@67c7967a1913960055e64c49c26c5f622c1f1ff1`
  - outcome: build failed before pytest while compiling `sglang-kernel-cpu`; the ARM64 `vaddq_f16` path hit a target-specific option mismatch.
  - conclusion: this does not disprove the SGLang FP4 KV gate patch, but the cheap CPU Docker verification route needs either a no-kernel pytest image or an ARM64 CPU-kernel build-flag fix.
- Fixed and validated the SGLang FP4 KV gate test setup.
  - fork commit: `jethac/sglang@eefe8aded`
  - result: `results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`
  - outcome: targeted Linux `aarch64` `KV4Compatibility` pytest passed: `3 passed, 56 deselected`.
  - limitation: this is Python-level argument compatibility only; native FP4 KV pools, backend wrapper plumbing, quality, capacity, and serving remain pending.
- Added `scripts/cuda_build_target_audit.py` for build/JIT log target evidence before `.so` inspection.
  - first smoke artifact: `results/llamacpp_gemma4_26b_q4_0_build_target_audit_20260608T0325JST.json`
  - result: the existing llama.cpp server log contains accepted Spark target evidence through `CUDA : ARCHS = 1210`.
- Ran Gemma 4 26B-shaped FlashInfer FA2 NVFP4 KV probes.
  - sliding/local artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_sliding_1024_20260608T0340JST.json`
  - global/full artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json`
  - outcome: sliding/local shape `H_q=16`, `H_kv=8`, `D=256`, `page=16` passed NHD/HND decode and prefill at `kv_len=1024`, `qo_len=128`.
  - blocker: global/full shape `H_q=16`, `H_kv=2`, `D=512`, `page=16` failed all NHD/HND decode/prefill operations with FlashInfer FA2 invalid configuration from `prefill.cuh:3215`.
  - conclusion: Gemma 4 26B NVFP4 KV cannot be called ready for serving until the global-attention path is fixed or routed to a proven fallback.
- Promoted Qwen to a first-class speed/capacity benchmark lane.
  - issue: https://github.com/jethac/dgx-spark-hijinks/issues/20
  - new doc: `docs/QWEN_ON_DGX_SPARK.md`
  - rationale: Qwen is the cleaner path for NVFP4 weights, fp8-vs-NVFP4 KV capacity, and DFlash measurement; Gemma remains the harder model-family compatibility target.
  - AEON prior art: Qwen3.6 NVFP4+DFlash and Gemma 4 NVFP4-weight recipes are useful external GB10 evidence, but they do not prove our FA2 NVFP4-KV fork or SGLang `fp4_e2m1` KV.
- Pushed vLLM Qwen/DFlash SM12x stability branch.
  - fork commit: `jethac/vllm@0667185d5adaec32ff8cc8289a4d7716f6cdf966`
  - branch: `spark/hijinks-020-aeon-qwen-dflash-sm121a`
  - changes: guarded lazy fallback import for `_C_stable_libtorch`; speculative-decode CUDA graph capture-size alignment now applies to every non-`NONE` graph mode, including pure `PIECEWISE`.
  - verification artifact: `results/vllm_qwen_dflash_sm121a_patch_verify_20260608T0330JST.md`
  - limitation: local pytest collection is blocked by missing vLLM dev dependencies; GB10 Qwen3.6 NVFP4+DFlash serving reproduction remains pending.
- Captured SGLang Qwen2.5 1.5B fp8-vs-fp4 KV evidence.
  - summary: `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`
  - fp8 row: NVIDIA SGLang 26.05 serves `Qwen/Qwen2.5-1.5B-Instruct` with FlashInfer attention, CUDA graphs enabled, hardware key `NVIDIA_GB10:sm_121:sms_48`, `3,113,713` KV tokens, and about `58-59 tok/s` decode.
  - stock fp4 FlashInfer row: fails at `KV4Compatibility`, rejecting FlashInfer for MHA FP4 KV.
  - stock fp4 Triton row: allocates `5,534,509` KV tokens, about `1.78x` fp8 capacity, then fails on missing `KVFP4QuantizeUtil`.
  - conclusion: fp8 is now a real SGLang Qwen comparator; stock fp4 KV is not a serving path yet, and the next after-row should use the `jethac/sglang` fork.
- Captured the matched SGLang Qwen BF16/auto comparator at `mem_fraction_static=0.40`.
  - artifacts: `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_openai_benchmark.json` and server log.
  - result: BF16/auto KV allocated `1,557,709` tokens and decoded `58.89`, `58.59`, and `57.73 tok/s` across the standard short, medium, and long-prefill cases.
  - conclusion: fp8 roughly doubles KV pool tokens over BF16/auto at matched memory fraction without materially changing decode speed on this small Qwen row.
- Pushed the SGLang FP4 KV alias fix and ran patched overlay serving attempts.
  - fork commit: `jethac/sglang@98ad46961`.
  - change: `KVFP4QuantizeUtil` is now an alias of `BlockFP4KVQuantizeUtil`, matching the historical import used by the MHA KV memory-pool path.
  - FlashInfer attention overlay: clears the stock compatibility/import failures, allocates `5,539,718` FP4 KV tokens, targets `compute_121a,code=sm_121a`, then fails compiling FlashInfer FP4 decode at `vec_dtypes.cuh(117)`.
  - Triton attention overlay: normal graph capture stalls; disabling only standard CUDA graphs still enters piecewise graph capture and stalls; disabling both graph modes serves.
  - no-graphs Triton FP4 KV result: `5,541,103` KV tokens, smoke passes, but `short_decode` is only `0.276 tok/s` with repetitive output.
  - conclusion: the FP4 KV capacity path is real, but SGLang FP4 KV remains unblessed until the clean fork/dependency stack serves with graphs and acceptable quality.
- Added `scripts/record_openai_serving_row.py`.
  - purpose: capture smoke, benchmark, optional runtime/CUDA audits, and a manifest for one already-running OpenAI-compatible server.
  - verification: `python -m py_compile` passed and `--dry-run` emits relative artifact paths and portable command records.
- Captured llama.cpp Qwen2.5 1.5B Q4_K_M serving evidence.
  - model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
  - binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`, build `308f61c31 (9536)`.
  - artifact prefix: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST`.
  - server log: `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, `BLACKWELL_NATIVE_FP4 = 1`.
  - compact OpenAI serving decode: `175.19`, `174.86`, and `166.66 tok/s` on short, medium, and long-prefill cases.
  - `llama-bench`: `pp512 12505.79 +/- 615.87 tok/s`, `tg128 178.10 +/- 0.95 tok/s`.
  - logprobs probe: still not lm-eval compatible; response exposes `logprobs.content` but not `tokens` and `token_logprobs`.
  - conclusion: practical Qwen GGUF serving is proven for llama.cpp on GB10; GGUF accuracy and native FP4/MXFP4 dispatch remain separate workstreams.
- Added the AEON vLLM reproduction runner and preflight.
  - new script: `scripts/run_aeon_vllm_reproduction.sh`
  - targets: `gemma26-dflash` and `qwen36-dflash`
  - preflight artifact: `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`
  - outcome: both AEON GHCR images resolve; Gemma/Qwen target and drafter HF repos are public and non-gated from the GB10 host.
  - limitation: no large model download or vLLM serving row has been started yet; this is readiness evidence for the next benchmark, not a performance result.
- Reproduced the AEON Gemma 4 26B NVFP4+DFlash vLLM row locally.
  - run ID: `aeon_gemma26_dflash_20260608T0436JST`
  - image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
  - model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
  - drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
  - artifact summary: `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`
  - warmed compact row: `47.91`, `53.60`, and `98.38 tok/s` across short, medium, and long-prefill cases.
  - backend evidence: `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, target `TRITON_ATTN`, drafter `FLASH_ATTN`, CUDA graphs, DFlash.
  - caveats: not a `jethac` fork speedup claim; PyTorch arch list reports `sm_120` but not explicit `sm_121`; server log warns about differing NVFP4 global scales across fused parallel layers; accuracy remains separate.
- Attempted the AEON Qwen3.6 NVFP4+DFlash vLLM reproduction.
  - run ID: `aeon_qwen36_dflash_20260608T0501JST`
  - model downloaded: `/home/jethac/models/aeon/qwen36-nvfp4`, about `22G`
  - drafter downloaded: `/home/jethac/models/aeon/qwen36-dflash`, about `905M`
  - blocker: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` did not finish/register after the initial pull or a bounded `timeout 900` retry.
  - artifact: `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`
  - interpretation: this is a container acquisition blocker, not yet a Qwen model-load, runtime, or kernel failure.
- Promoted the AEON Qwen vLLM runner default to the current `v2` image and recorded a stop point.
  - runner default: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
  - artifact: `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`
  - blocker: the `v2` pull had been started and partially observed, but the GB10 host later stopped answering SSH and ping before final pull status could be inspected.
  - interpretation: Qwen speed remains a required benchmark lane alongside Gemma; this stop point is an acquisition/reachability failure, not a Qwen model/runtime/kernel result.
- Advanced the vLLM Qwen/DFlash branch to cover the remaining AEON source patches that apply to the current fork.
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - artifact: `results/vllm_aeon_qwen_patch_port_20260608T0619JST.md`
  - changes: Qwen3.5/3.6 text registry entries, hybrid KV `block_size=None` safety, Mamba block-size fallback, and text-only M-RoPE fallback, on top of the existing lazy import and CUDA graph alignment fixes.
  - validation: Python compile and `git diff --check` passed; targeted pytest is blocked by missing local vLLM test dependency `tblib`.
  - interpretation: the fork is closer to the AEON Qwen runtime shape, but it is still not a Qwen speed result. The required next benchmark is still AEON Qwen36 NVFP4+DFlash serving, then a matched `jethac` fork row.
- Extended the llama.cpp native loglikelihood probe toward a GGUF lm-eval adapter.
  - script: `scripts/llamacpp_native_loglikelihood_probe.py`
  - artifact: `results/llamacpp_native_loglikelihood_probe_v2_selftest_20260608.json`
  - changes: explicit `--pair CONTEXT|||CONTINUATION` scoring, continuation-token logprob summation, greedy-match metadata, and an `lm_eval_loglikelihood_tuple` field.
  - validation: Python compile and self-test passed.
  - interpretation: this is adapter-shape readiness only. GGUF accuracy is still blocked until the native probe passes against a live llama-server and a tiny lm-eval task.
- Added a tiny llama.cpp native loglikelihood task harness.
  - script: `scripts/llamacpp_native_loglikelihood_task.py`
  - task file: `tasks/llamacpp_loglikelihood_smoke.jsonl`
  - dry-run artifact: `results/llamacpp_native_loglikelihood_task_dryrun_20260608.json`
  - validation: Python compile, task dry-run, no-server failure path, and `git diff --check` passed.
  - interpretation: this makes the next live GGUF accuracy proof one command after starting llama-server; it still does not prove paper-comparable GGUF accuracy until run against the server.
- Ran the llama.cpp native loglikelihood task against a live Qwen2.5 1.5B Q4_K_M llama-server.
  - summary: `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md`
  - server evidence: `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`.
  - result: likely continuations were scored, but the unlikely `zebra` continuation was missing from top-512 probabilities; task `ok=false`.
  - interpretation: the OpenAI schema blocker has a live native-endpoint follow-up, but the native top-N path still does not satisfy arbitrary continuation-token scoring for lm-eval.
- Added the public compatibility board.
  - doc: `docs/COMPATIBILITY_BOARD.md`
  - purpose: recurring runtime/status view covering vLLM, SGLang, llama.cpp, FlashInfer, LiteRT-LM, HF fallback, model lanes, live proof queue, and update cadence.
  - interpretation: this closes the documentation gap where issues existed but there was no single compatibility board; maintaining it remains part of every future row update.
- Added the wheel/container matrix.
  - doc: `docs/WHEEL_CONTAINER_MATRIX.md`
  - purpose: map proven, partial, blocked, debug-only, and side-runtime install/container paths to exact artifact evidence.
  - interpretation: the matrix moves solution area 2 from scattered notes to an explicit acceptance-evidence index, while still marking the clean Spark-blessed vLLM/SGLang/FlashInfer package set as missing.
- Tightened the benchmark protocol so Qwen speed/capacity is mandatory alongside Gemma for broad runtime claims.
  - Qwen is the clean first lane for SM121a throughput, speculative decode, NVFP4 weights, and fp8-vs-NVFP4 KV capacity.
  - Gemma remains required for the original workload and harder model-family compatibility path.
  - llama.cpp Qwen2.5 GGUF evidence is now recorded in the llama.cpp serving recipe instead of being left as a stale template-only section.
- Added a Qwen speed-lane runner.
  - script: `scripts/qwen_speed_lane.py`
  - sample rows: `tasks/qwen_speed_lane_sample.jsonl`
  - dry-run artifact: `results/qwen_speed_lane_dryrun_20260608_summary.json`
  - purpose: record already-running vLLM, SGLang, and llama.cpp Qwen servers through the shared `record_openai_serving_row.py` manifest path.
  - interpretation: this does not replace the missing vLLM Qwen36 live result; it makes the next live Qwen campaign repeatable once the host and image acquisition are healthy.
- Added the AEON prior-art port map.
  - doc: `docs/AEON_PRIOR_ART_PORT_MAP.md`
  - script: `scripts/aeon_prior_art_audit.py`
  - audit artifact: `results/aeon_prior_art_audit_20260608.json`
  - sources: local `CODEX_REPORT_AEON7.md`, AEON Qwen patch directory, AEON Gemma repo, AEON `vllm-dflash`, and current `jethac` forks.
  - conclusion: literal AEON patches are vLLM internals and are already represented in `jethac/vllm@6804e1b`; SGLang and llama.cpp need counterpart experiments, not blind vLLM patch copies.
  - SGLang nuance: current `jethac/sglang@98ad46961` already has DFlash-specific surfaces, so the next useful step is a serving proof after ordinary Qwen/Gemma rows are stable.
- Added a mechanical solution coverage audit.
  - script: `scripts/solution_coverage_audit.py`
  - audit artifact: `results/solution_coverage_audit_20260608.json`
  - status doc change: `docs/SOLUTIONS_STATUS.md` now preserves every numbered solution-plan row through `15` instead of folding benchmark design, observability, upstream coordination, forks, and recipes into unrelated row numbers.
  - issue tracker change: `docs/ISSUE_TRACKER.md` now records solution-plan IDs separately from GitHub issue numbers.
  - Qwen rule: Qwen speed/capacity is a required cross-cutting benchmark lane alongside Gemma, not a substitute for any numbered solution area and not optional for broad runtime claims.
- Added a serving-manifest claim-readiness audit.
  - script: `scripts/serving_manifest_audit.py`
  - audit artifact: `results/serving_manifest_audit_20260608.json`
  - converted the llama.cpp Qwen2.5 Q4_K_M row manifest from a manual note into strict JSON.
  - result: the llama.cpp Qwen row is now claim-ready for practical serving evidence; the AEON Gemma vLLM row remains not claim-ready in the strict audit because its build-target audit lacks accepted Spark target evidence.
  - dry-run Qwen lane manifests are explicitly classified as planning evidence only.
- Added a container target audit to separate family/PTX evidence from native Spark target proof.
  - script: `scripts/container_target_audit.py`
  - AEON Gemma artifact: `results/aeon_gemma26_dflash_20260608T0436JST_container_target_audit.json`
  - result: the AEON Gemma image/container has GB10 runtime evidence and SM120-family/PTX evidence through `TORCH_CUDA_ARCH_LIST=... 12.0+PTX` / `sm_120`, but no explicit native `sm_121` or `sm_121a` target evidence.
  - serving-manifest audit now records this as `family_or_ptx_count: 1` for Gemma while keeping strict Gemma claim readiness false.
- Added an NVFP4 checkpoint metadata audit before larger Qwen/Gemma speed work.
  - script: `scripts/nvfp4_checkpoint_audit.py`
  - sample fixture: `tasks/nvfp4_checkpoint_audit_sample`
  - sample artifact: `results/nvfp4_checkpoint_audit_sample_20260608.json`
  - purpose: classify compressed-tensors versus ModelOpt NVFP4 markers, flag quantized router/vision/visual tensors, and check Gemma EOS/control-token metadata without loading tensor data or using GPU time.
  - interpretation: this does not prove a live Qwen or Gemma speed row; it prevents future NVFP4-weight serving or GGUF conversion rows from silently using a bad checkpoint format.
- Added a counterpart evidence audit for AEON-derived non-vLLM work.
  - script: `scripts/counterpart_evidence_audit.py`
  - artifact: `results/counterpart_evidence_audit_20260608.json`
  - result at creation time: all seven counterpart proof rows were still missing, partial, or blocked. The later `jethac/vllm` Qwen3.6+DFlash row moved the vLLM requirement to claim-ready for serving evidence; the SGLang and llama.cpp counterpart rows remain missing or partial.
  - interpretation: AEON source-port coverage is useful but does not satisfy the SGLang/llama.cpp counterpart acceptance tests.
- Added live task contracts for the missing counterpart rows.
  - task file: `tasks/counterpart_evidence_tasks.jsonl`
  - validator: `scripts/counterpart_task_matrix.py`
  - artifact: `results/counterpart_task_matrix_20260608.json`
  - result: all seven missing or partial counterpart requirements have concrete command templates and expected claim artifacts.
  - interpretation: this is runbook readiness, not live proof. The counterpart evidence audit remains the source of truth for whether those rows have actually landed.
- Reconnected to the GB10 host through Tailscale and captured fresh doctor evidence.
  - reachable host: `thinkstationpgx-00b4.tail740c8d.ts.net`
  - reachable IP: `100.113.98.11`
  - stale/unreachable LAN IP from this client: `192.168.68.112`
  - artifacts: `results/spark_doctor_tailnet_reconnect_20260608T074035JST.json`, `results/spark_doctor_tailnet_reconnect_20260608T074035JST.md`, `results/counterpart_task_matrix_tailnet_reconnect_20260608T074035JST.json`
  - result: live doctor confirms Linux `aarch64`, `NVIDIA GB10`, compute capability `12.1`, driver `580.159.03`, and CUDA `13.0`.
  - caveat: system Python has no Torch, so this reconnect artifact does not record SM count; use a runtime/container doctor for SM-count-backed benchmark rows.
- Ran the first local AEON Qwen3.6 NVFP4+DFlash vLLM attempt after reconnect.
  - run id: `aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST`
  - image/model state: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2` was present; target and drafter weights were present under `/home/jethac/models/aeon`.
  - checkpoint audit: `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`, `ok=true`, compressed-tensors NVFP4, `124306` safetensors keys, `0` quantized sensitive keys.
  - backend evidence: server log resolved `Qwen3_5MoeForConditionalGeneration` and `DFlashDraftModel`, selected `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, and `585168` KV tokens.
  - failure: row manifest is `ok=false`; chat smoke produced `message.reasoning` but no normal content and no `spark-ok`; benchmark recorded completion-token counts but no valid output text, so this is not a speed row.
  - interpretation: image acquisition and model startup are no longer the vLLM Qwen blocker; the next blocker is response/content validation plus native-target proof.
- Fixed the AEON Qwen3.6 vLLM response/content row by disabling Qwen thinking through OpenAI `chat_template_kwargs`.
  - direct probe: `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`
  - finding: baseline and prompt-level `/no_think` stayed in `message.reasoning`, while API-level `{"enable_thinking": false}` returned normal `message.content` for both `qwen36-fast` and `qwen36-deep`.
  - passing row: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `ok=true`
  - compact decode: `50.37 tok/s` short, `55.84 tok/s` medium, `53.75 tok/s` long-prefill.
  - caveat: this is AEON's container/checkpoint, not a `jethac` fork speedup; build-target audit still lacks explicit native `sm_121`/`sm_121a` evidence and the server still warns about Marlin weight-only FP4.
- Built and smoke-started the matched `jethac/vllm` derived image for Qwen3.6.
  - derived image: `jethac-vllm-aeon-q36:6804e1b81`
  - base image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - build artifact: `results/jethac_vllm_aeon_q36_6804e1b81_image_build_20260608T0845JST.log`
  - stop-point artifact: `results/jethac_qwen36_dflash_depstop_20260608T0850JST_summary.md`
  - result: image imports `vllm 0.1.dev1+g6804e1b81`, but serving exits before health because the AEON base environment lacks `compressed_tensors.compressors.pack_quantized`.
  - interpretation: fork parity is now blocked on dependency/API drift, not image build, Qwen weights, or `sm_121` kernels.
- Advanced the matched `jethac/vllm` Qwen3.6 row past the dependency stop point.
  - passing image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - summary artifact: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`
  - image layers: `compressed-tensors==0.17.0`, `humming-kernels[cu13]==0.1.4` plus `pyelftools`, and AEON's original FA2 binary restored after a PyTorch ABI mismatch.
  - backend evidence: `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, and `1,251,446` KV tokens.
  - compact decode: `47.22 tok/s` short, `58.88 tok/s` medium, `61.62 tok/s` long-prefill.
  - interpretation: this is a passing fork-derived vLLM Qwen serving row, but not clean fork packaging and not native `sm_121a` target proof because it still depends on AEON's FA2 binary and only host-side audits were captured.
- Added vLLM clean-packaging hook and in-container target/JIT audit tooling.
  - fork commit: `jethac/vllm@db4b210c1`
  - new vLLM env knob: `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1`, which skips extracting bundled FA2/FA3 extensions from a precompiled wheel while preserving the rest of the precompiled extension set
  - campaign scripts: `scripts/run_vllm_incontainer_target_audit.sh` and `scripts/cuda_artifact_arch_audit.py`
  - prior in-container audit interpretation: the passing `jethac/vllm@6804e1b` Qwen image has GB10 runtime evidence but no inspected `sm_121`/`sm_121a` CUDA object evidence, so the next vLLM image must replace AEON FA2 with an ABI-matched clean build and rerun the no-think Qwen row.
- Advanced the vLLM clean FA2 image build to the next blocker.
  - fork commit: `jethac/vllm@a919d635d`
  - main repo script: `scripts/build_vllm_aeon_qwen_cleanfa2_image.sh`
  - artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion_summary.md`
  - raw log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion.log`
  - result: `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` successfully skipped bundled FA2/FA3, and `VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d` fixed the previous `setuptools-scm` failure from missing `.git` metadata in the Docker context.
  - native-target finding: top-level vLLM CMake accepted `12.1a` and printed `arch=compute_121a,code=sm_121a`, but the nested pinned vLLM FlashAttention project reduced its supported target to `12.0`, selected `FA2_ARCHS: 8.0+PTX`, and compiled `_vllm_fa2_C` with only `sm_80`/`compute_80`.
  - interpretation: the clean vLLM packaging path is past dependency/versioning, but not past native FA2. The builder now fails fast if nested FA2 does not select SM121/SM121a; next work is patching or forking the pinned vLLM FlashAttention source.
- Added and tested the vLLM-pinned FlashAttention FA2 SM121a patch path.
  - fork branch: `jethac/flash-attention@spark/hijinks-021-fa2-sm121a`
  - fork commit: `7d53245`
  - submodule: `third_party/vllm-flash-attention`
  - issue: [#21](https://github.com/jethac/dgx-spark-hijinks/issues/21)
  - build artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_summary.md`
  - result: patched nested vLLM FlashAttention CMake selected `CUDA supported target architectures: 12.1a`, `FA2_ARCHS: 12.1a`, and `_vllm_fa2_C` invoked `nvcc` with `arch=compute_121a,code=sm_121a`.
  - failure: the build then stopped because the copied FlashAttention tree lacked its nested `csrc/cutlass` submodule, causing missing `cute/tensor.hpp` and `cutlass/numeric_types.h`.
  - interpretation: the architecture-selection blocker is fixed; the current blocker is packaging the nested CUTLASS dependency into the clean build context. The builder now initializes `csrc/cutlass` before creating the Docker context.
- Completed the clean vLLM FA2 SM121a build and in-container target audit.
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - main repo commit before build: `jethac/dgx-spark-hijinks@6b33492`
  - vLLM fork commit: `jethac/vllm@a919d635d`
  - FlashAttention fork commit: `jethac/flash-attention@7d53245`
  - build artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass_summary.md`
  - audit artifact: `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`
  - result: `_vllm_fa2_C.abi3.so` built, installed, imported, and `cuobjdump` showed `sm_121a` cubins in the patched FA2 extension.
  - runtime evidence: `NVIDIA GB10`, compute capability `[12, 1]`, `48` SMs, Torch `2.12.0.dev20260408+cu130`, FlashInfer `0.6.9rc1`, vLLM `0.1.dev1+ga919d635d`.
  - remaining caveat: this is native-target proof for the vLLM FlashAttention extension only. Other vLLM objects still carry their existing mixed prebuilt architecture surface, including `sm_120`, `sm_100`, and `sm_90a`.
  - next gate: rerun the no-think Qwen3.6+DFlash serving row on this clean FA2 image.
- Ran the no-think Qwen3.6+DFlash serving row on the clean vLLM FA2 SM121a image.
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - run id: `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST`
  - summary artifact: `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_summary.md`
  - manifest: `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_row_manifest.json`, `ok=true`.
  - smoke: normal OpenAI `message.content` returned `spark-ok` with `chat_template_kwargs={"enable_thinking": false}`.
  - compact decode: `61.07 tok/s` short, `56.97 tok/s` medium, `60.10 tok/s` long-prefill.
  - backend evidence: `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, FlashInfer FP4 GEMM autotune, CUDA graphs, and `1,241,920` KV tokens.
  - interpretation: this closes the clean vLLM Qwen packaging gap and pairs serving evidence with separate FA2 `sm_121a` cubin proof. It is not a speedup claim and it still does not prove native FP4 weight/MoE compute; the server warns that the selected weight path is Marlin weight-only FP4.
  - failed first recorder pass: `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_20260608T2359JST` hit a PowerShell-to-SSH JSON quoting bug for `chat_template_kwargs`, producing a recorder `JSONDecodeError`. The server had started and served; classify that failed row as operator quoting, not runtime/model failure.
- Moved the clean SGLang FP4 KV row from corrupted graph serving to an autosafe no-graph capacity path.
  - fork branch: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
  - live image: `nvcr.io/nvidia/sglang:26.05-py3` with editable source overlay and local FlashInfer JIT headers/source.
  - graph-enabled negative artifacts: `results/sglang_qwen_fp4kv_decode_dtype_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_decode_dtype_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_decode_dtype_bad_output_20260608_server.log`.
  - layout probe: `results/sglang_nvfp4_kv_layout_probe_20260608.json`; both 4D and 3D scale-factor forms matched a faithful dequant reference at cosine `0.9999957`, so scale-rank was not the serving corruption root cause.
  - eager/no-graph positive artifacts: `results/sglang_qwen_fp4kv_eager_only_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_eager_only_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_eager_only_server_20260608.log`.
  - auto-safe default artifacts: `results/sglang_qwen_fp4kv_autosafe_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_autosafe_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_autosafe_server_20260608.log`.
  - result at this stage: the fork disables CUDA graph and piecewise graph capture for native FP4 KV unless `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` is set. The early autosafe smoke looked better than graph-enabled decode, but this was not enough evidence to bless quality.
  - interpretation: SGLang FP4 KV needed a matched fp8-vs-fp4 row with standardized quality checks before any serving claim.
  - related upstream note: TensorRT-LLM #11368 documents a separate GB10 FP4 GEMM problem where SM120 tile configs exceed GB10's 99 KiB shared-memory limit, explaining why SM12x dispatch/JIT targeting alone may not improve MoE-shaped FP4 GEMM performance.
- Recorded the matched SGLang Qwen autosafe fp8-vs-FP4 KV row.
  - summary: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md`
  - fp8 comparator: `3,101,822` KV tokens, no-graph policy match, decode `56.73`, `56.81`, and `57.10 tok/s`, raw `2+2` returns `4`.
  - FP4 KV: `5,519,481` KV tokens, auto-safe no-graph policy, `1.779x` fp8 capacity, and `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`.
  - negative result: standardized FP4 raw `2+2` and compact benchmark content fail quality, so this is a capacity proof rather than a blessed serving or speed row.

- Recorded the matched vLLM Qwen fp8-vs-NVFP4 KV capacity row.
  - summary: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
  - runtime ref: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d`
  - fp8 comparator: `6,364,935` KV tokens, `24.28x` max concurrency at 262k context, decode `43.001`, `42.512`, and `42.684 tok/s`.
  - NVFP4 KV: `11,146,226` KV tokens, `42.52x` max concurrency at 262k context, decode `43.014`, `42.615`, and `42.898 tok/s`.
  - backend evidence: `kv_cache_dtype='nvfp4'`, `Using nvfp4 data type to store kv cache`, `Using AttentionBackendEnum.FLASHINFER backend`, and `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
  - interpretation: this is the first end-to-end vLLM NVFP4-KV serving capacity proof on the Qwen lane (`1.751x` fp8 KV pool/concurrency with normal content). It is not a decode-speed win, not native FP4 weight/MoE proof, and not a Gemma proof. Future Gemma work follows `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md` via source overlay and the standalone `D=512` FlashInfer diagnosis rather than more image-build iteration.

- Set standing vLLM-lane direction toward Gemma 4 (and earlier Gemmas) on Spark with NVFP4 KV.
  - doc: `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`
  - framing: Gemma's serving blocker and the NVFP4-KV blocker are the same blocker — heterogeneous/dual head dims (local `D=256`, global `D=512`) plus alternating SWA force `TRITON_ATTN`, so the native FA2 `sm_121a` path never engages and the FA2 NVFP4-KV path fails Gemma global `D=512` at `prefill.cuh:3215` while passing local `D=256`.
  - sequencing: no vLLM image builds in the dev loop; iterate via standalone FlashInfer kernel harness (the keystone `D=512` head-dim-guard-vs-SMEM-overflow diagnosis) and source overlay on the proven `cleanfa2-patchedfa2-cutlass` image. Image bake is the final gated deliverable (Objective E), not a dev step.
  - SM120 ride-along: build on hikarioyama's SM120 prior art and keep patches SM12x-family-shaped so the eventual PR serves the larger RTX PRO 6000 audience (vllm #31085). Emit both `sm_120a` and `sm_121a` (arch-specific; `120f` cannot emit native FP4 MMA per #3170). `a` cubins are not cross-capability portable, so `sm_120a` cannot run on GB10 — SM120 stays compiled-but-unclaimed, validated by hikari, not us.
  - verified (NVIDIA Blackwell Tuning Guide, 2026-06): RTX PRO 6000 = `sm_120`/CC 12.0, GB10 = `sm_121`/CC 12.1; both CC 12.x have 128 KB SMEM/SM and a 99 KB/thread-block limit, vs B200 CC 10.0 at 228 KB/SM. So TRT-LLM #11368's >99 KiB FP4 GEMM tile overflow is a 12.x-family constraint, not a GB10 quirk — a 99 KB-fitting tile fix is correct for both sm_120 and sm_121.
  - tracker: added an "Upstream Issues Referenced" section to `docs/ISSUE_TRACKER.md` (vllm #31085, vllm #31128, TRT-LLM #11368) and a native-FP4 target note; linked the direction doc from `README.md`.

- Set standing SGLang-lane direction: convert the proven NVFP4-KV capacity row into a blessed quality row.
  - doc: `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`
  - framing: SGLang already proves ~1.78× fp8 KV capacity on GB10, but output corrupts even in eager/no-graph mode. The FlashInfer FA2 NVFP4-KV kernel is correct standalone (`e152cf4d`, cosine ≥ 0.99999946) and the layout probe cleared scale-rank (dequant cosine 0.9999957), so the bug is in SGLang's integration of a correct kernel, not the kernel math.
  - prime suspect: the turn-1 convention mismatch — the SGLang patch routes SM12x through `nvfp4_kv_quantize` (encode/multiply convention), but the GB10-runnable FlashInfer FA2 consumer may expect the `fp4_quantize` decode/divide convention; mismatch = off-by-`s_enc²` garbage, which matches the symptom. First decisive test: swap SM12x to `fp4_quantize` + inverted global scale and check whether raw `2+2` returns `4` in eager mode.
  - objectives: (A) root-cause eager corruption via a SGLang-vs-standalone-FlashInfer numerical bridge, testing convention → V-scale layout (SGLang symmetric-linear, not vLLM B2 swizzle) → calibration → decode kernel; (B) confirm the force-compiled FP4 decode kernel is numerically correct, not just building; (C) fix CUDA-graph-capture corruption so capacity isn't stuck on the slow no-graph path; (D) land a blessed matched fp8-vs-FP4 row with quality passing; (E) SWA/Gemma later (Qwen/non-SWA first).
  - methodology: no image builds in the dev loop — iterate via editable source overlay on stock `nvcr.io/nvidia/sglang:26.05-py3` and use the standalone FlashInfer reference as ground truth.
  - tracker: linked the direction doc from `README.md`; SGLang NVFP4 KV remains issue #18.

- Set standing llama.cpp-lane direction: unblock GGUF lm-eval accuracy and prove-or-deny native FP4.
  - doc: `docs/CODEX_DIRECTION_LLAMACPP.md`
  - framing: llama.cpp is the fastest practical runtime on the box and already blessed for serving (~175 tok/s Qwen2.5 1.5B, ~77 tok/s Gemma 4 26B), but two campaign deliverables stay red: GGUF lm-eval accuracy (blocked since day one, 120 `loader_failed` rows) and native `sm_121a` FP4 dispatch (`BLACKWELL_NATIVE_FP4=1` is compiled in but never proven to engage on k-quant models).
  - keystone (A): get exact per-continuation-token logprobs from supplied/prompt-echo tokens, not top-N. The `n_probs=512` native path is a cleared dead end — it misses unlikely continuations (the `zebra` failure). Decisive first test: probe every `b9536` endpoint for the exact logprob of a supplied unlikely token, rank-independent; if none returns it, escalate to a build-pin or a `jethac/llama.cpp` endpoint fork. Fixing this gives the whole campaign a quantization-accuracy oracle it currently lacks.
  - (B): prove or deny native FP4 via cuobjdump/dispatch evidence (a k-quant model on an FP4-capable build proves nothing), and chase MXFP4/GPT-OSS — since llama.cpp avoids triton it may be the only working MXFP4-on-Spark path where triton #8335 blocks `sm_121a`. (C) larger Qwen3/3.6 GGUF row; (D) keep serving recipes current.
  - methodology: native host binaries, not containers — iterate against the running `llama-server`, rebuild/fork only if the stock server can't expose supplied-token logprobs. SM120/RTX PRO 6000 is simpler here: source-built per machine (`CMAKE_CUDA_ARCHITECTURES=121` vs `120`), no non-portable `a` cubins to ship.
  - guardrails: three claim classes never merged (serving blessed / accuracy blocked / native-FP4 unproven); top-N is not loglikelihood. Issues #8 (accuracy), #17 (serving).

- Ran the SGLang standalone NVFP4-KV convention bridge on GB10.
  - script: `scripts/sglang_nvfp4_kv_convention_probe.py`
  - artifact: `results/sglang_nvfp4_kv_convention_probe_20260608.json`
  - runtime: `nvcr.io/nvidia/sglang:26.05-py3`, FlashInfer `0.6.10+cf494fca.nv26.05.cu132.50619265`, Torch `2.12.0a0+5aff3928d8.nv26.05`, device `NVIDIA GB10`, capability `[12, 1]`.
  - shape: page-size-1 decode, `tokens=40`, `query_heads=64`, `kv_heads=4`, `head_dim=128`.
  - result: `fp4_quantize` with encode scale (`1 / decode_scale`) and FA2 reader decode scale passed (`attention_cosine_vs_source=0.9950249`, `attention_cosine_vs_dequant=0.9999955`).
  - result: `nvfp4_kv_quantize` with decode scale and FA2 reader decode scale also passed with the same cosine values.
  - negative: `nvfp4_kv_quantize` with encode scale plus FA2 reader decode scale failed completely (`attention_cosine_vs_source=0.0`).
  - interpretation: the raw FA2 reader is convention-matched for the two viable pairs above, and the known `nvfp4_kv_quantize` encode/decode crossing is invalid. Because the serving overlay had already tried the `fp4_quantize` convention and still produced corrupt text, the next SGLang suspect is integration state: calibration scale plumbing, memory-pool/backend scale application, V-scale layout, or the forced-compiled decode path, not the standalone FA2 reader math.

- Ran the llama.cpp supplied-token echo-logprobs probe against the pinned `b9536` server.
  - script update: `scripts/gguf_logprobs_probe.py` now tokenizes `--context` and `--continuation`, sends `context + continuation` with `echo=true`, and only passes if prompt `tokens` plus `token_logprobs` cover the supplied continuation span.
  - artifacts: `results/llamacpp_gguf_echo_logprobs_probe_20260608_max0.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_max1.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_summary.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_server.log`.
  - target: Qwen2.5 1.5B Q4_K_M on `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`, prompt `The capital of Japan is zebra`.
  - tokenization: context tokens `[785, 6722, 315, 6323, 374]`; continuation tokens `[1147, 50213]`.
  - result: both `max_tokens=0` and `max_tokens=1` returned `choices[0].logprobs.content` for a generated token (`-striped`) and did not expose prompt `tokens` or `token_logprobs`; both rows have `ok=false`.
  - interpretation: the OpenAI echo path on pinned `b9536` cannot provide exact supplied-continuation logprobs for the `zebra` case. The llama.cpp accuracy lane needs either a newer server pin that exposes prompt-token logprobs or a `jethac/llama.cpp` endpoint fork.

- Re-ran the focused Gemma 4 26B global-attention FlashInfer FA2 NVFP4-KV blocker on the Qwen-proven vLLM image.
  - artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_nhd_debug_20260608.json`
  - image/source: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`, FlashInfer `0.6.13` from `/opt/jethac-flashinfer`, git `e152cf4d`.
  - shape: Gemma global/full attention NHD, `batch_size=2`, `kv_len=128`, `qo_len=16`, `page_size=16`, `H_q=16`, `H_kv=2`, `D=512`, `dtype=bfloat16`, vLLM-style swizzled V scale factors.
  - result: both operations still fail at `include/flashinfer/attention/prefill.cuh:3215`; decode trait is `NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4`, prefill trait is `NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1`.
  - interpretation: Gemma NVFP4-KV remains blocked below vLLM routing. The next work is a FlashInfer FA2 `D=512` trait/tile fix or a mixed-KV fallback for Gemma global layers.
  - trait audit: `DISPATCH_HEAD_DIM` already has `case 512`; the failure is `KernelTraits::IsInvalid()` clause `NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256`. For `D=512`, `NUM_MMA_D_VO=32`, so the term is `264` for decode and `272` for prefill. This is a fragment/register-shape guard, not a missing head-dim table or primarily a 99 KiB SMEM overflow. Mixed KV for global layers is the pragmatic next vLLM/Gemma path.

- Ran the SGLang FP4 pool bridge on GB10.
  - script: `scripts/sglang_fp4_pool_bridge_probe.py`
  - artifacts: `results/sglang_fp4_pool_bridge_probe_20260608.json`, `results/sglang_fp4_pool_bridge_probe_prefill_20260608.json`
  - runtime: `nvcr.io/nvidia/sglang:26.05-py3` with `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving` overlaid through `PYTHONPATH`; FlashInfer `0.6.10+cf494fca.nv26.05.cu132.50619265`; device `NVIDIA GB10`, capability `[12, 1]`.
  - shape: `tokens=40`, `query_heads=64`, `kv_heads=4`, `head_dim=128`, page size 1, token slots `1..40` so slot 0 remains the SGLang padded slot.
  - decode result: `all_ok=true`; `attention_cosine_vs_dequant=0.9999946356`, `attention_cosine_vs_source=0.9958496094`, finite output, `key_dequant_cosine_vs_source=0.9955501556`, `value_dequant_cosine_vs_source=0.9954957962`.
  - widened result: decode and paged prefill both pass; prefill has `attention_cosine_vs_dequant=0.9999946356`, `attention_cosine_vs_source=0.9957648516`, finite output, and `passed=true`.
  - interpretation: `MHATokenToKVPoolFP4` writes packed K/V and FP8 scale buffers that FlashInfer FA2 can consume through the same getters used by serving for decode and paged prefill. The remaining SGLang FP4-KV corruption is downstream of the basic pool contract: backend wrapper metadata, graph/capture state, stale calibration state, or a model-serving path not covered by the synthetic bridge.

- Added and ran SGLang FP4-KV backend trace instrumentation.
  - fork commit: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
  - knob: `SGLANG_FP4_KV_TRACE_BACKEND=1`
  - artifacts: `results/sglang_fp4_backend_trace_20260608T1536JST_summary.md`, `results/sglang_fp4_backend_trace_20260608T1536JST_server.log`, `results/sglang_fp4_backend_trace_20260608T1536JST_trace_excerpt.txt`, `results/sglang_fp4_backend_trace_20260608T1536JST_raw_2plus2.json`, `results/sglang_fp4_backend_trace_20260608T1536JST_chat_smoke.json`
  - command shape: NVIDIA SGLang 26.05 source overlay, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `--kv-cache-dtype fp4_e2m1`, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled.
  - result: server reached readiness, allocated `5,516,867` FP4 KV tokens, calibrated 28 layers, and traced all 28 decode layers through native FP4 KV with packed `uint8` K/V, FP8 scale buffers, and finite per-layer global scales.
  - sanity: raw `2+2 is` returned ` 4, 2+2 is 4, 2+2 is`; chat smoke returned exactly `spark-ok`.
  - interpretation at the time: this was quality-positive debug evidence and showed the backend decode call contract matched the cleared pool bridge, but it was not yet a blessed SGLang FP4-KV row. It is superseded by the matched fp8-vs-FP4 trace row below.

- Ran the matched SGLang fp8-vs-FP4 KV row on the backend trace branch.
  - fork commit: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`, plus fp8/fp4 row manifests, OpenAI benchmarks, raw `2+2`, chat smoke, server logs, runtime probes, build-target audits, and trace excerpts.
  - command shape: NVIDIA SGLang 26.05 source overlay, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `--attention-backend flashinfer`, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled; FP4 row adds `--kv-cache-dtype fp4_e2m1` and `SGLANG_FP4_KV_TRACE_BACKEND=1`; fp8 comparator uses `--kv-cache-dtype fp8_e4m3`.
  - capacity: fp8 allocated `3,105,240` KV tokens; FP4 allocated `5,517,572` KV tokens, or `1.7769x` fp8.
  - trace: FP4 row calibrated 28 layers and logged all 28 decode plus all 28 `extend_merge_paged` layers through packed `uint8` K/V and FP8 scale buffers.
  - smoke: fp8 raw `2+2 is` returned ` 4. 2+2 is 4. 2+2 is`; FP4 returned ` 4, 2+2 is 4, 2+2 is`; both chat smokes returned `spark-ok`.
  - benchmark: fp8 produced normal compact benchmark text at about `57 tok/s`; FP4 short/medium/long benchmark content remained degraded even though raw/chat smoke passed.
  - interpretation: this retires the missing matched-comparator/request-trace task, but does not bless SGLang FP4 KV quality or speed. The next SGLang task is quality localization on the degraded benchmark prompts.

- Ran the SGLang FP4-KV logprob quality probe on the same `d7d931f` source-overlay path.
  - script: `scripts/openai_quality_probe.py`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md`, `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp8_quality_probe.json`, `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_quality_probe.json`, and `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_compare.json`.
  - command shape: same NVIDIA SGLang 26.05 source overlay, same Qwen2.5 1.5B model, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled; probe requested generated-token logprobs for `short_decode` and `medium_decode`.
  - result: fp8 passed both probes. FP4 `short_decode` began with the same high-confidence prefix (`A local AI workstation`) then drifted into mixed Chinese/repetition; FP4 `medium_decode` diverged at token one (`the following code:` vs fp8 `**Engineering Note:`) and collapsed into repeated `import` text.
  - interpretation: the quality bug is now localized beyond "bad benchmark text": one standardized prompt is wrong from the first token, while another starts plausibly and corrupts later. Next useful SGLang work is a divergence-window trace around the failing `medium_decode` prompt, not another capacity row.

- Ran the SGLang native `/generate` logprob divergence probe for `medium_decode`.
  - script: `scripts/sglang_native_logprob_compare.py`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_summary.md`, `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_native_logprob_compare.json`, and server/trace logs with the same prefix.
  - command shape: two simultaneous NVIDIA SGLang 26.05 source-overlay servers on ports 30012 and 30013; fp8 comparator versus FP4 KV candidate; same model, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled. The probe rendered the Qwen chat template explicitly, then called native `/generate` with `return_logprob=true`.
  - result: fp8 and FP4 used the same 56-token rendered prompt and matched through output tokens 0-3 (`**`, `Engineering`, ` Note`, `:`). First divergence was token index 4: fp8 chose ` Valid`, FP4 chose ` Validate`; both alternatives appeared in both top-k lists, but FP4 reversed their rank.
  - interpretation: under native `/generate`, the failure is an early decode distribution perturbation that compounds, not a total first-token collapse. The next SGLang question is why OpenAI Chat Completions looked worse than native rendered-template `/generate`.

- Ran the SGLang OpenAI-vs-native prompt reconciliation probe.
  - script: `scripts/sglang_openai_native_reconcile.py`
  - artifacts: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_summary.md`, `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST.json`, and matching fp8/FP4 server logs, trace excerpts, and container inspect files.
  - command shape: two simultaneous NVIDIA SGLang 26.05 source-overlay servers on ports
    30012 and 30013; fp8 comparator versus FP4 KV candidate; same Qwen2.5 1.5B model,
    FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise
    graph disabled. The probe compared OpenAI Chat Completions prompt IDs against local
    Qwen chat-template rendering and replayed native `/generate` from the same prompt IDs.
  - result: fp8 and FP4 OpenAI prompt IDs matched the local render exactly: 56 tokens,
    SHA-256 `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`, no first
    prompt diff. FP4 OpenAI still diverged at token 4, while FP4 native `/generate` from
    the same prompt IDs diverged at token 0 (`**` -> `ark`).
  - interpretation: prompt serialization is retired as the cause. The remaining SGLang
    FP4-KV quality bug is endpoint/path-specific serving numerics or metadata; next inspect
    request metadata and pre-sampling logits/hidden state before touching kernel math again.
- Added the Gemma compatibility plan as a sequenced ladder across the whole family.
  - doc: `docs/GEMMA_COMPATIBILITY_PLAN.md`
  - rationale: "Gemma 4" is five models across four architectures (E2B/E4B dense-mobile+PLE+audio, 12B dense encoder-free multimodal, 26B-A4B MoE, 31B dense), plus Gemma 3 (dense, SWA, uniform head dim) and Gemma 3n (the superseded mobile line). "Fix Gemma" is a matrix, not a checkbox.
  - structure: one-new-complication-per-rung ladder. Rung -1 = per-variant config audit (settles where `D=512` lives). Main vLLM NVFP4-KV ladder: Rung 0 Qwen (done) -> Rung 1 Gemma 3 27B (isolate SWA / hybrid mixed-KV, also a shippable big-Gemma capacity win) -> Rung 2 Gemma 4 31B text-only (dense D=512 mixed-KV) -> Rung 3 Gemma 4 26B-A4B text-only (add MoE) -> Rung 4 Gemma 4 12B (encoder-free multimodal through decoder/KV). Mobile side track (llama.cpp/LiteRT): Gemma 3n -> Gemma 4 E2B/E4B (isolate PLE/audio/elastic).
  - discipline: prove-before-climb; assign each model to its natural runtime rather than filling all 15 cells; **measure attention geometry (per-layer head_dim, heads, KV heads, SWA layer map, KV bytes/token) from the running model every rung** — config is a hint, the running model is ground truth, and the mixed-KV layer classification must come from measured per-layer head_dim, not assumption.
  - provenance caveat: Gemma 4 lineup/architecture is operator-provided (post-cutoff) and must be confirmed by the rung -1 audit + per-rung measurement before building on it.

- Completed the Gemma Rung -1 config audit.
  - script: `scripts/gemma_rung_minus1_config_audit.py`
  - artifact: `results/gemma_rung_minus1_config_audit_20260608.json`
  - report: `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md`
  - result: `google/gemma-3-27b-it` normalizes to uniform `D=128` (not the earlier
    plan's `D=256`), with 52 sliding layers and 10 full layers; no `global_head_dim=512`.
  - result: `D=512` appears in all audited Gemma 4 server configs. 12B has 40 sliding
    `D=256` + 8 full `D=512` layers and vision/audio config blocks; 31B has 50 sliding
    `D=256` + 10 full `D=512` layers and is dense; 26B-A4B has 25 sliding `D=256` + 5
    full `D=512` layers plus MoE (`128` experts, top-`8`).
  - decision: 31B isolates dense `D=512` before 26B-A4B adds MoE. The later ladder
    correction moves Gemma 4 12B to the final multimodal-KV rung. Next vLLM live rung
    remains Gemma 3 27B; next SGLang live work remains Qwen FP4-KV quality.

- Tightened the Gemma Rung -1 audit against cached QAT/server variants.
  - artifact: `results/gemma_rung_minus1_config_audit_strict_20260608.json`
  - script update: `scripts/gemma_rung_minus1_config_audit.py` now records config wrapper
    status, operator architecture hints, PLE-like keys, and config-derived BF16 raw KV
    bytes/token totals.
  - result: 12B QAT variants preserve the base 12B geometry: 40 sliding `D=256` + 8 full
    `D=512`; 31B W4A16 preserves 50 sliding `D=256` + 10 full `D=512`; 26B-A4B
    QAT-unquantized preserves 25 sliding `D=256` + 5 full `D=512` and MoE.
  - correction: config wrapper fields alone do not decide whether modality is quarantined
    in an unfired encoder or fused into the decoder/KV. The ladder now follows the
    operator-provided architecture: 31B and 26B-A4B are text-only encoder-quarantine rungs;
    12B is the final encoder-free multimodal-KV rung.

- Integrated the first llama.cpp NVFP4 GGUF runtime smoke from the parallel lane.
  - artifact: `results/llamacpp_nvfp4_runtime_gate_20260608T1748JST_summary.md`
  - source: `jethac/llama.cpp@19bba67c1`, built for `sm_121a`
  - result: cached AEON Qwen3.6 NVFP4 converted to NVFP4 GGUF, loaded in `llama-server`,
    and returned `The capital of Japan is Tokyo.`
  - profiler evidence: Nsight summary shows `GGML_TYPE_NVFP4` matmul and
    `quantize_mmq_nvfp4` kernels. This is runtime dispatch/smoke evidence, not an accuracy
    or speed benchmark.

- Completed the first llama.cpp native FP4 arch-build checkpoint in parallel with the
  Gemma audit.
  - submodule/fork: `third_party/llama.cpp` -> `jethac/llama.cpp`, branch
    `spark/native-fp4-sm121-20260608`
  - pinned commit: `19bba67c1f4db723c60a0d421aa0788bf4ddc699`
  - artifact: `results/llamacpp_native_fp4_arch_20260608T164917JST_summary.md`
  - result: `CMAKE_CUDA_ARCHITECTURES=121a` configures/builds and emits `sm_121a` cubins
    with `2592` `mxf4nvf4.block_scale.scale_vec::4X` PTX hits.
  - result: `CMAKE_CUDA_ARCHITECTURES=121` is accepted but rewritten by this llama.cpp pin
    to `121a`, so it is not an independent non-`a` build.
  - result: `CMAKE_CUDA_ARCHITECTURES=120f` fails at CMake configure-time under this CUDA
    13.0/CMake 3.28.3 toolchain.
  - interpretation: native block-scale FP4 code emission on `sm_121a` is proven for the
    pinned source build; runtime dispatch, correctness, and speed on an actual NVFP4 GGUF
    remain the next gate.

- Tightened the vLLM Gemma 3 27B Rung 1 live packet after host preflight.
  - script: `scripts/prep_vllm_gemma3_27b_rung1.sh`
  - doc: `docs/VLLM_GEMMA3_27B_RUNG1_PREP_20260608.md`
  - preflight artifact: `results/vllm_gemma3_27b_rung1_preflight_20260608.md`
  - result: the Spark-class Linux endpoint is reachable, idle, and has the target vLLM
    image locally, but the older `/home/jethac/src/vllm` and `/home/jethac/src/flashinfer`
    assumptions are false, the existing Linux repo checkout is stale/dirty, and
    `google/gemma-3-27b-it` was not found in the bounded HF cache search.
  - fix: generated packets now stream `docker logs -f` into `_server.log`, write container
    IDs, wait for `/v1/models` before recording rows, and remove row containers on exit.
    This prevents the next Gemma Rung 1 run from recording only a Docker container ID as
    the server log or racing readiness.
  - next gate: create/sync a clean Linux run checkout with initialized vLLM/FlashInfer
    submodules and gated Gemma 3 access, then run the fp8 comparator row before the NVFP4
    candidate.

- Prepared the clean Linux checkout for the vLLM Gemma 3 27B Rung 1 live row.
  - checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
  - superproject: `595dfb6dba863088707afadbad816a511b803f81`
  - source overlays: `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24` and
    `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
  - generated packet:
    `docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`
  - artifact: `results/vllm_gemma3_27b_rung1_checkout_setup_20260608.md`
  - validation: generated packet passes `bash -n`.
  - remaining gate: `google/gemma-3-27b-it` still needs gated access/cache confirmation
    before the fp8 comparator row starts.

- Ran the bounded Hugging Face access probe for vLLM Gemma 3 27B.
  - script: `scripts/hf_model_access_probe.py`
  - artifact: `results/vllm_gemma3_27b_hf_access_probe_20260608T173133JST.json`
  - target: `google/gemma-3-27b-it`
  - container: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - result: `model_info` succeeds and reports `gated="manual"` at revision
    `005ad3404e59d6023443cb575daa05336842228a`, but a config/tokenizer-only
    `snapshot_download` fails with `GatedRepoError`.
  - environment: no `HF_TOKEN` was present inside the container; cache filesystem free
    space was about `2.50e12` bytes.
  - interpretation: the immediate blocker for the Gemma 3 fp8 comparator row is gated
    Hugging Face authentication/access, not disk headroom, target image availability, or
    source-checkout setup.

- Added the SGLang FP4 endpoint metadata localization packet.
  - script: `scripts/sglang_fp4_endpoint_metadata_probe.py`
  - first-token dump hook: `scripts/sglang_fp4_first_token_dump_patch.yaml`
  - artifact: `results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST_summary.md`
  - result: the FP4 OpenAI and native paths still share the same 56-token prompt hash, but
    OpenAI starts with `**` while native `/generate` starts with `ark` token `838`; current
    backend traces cover decode and `extend_merge_paged` but are not request-tagged.
  - interpretation: the next SGLang proof is a live one-token request-tagged dump of
    `next_token_logits` before and after `ModelRunner._preprocess_logits()`, not another
    capacity row and not Gemma.

- Closed the vLLM Gemma 3 Rung 1 geometry-hook prep gap.
  - vLLM fork commit: `jethac/vllm@3658ba7123c3eb2211f18a882af1b993112fadb1`
  - artifact: `results/vllm_gemma3_27b_rung1_auth_recheck_20260608T181438JST.md`
  - result: the remote run checkout now contains `SPARK_GEMMA_KV_GEOMETRY` and
    `SPARK_GEMMA_KV_SPEC` logging in the vLLM overlay; the host remains reachable and idle.
  - blocker: `google/gemma-3-27b-it` still has no token/cache on the host, so starting the
    fp8 comparator would only test gated Hugging Face access.

- Cleared the vLLM Gemma 3 Rung 1 Hugging Face auth/cache blocker.
  - artifact: `results/vllm_gemma3_27b_hf_access_probe_20260608T1832JST.md`
  - result: the `jethac` user now has a token-file/profile-loader setup, the container sees
    `HF_TOKEN`, and `google/gemma-3-27b-it` config/tokenizer snapshot download succeeds at
    revision `005ad3404e59d6023443cb575daa05336842228a`.
  - new blocker: the prepared packet fails during editable vLLM install because
    `wheels.vllm.ai` returns 404 for `8916796bc50926fd61e606718b194a71e2e31a24` under the
    `cu130` metadata paths. The next vLLM step is to rebase the geometry overlay onto a
    source/precompiled-wheel pair with published CUDA 13 metadata before running fp8.

- Rebased the vLLM Gemma 3 geometry hook onto the CUDA-13-proven Qwen lane.
  - vLLM fork commit: `jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
  - branch: `jethac/vllm:spark/hijinks-020-aeon-qwen-dflash-sm121a`
  - precompiled wheel base: `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
  - result: the env-gated `SPARK_GEMMA_KV_GEOMETRY`/`SPARK_GEMMA_KV_SPEC` hook now sits on
    the same line as the clean `a919d635d` Qwen packaging that already built with CUDA 13.
  - superseded next gate: the setup-only check and fp8 comparator have since run; the live
    gate is now the matching Gemma 3 NVFP4 row.

- Cleared the vLLM Gemma 3 Rung 1 setup-only packaging gate.
  - artifact: `results/vllm_gemma3_27b_rung1_setup_only_20260608T1855JST.md`
  - result: the repaired source/wheel pair clears the prior metadata 404, but the first
    setup probe allowed dependency resolution and downgraded Torch/FlashInfer.
  - caveat: that downgraded dependency state is rejected; live rows now use `--no-deps` and
    copy the ABI-matched FA2 extension from `/opt/jethac-vllm`.
  - superseded next gate: the fp8 comparator has since run; the live gate is now the
    matching Gemma 3 NVFP4 row.

- Captured the next SGLang Qwen FP4-KV quality localization step from the sidecar lane.
  - artifact: `results/sglang_qwen_fp4kv_first_token_logits_plan_20260608T1900JST.md`
  - result: the next proof is a one-token FP4-only endpoint split with
    `scripts/sglang_fp4_first_token_dump_patch.yaml`, dumping `next_token_logits` before and
    after `ModelRunner._preprocess_logits()` plus request metadata.
  - next gate: run the dump against the existing no-graph FP4 Qwen source-overlay server and
    compare whether divergence is present before logits preprocessing.

- Added the llama.cpp NVFP4 correctness/speed run packet.
  - packet: `tasks/llamacpp_nvfp4_correctness_speed_packet_20260608.md`
  - doc update: `docs/GGUF_LLAMA_CPP_STATUS.md`
  - result: the previous runtime gate is now treated as first NVFP4 GGUF dispatch proof,
    while correctness versus BF16/Q8 and matched PP/TG speed remain explicitly unproven.
  - next gate: run the packet on the Linux GB10 host with a same-lineage BF16 or Q8_0
    Qwen3.6 reference GGUF.

- Tightened the vLLM Gemma 3 Rung 1 install path after rejecting a dependency downgrade.
  - artifact updated: `results/vllm_gemma3_27b_rung1_setup_only_20260608T1855JST.md`
  - result: the repaired source/wheel pair still clears the prior CUDA 13 metadata 404, but
    the dependency-resolving setup probe downgraded Torch/FlashInfer and is no longer an
    accepted runtime environment.
  - fix: the command packet now uses `pip install --no-build-isolation --no-deps -e .`,
    preserves the clean image dependency stack, copies the ABI-matched FA2 extension from
    `/opt/jethac-vllm`, and adds `RUN_FP8` so fp8 and NVFP4 rows can be run separately.

- Captured the vLLM Gemma 3 27B Rung 1 fp8 comparator row.
  - artifact: `results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`
  - result: Gemma 3 27B text-only serves with fp8 KV and FlashInfer decoder attention.
  - measured geometry: `62` decoder layers, `52` local SWA layers, `10` full/global layers,
    uniform `heads=32`, `kv_heads=16`, `head_dim=128`, `head_dim_v=128`, and the expected
    `0-4 local / 5 full` repeating pattern.
  - capacity baseline: `882,851` fp8 KV tokens and `6.74x` maximum concurrency at
    `131,072` tokens/request.
  - benchmark: all three OpenAI cases completed with unflagged output, around
    `4.16-4.24 tok/s` decode.
  - next gate: run the matching NVFP4 row against this baseline.

- Captured the vLLM Gemma 3 27B Rung 1 NVFP4 candidate row.
  - command: `RUN_FP8=0 RUN_NVFP4=1 bash docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`
  - artifact: `results/vllm_gemma3_27b_rung1_nvfp4_20260608T1924JST.md`
  - capacity result: FlashInfer FA2 NVFP4 KV routes on SM12x and reports `1,568,861` KV
    tokens / `11.97x` concurrency versus fp8 `882,851` / `6.74x`, a `1.777x` pool gain.
  - speed result: decode is essentially parity with fp8 across the three benchmark cases.
  - quality result: red. Strict `spark-ok` smoke returns nonsensical mixed-script text, and
    the benchmark generations are also corrupted. The simple heuristic quality probe is too
    weak because it did not flag the garbage text.
  - next gate: add a stronger fp8-vs-NVFP4 correctness/localization probe for vLLM Gemma 3
    before climbing to the Gemma 4 31B rung.

- Added the vLLM Gemma 3 first-token corruption localization probe.
  - script: `scripts/openai_first_token_probe.py`
  - plan artifact: `results/vllm_gemma3_27b_first_token_probe_plan_20260608.md`
  - packet update: `scripts/prep_vllm_gemma3_27b_rung1.sh` now writes fp8/NVFP4
    first-token artifacts and a first-token compare, and it continues after a red manifest
    so diagnostic artifacts are still collected.
  - result: ready for the next live GB10 run with a fresh stamp. This should answer whether
    Gemma 3 NVFP4 diverges on the first generated token after prefill or only after decode
    state starts compounding.

- Captured a partial SGLang Qwen FP4-KV first-token logits dump.
  - script: `scripts/sglang_first_token_dump_summary.py`
  - artifact: `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_summary.md`
  - dump summary:
    `results/sglang_qwen_fp4kv_first_token_logits_20260608T2008JST_cleanup0_dump_summary.md`
  - result: the `DUMPER_CLEANUP_PREVIOUS=0` rerun captured `55` tensor dumps and the
    summarizer grouped `9` real-request forward passes plus `2` health-check passes.
  - finding: for the captured native `/generate` request, logits before and after
    `ModelRunner._preprocess_logits()` are identical (`max_abs_delta=0`, same argmax,
    top-20 Jaccard `1.0`).
  - caveat: this is not an OpenAI-vs-native comparator because the host probe failed due a
    missing `transformers` import before any `/v1/chat/completions` request completed.
  - next gate: rerun the endpoint probe inside the SGLang container or another environment
    with `transformers`, with explicit endpoint labels or split request runs.

- Captured the paired SGLang Qwen FP4-KV OpenAI/native first-token dump.
  - artifact: `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_summary.md`
  - endpoint probe: `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST.json`
  - dump summary:
    `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_dump_summary.md`
  - result: running the probe inside the SGLang container fixed the missing-`transformers`
    host failure and completed both `/v1/chat/completions` and `/generate`.
  - finding: prompt reconciliation still passes with the same 56-token prompt hash, but
    OpenAI returns `**` while native returns `ark` / token id `838`.
  - localization: candidate prefill dump groups already diverge before logits
    preprocessing: OpenAI argmax `334`, native argmax `838`; native argmax `838` matches
    the endpoint-visible native first token. `_preprocess_logits()` is not the primary
    cause.
  - next gate: compare fp8 vs FP4 under the same OpenAI/native sequencing and add explicit
    endpoint request tags or split runs to isolate endpoint metadata, scheduler/cache state,
    or FP4-KV endpoint-specific request handling.

- Captured the paired SGLang Qwen fp8-KV OpenAI/native first-token control.
  - artifact: `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_summary.md`
  - endpoint probe: `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST.json`
  - dump summary: `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_dump_summary.md`
  - result: under the same prompt, same OpenAI/native sequence, same no-graph policy, and
    `--kv-cache-dtype fp8_e4m3`, both endpoints return `**`.
  - localization: both endpoint candidate prefill dump groups argmax `334`, so the FP4
    row's native argmax `838` is FP4-KV-specific rather than a generic endpoint mismatch.
  - next gate: compare fp8 and FP4 tensor state at or before the first prefill attention/KV
    write for native `/generate`, with explicit endpoint request tags and no unrelated
    warmup request.

- Captured SGLang Qwen FP4-KV radix-cache isolation.
  - artifact: `results/sglang_qwen_fp4kv_radix_isolation_20260608T2038JST_summary.md`
  - rows:
    `results/sglang_qwen_fp4kv_first_token_noradix_20260608T2033JST.json`,
    `results/sglang_qwen_fp4kv_first_token_skipwarmup_20260608T2036JST.json`,
    `results/sglang_qwen_fp4kv_first_token_radixoff_20260608T2038JST.json`
  - result: `--skip-server-warmup` alone still fails (`**` vs `ark` / token id `838`),
    while `--disable-radix-cache` fixes the first-token split both with and without public
    warmup (`**` vs `**`).
  - localization: the FP4 native `/generate` failure is now narrowed to radix/prefix-cache
    reuse or FP4 cached-prefix read/merge behavior. Disabling radix cache is a diagnostic
    workaround, not a blessed serving fix.
  - next gate: instrument `tree_cache.match_prefix`, `prefix_indices`,
    `extend_prefix_lens_cpu`, `use_ragged`, `extend_no_prefix`, and `extend_merge_paged`
    for the default FP4 native request versus the radix-off FP4 native request.

- Added SGLang FP4-KV radix/reuse instrumentation.
  - fork commit: `jethac/sglang@ce1b6d15e76985240e91592a0f44c0f282fc65af`
  - branch: `spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
  - files: `schedule_batch.py`, `forward_batch_info.py`, `flashinfer_backend.py`
  - behavior: inactive unless `SGLANG_FP4_KV_TRACE_RADIX=1` is set.
  - purpose: request-tagged tracing from radix prefix match through `ForwardBatch` to
    FlashInfer prefill/extend path selection, so default FP4 and radix-off FP4 runs can
    prove whether packed FP4 KV bytes and FP8 scale buffers stay aligned under prefix reuse.
  - validation: `python -m py_compile` passes for all three touched files; no serving claim
    is made until the traced default/radix-off run is captured on GB10.

- Captured the vLLM Gemma 3 27B Rung 1 first-token diagnostic.
  - artifact: `results/vllm_gemma3_27b_rung1_first_token_20260608T205432JST.md`
  - packet: `docs/results/vllm_gemma3_27b_rung1_20260608T205432JST_command_packet.sh`
  - result: fp8 still serves cleanly; NVFP4 still selects FlashInfer FA2 and records
    `1,595,236` KV tokens / `12.17x` concurrency, but remains quality-red.
  - first-token finding: fp8 chooses `spark`, `4`, and `A`; NVFP4 chooses unrelated
    Cyrillic/CJK tokens with `0.0` top-logprob overlap for all three cases.
  - localization: Gemma 3 NVFP4 corruption is present before sampling on the first
    generated token, so the next vLLM diagnostic should trace SWA block lifecycle, slot
    mapping, NVFP4 split/view offsets, and FlashInfer read-side page IDs.

- Captured the SGLang Qwen FP4-KV radix metadata trace.
  - artifact: `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_summary.md`
  - fork commit: `jethac/sglang@ce1b6d15e76985240e91592a0f44c0f282fc65af`
  - rows:
    `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_default.json`,
    `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_radixoff.json`
  - result: default FP4 still fails (`OpenAI **` vs native `ark` / `838`), while
    `--disable-radix-cache` still passes (`**` vs `**` / `334`).
  - localization: the failing native request has `prefix_indices_len=55`,
    `extend_prefix_lens_cpu=[55]`, `extend_seq_lens_cpu=[1]`, and routes through
    `forward_extend_merge_paged`; the passing radix-off request has no prefix reuse,
    `extend_prefix_lens_cpu=[0]`, `extend_seq_lens_cpu=[56]`, and routes through
    `forward_extend_ragged_no_prefix`.
  - next gate: instrument cached-prefix page IDs and verify K-data/K-scale/V-data/V-scale
    page pairing for reused FP4 prefix pages.

- Updated the vLLM Gemma NVFP4-KV diagnosis from the read-only audit.
  - finding: the three Gemma 3 first-token prompts are `18`, `23`, and `24` tokens, below
    the `1024` sliding window, so the observed first-token corruption cannot require SWA
    eviction or window rotation.
  - implication: the next vLLM trace should still include SWA block lifecycle, but it must
    first prove base Gemma NVFP4 write/read/page pairing in `BlockTable`,
    `do_kv_cache_update`, `reshape_and_cache_nvfp4_dispatch`, and
    `nvfp4_kv_cache_split_views`.

- Added and ran SGLang FP4-KV cached page-pair tracing.
  - fork commit: `jethac/sglang@839cb7457`
  - env gate: `SGLANG_FP4_KV_TRACE_PAGE_PAIR=1`
  - artifact: `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_summary.md`
  - rows:
    `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_default.json`,
    `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_radixoff.json`
  - result: default FP4 still fails (`**` vs `ark` / `838`) with `cached_tokens=55`;
    radix-off still passes (`**` vs `**` / `334`) with `cached_tokens=0`.
  - localization: the default row's paged plan consumes the same 55 page IDs as the radix
    prefix (`4113..4167`), and all 28 layer page-pair traces report matching first
    dimensions for K data, V data, K scale, and V scale.
  - interpretation: gross cached-prefix page-list mismatch is not observed; next inspect
    actual FP4 data/scale bytes and `o1/s1/o2/s2` before `_safe_merge_state`.

- Recorded the vLLM Gemma NVFP4-KV trace plan from the parallel read-only audit.
  - doc: `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`
  - scope: Python-side JSONL instrumentation only, compatible with the source-overlay loop.
  - env gates: `VLLM_SPARK_KV_TRACE`, `VLLM_SPARK_KV_TRACE_FILE`,
    `VLLM_SPARK_KV_TRACE_LAYERS`, `VLLM_SPARK_KV_TRACE_LIMIT`,
    `VLLM_SPARK_KV_TRACE_VALUES`.
  - required first gate: prove short Gemma 3 prompts have
    `swa_skip.num_skipped_tokens == 0`; if corruption remains, focus on NVFP4 data/scale
    contents or V-scale swizzle/deswizzle rather than SWA eviction.

- Added and ran SGLang FP4-KV cached merge-state tracing.
  - fork commits: `jethac/sglang@ed1a7d6b1` added the merge-state trace,
    `jethac/sglang@991ac1e63` fixed the diagnostic dedupe key.
  - env gate: `SGLANG_FP4_KV_TRACE_MERGE_STATE=1`
  - artifact: `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_summary.md`
  - rows:
    `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_default.json`,
    `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_radixoff.json`
  - result: default FP4 still fails (`OpenAI **` vs native `ark` / `838`) with
    `cached_tokens=55`; radix-off still passes (`**` vs `**` / `334`) with
    `cached_tokens=0`.
  - localization: layer 0 samples readable, nonzero packed K/V bytes and FP8 K/V scale
    bytes at pages `4113..4116`; `o1/s1`, `o2/s2`, and merged tensors are finite.
  - interpretation: the defect is past gross page-list/view mismatch and past a non-finite
    merge-state failure. Next compare write-time bytes/scales from
    `MHATokenToKVPoolFP4.set_kv_buffer()` against the read-time page bytes, or build a
    same-prompt no-prefix reference for the paged-prefix contribution.

- Implemented vLLM Gemma NVFP4-KV trace hooks in the vLLM fork.
  - fork commit: `jethac/vllm@e2a8197a9`
  - files: `vllm/v1/attention/backends/flashinfer.py`,
    `vllm/v1/core/single_type_kv_cache_manager.py`
  - env gates: `VLLM_SPARK_KV_TRACE`, `VLLM_SPARK_KV_TRACE_FILE`,
    `VLLM_SPARK_KV_TRACE_LAYERS`, `VLLM_SPARK_KV_TRACE_LIMIT`,
    `VLLM_SPARK_KV_TRACE_VALUES`
  - events: `fi_metadata`, `kv_write_pre`, `kv_write_post_nvfp4`,
    `kv_read_views_nvfp4`, and `swa_skip`.
  - next gate: rerun Gemma 3 27B fp8/NVFP4 first-token packet with trace file enabled and
    verify slot/page metadata, NVFP4 data/scale offsets, sampled bytes, and short-prompt
    `swa_skip` before climbing to Gemma 4 31B.

- Recorded the Gemma Rung -1 sidecar audit.
  - artifact: `results/gemma_rung_minus1_audit_sidecar_20260608.md`
  - conclusion: Gemma 3 27B has measured vLLM geometry but red NVFP4-KV quality; Gemma 4
    31B is the next clean dense `D=512` rung after Gemma 3 is green; Gemma 4 26B-A4B then
    adds MoE; Gemma 4 12B remains the final fused multimodal/KV rung.

- Added and ran SGLang FP4-KV write/read pairing trace.
  - fork commit: `jethac/sglang@f76f80484`
  - env gate: `SGLANG_FP4_KV_TRACE_WRITE_READ=1`
  - artifact: `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_summary.md`
  - rows:
    `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_default.json`,
    `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_radixoff.json`
  - result: default FP4 still fails (`OpenAI **` vs native `ark` / `838`) with
    `cached_tokens=55`; radix-off still passes (`**` vs `**` / `334`) with
    `cached_tokens=0`.
  - localization: for sampled layer-0 cached pages `4113..4116`, K data, V data, K scale,
    and V scale all match write input bytes = stored bytes = read bytes.
  - interpretation: sampled cached-prefix bytes/scales are structurally correct; next
    build a cached-paged-vs-recomputed-ragged numerical comparator for the same 55-token
    prefix contribution.

- Recorded the vLLM Gemma 3 NVFP4 trace packet.
  - task: `tasks/vllm_gemma3_nvfp4_trace_packet_20260608.md`
  - scope: next GPU-slot command packet for `jethac/vllm@e2a8197a9`, using
    `VLLM_SPARK_KV_TRACE_FILE` and first-token probes before benchmark/manifest traffic.

- Added SGLang FP4-KV cached-prefix reference comparator.
  - fork commits: `jethac/sglang@a8e8de26d`, `0b0b6ca4a`, `2a228949a`
  - env gate: `SGLANG_FP4_KV_TRACE_PREFIX_REF=1`
  - task: `tasks/sglang_qwen_fp4kv_prefix_ref_trace_20260608.md`
  - purpose: compare FlashInfer native FP4 paged-prefix `o2/s2` against a torch reference
    computed from SGLang-dequantized cached FP4 prefix slots, after `f76f80484` proved
    sampled cached page bytes and scale bytes match write/stored/read.

- Ran SGLang FP4-KV cached-prefix reference comparator.
  - artifact: `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_summary.md`
  - parsed artifact: `results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_parsed.json`
  - result: default FP4 still fails (`OpenAI **` vs native `ark` / `838`) with
    `cached_tokens=55`; radix-off still passes (`**` / `334`) with `cached_tokens=0`.
  - comparator: cached paged-prefix `o2` matches the dequantized torch reference
    (`cosine=0.999997`, `max_abs=0.0078125`), FlashInfer-base `s2` matches after converting
    reference LSE to log2 units (`max_abs=0.001953125`), and manual `exp2` merge matches
    `_safe_merge_state` exactly at BF16 precision.
  - interpretation: the remaining SGLang FP4-KV quality bug is above FP4 paged-prefix
    read/dequant and merge math; next inspect calibration/quantization-error impact and
    request sequencing/state across the OpenAI/native radix-hit pair.

- Added and ran SGLang FP4-KV quant-error trace.
  - fork commit: `jethac/sglang@d4fe78078`
  - env gate: `SGLANG_FP4_KV_TRACE_QUANT_ERROR=1`
  - artifact: `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_summary.md`
  - parsed artifact: `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_analysis.json`
  - result: default FP4 still fails (`OpenAI **` vs native `ark` / `838`) with
    `cached_tokens=55`; radix-off still passes (`**` / `334`) with `cached_tokens=0`.
  - quantization: default and radix-off have identical sampled layer-0 dense-vs-dequant
    metrics for the 56-token fill (`K cosine=0.996669`, `V cosine=0.995715`) and the same
    global scales.
  - interpretation: calibration/global-scale selection and ordinary FP4 quant/dequant loss
    are not the distinguishing factor; next run
    `tasks/sglang_qwen_fp4kv_request_order_probe_20260608.md`.

- Added and ran SGLang FP4-KV request-order probe.
  - script: `scripts/sglang_fp4_request_order_probe.py`
  - artifact: `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_summary.md`
  - parsed artifact: `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_analysis.json`
  - result: OpenAI-first/native-second fails on native with `cached_tokens=55` and `ark`;
    native-first/OpenAI-second fails on OpenAI with `cached_tokens=55` and `ark`.
  - controls: flush-between and distinct `extra_key`/`cache_salt` namespaces both force
    `cached_tokens=0` and both endpoints emit `**`.
  - interpretation: the remaining SGLang blocker is endpoint-independent FP4 cached-prefix
    reuse. The next fix/probe is dense full-prefill versus FP4 cached-prefix quality,
    better calibration, or a selective no-reuse policy.

- Staged the next vLLM Gemma 3 FlashInfer paged-prefill diagnostic.
  - FlashInfer fork branch: `jethac/flashinfer@spark/hijinks-021-prefill-debug`
  - commit: `96be2fa8`
  - parent campaign commit: `4598919`
  - run packet: `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`
  - env gate: `FLASHINFER_PREFILL_DEBUG_ONCE=1`
  - purpose: print generated dtype identity, `REQUIRE_FP4_KV_CACHE`, FP4x2 carrier status,
    additional scale-factor tensor metadata, runtime page/head/window fields, and TensorView
    shape/stride/dtype from the generated C++ paged-prefill module.
  - hardening: identity and tensor lines now share `(path, call_id, module_uri, module_key)`,
    so the audit can prove the tensor dump came from the same generated module call as the
    identity line.
  - interpretation: this is the next live check before kernel-side fragment dumps. It tests
    whether the live generated module/C++ binding is the one the Python wrapper thinks it is.

- Rewired the SGLang FP4-KV matrix to use a source-built stack.
  - commit: `a4ff319`
  - script: `scripts/install_sglang_source_stack.sh`
  - callers: `scripts/prepare_sglang_source_stack_image.sh`,
    `scripts/run_sglang_fp4_request_order_matrix.sh`
  - result: the next SGLang matrix run installs editable patched FlashInfer and source-builds
    `sgl-kernel` from the `jethac/sglang` submodule, instead of using stale PyPI
    `flashinfer-cubin`, `flashinfer-jit-cache`, or `sglang-kernel` wheels.

- Recorded the GB10 host access stop point.
  - commit: `f120023`
  - artifact: `results/gb10_host_access_20260609_tailnet_stop_point.md`
  - result: the tailnet name resolves to `100.113.98.11`; the host briefly answered
    Tailscale ping earlier in the session, but the final check timed out on Tailscale ping
    and TCP/22 returned `False`.
  - interpretation: live vLLM/SGLang validation is queued, but this workspace cannot drive
    it until SSH/TCP 22 is reachable again.

- Refreshed offline status audits for 2026-06-09.
  - artifacts:
    `results/solution_coverage_audit_20260609.json`,
    `results/serving_manifest_audit_20260609.json`,
    `results/counterpart_evidence_audit_20260609.json`,
    `results/counterpart_task_matrix_20260609.json`
  - result: solution coverage is structurally OK; serving strictness is still false; the
    counterpart audit still has six missing/partial rows; the task matrix has no missing
    task contracts for those rows.

- Added the SGLang dense-vs-cached FP4 KV trace head.
  - SGLang fork branch: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
  - commit: `e631a13fd`
  - task packet: `tasks/sglang_qwen_fp4kv_dense_cache_trace_probe_20260609.md`
  - runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
  - env gate: `SGLANG_FP4_KV_TRACE_DENSE_CACHE=1`
  - purpose: compare full-prefill and cached-prefix Qwen FP4-KV tensors at FlashInfer
    attention, Qwen2 hidden states, raw/sampled logits, and sampler preprocessing.
  - follow-up hardening: the same branch now stamps `forward_pass_id`, `kind`, request ids,
    and logits/sampler `sample_rows` into trace payloads. The campaign comparator rejects
    missing schema fields, unknown rid/request binding, and metricless structural matches.
  - local verification: `python -m py_compile` for touched files and `git diff --check`
    passed for the fork; `bash -n scripts/run_sglang_fp4_dense_cache_trace.sh` and
    `git diff --check` passed for the campaign runner.
  - live status: queued. Later host-access probes show the node visible in the Tailscale
    control plane, but Tailscale ping, TCP/22, and SSH time out from this workspace, so no
    live trace row was run in this stop point.
  - tailnet retry artifact: `results/gb10_host_access_probe_tailnet_retry_20260609.md`;
    hostname and raw `100.113.98.11` SSH probes both timed out.

- Added the llama.cpp supplied-token loglikelihood contract.
  - task packet: `tasks/llamacpp_supplied_token_loglikelihood_contract_20260609.md`
  - purpose: pin the row-8 acceptance primitive after the negative `b9536` OpenAI echo
    and native top-N rows: context plus continuation in, continuation token ids and exact
    per-token logprobs out, with summed logprob and greedy-match boolean.
  - interpretation: top-N `/completion` output is not a fix unless every supplied
    continuation token is present; the `" zebra"` smoke row must score before GGUF
    lm-eval accuracy can leave blocked.

- Added the llama.cpp loglikelihood contract auditor.
  - script: `scripts/llamacpp_loglikelihood_contract_audit.py`
  - red artifacts:
    `results/llamacpp_native_loglikelihood_20260608T1331JST_contract_audit.json`,
    `results/llamacpp_native_loglikelihood_task_dryrun_contract_audit_20260609.json`
  - result: the live top-512 artifact is rejected because token id `1147` from
    `" zebra"` was not scored; the dry-run artifact is rejected because it has no token
    logprobs. A synthetic direct supplied-token artifact passes, proving the audit can
    accept the intended future endpoint shape.

- Added a llama.cpp source-level loglikelihood audit.
  - script: `scripts/llamacpp_source_loglikelihood_audit.py`
  - artifact: `results/llamacpp_source_loglikelihood_audit_20260609.md`
  - checkout: `jethac/llama.cpp@19bba67c1`
  - result: `stock_server_contract_capable=false`; server logprobs are generated-token /
    top-N oriented, OpenAI `echo` is rejected, prompt processing extracts only the last
    prompt token's logits, and `tools/perplexity/perplexity.cpp` contains the reusable
    logits+target-token scoring primitive for a future supplied-token endpoint.

- Implemented the llama.cpp supplied-token loglikelihood endpoint branch.
  - fork branch: `jethac/llama.cpp@spark/hijinks-008-supplied-loglikelihood`
  - commit: `aa6a5961977139f23ae54dc8279fdac3d1494a77`
  - artifact: `results/llamacpp_supplied_loglikelihood_endpoint_patch_20260609.md`
  - endpoints: `POST /loglikelihood`, `POST /v1/loglikelihood`
  - implementation: queued server task, direct target-token log-softmax from logits,
    summed continuation logprob, and lm-eval-style greedy boolean.
  - local verification: `git diff --check` passed; CPU-only WSL `llama-server` build
    passed through `server-context.cpp`, `server-task.cpp`, `server.cpp`, and final link.
  - live status: GB10 runtime validation remains pending because the Tailnet host is
    visible but not reachable over Tailscale ping, TCP/22, or SSH from this workspace.
- Root-caused the 2026-06-09 host outage: **vLLM unified-memory OOM → kernel hung-task deadlock, NOT thermal.**
  - doc: `docs/INCIDENT_20260609_OOM_DEADLOCK.md`
  - evidence (live SSH after recovery): global OOM-killer killed `VLLM::EngineCor` (pid 400721, in a Docker container) mapping ~123 GB on a 119 GiB box; the killed process held an mmap rw-semaphore in the NVIDIA driver, so `jbd2`/`gnome-shell`/`gsd-color`/`systemd-journal` blocked 122→368 s and never cleared. `kernel.hung_task_panic=0` → no auto-reboot → required a physical power-cycle. GPU 40 °C, zero thermal/throttle events — heat exonerated. Recurring: NVRM OOM + OOM-kills on Jun 06, 07, and 09.
  - root cause: GB10 unified memory — `--gpu-memory-utilization` is a fraction of the *shared* 119 GiB pool, not separate VRAM. Big model + NVFP4 KV pool (especially the matched fp8-vs-nvfp4 comparator running two servers at once) exceeds the pool; global OOM + driver-held mmap_lock deadlocks the kernel (vs a clean CUDA-process kill on a discrete-VRAM box).
  - standing mitigations: leave ≥15–20 GiB OS headroom (don't run 0.85+ mem-fraction); never run the fp8+nvfp4 comparators concurrently at high mem-fraction (sequential, or cap so the sum < ~100 GiB); cgroup-limit the container (`--memory`) so a runaway is killed in-cgroup instead of wedging the kernel; optionally `kernel.hung_task_panic=1`+`kernel.panic=30` so a headless hang auto-reboots. Linked from README Start Here.

- Recovered the live GB10 host over Tailnet and completed the vLLM Gemma 3 FlashInfer prefill debug packet.
  - host access: Tailscale ping, TCP/22, and key-based SSH worked through `thinkstationpgx-00b4` / `100.113.98.11`.
  - artifact: `results/vllm_gemma3_flashinfer_prefill_debug_20260609T143948JST_summary.md`
  - run checkout: campaign `0713537`, `jethac/vllm@1fabc6649`, `jethac/flashinfer@96be2fa8`
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - result: Gemma 3 NVFP4-KV first-token corruption reproduced (` Reigns`, Gujarati text, `ioane`).
  - FlashInfer audit: the corrected parser now sees `4` paged-prefill identity lines and `4` bound tensor lines across SWA (`window_left=1023`) and global (`window_left=-1`) calls. Every live generated module is compiled as `dtype_kv=uint8_t`, `require_fp4_kv=0`, `is_kv_fp4x2=0`, with empty `additional_tensors` and no `maybe_k_cache_sf` / `maybe_v_cache_sf`.
  - interpretation: this stop point moves the vLLM Gemma 3 failure from "mysterious byte-like FA2 output" to a concrete vLLM/FlashInfer Python/JIT binding bug: the paged-prefill module is not being specialized as FP4 KV and is not receiving scale tensors. Do not chase lower FA2 prefill math until that binding path is fixed and re-run.

- Applied GB10 memory guardrails to active serving runners after the OOM-deadlock incident.
  - lowered vLLM runner defaults from `--gpu-memory-utilization 0.85` to `0.72` for the Gemma 3 debug runner, Qwen PPL pair, and AEON vLLM reproduction runner.
  - added Docker cgroup caps (`GB10_DOCKER_MEMORY=100g`, `GB10_DOCKER_MEMORY_SWAP=100g`) to active vLLM and SGLang serving/debug runners.
  - kept matched comparator runs sequential where the runner already does that; future packets must not run two large fp8/nvfp4 servers concurrently on the shared memory pool.

- Ran the follow-up vLLM Gemma 3 FlashInfer binding-fix probe under the new memory rules.
  - artifact: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_summary.md`
  - run checkout: temp Spark checkout with `jethac/vllm@1fabc6649` and `jethac/flashinfer@96be2fa8`
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - safety: single server, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - positive: the pre-serve import probe now maps `torch.uint8` KV to `__nv_fp4x2_e2m1` in both FlashInfer JIT dtype tables and names the safe URI component `fp4x2_e2m1`.
  - red result: serving output remains corrupt (` Reigns`, Gujarati token, `ioane`), and the worker-side paged-prefill debug lines still show `dtype_kv=uint8_t`, `fp4_kv=0`, `require_fp4_kv=0`, empty `module_uri`, and no scale tensors.
  - interpretation: the import path can be patched, but the actual EngineCore paged-prefill binding path still resolves a stale/raw-byte generator or cached module. The next fix must instrument/force the worker-side FlashInfer binding path before returning to FA2 math, SWA reuse, or quality-gate work.

- Ran the vLLM Gemma 3 FlashInfer worker-binding probe that rebinds the standard batch-prefill generator path.
  - artifact: `results/vllm_gemma3_flashinfer_worker_bind_20260609T1552JST_summary.md`
  - patch under test: `sitecustomize.py` now rebinds `flashinfer.jit.gen_batch_prefill_module` and `flashinfer.prefill.gen_batch_prefill_module` to the local patched FlashInfer source, in addition to the customize generator; the runner clears all cached/generated batch-prefill modules across FlashInfer versions.
  - safety: single reduced Gemma 3 server, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - positive: the pre-serve probe confirms both dtype maps and both standard generator bindings point at the patched FlashInfer path.
  - result: the server fails during FlashInfer warmup with `TypeError: Mismatched number of arguments ... Expected 29 but got 27 arguments`.
  - interpretation: the failure mode moved forward. FlashInfer now generates an FP4-KV-aware paged-prefill module that expects scale-factor tensor arguments, but vLLM still calls it with the old raw-KV argument list. Next vLLM work is to pass `maybe_k_cache_sf` / `maybe_v_cache_sf` from the split NVFP4 KV views into `prefill_wrapper.run(...)`, then rerun the Gemma quality gate.

- Ran the vLLM Gemma 3 no-force prefill call-site probe.
  - artifact: `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_summary.md`
  - patch under test: external FlashInfer prefill generator forcing disabled (`SPARK_FLASHINFER_FORCE_PREFILL_MODULE=0`); vLLM prefill metadata appends `maybe_k_cache_sf` / `maybe_v_cache_sf`; the FlashInfer prefill wrapper includes K/V scale tensors in JIT optional-argument preparation.
  - safety: single reduced Gemma 3 server, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - positive: the server reaches readiness and the worker-bind `Expected 29 but got 27 arguments` warmup mismatch is gone.
  - red result: first-token output remains corrupt (` Reigns`, a Gujarati token, `ioane`), and the live C++ prefill debug lines still select the raw-byte module for both SWA and global prefill (`dtype_kv=uint8_t`, `require_fp4_kv=0`, `is_kv_fp4x2=0`, empty `additional_tensors`).
  - interpretation: scale-argument plumbing is necessary but not sufficient on this installed FlashInfer stack. The next fix is FlashInfer prefill module-selection/metadata/generator plumbing so the natural path selects the FP4-KV module and then receives the scale tensors; do not reopen page-byte pairing or generic FA2 math.

- Completed the vLLM Gemma 3 FP4 paged-prefill struct/scalar fix and cleared the short first-token gate.
  - artifact: `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_summary.md`
  - FlashInfer fork: `jethac/flashinfer@0919cdda` fixed the FP4 SWA paged-prefill `PagedParams` scale-stride compile hole; `jethac/flashinfer@c3dae30f` completes the Python wrapper plumbing for JIT internal tensors and scalar args.
  - safety: single reduced Gemma 3 server, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - result: server reaches readiness; the previous `Expected 29 but got 24/26 arguments` warmup failures are gone; runtime debug shows FP4 modules for both SWA (`window_left=1023`) and global (`window_left=-1`) prefill.
  - quality: `scripts/gemma_nvfp4_kv_quality_gate.py` passes against the prior fp8 first-token baseline. Candidate first tokens are `spark`, `4`, `A`; minimum top-logprob overlap ratio is `0.772727`.
  - caveat: this is a short first-token/logprob gate, not a long-context PPL or throughput blessing. The debug audit script is stale for the current raw line format and fails despite visible raw evidence.

- Attempted the immediate SGLang cross-lane retest after the FlashInfer prefill fix.
  - artifact: `results/sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST_summary.md`
  - intended test: default Qwen FP4-KV cached-prefix/radix row using editable FlashInfer with the new prefill wrapper fix.
  - outcome: inconclusive before serving. The runner was still rebuilding `sglang-kernel` after ~26 minutes (`82/127` build targets) and had not launched SGLang; no request JSON or dense-cache comparison was produced.
  - interpretation: this does not test or falsify the cross-lane FlashInfer hypothesis. Next SGLang work should prepare a reusable source-stack image or narrower kernel build target, then rerun the default radix row from that prepared stack.
  - build warning to track: SGLang `sglang-kernel` emits repeated `ptxas` warnings for `compute_120a`/`compute_121a` about `.multicast::cluster` on `cp.async.bulk{.tensor}` being intended for datacenter targets, plus `setmaxnreg` compatibility warnings.

- Ran the first vLLM Gemma 3 sequential prompt-logprob PPL gate after the FlashInfer prefill fix.
  - artifact: `results/vllm_gemma3_27b_ppl_20260609T1852JST_ctx512_summary.md`
  - runner: `scripts/run_vllm_gemma3_ppl_pair.sh`
  - checkout: campaign `dad4468`, `jethac/vllm@0c278200e`, `jethac/flashinfer@c3dae30f`
  - safety: fresh checkout, sequential fp8 then NVFP4 servers, `CTXS=512`, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - fp8: PPL `115.4583`, mean NLL `4.7489` nats/token, `511 / 511` supplied prompt tokens scored.
  - NVFP4: PPL `119.8578`, mean NLL `4.7863` nats/token, `511 / 511` supplied prompt tokens scored.
  - delta: `+4.3995` PPL, `+0.0374` nats/token.
  - interpretation: this is the first Gemma 3 supplied-token PPL evidence that the fixed NVFP4-KV path is quality-sane on short text-only prompts. It strengthens the first-token green gate, but it is not yet a long-context/SWA-window stress test, throughput row, or full Gemma 3 blessing.

- Ran the larger vLLM Gemma 3 sequential prompt-logprob PPL gate.
  - artifact: `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_summary.md`
  - runner: `scripts/run_vllm_gemma3_ppl_pair.sh`
  - safety: sequential fp8 then NVFP4 servers, `CTXS="1024 2048"`, `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=4096`, vLLM memory fraction `0.72`, Docker `100g` cgroup cap; host returned to ~115 GiB available after teardown.
  - fp8 1024: PPL `35.0895`, mean NLL `3.5579` nats/token, `1023 / 1023` supplied prompt tokens scored.
  - NVFP4 1024: PPL `35.2563`, mean NLL `3.5626` nats/token, `1023 / 1023` supplied prompt tokens scored.
  - fp8 2048: PPL `20.5861`, mean NLL `3.0246` nats/token, `2047 / 2047` supplied prompt tokens scored.
  - NVFP4 2048: PPL `20.4757`, mean NLL `3.0192` nats/token, `2047 / 2047` supplied prompt tokens scored.
  - delta: `+0.0047` nats/token at 1024, `-0.0054` nats/token at 2048.
  - interpretation: Gemma 3 NVFP4-KV stays quality-sane through 2048 supplied-token PPL on this corpus slice, including a row that crosses the 1024-token SWA window. The next vLLM Gemma 3 gate is capacity/concurrency and throughput, not another short quality smoke.

- Ran the current-head SGLang Qwen FP4-KV dense-vs-cached trace from a reusable source-stack image.
  - artifact: `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_summary.md`
  - runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
  - checkout: campaign `2c0a475`, `jethac/flashinfer@c3dae30f`, `jethac/sglang@e631a13fd`
  - image: `sglang-source-stack-c3dae30f-e631a13fd`
  - stack: editable FlashInfer `0.6.13`, editable SGLang `0.5.12.post2.dev1022+ge631a13fd`, source-built `sglang-kernel 0.4.3`, Torch `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`
  - result: red. Default FP4 cached-prefix reuse still flips the first token: no-cache rows emit `**` at logprob `-0.7235294580459595`; 55-token radix-hit rows emit `ark` / `838` at logprob `-0.5874708890914917`.
  - controls: flush-between and namespace-isolated rows keep `cached_tokens=0` and remain clean (`**` at the same no-cache logprob).
  - trace status after parser repair: `733` trace events total, `301` warmup/health-check events ignored, `432` request-bound events compared (`324` dense, `108` cached, `0` unknown), `648` matched tensor comparisons, `0` metricless rows, no schema findings. First localized request-bound divergence is attention output at layer 0: dense full-prefill `o_rows` versus cached-prefix merged `merged_rows` (`cosine=0.006467887232207366`, `max_abs=0.318359375`, `rms=0.13599727805129772`). The next SGLang step is to inspect why FP4 cached-prefix attention output disagrees despite the earlier paged-prefix read/LSE/merge references passing.
  - build/perf note: the source build succeeds but emits repeated SM12x warnings (`242` `.multicast::cluster` advisories, `109` `compute_120a` refs, `109` `compute_121a` refs, `74` `setmaxnreg` compatibility warnings). This is separate from FP4-KV correctness and should be tracked as an SGLang SM12x performance-portability issue.

- Ran the SGLang Qwen mixed-KV default radix row.
  - artifact: `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`
  - runner: `scripts/run_sglang_fp4_dense_cache_trace.sh` with `MIXED_KV=1`
  - runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
  - launch shape: `--attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --page-size 1 --mem-fraction-static 0.40`, CUDA graph and piecewise graph disabled.
  - storage policy: FP8 K plus packed NVFP4 V; server log confirms `SGLang FP4 KV mixed mode enabled`.
  - result: green for the targeted radix first-token gate. The old second-request cached-prefix failure (`ark` / token `838`) is gone: both cached-prefix second requests now emit `**` / token `334` at logprob `-0.7601577043533325`, while fresh controls remain `**` at `-0.7235294580459595`.
  - audit: dense-cache summary audit `ok: true`, `733` trace events, `648` metric comparisons, no findings.
  - capacity log: `#tokens: 5573469`, `K size: 37.21 GB`, `V size: 20.93 GB`, `max_total_num_tokens=5573469`.
  - caveat: this is not exact tensor equality or long-form quality. The first request-bound dense-vs-cached tensor delta remains layer-0 attention output (`cosine=0.4661444810372346`, `max_abs=0.2578125`, `rms=0.11784679304779001`), but it no longer flips the tested first token. Next gates are a fresh sequential fp8 comparator, long-form/benchmark quality, throughput, and graph-safety.

## First Benchmark Campaign Summary

The initial personal Gemma 4 benchmark run was run on `thinkstationpgx-00b4` in `/home/jethac/gemma4-evals`.

At the last local sync:

- smoke rows: 152 complete
- smoke `ok`: 21
- smoke `eval_failed`: 11
- smoke `loader_failed`: 120
- full eval records: 70
- full eval `ok`: 65
- full eval failed: 5
- throughput rows observed: 2
- MTP rows observed: 2

That personal benchmark run was still in full accuracy when monitoring stopped. It was not killed.

<!-- source: docs/CAMPAIGN_PLAN_EPOCH2.md -->

# Campaign plan, epoch 2: NVFP4 KV everywhere + DiffusionGemma

Date: 2026-06-11. Branch: `epoch2` is THE working branch for both agents
(replaces the 022/docs split; both lanes commit here; `git pull --rebase`
before every push). Blog/Colab/upstream filing remain gated on completing
the FULL ladder (Jetha's standing call - no publication-date sequencing).

## Why epoch 2
DeepMind released DiffusionGemma (2026-06-10): block-autoregressive discrete
diffusion on the Gemma 4 26B-A4B base. Day-zero NVFP4 support is
WEIGHTS-ONLY (Model Optimizer checkpoint + vLLM playbooks). Its decoding
re-reads the cached prefix KV 12-48x per canvas - KV bandwidth becomes the
dominant decode cost - so NVFP4 KV (our territory, still nobody else's) is
worth MORE here than on autoregressive models. Our assets map directly:
VO-split + dispatcher fix (76af7982) if it inherits D=512 globals;
bidirectional-canvas masking = the mm-prefix custom-mask problem;
harnesses/provenance gates carry over unchanged.

## Rungs (additions; the existing Gemma ladder rungs all remain)
- DG-0 baseline: stock day-zero DiffusionGemma NVFP4-weights on Spark vLLM.
  Measured: tok/s vs context length, KV pool, attention backend + per-layer
  geometry from SERVING DISPATCH, canvas/prefix cache implementation notes.
- DG-1 cache analysis: how vLLM's DiffusionGemma manages prefix KV + canvas
  attention; whether reshape_and_cache + our NVFP4 writer/reader engage
  unmodified; what masking the denoising steps demand.
- DG-2 full NVFP4 KV: knobs on; canvas masking enablement if required
  (FlashInfer custom-mask = generalization of task 21's deferred half).
- DG-3 the killer benchmark: decode tok/s vs context length, NVFP4-KV vs
  fp8/bf16-KV - the KV-read-amplification thesis measured. Capacity +
  quality rows per campaign standards.
- DG-4: AR Gemma 4 26B-A4B rung inherits everything from DG-2/3.

## Lane split
**Claude (vLLM + FlashInfer kernel lane):**
1. Upstream-overlap audit (vLLM PR #40082, 0.19 diffs near our files) ->
   rebase decision + plan.
2. Dispatcher-fix validation window (rt-base/rt5 flip green + regression
   slice) -> unblocks bf16 anchor -> quality table -> Triton-retirement
   benchmarks (existing ladder debt, stays scheduled).
3. DG-0/DG-1 on vLLM.
4. FlashInfer custom-mask (canvas/mm-prefix) enablement; split-dtype module
   keying (task 22, Codex's graph-gate consumer).
**Codex (SGLang + images/infra lane):**
1. Finish SGLang Gemma 4 rung 0/1 per existing goal (mid-flight).
2. r9 image: r8 + flashinfer 76af7982 (+ latch diag + cache hygiene gates
   as in r8).
3. SGLang DiffusionGemma feasibility: does SGLang have/want day-zero
   support? Scope what serving it takes (its diffusion scheduling is
   nontrivial in any engine).
4. Inherits split-dtype keying when Claude lands it -> mixed-KV graph gate.

## Protocols
- Box: CLAUDE_WINDOW_OPEN marker unchanged (present = Claude holds Spark).
- Messages: mail/ on epoch2 (see mail/README.md). Jetha no longer relays.
- Evidence/provenance/memory guardrails: unchanged from epoch 1.

## Post-ladder capstone (gated; Jetha 2026-06-11)
llama.cpp NVFP4 KV cache contribution starts ONLY after vLLM + SGLang are
verified across Gemma 3, all Gemma 4 sizes, and DiffusionGemma. Approach:
extend ggml's existing GGML_TYPE_NVFP4 to KV (writer kernel + fattn
converter + plumbing), contributed upstream with our layout/provenance
lessons and the cross-implementation quality evidence - NOT a FlashInfer
port. Details in task 28.

<!-- source: docs/CLAUDE_GOAL_EPOCH2.md -->

# Claude /goal, epoch 2 (rev 2, 2026-06-11)

Branch: `epoch2` (single working branch; pull --rebase before push).
Plan: docs/CAMPAIGN_PLAN_EPOCH2.md. Mail: mail/ per mail/README.md - check
at session start, stop points, box windows; coordinate with Codex by mail.

OPERATING MODE: autonomous. Work the queue continuously; do not stop to ask
for go-aheads between items. Stop ONLY when blocked on something only Jetha
can do (account access, hardware power, purchases, publication decisions)
or a genuine scope change. Report progress as work lands, not as requests.

PLATFORM POLICY (rev 2): Colab lane is PAUSED - do not iterate on the
notebook or the sm_120a wheel unless Jetha reopens it. sm_120 verification
runs on the LOCAL P520 (RTX 5060 Ti via WSL2 Ubuntu; rerunnable bring-up at
B:\workshop\wsl_sm120\, results pattern in
results/wsl_sm120_fix_validation_20260611/). The P520 is Claude-driven,
zero-contention; use it for all probe/kernel-level validation before any
Spark window. Spark = serving/capacity rows only, via marker protocol at
Codex's gaps.

QUEUE (work top to bottom; interleave Spark windows opportunistically):
1. DG-0 window packet (author offline): docker-pinned baseline per
   docs/DG0_SERVING_STACK_RECON.md + task 24; combined with the 31B bf16
   anchor row (r9 image, task 17) in ONE Spark window at Codex's next gap.
2. Run that window -> close the epoch-1 quality table + open the
   DiffusionGemma ledger. Mail Codex results.
3. FlashInfer canvas-mask enablement (bidirectional-in-canvas + causal
   prefix; generalizes the mm-prefix custom-mask work) - author + validate
   on the P520.
4. Split-dtype module keying (task 22) - author + P520-validate; mail Codex
   to unpark its graph gate.
5. DG-1 cache analysis -> DG-2 (full NVFP4 KV on DiffusionGemma) -> DG-3
   (KV-read-amplification benchmark).
6. Epoch-1 ladder debt as windows allow: E4B/12B AFTER rows, fp8 1-byte
   guard term, M4 (low).
Standing: keep docs/RESULTS_LEDGER.md and the support matrix current as
rungs flip; provenance gates (latch diag, EXT_PATH, md5) on every run; blog
remains gated on the full ladder.

<!-- source: docs/CLAUDE_GOAL_VLLM_DIFFUSIONGEMMA_NVFP4_KV.md -->

# GOAL (Claude / vLLM lane): DiffusionGemma NVFP4 KV-cache on vLLM (DG-V rungs)

**One-liner:** bring full-NVFP4 K+V KV cache (3.556x, format-exact) to DiffusionGemma
26B-A4B on vLLM/Blackwell (sm_120 + sm_121), to parity with SGLang's DG-R5/R6 receipts.
Closes the gap that the engine with the *official* DiffusionGemma recipe serves it bf16-only.

## Why this exists
vLLM supports DiffusionGemma (official recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)
but in bf16 only -- no NVFP4 KV, no Blackwell, no FlashInfer. We routed ALL DiffusionGemma
work to Codex's SGLang lane and never applied the NVFP4-KV campaign to it on vLLM. DiffusionGemma
is the strongest 4-bit story we have (the decoder re-reads the full prefix every denoise step,
so 3.556x KV capacity compounds harder than anywhere in the AR ladder) -- so it should NOT be
missing from the more-visible vLLM path.

## Reference (do NOT re-derive -- match confirmed-good)
- vLLM AR NVFP4-KV impl (`spark/hijinks-e2-vllm`): NVFP4 writer, `config.py` per-layer
  routing, `flashinfer.py` VO-split. Proven across E2B->31B on Pro 6000 + Spark.
- SGLang DG-R5/R6 (Codex): proves the NVFP4 read path tolerates block-diffusion attention.
  Format is geometry-independent; these are the parity target.
- Upstream vLLM DiffusionGemma recipe: the model class + serving constraints
  (`--max-num-seqs 4`, `--gpu-memory-utilization 0.85`, entropy-bound sampling overrides,
  256-token block denoise loop).

## Gate 0 -- PROBE FIRST, decide before building (the one real unknown)
Does vLLM's DiffusionGemma attention backend expose a hook to the FlashInfer NVFP4 path
the way the AR models do?
- (a) Does the block-diffusion decoder route through the same paged-prefill/decode wrappers
  our VO-split patches?
- (b) Does the bidirectional-within-block mask survive the VO-split two-pass over V halves?

**If no clean hook -> STOP, write the blocker up, escalate. Do not force it.**
If yes -> proceed to Build.

## Build (only past Gate 0 -- wiring, not new kernels)
1. Wire vLLM's DiffusionGemma model class into the existing NVFP4-KV config routing
   (`kv_data_type=uint8`, D=512 VO-split, linear-V-SF knob).
2. DG-specific allocator accounting: the 256-token-block denoise loop re-reads the full
   prefix every step -- confirm the NVFP4 page budget (`9*head/16`) holds under that
   re-read pattern with `--max-num-seqs 4`.

## Per-rung green bar (zero-bug)
- **DG-V5** (= SGLang DG-R5): full-NVFP4 K+V serves coherent generations, **>=3.5x KV
  capacity** vs bf16 denominator, **double-run bitwise-deterministic**.
- **DG-V6** (= DG-R6): perf pair (NVFP4 vs bf16 throughput/latency at matched batch);
  quantify the compounding win from per-denoise-step prefix re-read.
- Both on **sm_120 (Colab Pro 6000) AND sm_121 (Spark)**; bf16 denominator captured each.

## Done
DG-V5 + DG-V6 green on both silicon, receipts banked to `results/`, the two vLLM
DiffusionGemma cells flip (gap -> green). DiffusionGemma is then cross-engine complete and
the FlashInfer surface is proven on the diffusion path in BOTH stacks -- clearing the last
DG-shaped item under the "surface stable across all Gemma variants before filing" PR gate.

## Gate 0 VERDICT (2026-06-12): GO -- lift smaller than assumed
Probe done. DiffusionGemma routes through machinery we already own and proved:
- **Model id** `google/diffusiongemma-26B-A4B-it`; arch `DiffusionGemmaForBlockDiffusion`,
  `model_type: diffusion_gemma`. Block-diffusion MoE (128 experts top-8) + vision,
  30 layers sliding/full alternating, `max-model-len 262144` (256k -> NVFP4 KV pays off hard).
- **It is NOT mainline-plain:** the recipe needs "a vLLM build with diffusion support in the
  Gemma docker image." Upstream vLLM main DOES have `vllm/model_executor/models/diffusion_gemma.py`;
  **our fork (e2-vllm @ e32459eea) predates it** -> port it in, don't author it.
- **diffusion_gemma.py delegates its text decoder to `Gemma4Model`** -- the exact backbone our
  NVFP4 writer + config routing + VO-split already cover. Attention goes through vLLM's standard
  `Attention` + `build_attn_metadata` + `kv_cache_dtype` (NOT a bespoke attention call).
- **The block-diffusion mask is a per-request `causal` flag** in attn_metadata: encoder phase=causal,
  decoder phase=bidirectional (`causal=False`). Standard non-causal prefill -- FlashInfer supports it;
  SGLang DG-R5 already proved the NVFP4 read path tolerates the bidirectional decoder phase.
- **`attention_k_eq_v` wrinkle is ALREADY HANDLED in our backbone:** DiffusionGemma full-attn layers
  have no v_proj (V=K pre-k_norm). Our `gemma4.py` (use_k_eq_v, lines ~431/462/570/626) loads K into
  both K and V qkv_proj slots, so V==K flows automatically and the NVFP4 V-writer sees a normal V.
- **head_dim 256 (16 heads / 8 KV) per config** -> VO-split likely NOT on the critical path here;
  whatever the gemma4 text_config sets, our head-dim-conditional VO-split engages or no-ops itself.

**Revised lift (wiring, confirmed):** (1) port `diffusion_gemma.py` + its config plumbing + the
block-diffusion scheduling (diffusion_states, per-req causal buf, denoise loop) from upstream main
onto our NVFP4-KV branch; (2) register the `DiffusionGemmaForBlockDiffusion` arch; (3) NVFP4 routing
is automatic (Gemma4Model underneath) -- confirm it engages for the new arch name; (4) empirically
confirm the bidirectional (causal=False) decoder phase reads NVFP4 correctly on vLLM (the one genuine
remaining unknown, de-risked by SGLang DG-R5). No new kernel work identified.

## CORRECTION (2026-06-12): this is a RECONCILIATION, not a port
DiffusionGemma-on-vLLM was NOT un-started. Branch `spark/hijinks-e2-dgemma` (local + origin)
already carries:
- the full model stack: `diffusion_gemma.py`, `config/diffusion.py`,
  `transformers_utils/configs/diffusion_gemma.py` (the DG-0 baseline, task #24);
- commit `6fe6d798f` "spark-hijinks epoch-2 patch set on the dgemma branch (DiffusionGemma stack)"
  -- the runner/scheduler/sampler block-diffusion denoise integration;
- commit `dfb427952` "DG-2: per-request causal grouping in the FlashInfer backend + gate lifts"
  -- config.py +25, **flashinfer.py +313**: the NVFP4 diffusion attention path, WIRED but never
  verified-and-banked (no receipts on the branch; task #27 still pending).

It never merged to the integrated NVFP4 line, and it diverged: dgemma is 23 commits off our
HEAD, and the integrated line moved **44 commits** since (AR-ladder completion, MTP, Triton
retirement, mm-prefix). Only `k_eq_v` reached mainline (via the MTP merge) -- which is why the
integrated branch showed DiffusionGemma as a gap.

**So DG-V = reconcile the e2-dgemma DiffusionGemma stack onto the current confirmed-good
integrated NVFP4 HEAD, then verify DG-V5/V6 + bank receipts. NOT a from-scratch port.**

**Crux conflict:** `flashinfer.py` three-way merge -- dgemma's DG-2 causal-grouping (+313) vs
HEAD's 44-commit evolution of the SAME file (VO-split refinements, MTP backend pin, mm-prefix
masking, the max_mma_kv dispatcher fix). Plus `config.py` routing. The model/config files are
clean adds (no conflict).

**Plan:** off a working branch from HEAD: (1) bring the clean model/config files over as-is;
(2) cherry-pick the two MEANINGFUL dgemma commits (6fe6d798f stack + dfb427952 DG-2), discarding
the noise commits (race-fix/cleanup/revert churn), resolving the flashinfer.py + config.py
three-way by hand against the current surface; (3) register the arch; (4) build the wheel;
(5) run DG-V5/V6 on sm_120 + sm_121 and bank. The DG-2 causal grouping is the asset to preserve
through the merge -- it's the diffusion-specific FlashInfer logic, already authored.

## EXECUTION STATUS (2026-06-12)
- Working branch `spark/hijinks-e2-dgv` created off integrated HEAD (e32459eea).
- **Relief found:** HEAD already has the gpu/ v2 runner infra (model_states/, attn_utils,
  sample/, buffer_utils) that diffusion_gemma.py imports -> NO runner subsystem to port.
  Same v2-runner base as upstream PR eb28452b1, so the CLEAN upstream PR is the preferred
  source for shared-file diffusion hunks (over the entangled dgemma branch).
- DONE: landed the 3 clean new files (diffusion_gemma.py 1359L + 2 config files) on e2-dgv.
- NEXT: (a) shared-file diffusion hunks from PR eb28452b1 -> registry, model/config.py routing,
  transformers config registration, model_runner denoise loop, model_states diffusion ModelState,
  scheduler/sampler/cudagraph hooks; (b) replay DG-2 flashinfer causal grouping (dfb427952,
  +313) onto the current flashinfer.py -- the crux, hand-resolved; (c) arch register; (d) build
  sm120a wheel; (e) DG-V5/V6 both silicon + bank.

## CODE INTEGRATION COMPLETE (2026-06-12) -- branch spark/hijinks-e2-dgv @ 52bfe5c34
Pushed to origin. Delta vs integrated HEAD: 53 files, +3500/-317. Three commits:
1. `291621ecb` cherry-pick upstream PR eb28452b1 (DiffusionGemma model + runner denoise
   integration + FA/Triton mixed-causal base) onto our NVFP4 HEAD. Auto-merged almost
   entirely (v2-runner match); only 2 trivial conflicts (docs table + the diffkv triton
   file kept from PR). py_compile clean. Arch `DiffusionGemmaForBlockDiffusion` registered.
2. `e8acd5236` **unify FlashInfer prefill grouping** -- the crux. Merged HEAD's
   FIPrefillMMGroup (image-span packed-mask) + DG-2's FIPrefillCausalGroup (per-request
   causal) into ONE `FIPrefillGroup` keyed by (is_mm, causal). Handles mm-only, causal-only,
   AND composed mm x causal (DiffusionGemma is multimodal -> needs both). Legacy scalar-causal
   no-mm path byte-identical behind `prefill_groups is None`. NVFP4 jit_args + head_dim_vo
   VO-split preserved per group. DG-2 guards + supports_non_causal=True preserved. (Subagent
   authored under precise spec; I reviewed the full diff -- dispatch gate, impl gather/run/
   scatter loop, VO-split-per-group, coherent rename all verified.)
3. `52bfe5c34` knob-gate the DiffusionGemma FLASHINFER allowance (DG-2 routing) -- replaces
   upstream's hard "unsupported" raise with the VLLM_FLASHINFER_VOSPLIT/VLLM_NVFP4_KV_VOSPLIT
   gated path.

**Verified statically:** py_compile clean on all changed Python; coherent rename (grep clean);
legacy AR-ladder path structurally untouched. **NOT yet verified:** anything numeric (no build/GPU).

## REMAINING (build + GPU -- needs marker coordination before touching a card)
1. Build sm120a + sm121a-arm64 wheels off e2-dgv.
2. **Composed mm x non-causal mask GPU cosine** -- the one flagged-untested unification branch
   (causal_base=False). Moot at DG shipping config (canvas 256 <= window 1024) but verify.
3. DG-2 4-request mixed-batch cosine harness re-run on the unified path (sanity vs the prior
   0.999998 baseline).
4. **AR-ladder regression**: re-run a Gemma 4 NVFP4 row (e.g. E4B) to confirm the unification
   didn't disturb the proven path (legacy + grouped both exercised).
5. DG-V5 (full-NVFP4 K+V coherent + >=3.5x + bitwise) + DG-V6 (perf pair) on sm_120 + sm_121.
6. Bank receipts to results/, flip the two vLLM DG cells green.

## VALIDATION PROGRESS (2026-06-12)
- **CI wheels building** off e2-dgv: sm120a (run 27404118729) + sm121a-arm64 (27404118751),
  triggered via push-branch add. ~2-5h compile. These are the clean deployable wheels +
  the proper base for all serve validation.
- **Overlay validated ABI-safe**: only non-Python drift base-wheel(6adc00f70)..e2-dgv is one
  cmake build-config file (vllm_flash_attn.cmake, build-time only); ZERO .cu/.cpp drift. So a
  Python-only overlay (wheel .so + e2-dgv .py at ~/dgv_overlay) is sound for bf16/legacy-path
  checks (the cmake bump only matters for the DG FA-diffkv kernel -> validated on Spark).
- **LIVE IMPORT + REGISTRATION SMOKE GREEN** (P520 WSL2, overlay, GPU-free): vllm imports e2-dgv
  with real _C + flashinfer source; unified `FIPrefillGroup` present + `supports_non_causal=True`;
  `DiffusionGemmaForBlockDiffusion` registers; diffusion configs import. Beyond py_compile --
  confirms the reconciliation loads in a real vLLM runtime (no import-time breakage). P520 GPU
  released (never occupied -- import-only).
- **DEFERRED to the e2-dgv CI wheel + proper harness** (the wheel env lacks pytest + the direct
  1B serve harness; ad-hoc reconstruction risks a false-green): the bf16 AR-ladder PPL regression
  (target: match banked g3-1b FLASHINFER bf16 2.3571850630239095 -> proves legacy path byte-id on
  silicon), the DG-2 4-request cosine, and (Spark, sm_121) DG-V5/V6 nvfp4 coherence.

## WHEEL GREEN + P520 LEGACY-PATH REGRESSION GREEN (2026-06-12)
- Both CI wheels built SUCCESS off e2-dgv @ 98cd3e59f: `sm120a-wheels-98cd3e59f`,
  `sm121a-arm64-wheels-98cd3e59f` (Latest). On-CI, no GPU-box build (per Jetha).
- **P520 (sm_120) bf16 legacy-path regression: GREEN, byte-identical.** Served Gemma 3 1B
  bf16 FLASHINFER on the e2-dgv overlay (VLLM_BUILD_CHECK -> dgv_overlay/vllm; backend proof
  = AttentionBackendEnum.FLASHINFER; ready 99s, no wedge). C1 ctx8191 x2:
  **mean_nll_nats = 2.3571850630239095 (both rows)** == banked baseline 2.3571850630239095
  to 16 figures. Coherent ("The capital of Japan is Tokyo."). => the unified FIPrefillGroup
  refactor does NOT disturb the legacy/AR-ladder serving path on real silicon. Crown-jewel
  safety gate PASSED. Artifacts: ~/g3_1b_retest/results/claude_p520_dgv_fi_bf16_*.

## REMAINING: Spark DG-V5/V6 nvfp4 coherence (sm_121, the headline)
Serve the sm121a-arm64-wheels-98cd3e59f wheel clean on Spark (no overlay -- divergent image
lineage; fresh container + pip install wheel). DiffusionGemma 26B-A4B FLASHINFER + nvfp4 KV +
VO-split knobs. Targets (SGLang DG-R5/R6 parity): coherent (Tokyo/2+2/DGX Spark), full-NVFP4
proof (kv uint8, mixed_kv False), VO-split proof (global head_dim 512 -> vo 256), ~3.56x
capacity vs bf16, double-run bitwise; DG-V6 perf pair. Harness staged:
scripts/run_vllm_dgemma_dgv_spark.sh.

## SPARK DG-V5 BLOCKED ON ARM64-WHEEL PORTABILITY (2026-06-12)
Tried to serve the e2-dgv arm64 wheel on Spark; hit a 3-layer env mismatch retrofitting the
existing r10 container, fully diagnosed:
1. lineage: existing Spark gemma4 image is 9759e3b06 (022 line, divergent) -> can't overlay e2-dgv
   .py (MoE compiled drift). Used the WHEEL instead (clean swap).
2. torch: wheel needs torch 2.12 (CI pin); both Spark images ship torch 2.11. `_C.abi3.so`
   undefined-symbol until I upgraded torch->2.12 in-container (then `_C` loaded).
3. glibc: wheel's `_C_stable_libtorch.abi3.so` needs GLIBC_2.38; both Spark images are
   Ubuntu 22.04 / glibc 2.35. HARD WALL.
ROOT CAUSE: the **sm121a-arm64 CI workflow lacks the glibc gate sm120a has**. sm120a builds on
`ubicloud-standard-30-ubuntu-2204` + fails if any .so needs > GLIBC_2.35; the arm64 workflow
builds on `ubicloud-standard-30-arm` (Ubuntu 24.04), no gate -> glibc-2.38 wheel.
FIX: mirror sm120a in the arm64 workflow -> build on a 22.04 arm runner + add the glibc-2.35
gate, rebuild. Then the glibc-2.35 + torch-2.12 wheel loads in the r10 container after an
in-container `pip install torch==2.12.0 cu130` (proven: `_C` loads once torch matches). Then
serve DG-V5 per scripts/run_vllm_dgemma_dgv_spark.sh. (Alternative: build a full Ubuntu-24.04
arm e2-dgv serving image so no glibc/torch retrofit is needed.) Either way it's a CI/Ubicloud
build, not a Spark build. NOTE: this arm64-wheel glibc bug affects ALL Spark wheel deploys,
not just DG-V -- worth fixing regardless.

## SPARK IMAGE DECISION REVISED (2026-06-12): match r10, don't rebuild from scratch
Jetha: "wouldn't it make more sense to build spark containers that are ubuntu 22.04 + torch 2.11."
YES -- better. The r10 image (22.04/glibc-2.35/torch-2.11/flashinfer-76af7982/transformers-5.11.0)
already serves the full Gemma 4 nvfp4 ladder green. So build the arm64 wheel to FIT r10, then
bake r11 = r10 + the e2-dgv wheel (swap only vLLM; reuse the validated stack). Implemented:
`build-sm121a-arm64-wheel.yml` retargeted to `ubicloud-standard-30-arm-ubuntu-2204` + torch 2.11.0
+ a GLIBC_2.35 gate. Rebuild: run 27409465227. The from-scratch 24.04/torch-2.12 image
(Dockerfile.spark + build-spark-image.yml) is PARKED dispatch-only (the 24.04 build failed anyway;
it's the heavier "full CI-repro" end-state). Wheel matrix: x64/Colab=22.04/torch-2.12 bare;
arm64/Spark=22.04/torch-2.11 baked into r11. Decision mailed to Codex (mail 0089).
NEXT: when the retargeted wheel lands, bake r11 on Spark (r10 + wheel layer-add, no compile),
serve DG-V5 via scripts/run_vllm_dgemma_dgv_spark.sh.

## (superseded) SPARK-READY IMAGE PATTERN (2026-06-12) -- the deterministic fix
Jetha: "adjust our work to build images that work as-is on spark." Implemented in the vLLM
repo (branch e2-dgv):
- `docker/Dockerfile.spark`: self-contained image -- Ubuntu 24.04 (glibc 2.39) + CUDA 13 cu130
  + torch 2.12 + the e2-dgv arm64 wheel + campaign FlashInfer source (JIT) + transformers 5.11.0
  + VO-split env defaults + GPU-free build-time provenance asserts. One consistent stack, so the
  glibc-2.38 wheel loads natively and it runs AS-IS on GB10 (host supplies only the driver).
- `.github/workflows/build-spark-image.yml`: builds on `ubicloud-standard-30-arm`, consumes an
  already-published arm64 wheel release (decoupled from compile), pushes to
  `ghcr.io/jethac/spark-vllm:e2-dgv-<sha>` via the built-in token. First run: 27408638063.
This kills the bare-wheel host-mismatch (lineage/torch/glibc) that blocked DG-V5. Spark just
`docker pull` + `docker run` -- no torch upgrade, no overlay, no glibc retrofit. Once the image
publishes, DG-V5/V6 serve via scripts/run_vllm_dgemma_dgv_spark.sh inside it.
Full end-state (later): move the rebuilt-C base build to Ubicloud too for end-to-end CI repro.

## DG-V5 SERVE: pipeline GREEN to the runtime; block-diffusion PROFILING HANGS (2026-06-12)
The whole serving pipeline now works end-to-end up to the serving runtime:
- Pivot validated ON HARDWARE: 22.04/torch-2.11 wheel (sm121a-arm64-wheels-3d6a0d507) loads in r10
  with ZERO retrofit (FIPrefillGroup/supports_non_causal/DiffusionGemma all True, no torch/glibc fight).
- **r11 baked + provenance-green**: jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11 (r10 + e2-dgv
  vLLM, swap-only). Dockerfile.r11 + docker/Dockerfile.r11 pushed.
- DiffusionGemma serve INITIALIZES correctly: nvfp4 KV engaged, FLASHINFER backend, **VO-split +
  VLLM_NVFP4_KV_LINEAR_V_SF=1 proof line present** ("NVFP4 V scale factors are linear"), 26B MoE
  loads (48.6 GiB). Two config gates surfaced + satisfied: enforce-eager (cudagraph OOM on 26B MoE),
  VLLM_NVFP4_KV_LINEAR_V_SF=1 (512-wide-head VO split).
- **BLOCKER: serve HANGS in the post-load memory-profiling dummy forward** (frozen right after
  "Using FlashInfer for top-p & top-k sampling"; no "GPU KV cache size" line; generation empty;
  container alive, no error). NOT fixed by --no-enable-prefix-caching --no-enable-chunked-prefill.
  This is a vLLM **block-diffusion serving-runtime** issue (the profiling forward vs the denoise /
  model_states integration), NOT our NVFP4-KV code (which is proven: P520 byte-identical + engages
  here correctly). Iteration cost ~6min/model-reload.

### NEXT (decisive bisect): serve **bf16** DiffusionGemma (--kv-cache-dtype auto) in r11.
- bf16 ALSO hangs -> block-diffusion serving-runtime general (upstream DG-in-our-base profiling
  path); investigate the profiling dummy-run vs the diffusion model_states/denoise setup.
- bf16 SERVES but nvfp4 hangs -> our nvfp4+VO-split+diffusion-attention interaction during profiling.
Either way it's a scoped serving-runtime follow-up; the NVFP4-KV contribution + the whole image/
wheel/r11 pipeline are done. (Note: upstream's recipe serves DG via `vllm serve` in the Gemma
docker image -> their newer vLLM base may profile DG correctly where our cherry-picked PR-on-older-
base has a gap.)

## BISECT RESULT (2026-06-12): hang is BLOCK-DIFFUSION-GENERAL, not nvfp4. KV code cleared.
Served bf16 DiffusionGemma (--kv-cache-dtype auto) in r11: **hangs IDENTICALLY** to nvfp4
(loaded 48.6 GiB, frozen at "Using FlashInfer for top-p & top-k sampling", no "GPU KV cache size"
line, generation empty, no error). => the post-load memory-profiling hang is independent of KV
dtype. Decisive corroboration: vLLM's profiling dummy run sets **skip_attn=True**
(model_runner.py:511-513) -> attention is SKIPPED during profiling -> the hang cannot be in our
nvfp4/VO-split/FlashInfer path. **The DG-V NVFP4-KV contribution is fully cleared of this bug.**

### Root-cause hypothesis (code-localized, no repro needed):
profile_run -> _dummy_run builds a `SchedulerOutput.make_empty()` (model_runner.py:533) with synthetic
`_dummy_req_i` ids but NO `new_requests` -> `model_state.add_request` (runner:778) is never called for
the dummy reqs -> `DiffusionGemmaModelState.diffusion_states` (canvas / is_encoder_phase / step) stay
uninitialized -> the block-diffusion forward (diffusion_gemma.py:312, the denoise/canvas/state-update
path ~554-633) hangs on uninitialized/empty state during the profiling dummy forward. This is a
vLLM **block-diffusion serving-runtime** gap -- almost certainly the upstream DiffusionGemma PR's
diffusion-aware dummy/profile handling not fully landing on our OLDER runner base (upstream serves DG
fine via `vllm serve` in the Gemma docker image, which runs a newer vLLM).

### FIX DIRECTION (scoped follow-up, separate from the KV goal):
Diff our merged `model_runner.py` profile_run/_dummy_run/execute_model + `diffusion_gemma.py`
forward/model_state against the upstream PR eb28452b1 to find the dummy/profile diffusion gap. Likely
one of: (a) make the diffusion model_state register/handle the dummy reqs during profiling, or
(b) short-circuit the denoise/canvas logic in the DG forward when dummy_run/is_profile is set. Then
rebuild r11 + re-serve. ALT (cleaner long-term): rebase the DG cherry-pick onto a newer vLLM base
that already serves DG (the upstream Gemma-docker recipe lineage).

### STATE OF DG-V (honest):
DONE + PROVEN: Gate 0; full code reconciliation (unified FIPrefillGroup); P520 bf16 legacy-path
byte-identical (2.3571850630239095); the entire image/wheel/r11 deterministic Spark pipeline (pivot
hardware-validated); DiffusionGemma nvfp4 serve INITIALIZES correctly on sm_121 (nvfp4 + VO-split +
linear-V-SF all engage; 26B MoE loads). REMAINING (one item, isolated): the block-diffusion
profiling-forward hang -- a serving-runtime gap, NOT the NVFP4-KV code -- blocks the final DG-V5/V6
serving receipt.

## CORRECTED DIAGNOSIS (2026-06-12, via py-spy): NOT a hang -- FlashInfer CUTLASS MoE JIT compile
py-spy dump of the "hung" EngineCore (served with --cap-add SYS_PTRACE):
  run_ninja (flashinfer/jit/cpp_ext.py:368) <- subprocess.communicate, COMPILING
  get_cutlass_fused_moe_module (flashinfer/fused_moe/core.py:278)
  cutlass_fused_moe -> flashinfer_cutlass_moe.apply -> ... -> gemma4.py:419 (MoE layer)
  -> diffusion_gemma.py:322 forward -> execute_model -> _dummy_run (profiling)
The "hang" was the **FlashInfer CUTLASS fused-MoE kernel JIT-compiling on first invocation**
(26B-A4B = 128-expert MoE; moe_backend=auto picked flashinfer_cutlass). CUTLASS MoE builds take
10-30min COLD. **Every earlier attempt was KILLED premat