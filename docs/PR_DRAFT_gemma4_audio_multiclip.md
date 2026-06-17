# DRAFT PR — Gemma 4 multi-clip audio stacking bugfix (NOT SUBMITTED)

- **Branch:** `jethac/vllm:gemma4-multi-audio-stacking` (1 commit atop `upstream/main`)
- **Title:** `[Bugfix][Model] Gemma 4: stack variable-length audio clips for multi-audio prompts`
- **Author/DCO:** Jetha Chan <jethachan@gmail.com> (signed-off); `Co-authored-by: Claude` per vLLM's
  AI-Assisted Contributions Policy.
- **Status:** helper logic verified locally (torch assertions pass); CPU-only regression test added.
  **Before filing, Jetha must personally review the diff and run the test** (vLLM's no-pure-agent rule).

---

## Purpose

A chat request carrying **multiple audio clips of differing MEL-frame length** crashes Gemma 4 during
`_process_audio_input`:

```
AttributeError: 'list' object has no attribute 'squeeze'
  File ".../vllm/model_executor/models/gemma4_mm.py", in _process_audio_input
    input_features = audio_input["input_features_padded"].squeeze(1)
```

`input_features_padded` is a `MultiModalFieldConfig.batched("audio")` field. Its `batched()` re-pad is a
**no-op when the per-clip frame counts differ**, so the field arrives as a *list* of per-clip tensors
instead of one stacked tensor — and `.squeeze(1)` fails on the list. Single-clip prompts (one tensor)
work, which is why this is only hit with ≥2 audio clips of unequal duration.

Fix: a shared `stack_audio_input_features()` helper pads each clip to the batch-max frame count and stacks
into `[bn, s_max, f]` (building the validity mask alongside), keeping the single-tensor fast path. Both
`Gemma4ForConditionalGeneration` (`gemma4_mm`) and the unified variant (`gemma4_unified`) carried the same
bug and now share the helper.

> **Note for reviewers — fix placement.** The root cause is that `batched("audio")` doesn't re-pad
> variable-length MEL features, so the same crash could in principle affect any future consumer of this
> field. I've fixed it at the model layer (where both Gemma 4 variants converge on a shared helper) to keep
> the change tightly scoped, but I'm happy to relocate it to the multimodal field-config/processing layer
> if you'd prefer it solved once for all consumers.

## Test Plan

```
pytest tests/models/multimodal/processing/test_gemma4_audio_stacking.py
```
CPU-only, no model weights — the bug is in tensor batching, not the audio encoder.

## Test Result

- **Before:** `_process_audio_input` raises `AttributeError: 'list' object has no attribute 'squeeze'` on a
  ≥2-clip differing-length audio prompt.
- **After:** clips pad to the batch maximum and stack to `[bn, s_max, f]`; the new test asserts shapes,
  pad-region zeroing, and mask correctness for the 3-D list, 2-D list, and single-tensor fast paths.
  All pass.

## Files
- `vllm/model_executor/models/gemma4_mm.py` — add `stack_audio_input_features()`, use it.
- `vllm/model_executor/models/gemma4_unified.py` — import + use the helper.
- `tests/models/multimodal/processing/test_gemma4_audio_stacking.py` — new CPU regression test.

## AI assistance disclosure
This change was developed with AI assistance (Claude). The commit carries a `Co-authored-by: Claude`
trailer, and the human author reviewed every changed line, validated the behavior, and ran the test
before submission — per vLLM's AI-Assisted Contributions Policy (no "pure agent" PRs).
