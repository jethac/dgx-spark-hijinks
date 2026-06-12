# SGLang Gemma 4 MTP Identity Gate - NVFP4 RED

Status: RED

## Scope

Greedy spec-off vs spec-on identity for SGLang Gemma 4 Frozen-KV MTP with full NVFP4 K+V target cache. Servers are run sequentially; this is not a speedup claim.

## Provenance

- Run: `sglang_gemma4_mtp_identity_nvfp4_20260612T135033JST`
- Target model: `google/gemma-4-E2B-it`
- Draft model: `google/gemma-4-E2B-it-assistant`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `0211f87b234b0f05eb10005b3791d81a962883ca`
- FlashInfer: `f99323bd7d1c`
- KV cache dtype: `fp4_e2m1`
- Graphs disabled: `1`

## Result

- Spec-off server reached readiness and captured 3 prompt rows.
- Spec-on server did not reach readiness.

## Red Reason

The Frozen-KV MTP draft worker attempted NVFP4 KV-cache calibration even though the native assistant has no own KV cache and reads the target KV cache. The calibration warmup first logged that `Gemma4AssistantForCausalLM` needed `set_embed_and_head()`, then the cleanup path hit a CUDA device-side assert while freeing shared SWA allocator slots:

```text
NVFP4 KV cache calibration warmup failed; first real prefill will remain the fallback for auto-calibration. Error: Gemma4AssistantForCausalLM requires set_embed_and_head() before token-id forward.
...
self._calibrate_nvfp4_kv_cache()
allocator.free(allocated_out_cache_loc)
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Follow-up fix: `jethac/sglang@98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968` skips NVFP4 KV-cache calibration for `is_draft_worker and spec_algorithm.is_frozen_kv_mtp()`, leaving calibration to the target worker whose cache the assistant reads.

## Artifacts

- `preflight.log`
- `checkout.log`
- `spec_off.capture.json`
- `spec_off.server.log`
- `spec_on.server.log`
- `docker_ps_after.txt`
