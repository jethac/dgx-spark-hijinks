# SGLang Gemma 4 MTP recon

Date: 2026-06-12 JST
Branch: `epoch2`
Scope: mail/0030 + mail/0031 MTP recon for the SGLang lane. This is code recon only; no serving claim.

## Verdict

SGLang on this branch has a real Gemma 4 native-assistant MTP path. This is not gap-only support.

The path is `FROZEN_KV_MTP`: when the draft model config advertises `Gemma4AssistantForCausalLM` or `Gemma4UnifiedAssistantForCausalLM`, `--speculative-algorithm NEXTN` or `EAGLE` is promoted to `FROZEN_KV_MTP` in `sglang/srt/arg_groups/speculative_hook.py`. `EAGLE3` is rejected for these assistants.

## Code anchors

- `third_party/sglang/python/sglang/srt/arg_groups/speculative_hook.py`
  - Detects Gemma 4 assistant architectures.
  - Promotes `NEXTN` / `EAGLE` to `FROZEN_KV_MTP`.
  - Routes `FROZEN_KV_MTP` through `_handle_frozen_kv_mtp`.
- `third_party/sglang/python/sglang/srt/speculative/spec_info.py`
  - Defines `SpeculativeAlgorithm.FROZEN_KV_MTP`.
  - Creates `FrozenKVMTPWorker`.
  - Explicitly rejects overlap/spec-v2 for `FROZEN_KV_MTP`; this lane should start non-overlap.
- `third_party/sglang/python/sglang/srt/models/gemma4_mtp.py`
  - Defines `Gemma4AssistantForCausalLM` and `Gemma4UnifiedAssistantForCausalLM`.
  - Binds the target embedding to the assistant.
  - Maps assistant logical layers to the target physical KV owner layers by layer type.
  - Marks assistant attention layers as KV-shared and suppresses assistant KV writes.
- `third_party/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
  - Reuses target `req_to_token_pool` and `token_to_kv_pool_allocator`.
  - Builds only a dummy draft pool config.
  - Draft steps run under `target_kv_pool_view(...)`, swapping the draft attention backend to the target KV pool.
  - Verify calls `target_worker.forward_batch_generation(batch, is_verify=True)` with `ForwardMode.TARGET_VERIFY`.
- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  - `TARGET_VERIFY` routes through paged prefill wrappers, not ordinary decode.
  - The existing SGLang NVFP4/VO-split work therefore has to cover target-verify prefill plans too.

## NVFP4 touchpoints

For Gemma 4 native assistants, the drafter should read the target KV cache. There is no independent assistant KV pool in the normal frozen-KV MTP path. Therefore the main NVFP4 risk is not drafter KV writing; it is whether draft reads and target verify read the same target pages through the same correct FP4-K/V scale and VO-split wrapper configuration as ordinary target serving.

Concrete surfaces to gate:

1. Draft read path: `FrozenKVMTPWorker.draft_forward()` enters `target_kv_pool_view(...)` and forwards the assistant through the draft attention backend against target-owned pages.
2. Verify path: `FrozenKVMTPWorker.verify()` installs `FrozenKVMTPVerifyInput`, sets `ForwardMode.TARGET_VERIFY`, and calls the target worker. In FlashInfer this uses paged prefill metadata/wrappers.
3. Sliding/global dispatch: Gemma 4 assistants bind by target layer type. The same target-vs-drafter backend decision must apply for SWA and global layers, especially D=512 global layers that require VO-split.

## First live gate

Start with a small model to avoid conflating MTP with model size:

- Target: `google/gemma-4-E2B-it`
- Draft: `google/gemma-4-E2B-it-assistant`
- Algorithm CLI: `--speculative-algorithm NEXTN` or `EAGLE`; expected runtime promotion to `FROZEN_KV_MTP`
- KV rows: bf16/auto first, then NVFP4 after ordinary non-spec E2B row is green.
- Spec settings: non-overlap, topk=1 first.
- Gate: temperature-0 greedy spec decode output must be byte-identical to non-spec greedy for the same prompts. Any divergence is RED.

If the identity gate fails, localize by phase:

- If draft crashes before verify, inspect `target_kv_pool_view(...)` and drafter attention backend selection.
- If verify diverges or crashes, inspect `TARGET_VERIFY` paged-prefill wrapper plans and VO-split/NVFP4 dtype arguments.
- If only NVFP4 fails, compare target decode vs `TARGET_VERIFY` selected wrappers and scale-factor plumbing.

## Current claim boundary

This recon proves the code path exists and identifies the gates. It does not prove SGLang Gemma 4 MTP works on GB10, and it does not prove NVFP4 MTP correctness. The live serving identity gate is still required before any support claim.
