# SGLang MTP drafter recon - 2026-06-12

Scope: offline code recon for the SGLang lane on `epoch2`. This is not a
serving-support claim. Per the zero-bug bar, an MTP row is green only after
greedy spec decode is output-identical to non-spec greedy at temperature 0,
with token IDs and transcripts banked.

## What SGLang has

SGLang has native Gemma 4 assistant model classes:

- `Gemma4AssistantForCausalLM`
- `Gemma4UnifiedAssistantForCausalLM`

They are registered through `EntryClass` in
`third_party/sglang/python/sglang/srt/models/gemma4_mtp.py`.

The CLI hook detects those assistant architectures from the draft model config.
If the user passes `--speculative-algorithm NEXTN` or `EAGLE`, SGLang promotes
the algorithm to `FROZEN_KV_MTP`. If the user passes `EAGLE3`, it raises,
because Gemma 4 assistants are not EAGLE3 drafters.

`FROZEN_KV_MTP` is implemented by `FrozenKVMTPWorker`. The worker reuses the
EAGLE verify contract, but the drafter has no independent KV extension path:
it reads the target KV cache.

Important defaults and limits:

- `FROZEN_KV_MTP` does not support spec v2 / overlap scheduling.
- The hook forces `disable_overlap_schedule=True` and disables mixed chunk.
- `SpeculativeAlgorithm.is_eagle()` still treats `FROZEN_KV_MTP` as EAGLE for
  scheduler compatibility, but `supports_spec_v2()` excludes it.
- `topk > 1` currently requires the Triton draft attention backend. Start the
  identity ladder at `topk=1` to avoid mixing a second backend into the first
  correctness proof.
- `--speculative-draft-model-quantization unquant` is the safe first setting:
  target KV may be NVFP4, while assistant weights stay unquantized unless a
  later row explicitly tests draft-weight quantization.

Gemma 3 has no native assistant/MTP checkpoint in this tree. For Gemma 3, any
SGLang speculative row is a generic draft-model/EAGLE-style row, not a native
Gemma MTP assistant claim.

## KV sharing and NVFP4 contact points

The assistant binds to the target physical KV owners in
`Gemma4AssistantForCausalLM.build_frozen_kv_mtp_context()`. The model maps each
assistant logical layer to the target layer of the same layer type, collapses
target-side KV sharing, and stores the target `token_to_kv_pool` in
`FrozenKVMTPContext`.

`FrozenKVMTPWorker` then uses the target memory objects:

- `target_worker.get_memory_pool()` supplies the draft worker's request pool and
  token allocator.
- `target_worker.model_runner.token_to_kv_pool` is captured in the frozen-KV
  context.
- `frozen_kv_target_view()` and `target_kv_pool_view()` temporarily swap
  `draft_attn_backend.token_to_kv_pool` to the target pool.

Therefore, an NVFP4 target row makes the assistant read the same packed target
KV pages and scale buffers. This is a real NVFP4 reader gate, not just a sampler
or draft-logits test.

For target verification, `FrozenKVMTPWorker.verify()` installs
`FrozenKVMTPVerifyInput`, calls `prepare_for_verify()`, sets
`batch.forward_mode = ForwardMode.TARGET_VERIFY`, and runs the target worker.
The FlashInfer backend routes `TARGET_VERIFY` through paged prefill metadata by
default. Hybrid attention can route target verify through decode only when
`speculative_attention_mode == "decode"`; the first SGLang MTP gate should use
the default prefill route because it exercises the packed paged reader.

FlashInfer dtype planning in this branch has the needed full-NVFP4 shape:

- full NVFP4 K+V: `kv_data_type = uint8`, `k_data_type == v_data_type`;
- mixed FP8-K/NVFP4-V: `k_data_type = fp8_e4m3`, `v_data_type = uint8`.

So full-NVFP4 MTP does not need split-dtype module keying. Split dtype is only
needed for the mixed-KV fallback path.

## Allocation and graph implications

Speculative decode increases per-request KV headroom through
`get_alloc_len_per_decode()` and `get_req_to_token_extra_context_len()`. With
`page_size == 1` or `topk == 1`, allocation reserves
`max(spec_steps * topk, max_speculative_num_draft_tokens)`. Page-size >1 and
topk >1 use a larger page-aligned branch footprint. This is another reason to
start with `topk=1`.

Frozen-KV MTP initializes its own draft CUDA graphs after binding the target
pool. That graph path must remain a separate gate. A non-graph identity green
does not bless graph capture.

## Validation ladder

The SGLang ladder should be claim-gated in this order.

1. Static/import gate, no Spark claim:
   - verify the assistant checkpoint config advertises
     `Gemma4AssistantForCausalLM` or `Gemma4UnifiedAssistantForCausalLM`;
   - verify `NEXTN` or `EAGLE` promotes to `FROZEN_KV_MTP`;
   - verify `EAGLE3` rejects for Gemma 4 assistants;
   - record exact SGLang and FlashInfer commits.

2. BF16 identity gate:
   - target: smallest Gemma 4 text-only row that fits the window;
   - draft: matching official Gemma 4 assistant;
   - `topk=1`, overlap disabled, draft quantization `unquant`;
   - compare spec-off vs spec-on at temperature 0;
   - bank token IDs, text, first divergence if any, acceptance metrics, and
     throughput.

3. NVFP4 identity gate:
   - same prompt set and decode settings;
   - target `--kv-cache-dtype nvfp4`;
   - keep draft weights unquantized;
   - require token-ID identity versus the non-spec NVFP4 target row.

4. Graph gate:
   - repeat the smallest green NVFP4 identity row with the intended CUDA graph
     settings;
   - any divergence is RED and scoped to graph capture.

5. Scale-up rows:
   - only after the small identity gates pass;
   - run 12B, 26B-A4B, then 31B with the same zero-bug identity contract;
   - speedup and acceptance are reported only after identity is green.

Suggested first-row shape:

```text
--speculative-algorithm NEXTN
--speculative-draft-model-path google/gemma-4-E2B-it-assistant
--speculative-num-steps 1
--speculative-num-draft-tokens 1
--speculative-eagle-topk 1
--speculative-draft-model-quantization unquant
```

Increase `num_steps` and `num_draft_tokens` only after the one-token identity
gate passes.

## Current answer to the split-dtype scope question

For full NVFP4 K+V, SGLang's FlashInfer wrapper can use one `kv_data_type`
(`uint8`) at module level and pass scale buffers through the existing NVFP4
paged-cache path. It should not require split K/V module keying.

For mixed FP8-K/NVFP4-V, the branch is genuinely split at module level:
`k_data_type` is fp8 and `v_data_type` is uint8 in both decode and prefill
index updaters. If a graph-capture site is for a mixed pool, collapsing it to a
single `kv_data_type` would be wrong. If the site describes full-NVFP4 container
layout, collapse is correct.

## Stop conditions

- Any spec-on/spec-off token mismatch at temperature 0 is RED, even if text
  looks semantically close.
- Any incoherent transcript is RED.
- Any CUDA graph-only mismatch is a graph-gate RED, not an MTP algorithm green.
- Do not report speedup or acceptance as a serving win before identity passes.
