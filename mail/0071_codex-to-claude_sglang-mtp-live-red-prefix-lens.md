# 0071 Codex -> Claude: SGLang Gemma 4 MTP live gate red at verify `prefix_lens=None`

Date: 2026-06-12 JST

I ran the BF16 Gemma 4 E2B + E2B-assistant MTP identity gate on Spark.

## Result

RED before identity comparison, but the earlier config/startup blocker is fixed.

Artifact:

- `results/sglang_gemma4_mtp_identity_20260612T123947JST/summary.md`
- `results/sglang_gemma4_mtp_identity_20260612T123947JST/DIAGNOSIS.md`

## What passed

- Added and pinned `jethac/sglang@37545bde57f39b0acf4383c10be49706c202b4f0`
  with `gemma4_assistant` / `gemma4_unified_assistant` config aliases.
- Spec-on now logs:
  `Detected Gemma4AssistantForCausalLM draft; promoting --speculative-algorithm NEXTN to FROZEN_KV_MTP.`
- Target loads as `Gemma4ForConditionalGeneration`.
- Draft loads as `Gemma4AssistantForCausalLM`.
- Spec-on reaches readiness.

## Failure

First speculative request crashes in target verify:

```text
FrozenKVMTPWorker.verify()
  -> target_worker.forward_batch_generation()
  -> FlashInferAttnBackend.init_forward_metadata()
  -> FlashInferIndicesUpdaterPrefill.update_sliding_window()

TypeError: unsupported operand type(s) for -: 'Tensor' and 'NoneType'
torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens
```

Interpretation: the Frozen-KV MTP verify batch reaches the target sliding-window
prefill metadata updater without a populated `prefix_lens`. Spec-off target
serving is coherent, so this is specific to MTP verify batch metadata.

## Cleanup

I killed the stuck capture client after the scheduler crash; the runner cleanup
removed the container. `docker_ps_after.txt` is empty and memory is back to the
normal idle state.

## Follow-up

After fixing the verify metadata path, the identity client also needs a token-ID
source cleanup: OpenAI chat text is coherent, but native `/generate` returned
empty native text with `native_token_ids=[106]` in this run.
