# Diagnosis: Frozen-KV MTP Verify Drops `prefix_lens`

The live BF16 SGLang Gemma 4 MTP identity gate is blocked in the target verify
path, not in model loading or assistant detection.

## What Is Proven Green

The SGLang alias patch at
`jethac/sglang@37545bde57f39b0acf4383c10be49706c202b4f0` fixes the previous
startup blocker:

```text
Detected Gemma4AssistantForCausalLM draft; promoting --speculative-algorithm NEXTN to FROZEN_KV_MTP.
```

The spec-on server loads both models:

```text
type=Gemma4ForConditionalGeneration
type=Gemma4AssistantForCausalLM
```

It reaches readiness and begins the first speculative request.

## Crash

The scheduler crashes while `FrozenKVMTPWorker.verify()` calls the target worker
for verification. The target worker enters `forward_extend()`, then
`FlashInferAttnBackend.init_forward_metadata()`, then:

```text
sglang/srt/layers/attention/flashinfer_backend.py:3674
torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens
TypeError: unsupported operand type(s) for -: 'Tensor' and 'NoneType'
```

This means the Frozen-KV MTP verify batch is reaching the sliding-window
prefill metadata updater without a populated `prefix_lens`. The same target
model in spec-off mode serves normally, so this is specific to the MTP verify
batch construction/metadata path.

## Next Fix Candidate

Trace the `ForwardBatch` produced for `FrozenKVMTPWorker.verify()` and ensure
the target verify extend path supplies a valid prefix-length tensor before it
enters `FlashInferIndicesUpdaterPrefill.update_sliding_window()`.

The fix should stay scoped to Frozen-KV MTP / speculative verify batch metadata.
Do not weaken the generic FlashInfer SWA prefix handling without a non-MTP
regression gate.

## Secondary Follow-Up

After the crash is fixed, tighten the identity client token gate. In this run,
OpenAI chat text is coherent, but native `/generate` token extraction returned
empty text and `native_token_ids=[106]`, so it is not yet a trustworthy token-ID
identity source.
