# SGLang Gemma 4 MTP Identity Gate: RED

Date: 2026-06-12 JST

## Scope

- Target: `google/gemma-4-E2B-it`
- Draft: `google/gemma-4-E2B-it-assistant`
- Runtime: SGLang source overlay on Spark, image
  `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `37545bde57f39b0acf4383c10be49706c202b4f0`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`
- KV: BF16 target, unquantized draft, `NEXTN`, `topk=1`,
  `num_steps=1`, `num_draft_tokens=1`, CUDA graphs disabled.

## Verdict

RED before identity comparison. The spec-off target served coherent text, and
the spec-on server now gets past the prior assistant-config failure:

- `NEXTN` is promoted to `FROZEN_KV_MTP`.
- Target `Gemma4ForConditionalGeneration` loads.
- Draft `Gemma4AssistantForCausalLM` loads.
- The server reaches readiness.

The first speculative request then crashes during Frozen-KV MTP verify in
FlashInfer sliding-window metadata setup:

```text
TypeError: unsupported operand type(s) for -: 'Tensor' and 'NoneType'
at sglang/srt/layers/attention/flashinfer_backend.py:update_sliding_window()
torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens
```

## Evidence

- Spec-off capture: `spec_off.capture.json`
- Spec-on crash log: `spec_on.server.log`
- Cleanup proof: `docker_ps_after.txt`, `free_after.txt`
- Earlier host-side HF probes are advisory red because host Python lacks
  `huggingface_hub`; the container-side target/draft loads are the runtime
  evidence.

## Notes

The OpenAI spec-off responses were coherent:

- `capital_japan`: `The capital of Japan is Tokyo.`
- `arithmetic`: `2 + 2 equals 4.`
- `spark_use`: coherent one-sentence DGX Spark utility answer.

Native `/generate` token extraction was not useful in this run
(`native_token_ids=[106]`, empty native text), so the client token-ID gate still
needs tightening after the MTP verify crash is fixed.
