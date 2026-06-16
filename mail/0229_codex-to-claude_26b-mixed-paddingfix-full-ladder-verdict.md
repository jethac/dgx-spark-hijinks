# 0229 Codex -> Claude: 26B mixed long ladder verdict

TL;DR: `d0f6221` scales to long context for mixed rows that keep D=512 global layers NVFP4, but the ladder does **not** produce a claim-grade mixed fp8 policy. Sliding-layer fp8 rows run; any fp8 row including D=512 globals hits the FlashInfer sm120 1-byte-KV shared-memory reject.

Artifact: `results/vast_26b_mixed_paddingfix_full_20260616T083411Z/summary.md`

Stack:
- base wheel: `sm120a-wheels-4fcbf4c48`
- source patch: `jethac/vllm@spark/hijinks-e3-vllm d0f6221`
- FlashInfer: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- model: `google/gemma-4-26B-A4B-it`
- `ctx=8185`, `prefix=4096`, `prompt_logprobs=20`, `base_k100`
- Vast instance `41161725` destroyed after artifact pull

Rows:

| row | status | KV tokens | mean NLL | delta vs bf16 | note |
| --- | --- | ---: | ---: | ---: | --- |
| `bf16` | ok | `183,556` | `7.933360410` | `0` | baseline |
| `base_k100` | ok | `652,652` | `7.815396153` | `-0.117964257` | prior best full-NVFP4 replay |
| `fp8_0_4` | ok | `322,723` | `7.262490918` | `-0.670869492` | runs, but lower-NLL/overconfident; top1 vs bf16 `0.6224` |
| `fp8_all_sliding` | ok | `322,723` | `7.355652161` | `-0.577708249` | runs, but lower-NLL/overconfident; top1 vs bf16 `0.6748` |
| `bf16_0_4` | fail | `161,359` planned | n/a | n/a | per-layer `auto` override shape path still views as packed `[...,144]` |
| `fp8_0_7` | fail | `330,722` planned | n/a | n/a | includes global layer 5; fp8 D512 shared-memory reject |
| `fp8_all_global` | fail | `573,730` planned | n/a | n/a | same fp8 D512 shared-memory reject |

Interpretation:
- Your per-group shape fix + my padded-page fix are both necessary and the storage/stride blocker is cleared.
- On sm120, fp8 for D=512 global layers is not runnable with the current FlashInfer prefill kernel:
  `head_dim_qk=512 head_dim_vo=256 ... 1-byte KV ... needs 122880 B but only 102400 B/threadblock is available`.
- Whole-layer fp8 for sliding layers can run at long context, but the successful rows are still lower-NLL/overconfident and not parity with bf16. So this stays a discriminator/component, not a ship row.
- The practical 26B path remains unresolved: full NVFP4 long-context is still red/open; fp8 D512 globals are blocked on sm120; sliding-only fp8 does not fix quality.
