# Codex -> Claude: SGLang Gemma 4 MTP live identity is green

Stop point: `2026-06-12T14:10JST`

The BF16 Frozen-KV MTP identity row is now GREEN on Spark:

- Artifact: `results/sglang_gemma4_mtp_identity_20260612T133519JST/summary.md`
- Target: `google/gemma-4-E2B-it`
- Draft: `google/gemma-4-E2B-it-assistant`
- SGLang: `0211f87b234b0f05eb10005b3791d81a962883ca`
- FlashInfer: `f99323bd7d1c`
- Spec: `NEXTN`, topk1, one draft token, graphs disabled

What changed since the first live red:

- `bc3979870d`: passed verify `prefix_lens` into FlashInfer SWA metadata for `FROZEN_KV_MTP_VERIFY`.
- `da2e49b05a`: narrowed one-token draft seed MRoPE positions.
- `0211f87b23`: narrowed draft seed `out_cache_loc` to the seed token. The diagnostic guard had pinned the stale tensor as `GraphSlot 'out_cache_loc' copy shape mismatch: dst=(1,) src=(19,)`.

Gate result:

- OpenAI chat text matches spec-off for all three prompts.
- Native `/generate` exposes matching token IDs for all three prompts.
- Native `/generate` text is empty on both sides because the native prompt immediately emits the stop token; the harness now records that as not comparable instead of failing text identity.

Scope:

- This is an identity/load checkpoint only.
- No speedup claim.
- No graph-capture claim yet.
- The graph gate is the next MTP-specific gate once we are ready to spend another Spark window.

Box state when I checked: marker absent, `docker ps` empty.
