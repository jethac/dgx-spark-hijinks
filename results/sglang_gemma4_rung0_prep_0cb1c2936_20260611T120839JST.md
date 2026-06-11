# SGLang Gemma 4 Rung 0 Prep Stop Point

Date: 2026-06-11 12:08 JST

## Scope

Prepared the SGLang Gemma 4 rung-0 FlashInfer path for the next live serving attempt. This is a code/prep artifact, not a serving result.

Target rung:
- Model: `google/gemma-4-E4B-it`
- Runtime: SGLang source overlay
- KV mode: no KV quantization for rung 0
- Attention: FlashInfer with `SGLANG_FLASHINFER_VOSPLIT=1`

## Commits

- Parent repo: `jethac/dgx-spark-hijinks@0b1e850`
- SGLang fork: `jethac/sglang@0cb1c2936`
- FlashInfer overlay target remains: `jethac/flashinfer@8d85fff9`

## Change

`jethac/sglang@0cb1c2936` changes `python/sglang/srt/layers/attention/flashinfer_backend.py` to:

- plan FlashInfer wrappers with per-wrapper geometry instead of one model-wide `head_dim`;
- use `swa_head_dim` and `swa_num_key_value_heads` for sliding-window wrapper planning when the HF config exposes them;
- keep ordinary/uniform models on the existing geometry;
- route non-FP4 paged prefill and cached-prefix paged prefill through the existing VO-split helper, so bf16/fp8 rung-0 calls can exercise the two-pass `D_QK=512, D_VO=256` path;
- add `SGLANG_GEMMA4_TRACE_GEOMETRY=1` runtime proof lines for layer geometry, wrapper geometry, VO-split selection, KV cache views, and SF views.

## Local Validation

Passed:

```text
python -m py_compile third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
git -C third_party/sglang diff --check
```

## Live Run Status

Not started. The Spark was reachable over the tailnet as `jethac@thinkstationpgx-00b4`, but Claude's marker was present and a container was running:

```text
marker=PRESENT
docker_count=1
Mem: 119Gi total, 75Gi used, 35Gi free, 44Gi available
```

Per the marker protocol and GB10 memory rules, no serving run was launched.

## Next Gate

When the marker is absent and `docker ps` is empty:

1. Sync the live checkout to parent `0b1e850`, SGLang `0cb1c2936`, and FlashInfer `8d85fff9`.
2. Launch a single SGLang Gemma 4 E4B rung-0 server with `SGLANG_FLASHINFER_VOSPLIT=1`, `SGLANG_GEMMA4_TRACE_GEOMETRY=1`, `FLASHINFER_PREFILL_DEBUG_ONCE=1`, and the FlashInfer source-tree sitecustomize shim.
3. Keep Docker capped with `--memory=100g --memory-swap=100g`.
4. First run a short max-new-tokens=1 smoke to validate prefill/logit coherence and capture geometry proof lines.
5. If the server hits `Unsupported max_mma_kv: 0`, stop and preserve the debug dump for the shared FlashInfer dispatcher fix.
