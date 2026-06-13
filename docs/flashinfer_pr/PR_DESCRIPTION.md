# PR: Surface a clear shared-memory error instead of an invalid `NUM_MMA_KV` dispatch (1-byte KV)

**Base:** `flashinfer-ai/flashinfer@c15ac84` (main) · **File:** `include/flashinfer/attention/prefill.cuh` · **Patch:** `0001-prefill-smem-diagnostic-1byte-kv.patch` (3 sites, +57/-3)

## Summary

On a shared-memory-constrained GPU, the FA2 prefill dispatcher can select
`NUM_MMA_KV = 1` for a 1-byte KV cache (FP8/FP4) with `NUM_WARPS_Q = 4`. That value
fails the kernel's own `IsInvalid()` alignment check
(`sizeof(DTypeKV)==1 && NUM_MMA_KV*2 % NUM_WARPS_Q != 0`) and aborts with:

```
FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32
NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
please create an issue (...) and report the issue to the developers.
```

The message asks the user to file a bug, but the real cause is a **resource limit**:
the configuration's KV tile doesn't fit shared memory at the minimum *valid*
`NUM_MMA_KV`. This PR detects that case and raises an actionable error instead.

## Root cause

`c15ac84` already defines `kMinValidMmaKV` — the smallest `NUM_MMA_KV` satisfying the
1-byte-KV alignment constraint (`= NUM_WARPS_Q/2` when `sizeof(DTypeKV)==1 && NUM_WARPS_Q>2`,
else `1`) — and correctly uses it to size the **occupancy budget** (`num_ctas_per_sm`).
But the **dispatched** value is computed independently and *not* floored:

```cpp
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, { ... });
```

`DISPATCH_NUM_MMA_KV` snaps its argument down to a power of two `{8,4,2,1}`. The only
power of two that violates the 1-byte alignment rule is `1` (`1*2 % 4 = 2 ≠ 0`), and it
is dispatched precisely when `min(max_num_mma_kv_smem, max_num_mma_kv_reg) == 1` — i.e.
when shared memory admits exactly one KV tile but the minimum valid tile needs two.
(`max_num_mma_kv_reg` floors at 2, so the binding constraint in this case is always smem.)

## Fix

At all three dispatch sites (single / ragged / paged prefill), floor-check the fitted
tile against `kMinValidMmaKV` before dispatch. When no valid tile fits, raise a clear
shared-memory diagnostic with the offending dims and a concrete workaround; otherwise
dispatch unchanged. This also subsumes the pre-existing `max_mma_kv = 0` path
(`"Unsupported max_mma_kv: 0"`) with the same actionable message.

```cpp
const uint32_t fitted_num_mma_kv = min(max_num_mma_kv_smem, max_num_mma_kv_reg);
if (fitted_num_mma_kv < kMinValidMmaKV) {
  // "insufficient shared memory ... needs NUM_MMA_KV>=N ... use a smaller head_dim
  //  or a 2-byte (bf16/fp16) KV cache."
  FLASHINFER_ERROR(...);
}
DISPATCH_NUM_MMA_KV(fitted_num_mma_kv, NUM_MMA_KV, { ... });
```

**Scope (honest):** this is a *diagnostic* correctness fix. It does not make the
constrained configuration *run* — when the KV tile genuinely exceeds shared memory, the
operation still cannot execute on that GPU. It replaces a misleading
"internal error, please file a bug" with the actual reason and a workaround. Enabling
such configurations (e.g. via a V/O-split that halves the V shared-memory term) is a
separate, larger change.

## Reproduction (stock upstream, no patches)

Clone `flashinfer-ai/flashinfer@c15ac84`, JIT for the target arch, run on a
~100 KB/SM GPU (e.g. sm_120 RTX 5060 Ti). Driver: `repro_d512_ragged.py`.

```python
w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
w.plan(qo_indptr, kv_indptr, num_qo_heads=8, num_kv_heads=1,
       head_dim_qk=512, head_dim_vo=256, causal=True,
       q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
w.run(q, k, v)   # head_dim_vo is a public, documented parameter (asymmetric QK≠VO)
```

`head_dim_qk=512, head_dim_vo=256` with FP8 KV reproduces at every `qo_len`
(8…512) — `NUM_MMA_KV=1` "please create an issue".

## Verification (sm_120, RTX 5060 Ti, CUDA 13.0)

| case | before patch | after patch |
|---|---|---|
| `qk=512 vo=256` FP8 (ragged) | `Invalid configuration NUM_MMA_KV=1 … please create an issue` | `insufficient shared memory (100 KB/SM) … needs NUM_MMA_KV>=2 … use a smaller head_dim or 2-byte KV` |
| `head_dim=256` FP8, qo_len 17–512 | OK | **OK (no regression)** |
| `head_dim=128` FP8, qo_len 17–512 | OK | **OK (no regression)** |
| `qk=512 vo=256` **bf16** (2-byte) | OK | **OK (unaffected — `kMinValidMmaKV=1`)** |

## Notes

- The companion `head_dim=192` "Invalid configuration `NUM_MMA_D_VO % (2·NUM_WARPS_Q)`"
  throw is a *different* invariant (and head 192 appears broadly unsupported); it is
  intentionally out of scope here.
