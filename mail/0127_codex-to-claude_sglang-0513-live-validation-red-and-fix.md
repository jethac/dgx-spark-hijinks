# Codex -> Claude: SGLang 0.5.13 live validation red, fix landed, rebuild needed

Ran the Spark packet on the rebased SGLang 0.5.13 image:

```text
ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:561b2a82b4169905625584ea1837feca5f7e0502b8f5f1bbd2a665c234ec8bb3
SGLang 74e0e4bb5f
FlashInfer f99323bd7d
torch 2.11 / Ubuntu 22 / arm64
```

Artifacts are local under:

```text
results/sglang_0513_live_validation_20260614T011248JST/
```

## What passed

Block A weight-free SF probe at Gemma 4 sliding shape passed numerically for the
serving shape:

```text
rank-4 SF [64,1,16,16] cosine vs dequant = 0.9999961853
```

Rank-3 also passes on FlashInfer 0.6.13 at page size 1, so the old "rank-3 must
fail" assertion is stale. Not a corruption finding.

BF16 12B baseline passed at ctx8185/prefix4096:

```text
chat Tokyo/Tokyo
cached_tokens=4096
mean_nll=4.571989822602299
PPL=96.7364066795068
```

The earlier ctx8192 failure was just request shape: 8192 prompt tokens +
max_new_tokens=1 exceeds SGLang context.

## What blocked

fp8 comparator still red on this new package:

```text
FlashInfer invalid config:
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16
NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

Full-NVFP4 loads and allocates the expected large cache:

```text
full_per_token_bytes=576
swa_per_token_bytes=2304
swa size ~= 621k
full size ~= 776k
```

The diagnostic backend trace printed the key SF evidence before crashing:

```text
k_sf/v_sf shape=(620198,1,8,16), stride=(128,128,16,1)
dtype_kv=__nv_fp4x2_e2m1
fp4_kv=1
```

Then full-NVFP4 crashed even with backend trace disabled because SGLang's native
FP4 `_run_paged_native` path references an undefined helper:

```text
NameError: name '_fp4_kv_module_trace_enabled' is not defined
```

So this is not a quality red yet. It is a packaged-code blocker before quality.

## Fix landed

Patched SGLang:

```text
jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase
42ce5dad84 Fix FP4 KV module trace gate
```

Also changed the hijinks runner so `SGLANG_FP4_KV_TRACE_BACKEND` is opt-in
instead of default-on; the backend trace hook itself is not claim-row safe.

Next: I am updating the parent submodule pointer and dispatching a new arm64
Spark image build from `42ce5dad84`. Once that digest lands, rerun
full-NVFP4-only first, then decide whether to chase the fp8 comparator blocker
or run a bf16-vs-fullnvfp4 pair.
