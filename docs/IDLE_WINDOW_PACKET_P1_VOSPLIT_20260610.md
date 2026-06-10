# Idle-window packet — P1 VO-split gates, linear-writer regression, vosplit smoke

Date authored: 2026-06-10 JST. Owner: Claude lane. Run only inside a granted window
(CLAUDE_WINDOW_OPEN marker handshake); memory guardrails apply throughout
(single server, `--gpu-memory-utilization <= 0.72`, Docker `--memory 100g`,
sequential comparators, never concurrent fp8+nvfp4).

Code state this packet validates:
- `jethac/flashinfer@spark/hijinks-022-fa2-d512` head `fb7d62ea` (FP4 out-width from V).
- `jethac/vllm@spark/hijinks-022-gemma4-mixed-kv` head `f9159f41b` (P3 VO-split
  orchestration; `58d72be7e` linear V-SF writer knob).
- Probe: `scripts/flashinfer_nvfp4_kv_probe.py` on this branch (`--vo-split` mode).

## Hard dependency, decide first

Block C and D below exercise the **C++ cache writer** (`VLLM_NVFP4_KV_LINEAR_V_SF=1`
branch in `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`, commit `58d72be7e`). A
Python source overlay does NOT carry it — the vLLM extension must be rebuilt (new image
tag or in-container build of the `_C` extension). If the window is short, run blocks
A+B only (flashinfer-source-only; no vLLM rebuild needed) and bank C+D for a long window.

## Host prep (once per window)

```bash
# sync sources already pushed from the workshop
git -C /home/jethac/spark_tmp/flashinfer-fa2-d512 fetch origin && \
  git -C /home/jethac/spark_tmp/flashinfer-fa2-d512 checkout fb7d62ea
# sync this repo's scripts/ to wherever the runner mounts /work from
```

Container base: the `run_p0b.sh` pattern (image
`jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`, flashinfer source
mounted at `/fisrc`, `--rm`, capped memory). **`--output` must be a container path
(`/work/...`)** — the P0b runner bug.

## Block A — P1 probe gates (no rebuild needed)

All with `FLASHINFER_PREFILL_DEBUG_ONCE=1` on first run of each shape. Gate:
`cosine >= 0.9999` per pass, finite outputs, both layouts.

```bash
P=scripts/flashinfer_nvfp4_kv_probe.py
COMMON="--vo-split 2 --head-dim 512 --kv-container tuple --causal \
  --v-scale-layout linear --no-deswizzle-flag --layouts NHD HND \
  --cosine-threshold 0.9999 --flashinfer-source-root /fisrc"

# A1: batch>1 + multi-page (revalidates zero-copy view halves post out-width fix)
python $P $COMMON --batch-size 4 --kv-len 96 --qo-len 16 --page-size 16 \
  --output /work/p1_vosplit_batch4_$(date +%Y%m%dT%H%M).json
# A2: qo_len=1 — the decode-as-prefill shape the vLLM orchestration relies on
python $P $COMMON --batch-size 4 --kv-len 96 --qo-len 1 \
  --output /work/p1_vosplit_qolen1_$(date +%Y%m%dT%H%M).json
# A3: signed E2M1 values (harder numerics than the default positive nibbles)
python $P $COMMON --batch-size 2 --kv-len 64 --qo-len 16 --signed-values \
  --output /work/p1_vosplit_signed_$(date +%Y%m%dT%H%M).json
# A4: non-split control at head 128 with linear V-SF (writer-layout regression at
# probe level: linear must match the swizzled baseline numbers)
python $P --head-dim 128 --kv-container tuple --causal --v-scale-layout linear \
  --no-deswizzle-flag --layouts NHD HND --cosine-threshold 0.9999 \
  --flashinfer-source-root /fisrc \
  --output /work/p1_linear_sf_d128_control_$(date +%Y%m%dT%H%M).json
```

## Block B — M1 arch smoke (no new code paths)

`scripts/run_vllm_gemma4_12b_unified_probe.sh` (gemma4_unified, transformers >= 5.5
pin). Gate: model loads + generates; record backend selection lines.

## Block C — linear-writer regression (needs rebuilt extension)

`VLLM_NVFP4_KV_LINEAR_V_SF=1` on the **existing green rungs** — numbers must match the
swizzled-path rows (the knob changes layout, not values):
1. Qwen NVFP4-KV quality gate (`scripts/gemma_nvfp4_kv_quality_gate.py` flow) +
   prefix-reuse row.
2. Gemma 3 27B first-token + PPL pair (`scripts/run_vllm_gemma3_ppl_pair.sh` flow).

## Block D — Gemma 4 31B full-NVFP4 VO-split serving smoke (needs rebuilt extension)

```bash
VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  ... --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.72   # NO skip-layers
```
Proof lines to capture from the server log:
- `NVFP4 KV VO split: head_size 512 runs as 2 FA2 passes of head_dim_vo=256`
- per-layer KV dtype lines showing nvfp4 on ALL layers (no fp8/bf16 global fallback)
- Gemma4Config line: "keeping per-layer attention backend resolution ... VO split"
- absence of the decode-pathway assert.
Then short generation sanity (first-token gate) only — full P4 rung row (capacity
comparator + PPL) is its own window.

## Evidence

Copy `/work/*.json` back into `results/` on this branch (`git add -f results/...`),
one summary md per block, update `docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md` status.
