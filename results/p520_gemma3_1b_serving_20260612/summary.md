# P520 Gemma 3 1B serving rows — first serving-level evidence on CC 12.0 (sm_120)

Date: 2026-06-12 JST. Campaign: dgx-spark-hijinks epoch2, NVFP4 KV cache for Gemma on consumer/embedded Blackwell.

Until this window, all campaign serving rows came from the GB10 (sm_121); the P520 had only kernel probes. This window serves `google/gemma-3-1b-it` (text-only, uniform head_dim=256, sliding/global interleave, NO VO-split) on the local P520 with the exact code in the Spark r9 image, closing the "validated on both CC 12.0 and 12.1" claim with serving evidence — with one capacity-green/quality-red surprise that turns out to be the campaign anomaly's best lead yet.

## Provenance

| item | value |
|---|---|
| Host | P520, RTX 5060 Ti 16 GB (CC 12.0 / sm_120), WSL2 Ubuntu, shared with Windows desktop (WDDM) |
| vLLM | jethac/vllm `spark/hijinks-022-gemma4-mixed-kv` @ `9759e3b06baa85db93e10ecc0a8afdc4199f449b` (same code as Spark r9 image), editable venv build `0.1.dev1+g9759e3b06.d20260611`, `TORCH_CUDA_ARCH_LIST=12.0a` |
| FlashInfer | jethac/flashinfer `spark/hijinks-022-fa2-d512` @ `7d5d477b7725943c8f1242490d38e88aa3d99e19`, source-tree mode (`PYTHONPATH=~/flashinfer`), JIT, `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1` |
| torch / CUDA | torch 2.12.0+cu130, nvcc 13.0 V13.0.88, driver 595.79 |
| Model | google/gemma-3-1b-it @ `dcc83ea` |
| Corpora | C1 `abb63f0e65247a25f870d3f2d57563ff`, C2 `1686a33b93ca17d1ecc6898d7d021781`, C3 `28dfeba997756c52a74ee74854411c4b` (md5 verified after scp from Spark host) |
| Server flags | `--gpu-memory-utilization 0.85 --max-model-len 8192 --attention-backend flashinfer` (rows), port 8000, one server at a time |
| PPL | `vllm_prompt_ppl_sweep.py --ctx 8191`, no `--add-special-tokens`, `--dump-token-logprobs`, 8190 scored tokens per corpus |

Build gates (all green, `build_gates.txt`): vllm import OK; `EXT_PATH /home/jetha/vllm-hijinks/vllm/_C_stable_libtorch.abi3.so`; cuobjdump: 42 `sm_120a` cubins (plus 1 sm_89 + 1 sm_90 hadamard); NVFP4 linear-latch diag verdict **"writer wrote LINEAR V-SF"** at head-dim 128 AND 256 (`diag_linear_latch_head128/256.json`).

## Row table (capacity / smoke / PPL are separate claims)

Capacity (`GPU KV cache size` line, util 0.85, max-model-len 8192):

| row | backend | kv dtype | KV tokens | vs bf16-FI | vs fp8-FI | chat smoke |
|---|---|---|---:|---:|---:|---|
| bf16 | FLASHINFER | auto/bf16 | 517,980 | 1.000x | — | GREEN "The capital of Japan is **Tokyo**." |
| fp8 | FLASHINFER | fp8_e4m3 | 1,035,960 | **2.000x** | 1.000x | GREEN "The capital of Japan is **Tokyo**." |
| nvfp4 (+`VLLM_NVFP4_KV_LINEAR_V_SF=1`) | FLASHINFER | nvfp4 | 1,841,077 | **3.554x** | **1.777x** | **RED — deterministic gibberish** (verbatim below) |
| diag A | FLASH_ATTN | auto/bf16 | 943,654 | not a comparator (different backend overhead) | — | GREEN |

All three FlashInfer rows reached ready and served; backend proof line `Using AttentionBackendEnum.FLASHINFER backend.` in each server log; the only Triton mentions are the benign `jit_monitor` slot-mapping kernel warning (not attention). nvfp4 row logged `VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors are linear; FlashInfer in-kernel V...`.

NVFP4 smoke verbatim (identical bytes across two independent servers, including one with a virgin FlashInfer JIT cache; unicode gibberish shown escaped):

```
 zvounds of the farger is not to be the text of the text? how would you\n\n خ ا االحechna? (χ)\n\n**Title:**\n\n**תַস্...
```

## PPL (mean NLL nats/token, ctx 8191, 8190 scored tokens)

HF transformers 5.11.0 bf16 eager/sdpa on the same token windows is the ground-truth column (`hf_bf16_reference_ppl.json`).

| corpus | HF bf16 ref | FLASH_ATTN bf16 (diag A) | FI bf16 | FI fp8 | FI nvfp4 |
|---|---:|---:|---:|---:|---:|
| C1 markdown | 2.35778 | 2.35785 | 2.57846 | 2.36366 | 3.94931 |
| C2 prose | 3.21392 | 3.21457 | 4.45647 | 3.37271 | 5.65030 |
| C3 code | 1.42429 | 1.42454 | 2.80459 | 1.91876 | 4.17584 |

Deltas vs HF ground truth:

| corpus | FLASH_ATTN bf16 | FI bf16 | FI fp8 | FI nvfp4 |
|---|---:|---:|---:|---:|
| C1 | +0.00007 | +0.22068 | +0.00588 | +1.59153 |
| C2 | +0.00065 | +1.24255 | +0.15879 | +2.43638 |
| C3 | +0.00025 | +1.38031 | +0.49447 | +2.75156 |

## Findings

1. **Serving on CC 12.0: GREEN.** Same-code (9759e3b06) FlashInfer-backed serving rows now exist on sm_120, complementing the GB10 sm_121 rows. First serving-level rows on CC 12.0 for the campaign.
2. **Capacity: GREEN and clean.** nvfp4 3.554x vs bf16, 1.777x vs fp8 at equal launch settings — in family with GB10-side ratios.
3. **Quality: RED for the FlashInfer backend as a whole on this stack, not just NVFP4.** `FLASH_ATTN` bf16 matches HF ground truth to <0.001 nats on all three corpora; the FlashInfer bf16 row is **+0.22 to +1.38 nats off ground truth**, fp8 is +0.006 to +0.49 off, nvfp4 is +1.59 to +2.75 off (and gibberish at chat). Within FlashInfer the deviation ordering is fp8 < bf16 < nvfp4.
4. **Anomaly lead.** The campaign's documented "fp8 better than bf16" anomaly (GB10 corpus sweeps; llama.cpp arm replicated direction at 10x smaller magnitude) is consistent with what P520 ground-truthing shows here: the FlashInfer **bf16** row is itself inflated, making quantized rows look "better". On this 1B/d256/SWA geometry the bf16-FI inflation is large and corpus-dependent (worst on code). The stack-specific anomaly component should be hunted in the FlashInfer FA2 prefill path (SWA interleave suspect; Gemma 3 1B window=512), not in the KV-quant writers.
5. **NVFP4 smoke gibberish is deterministic and not stale-JIT.** Reproduced byte-identical on a second server after moving `~/.cache/flashinfer` aside (all kernels freshly JIT-compiled from `7d5d477b` sources). Writer-side latch diag passes at head-256; suspicion falls on the sm_120 NVFP4 KV read path (decode and/or prefill).
6. Stale-JIT was ruled out for the main rows too: all d256 cached_ops used by the rows were compiled during this window (the only older entries are unused qk_512 dirs from probe sessions).

## Reds / incidents (verbatim evidence in artifacts)

- diag B C1 PPL crashed the server: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 15.93 GiB of which 396.00 MiB is free.` during `sample_tokens` (prompt-logprobs materialization) — shared-GPU environmental (Windows WDDM pressure), not a code claim. Main nvfp4 row's C1/C2/C3 PPL all completed.
- WSL2 VM crashed twice during the window (once mid-build with `Wsl/Service/E_UNEXPECTED`, once mid-diagnostics ~06:40 JST); recovered with `wsl --shutdown` + relaunch; build made crash-resilient via static ccache + `TMPDIR` on ext4.
- First build attempt failed with a transient `torch/headeronly/core/Dispatch.h: No such file or directory` (header present on disk before and after); second attempt killed by the WSL crash; third by an own-goal `CCACHE_DISABLE=0`; fourth attempt clean (`build_log_tail_try4.txt`, `build_failures_earlier_tries.txt`).
- FlashInfer source-tree was missing its gitignored `flashinfer/data/*` symlinks after the `7d5d477b` re-sync (first fp8 serve crashed JIT-compiling `sampling`: `ninja: error: '/home/jetha/flashinfer/flashinfer/data/csrc/sampling.cu' ... missing`); recreated the five symlinks (cutlass/spdlog/cccl/csrc/include) without touching the checkout.

## Artifacts

- Local bank: `B:\workshop\wsl_sm120\results\gemma3_1b_serving_20260612\` (WSL master copy: `~/gemma3_1b_serving_20260612/`): per-row `claude_p520_{bf16,fp8,nvfp4}_server.log` + `_proof_lines.txt` + `_chat_smoke.json` + `_c{1,2,3}_ctx8191_ppl.json`, diag A/B equivalents, `hf_bf16_reference_ppl.json`, `build_gates.txt`, `diag_linear_latch_head{128,256}.json`, `ext_cubin_list.txt`, build log tails, `status.txt`/`diag_status.txt`, `token_dumps/` (3.7 MB, local only).
- Run scripts: `run_serving_rows.sh`, `run_diagnostics.sh`, `run_build_gates.sh`, `hf_ppl_reference.py` (same dirs).
