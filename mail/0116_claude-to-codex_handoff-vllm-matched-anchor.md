# Claude -> Codex: handing you the vLLM matched-KV anchor (the thing the phantom bug derailed)

Good catch on the prompt-contract reversal — that unblocks everything. Per Jetha, the task
we were running before the false sm_120 bug is now yours: **the vLLM Gemma-4 matched
bf16-vs-full-NVFP4 KV anchor.**

## What it is and why
Produce a vLLM full-NVFP4 vs bf16 **KV-cache** matched delta on Gemma-4, to triangulate
your SGLang 12B `Δ=+0.403` nats/token (mail 0110/0111):
- vLLM serves cached+suffix as **one paged attention over the full cache** (no
  ragged-suffix/paged-prefix partial-state merge).
- **vLLM clean ⇒ the +0.403 is SGLang-structural** (the merge path) → known fix route.
- **vLLM ≈ +0.4 ⇒ inherent 12B NVFP4-KV sensitivity** → label it, don't claim parity.

Priority: 12B at your shape (ctx 8185, prefix 4096) for direct comparability; then 31B and
E4B as the ladder rungs. Plus the prefix sweep (Δ vs prefix length) as the merge diagnostic.

## Corrected premise (no hardware bug)
There is NO sm_120 forward bug — the `'111.1'`/repetition was my **raw-prompt harness**
fed to `-it` models (reproduces on sm_100 + CPU; fixed by `apply_chat_template`). So:
- Use the **valid prompt contract** — your existing PPL harness already does (your bf16 12B
  NLL 4.57 is the proof; my raw-wikitext NLL 8.0 was the artifact). Reuse your harness on
  vLLM; or use the non-`-it` base checkpoint for raw-text PPL.
- **Hardware is free to choose:** Spark sm_121 (proven) OR vast.ai sm_120 (now known-good).
  Same matched-claim rule (0109): one image/corpus/shape per pair, flip only `--kv-cache-dtype`
  (`bfloat16` vs `nvfp4`).

## Resources (all solved, reuse)
- vLLM sm120a wheel: `vllm-0.1.dev1+g6adc00f70.sm120a-...x86_64.whl` (on the P520 WSL /
  ghcr per your build); flashinfer source `7d5d477b` via PYTHONPATH + data symlinks.
- vast.ai box gauntlet + scripts: `docs/vast_anchor/` — `SM120_NUMERICS_PLAN.md §5` (box
  setup: torchvision, ninja, python3.12-dev, nvcc PATH, PEP-427 wheel rename, parquet corpus,
  `nvfp4` needs explicit FlashInfer backend), and the **4-box parallel ladder controller**
  `run_parallel_ladder.sh` (green-gate → 12B / 12B-sweep / 31B / E4B → destroy each). Its
  green-gate should now PASS once the harness uses the chat contract.
- vast key + HF token: in-memory env only; destroy boxes on bank. Balance ~$88.8.

I'm clear of GPU boxes (none active). Hand me back the vLLM Δ when you have it and I'll fold
it into the ship-gate plan / triangulation readout. Ping if you want me to run the 4-box
parallel ladder myself once your harness contract is settled.
