# Window packet — 31B flagship quality gate (PPL pair) 

Authored 2026-06-11. The run that converts COHERENT + 1.70x into the complete
flagship ledger row. ~75 min. Guardrails/protocol unchanged.

## Code state
- Image: r8 (`jethac-vllm-aeon-gemma4:e08a6f3ae-rebuiltc-fb7d62ea-sm121a-r8`).
- PLUS Python overlay @ `jethac/vllm@9759e3b06` (envs.py knob registration +
  the --language-model-only mm-prefix waiver). Overlay rules: `-w /work`,
  PYTHONPATH prepend; verify `vllm.__file__` points at the overlay.
- NO convertor patch, NO CLAUDE_DISABLE_GEMMA4_MM_PREFIX: serve with
  `--language-model-only` instead. PROOF REQUIREMENT: server starts with
  FlashInfer selected and no mm-prefix rejection, no bypass env present.
  Also confirm the two startup warnings are gone (knobs now registered) and
  the V-SF mode log line reads "linear, in-kernel deswizzle disabled".

## The pair (sequential, identical except KV dtype + knobs)
Model `google/gemma-4-31B-it`, `--max-model-len 8192 --gpu-memory-utilization
0.72 --language-model-only`, PPL harness = the Block C pattern
(vllm_prompt_ppl_sweep.py, ctx 8192) + first-token gate prompts:
1. **fp8 comparator**: `--kv-cache-dtype fp8_e4m3`, no knobs (Gemma4Config
   forces TRITON_ATTN - expected, record the line).
2. **full NVFP4**: `--kv-cache-dtype nvfp4`, env `VLLM_NVFP4_KV_VOSPLIT=1
   VLLM_NVFP4_KV_LINEAR_V_SF=1`.
Record: PPL + nats/token both rows, delta; KV token counts both rows
(capacity ratio should reproduce ~1.70x); decode tok/s 3 reps each (parity
check - comparator is Triton, expect NVFP4/FlashInfer >= Triton).

## Gates for the ledger row
- Quality: delta nats/token within the campaign gate (reference points:
  Gemma 3 +0.005; Qwen linear-mode cost +0.0085 nats/tok at ctx 8192 - if
  31B's delta lands near the sum of quantization + linear costs, that is
  the honest expected band; a gibberish-scale blowup means stop and trace).
- Capacity: ratio from THIS pair (same harness, same day).
- Provenance: server logs, JSONs, proof lines, manifest - the full row needs
  no asterisks and no bypass patches.

## If green
The flagship row is complete: update RESULTS_LEDGER.md + the D=512 plan
status to P4 GREEN; next ladder rungs (12B text via --language-model-only on
the SAME branch zero-code; E4B NVFP4) inherit this exact packet shape.
