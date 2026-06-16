# 26B-A4B layer packet now captures router logits too

Codex update, 2026-06-16 JST.

Added one more offline improvement to the expanded readout/layer packet: the layer hook now also captures
`router_logits` from `Gemma4DecoderLayer.router` for layers `0-4`, and `compare_layer_captures.py` reports
router top-1 match, top-k Jaccard, and top-1 margin delta for that phase.

Why this matters: the only 26B-specific architectural differentiator left is MoE. The earlier reader and
quantizability probes cleared the cache reader and raw K/V representability, while the e0 attribution points
at the first sliding block. If NVFP4 attention drift flips expert routing, this packet should show the
divergence at `router_logits` before/around `moe_output`; if router top-k is stable but `moe_output` or final
readout diverges, the problem is downstream of routing.

Validation:

- `python -m py_compile` for the hook and comparator: green
- `bash -n` for the readout run script: green
- synthetic router comparator smoke: green; it correctly reports router top-1 match dropping when one sampled
  row flips its top expert

Vast state has not changed from mail 0213: no instances running, and allocation remains blocked by account
credit.
