# SGLang Qwen mixed-KV source-overlay eager repro attempt

Date: 2026-06-12 JST

This was a narrow follow-up to the graph-gate failure, intended to determine whether the source-overlay mixed-KV failure also reproduces with CUDA graphs disabled.

The attempt did not produce a usable eager traceback. The container was torn down after the graph-gate failure was already established, and the durable `server.log` only contains Docker's missing-container message. Treat this directory as an inconclusive repro attempt, not as root-cause evidence.

The durable root-cause artifact for the graph gate is:

- `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_graph_mixed_server.log`

