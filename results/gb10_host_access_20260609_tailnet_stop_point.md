# GB10 Host Access Stop Point

Date: 2026-06-09 JST

Status: live GB10 validation is blocked by unstable tailnet reachability and closed or
unresponsive SSH on `100.113.98.11`.

## Evidence

- Tailscale name resolution for the GB10 node resolves to `100.113.98.11`.
- Earlier in this session, `tailscale ping --c 3 --timeout=5s` reached the node once with
  an 80 ms pong over a direct tailnet path.
- During the final check after staging the next FlashInfer and SGLang work:
  - `tailscale ping --c 1 --timeout=5s` timed out with `no reply`;
  - `Test-NetConnection 100.113.98.11 -Port 22 -InformationLevel Quiet` returned `False`.
- Normal SSH and Tailscale SSH both depend on TCP/22 here and have timed out from this
  Windows workspace.

## Work Ready When Access Returns

- Parent campaign repo: `jethac/dgx-spark-hijinks@a4ff319`.
- FlashInfer fork branch: `jethac/flashinfer@spark/hijinks-021-prefill-debug`, commit
  `1230341d`.
- vLLM Gemma 3 debug packet:
  `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`.
- SGLang matrix runner now uses source-built patched FlashInfer and source-built
  `sgl-kernel` via `scripts/install_sglang_source_stack.sh`, avoiding stale PyPI
  `flashinfer-cubin`, `flashinfer-jit-cache`, and `sglang-kernel` wheels.

## Next Live Action

When SSH returns, run the vLLM Gemma 3 FlashInfer prefill debug packet first. It is the
current critical-path diagnostic for the Gemma 3 NVFP4-KV quality failure. Then rebuild or
prepare the SGLang source-stack image and rerun the Qwen FP4-KV request-order matrix.
