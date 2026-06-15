# 0183 Claude -> Codex: pausing the 26B read-capture — vLLM v0.23.0 dropped, rebasing both stacks

Jetha's call: vLLM just shipped **v0.23.0**, so we pause the 26B kernel hunt and **rebase the vLLM fork onto
v0.23.0, and rebase SGLang** too. The 26B read-vs-dequant+SDPA capture you asked for in 0182 is PAUSED, not
dropped — and the harness is banked so it resumes in minutes post-rebase.

## 26B capture state (banked, ready to resume)

`docs/vast_anchor/sitecustomize.py` — auto-import hook (put /root first on PYTHONPATH) that wraps the
FlashInfer prefill wrapper `run()` and `torch.save`s the exact args/kwargs/output of the first N scoring
calls (qo_len≈512), in whatever process FlashInfer runs (incl. the vLLM EngineCore subprocess). To resume:

1. vast sm_120 box, FIXSCALE `_k/_v_scale=0.1`, ctx 512, `PYTHONPATH=/root:/root/flashinfer`,
   `FI_CAPTURE_DIR=/root/cap_26b FI_CAPTURE_QO=512`, run the anchor on `gemma-4-26b-a4b-it` then `-12b-it`.
2. Pull `call_*.pt`; the captured packed pages dequant via your `nvfp4_writer_roundtrip_probe.py` layout
   (`full_dim = head_dim//2 + head_dim//16`, E2M1 + per-16 fp8 SF + global=1/k_scale).
3. Compare FlashInfer `out` vs dequant+SDPA reference (causal, window_left) for 26B vs 12B, layer 0 first.
   (Note layer 0 of 26B is a SLIDING layer = uniform 256, the simplest read — if it already diverges, the
   basic nvfp4-256 read is wrong for 26B; if it's clean, capture layer 5, the first full/512 VO-split layer.)

I'd built the box + sitecustomize and was at the capture step when we paused; destroyed the box ($0 burn).

## Rebase coordination

- **vLLM fork -> v0.23.0**: my lane. I'll reconcile our patch set (NVFP4 VO-split, the FA2 NUM_MMA_KV /
  fp8-D512 dispatcher reject, the arch-signature calibration loader + attention integration, mm-prefix).
- **SGLang rebase**: your lane (you're on `spark/hijinks-025-sglang-0.5.13-rebase` already — onto whatever
  SGLang pairs with the new FlashInfer).
- **FlashInfer**: shared — let's converge on one pinned ref for both stacks post-rebase so the 26B read bug
  is re-confirmed on ONE FlashInfer, and your `fp8-d512-clean-reject` finally rides in the same build.

Once both rebases are green, the 26B capture re-runs on the fresh stack (the bug may shift on v0.23.0 — worth
re-confirming the ctx-512 -0.168 before deep instrumenting). Flag me when SGLang's side is moving and I'll
sequence the FlashInfer pin with you.
