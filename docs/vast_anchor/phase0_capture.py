"""Phase 0 capture (put /root first on PYTHONPATH). Wraps the KV-cache WRITER
torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, ...) and torch.saves the TRUE post-RoPE
bf16 K/V (the `key`/`value` args, [num_tokens, num_kv_heads, head_dim], pre-paging + pre-quant) for the
first PHASE0_MAX scoring-prefill layers (num_tokens ~= PHASE0_QO). This is the faithful source for the
per-channel-outlier thesis test — no dequant, no paging gymnastics.

Env: PHASE0_DIR (enable), PHASE0_QO=512, PHASE0_MAX=8.
"""
import os
import functools

_DIR = os.environ.get("PHASE0_DIR")
if _DIR:
    import torch

    os.makedirs(_DIR, exist_ok=True)
    _QO = int(os.environ.get("PHASE0_QO", "512"))
    _MAX = int(os.environ.get("PHASE0_MAX", "8"))
    _n = [0]

    def _save(key, value):
        if _n[0] >= _MAX:
            return
        if not (torch.is_tensor(key) and key.dim() == 3):
            return
        if abs(int(key.shape[0]) - _QO) > 16:
            return
        torch.save(
            {"i": _n[0], "key": key.detach().to("cpu"), "value": value.detach().to("cpu")},
            os.path.join(_DIR, f"wkv_{_n[0]:03d}.pt"),
        )
        _n[0] += 1

    _wrapped = []

    # Primary: shadow the torch op on its namespace (catches the direct-op call path vLLM uses).
    try:
        _ns = torch.ops._C_cache_ops
        _orig = _ns.reshape_and_cache_flash  # triggers __getattr__, caches the real op

        @functools.wraps(_orig)
        def _w(*a, **k):
            try:
                if len(a) >= 2:
                    _save(a[0], a[1])
            except Exception as e:  # never break the forward
                print("PHASE0 save err:", e, flush=True)
            return _orig(*a, **k)

        _ns.reshape_and_cache_flash = _w
        _wrapped.append("torch.ops._C_cache_ops.reshape_and_cache_flash")
    except Exception as e:
        print("PHASE0 torch-op wrap failed:", e, flush=True)

    # Fallback: vllm._custom_ops Python wrapper, in case the model path goes through it.
    try:
        import vllm._custom_ops as _co

        if hasattr(_co, "reshape_and_cache_flash"):
            _co_orig = _co.reshape_and_cache_flash

            @functools.wraps(_co_orig)
            def _cw(*a, **k):
                try:
                    if len(a) >= 2:
                        _save(a[0], a[1])
                except Exception as e:
                    print("PHASE0 co save err:", e, flush=True)
                return _co_orig(*a, **k)

            _co.reshape_and_cache_flash = _cw
            _wrapped.append("vllm._custom_ops.reshape_and_cache_flash")
    except Exception as e:
        print("PHASE0 co wrap note:", e, flush=True)

    print("PHASE0 armed:", _wrapped, "dir=%s qo=%d max=%d" % (_DIR, _QO, _MAX), flush=True)
