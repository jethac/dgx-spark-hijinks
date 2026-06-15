"""Auto-imported capture hook (put /root first on PYTHONPATH). When FI_CAPTURE_DIR is set, wraps the
FlashInfer prefill wrapper's run() and torch.saves the exact args/kwargs/output of the first FI_CAPTURE_MAX
calls whose q has qo_len == FI_CAPTURE_QO (=512 scoring prefill; profiling uses 4096, chat smoke is short).
This captures the real serving FlashInfer nvfp4 read inputs+output for layer 0..N of the scoring forward,
in whatever process FlashInfer runs (incl. the vLLM EngineCore subprocess, which inherits PYTHONPATH)."""
import os, functools

_CAPDIR = os.environ.get("FI_CAPTURE_DIR")
if _CAPDIR:
    import torch
    os.makedirs(_CAPDIR, exist_ok=True)
    _TARGET = int(os.environ.get("FI_CAPTURE_QO", "512"))
    _MAX = int(os.environ.get("FI_CAPTURE_MAX", "6"))
    _n = [0]

    def _t(x):
        return x.detach().to("cpu") if torch.is_tensor(x) else None

    _MAXNUMEL = int(os.environ.get("FI_CAPTURE_MAXNUMEL", "0"))  # 0 = no cap; else skip bigger tensors

    def _put(d, key, x):
        """Recurse into tuples/lists so paged_kv_cache=(k,v) and kv_cache_sf=(k_sf,v_sf) are saved.
        With FI_CAPTURE_MAXNUMEL set, huge paged caches are skipped (store shape repr) — used for the
        bf16 run where only q (arg0) + out are needed and the full bf16 cache would blow the disk."""
        if torch.is_tensor(x):
            if _MAXNUMEL and x.numel() > _MAXNUMEL:
                d[key + "_repr"] = "TENSOR%s %s (skipped, numel=%d)" % (tuple(x.shape), x.dtype, x.numel())
            else:
                d[key] = x.detach().to("cpu")
        elif isinstance(x, (tuple, list)):
            for i, e in enumerate(x):
                _put(d, f"{key}_{i}", e)
        else:
            d[key + "_repr"] = repr(x)[:200]

    def _save(method, args, kwargs, out, slf=None):
        d = {"_method": method, "_call_index": _n[0]}
        for i, a in enumerate(args):
            _put(d, f"arg{i}", a)
        for k, v in kwargs.items():
            _put(d, f"kw_{k}", v)
        _put(d, "out", out)
        # plan state lives on the wrapper (page table, sm_scale, window, causal) — dump scalars + tensors
        if slf is not None:
            for k in dir(slf):
                if k.startswith("__"):
                    continue
                try:
                    v = getattr(slf, k)
                except Exception:
                    continue
                if torch.is_tensor(v):
                    d["self_" + k] = v.detach().to("cpu")
                elif isinstance(v, (int, float, bool, str)) and not callable(v):
                    d["self_" + k + "_repr"] = repr(v)[:120]
        torch.save(d, os.path.join(_CAPDIR, f"call_{_n[0]:03d}.pt"))

    def _wrap(cls, name):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_fi_captured", False):
            return
        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                if _n[0] < _MAX:
                    q = next((a for a in args if torch.is_tensor(a) and a.dim() == 3), None)
                    if q is None:
                        q = next((v for v in kwargs.values() if torch.is_tensor(v) and v.dim() == 3), None)
                    if q is not None and abs(int(q.shape[0]) - _TARGET) <= 16:
                        _save(f"{cls.__name__}.{name}", args, kwargs, out, slf=self)
                        _n[0] += 1
            except Exception as e:  # never break the forward
                print("FI_CAPTURE save err:", e, flush=True)
            return out
        wrapped._fi_captured = True
        setattr(cls, name, wrapped)

    try:
        import flashinfer.prefill as _fp
        for _cn in dir(_fp):
            _c = getattr(_fp, _cn)
            if isinstance(_c, type) and "Prefill" in _cn and hasattr(_c, "run"):
                _wrap(_c, "run")
        print("FI_CAPTURE armed dir=%s qo=%d max=%d" % (_CAPDIR, _TARGET, _MAX), flush=True)
    except Exception as _e:
        print("FI_CAPTURE wrap failed:", _e, flush=True)
