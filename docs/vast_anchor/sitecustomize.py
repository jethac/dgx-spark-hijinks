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

    def _save(method, args, kwargs, out):
        d = {"_method": method, "_call_index": _n[0]}
        for i, a in enumerate(args):
            t = _t(a)
            d[f"arg{i}"] = t if t is not None else None
            if t is None:
                d[f"arg{i}_repr"] = repr(a)[:200]
        for k, v in kwargs.items():
            t = _t(v)
            d[f"kw_{k}"] = t if t is not None else None
            if t is None:
                d[f"kw_{k}_repr"] = repr(v)[:200]
        if torch.is_tensor(out):
            d["out"] = _t(out)
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                d[f"out{i}"] = _t(o)
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
                        _save(f"{cls.__name__}.{name}", args, kwargs, out)
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
