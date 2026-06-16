"""Capture active paged-KV pages from FlashInfer prefill wrapper calls.

Enable with ``FI_ACTIVE_KV_CAPTURE_DIR`` on PYTHONPATH before vLLM imports
FlashInfer.  The hook saves only the pages used by the current request, which
keeps bf16 captures tractable and lets offline comparators mix bf16 and NVFP4
K/V references.
"""

from __future__ import annotations

import functools
import os
from typing import Any

_CAPDIR = os.environ.get("FI_ACTIVE_KV_CAPTURE_DIR")

if _CAPDIR:
    import torch

    os.makedirs(_CAPDIR, exist_ok=True)
    _TARGET = int(os.environ.get("FI_ACTIVE_KV_CAPTURE_QO", "512"))
    _MAX = int(os.environ.get("FI_ACTIVE_KV_CAPTURE_MAX", "8"))
    _MAX_SLICE_NUMEL = int(os.environ.get("FI_ACTIVE_KV_CAPTURE_MAX_SLICE_NUMEL", "90000000"))
    _SMALL_NUMEL = int(os.environ.get("FI_ACTIVE_KV_CAPTURE_SMALL_NUMEL", "50000000"))
    _n = [0]

    def _scalar(v: Any) -> Any:
        if isinstance(v, (int, float, bool, str)):
            return v
        if torch.is_tensor(v) and v.numel() == 1:
            return v.detach().cpu().item()
        return None

    def _get_self_tensor_or_scalar(slf: Any, *needles: str) -> Any:
        for name in dir(slf):
            if name.startswith("__"):
                continue
            lname = name.lower()
            if not all(n in lname for n in needles):
                continue
            try:
                value = getattr(slf, name)
            except Exception:
                continue
            if torch.is_tensor(value):
                return value.detach()
            scalar = _scalar(value)
            if scalar is not None:
                return scalar
        return None

    def _request_pages(slf: Any) -> tuple[torch.Tensor | None, int | None]:
        indptr = _get_self_tensor_or_scalar(slf, "indptr")
        if indptr is None:
            indptr = _get_self_tensor_or_scalar(slf, "kv_indptr")
        indices = _get_self_tensor_or_scalar(slf, "kv_indices")
        if indices is None:
            indices = _get_self_tensor_or_scalar(slf, "indices")
        last_len = _get_self_tensor_or_scalar(slf, "last_page")
        if not torch.is_tensor(indptr) or not torch.is_tensor(indices) or indptr.numel() < 2:
            return None, None
        start, end = int(indptr.detach().cpu().long()[0]), int(indptr.detach().cpu().long()[1])
        pages = indices.detach().cpu().long()[start:end]
        if torch.is_tensor(last_len):
            ll = int(last_len.detach().cpu().reshape(-1)[0])
        elif last_len is None:
            ll = None
        else:
            ll = int(last_len)
        return pages, ll

    def _looks_page_indexed(x: torch.Tensor, pages: torch.Tensor) -> bool:
        if x.dim() < 4 or pages.numel() == 0:
            return False
        # Paged tensors are normally [num_pages, page_size, heads, dim] (or a
        # close variant).  The active page ids must fit in dim 0.
        return int(pages.max()) < int(x.shape[0])

    def _put_tensor(d: dict[str, Any], key: str, x: torch.Tensor, pages: torch.Tensor | None) -> None:
        d[f"{key}_shape"] = tuple(x.shape)
        d[f"{key}_dtype"] = str(x.dtype)
        if x.dim() == 3 and x.numel() <= _SMALL_NUMEL:
            d[key] = x.detach().cpu()
            return
        if pages is not None and _looks_page_indexed(x, pages):
            sliced = x.index_select(0, pages.to(device=x.device))
            d[f"{key}_active"] = sliced.detach().cpu()
            return
        if x.numel() <= _SMALL_NUMEL:
            d[key] = x.detach().cpu()
        else:
            d[f"{key}_repr"] = f"TENSOR{tuple(x.shape)} {x.dtype} skipped numel={x.numel()}"

    def _put(d: dict[str, Any], key: str, value: Any, pages: torch.Tensor | None) -> None:
        if torch.is_tensor(value):
            if value.numel() > _MAX_SLICE_NUMEL and not (
                pages is not None and _looks_page_indexed(value, pages)
            ):
                d[f"{key}_repr"] = f"TENSOR{tuple(value.shape)} {value.dtype} skipped numel={value.numel()}"
            else:
                _put_tensor(d, key, value, pages)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                _put(d, f"{key}_{i}", item, pages)
        else:
            scalar = _scalar(value)
            d[f"{key}_repr"] = repr(scalar if scalar is not None else value)[:200]

    def _save(method: str, args: tuple[Any, ...], kwargs: dict[str, Any], out: Any, slf: Any) -> None:
        pages, last_len = _request_pages(slf)
        d: dict[str, Any] = {
            "schema": "flashinfer-active-kv-capture/v1",
            "_method": method,
            "_call_index": _n[0],
            "last_page_len": last_len,
        }
        if pages is not None:
            d["seq_pages"] = pages
        for name, needles in {
            "sm_scale": ("sm_scale",),
            "window_left": ("window",),
            "causal": ("causal",),
        }.items():
            value = _get_self_tensor_or_scalar(slf, *needles)
            scalar = _scalar(value)
            if scalar is not None:
                d[name] = scalar
        for i, arg in enumerate(args):
            _put(d, f"arg{i}", arg, pages)
        for key, value in kwargs.items():
            _put(d, f"kw_{key}", value, pages)
        _put(d, "out", out, None)
        torch.save(d, os.path.join(_CAPDIR, f"call_{_n[0]:03d}.pt"))

    def _wrap(cls: type, name: str) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_active_kv_captured", False):
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
                        _save(f"{cls.__name__}.{name}", args, kwargs, out, self)
                        _n[0] += 1
            except Exception as exc:
                print("FI_ACTIVE_KV_CAPTURE save err:", exc, flush=True)
            return out

        wrapped._active_kv_captured = True
        setattr(cls, name, wrapped)

    try:
        import flashinfer.prefill as _prefill

        for _class_name in dir(_prefill):
            _class = getattr(_prefill, _class_name)
            if isinstance(_class, type) and "Prefill" in _class_name and hasattr(_class, "run"):
                _wrap(_class, "run")
        print(
            f"FI_ACTIVE_KV_CAPTURE armed dir={_CAPDIR} qo={_TARGET} max={_MAX}",
            flush=True,
        )
    except Exception as _exc:
        print("FI_ACTIVE_KV_CAPTURE wrap failed:", _exc, flush=True)
