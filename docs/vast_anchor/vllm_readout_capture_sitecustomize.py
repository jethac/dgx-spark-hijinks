"""Opt-in vLLM readout capture hook for Gemma 4 26B-A4B NVFP4 debugging.

Put this file on ``PYTHONPATH`` as ``sitecustomize.py`` and set
``VLLM_READOUT_CAPTURE_DIR``.  It wraps Gemma4ForCausalLM.compute_logits and
saves sampled final hidden-state rows plus raw top-k logits for each prompt
logprob chunk.  The hook intentionally skips one-row decode/sample calls and
never stores full vocab logits.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


_CAPTURE_DIR = os.environ.get("VLLM_READOUT_CAPTURE_DIR")

if _CAPTURE_DIR:
    import torch

    _OUT = Path(_CAPTURE_DIR)
    _OUT.mkdir(parents=True, exist_ok=True)
    _MAX_CALLS = int(os.environ.get("VLLM_READOUT_CAPTURE_MAX_CALLS", "16"))
    _MIN_ROWS = int(os.environ.get("VLLM_READOUT_CAPTURE_MIN_ROWS", "64"))
    _DENSE_ROWS = int(os.environ.get("VLLM_READOUT_CAPTURE_DENSE_ROWS", "64"))
    _ROW_STRIDE = int(os.environ.get("VLLM_READOUT_CAPTURE_ROW_STRIDE", "64"))
    _TOPK = int(os.environ.get("VLLM_READOUT_CAPTURE_TOPK", "64"))
    _counter = {"n": 0, "seen": 0}

    def _select_rows(num_rows: int, device: torch.device) -> torch.Tensor:
        rows = set(range(min(_DENSE_ROWS, num_rows)))
        if _ROW_STRIDE > 0:
            rows.update(range(0, num_rows, _ROW_STRIDE))
        rows.add(num_rows - 1)
        return torch.tensor(sorted(rows), device=device, dtype=torch.long)

    def _capture(hidden_states: torch.Tensor, logits: torch.Tensor) -> None:
        call_seen = _counter["seen"]
        _counter["seen"] += 1
        if _counter["n"] >= _MAX_CALLS:
            return
        if hidden_states.ndim != 2 or logits.ndim != 2:
            return
        if hidden_states.shape[0] < _MIN_ROWS:
            return

        call_index = _counter["n"]
        _counter["n"] += 1
        with torch.inference_mode():
            rows = _select_rows(int(hidden_states.shape[0]), hidden_states.device)
            rows_cpu = rows.detach().cpu()
            hidden_sel = hidden_states.index_select(0, rows)
            logits_sel = logits.index_select(0, rows).float()
            k = min(_TOPK, int(logits_sel.shape[-1]))
            top_vals, top_ids = torch.topk(logits_sel, k=k, dim=-1)

            hidden_float = hidden_states.float()
            payload = {
                "schema": "vllm-readout-capture/v1",
                "call_index": call_index,
                "compute_logits_seen_index": call_seen,
                "pid": os.getpid(),
                "hidden_shape": tuple(hidden_states.shape),
                "hidden_dtype": str(hidden_states.dtype),
                "logits_shape": tuple(logits.shape),
                "logits_dtype": str(logits.dtype),
                "selected_local_rows": rows_cpu,
                "hidden_selected": hidden_sel.detach().to("cpu", dtype=torch.float16),
                "hidden_row_rms": torch.sqrt(
                    torch.mean(hidden_float * hidden_float, dim=-1)
                ).detach().cpu(),
                "hidden_row_mean_abs": torch.mean(
                    torch.abs(hidden_float), dim=-1
                ).detach().cpu(),
                "hidden_row_max_abs": torch.max(
                    torch.abs(hidden_float), dim=-1
                ).values.detach().cpu(),
                "logits_selected_top_ids": top_ids.detach().cpu(),
                "logits_selected_top_values": top_vals.detach().cpu(),
                "logits_selected_max": torch.max(logits_sel, dim=-1).values.detach().cpu(),
                "logits_selected_lse": torch.logsumexp(logits_sel, dim=-1)
                .detach()
                .cpu(),
            }
        torch.save(payload, _OUT / f"readout_call_{call_index:03d}.pt")

    def _patch() -> None:
        from vllm.model_executor.models import gemma4 as _gemma4

        cls = _gemma4.Gemma4ForCausalLM
        orig = getattr(cls, "compute_logits", None)
        if orig is None or getattr(orig, "_vllm_readout_capture", False):
            return

        @functools.wraps(orig)
        def wrapped(self, hidden_states, *args, **kwargs):
            logits = orig(self, hidden_states, *args, **kwargs)
            try:
                if torch.is_tensor(hidden_states) and torch.is_tensor(logits):
                    _capture(hidden_states, logits)
            except Exception as exc:
                print(f"VLLM_READOUT_CAPTURE save error: {exc}", flush=True)
            return logits

        wrapped._vllm_readout_capture = True
        setattr(cls, "compute_logits", wrapped)
        print(
            "VLLM_READOUT_CAPTURE armed "
            f"dir={_OUT} max_calls={_MAX_CALLS} min_rows={_MIN_ROWS} topk={_TOPK}",
            flush=True,
        )

    try:
        _patch()
    except Exception as exc:
        print(f"VLLM_READOUT_CAPTURE patch failed: {exc}", flush=True)
