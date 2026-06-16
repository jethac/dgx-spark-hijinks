"""Opt-in vLLM capture hook for Gemma 4 26B-A4B NVFP4 debugging.

Put this file on ``PYTHONPATH`` as ``sitecustomize.py`` and set
``VLLM_READOUT_CAPTURE_DIR``.  It wraps Gemma4ForCausalLM.compute_logits and
saves sampled final hidden-state rows plus raw top-k logits for each prompt
logprob chunk.  The hook intentionally skips one-row decode/sample calls and
never stores full vocab logits.

Optionally also set ``VLLM_LAYER_CAPTURE_DIR`` to capture selected rows from
Gemma4DecoderLayer layers named by ``VLLM_LAYER_CAPTURE_LAYERS``.  This is used
for the 26B-A4B first-sliding-block attribution path and includes router logits
when the layer has an MoE router.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


_CAPTURE_DIR = os.environ.get("VLLM_READOUT_CAPTURE_DIR")
_LAYER_CAPTURE_DIR = os.environ.get("VLLM_LAYER_CAPTURE_DIR")

if _CAPTURE_DIR or _LAYER_CAPTURE_DIR:
    import torch

    _OUT = Path(_CAPTURE_DIR) if _CAPTURE_DIR else None
    if _OUT is not None:
        _OUT.mkdir(parents=True, exist_ok=True)
    _LAYER_OUT = Path(_LAYER_CAPTURE_DIR) if _LAYER_CAPTURE_DIR else None
    if _LAYER_OUT is not None:
        _LAYER_OUT.mkdir(parents=True, exist_ok=True)
    _MAX_CALLS = int(os.environ.get("VLLM_READOUT_CAPTURE_MAX_CALLS", "16"))
    _MIN_ROWS = int(os.environ.get("VLLM_READOUT_CAPTURE_MIN_ROWS", "64"))
    _DENSE_ROWS = int(os.environ.get("VLLM_READOUT_CAPTURE_DENSE_ROWS", "64"))
    _ROW_STRIDE = int(os.environ.get("VLLM_READOUT_CAPTURE_ROW_STRIDE", "64"))
    _TOPK = int(os.environ.get("VLLM_READOUT_CAPTURE_TOPK", "64"))
    _LAYER_MAX_CALLS = int(os.environ.get("VLLM_LAYER_CAPTURE_MAX_CALLS", "64"))
    _LAYER_NAMES = {
        int(x.strip())
        for x in os.environ.get("VLLM_LAYER_CAPTURE_LAYERS", "0,1,2,3,4").split(",")
        if x.strip()
    }
    _counter = {"n": 0, "seen": 0}
    _layer_counter = {"n": 0, "seen": 0}

    def _select_rows(num_rows: int, device: torch.device) -> torch.Tensor:
        rows = set(range(min(_DENSE_ROWS, num_rows)))
        if _ROW_STRIDE > 0:
            rows.update(range(0, num_rows, _ROW_STRIDE))
        rows.add(num_rows - 1)
        return torch.tensor(sorted(rows), device=device, dtype=torch.long)

    def _tensor_summary(x: torch.Tensor, rows: torch.Tensor | None = None) -> dict:
        xf = x.float()
        if rows is None:
            rows = _select_rows(int(x.shape[0]), x.device)
        selected = x.index_select(0, rows).detach().to("cpu", dtype=torch.float16)
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "selected_local_rows": rows.detach().cpu(),
            "selected": selected,
            "row_rms": torch.sqrt(torch.mean(xf * xf, dim=-1)).detach().cpu(),
            "row_mean_abs": torch.mean(torch.abs(xf), dim=-1).detach().cpu(),
            "row_max_abs": torch.max(torch.abs(xf), dim=-1).values.detach().cpu(),
        }

    def _capture_readout(hidden_states: torch.Tensor, logits: torch.Tensor) -> None:
        if _OUT is None:
            return
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
            logits_sel = logits.index_select(0, rows).float()
            k = min(_TOPK, int(logits_sel.shape[-1]))
            top_vals, top_ids = torch.topk(logits_sel, k=k, dim=-1)

            hidden_summary = _tensor_summary(hidden_states, rows)
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
                "hidden_selected": hidden_summary["selected"],
                "hidden_row_rms": hidden_summary["row_rms"],
                "hidden_row_mean_abs": hidden_summary["row_mean_abs"],
                "hidden_row_max_abs": hidden_summary["row_max_abs"],
                "logits_selected_top_ids": top_ids.detach().cpu(),
                "logits_selected_top_values": top_vals.detach().cpu(),
                "logits_selected_max": torch.max(logits_sel, dim=-1).values.detach().cpu(),
                "logits_selected_lse": torch.logsumexp(logits_sel, dim=-1)
                .detach()
                .cpu(),
            }
        torch.save(payload, _OUT / f"readout_call_{call_index:03d}.pt")

    def _capture_layer(
        layer_idx: int,
        positions: torch.Tensor,
        layer_input: torch.Tensor,
        layer_output: torch.Tensor,
        phase_tensors: dict[str, torch.Tensor],
    ) -> None:
        if _LAYER_OUT is None:
            return
        if layer_idx not in _LAYER_NAMES:
            return
        call_seen = _layer_counter["seen"]
        _layer_counter["seen"] += 1
        if _layer_counter["n"] >= _LAYER_MAX_CALLS:
            return
        if layer_input.ndim != 2 or layer_input.shape[0] < _MIN_ROWS:
            return
        call_index = _layer_counter["n"]
        _layer_counter["n"] += 1
        rows = _select_rows(int(layer_input.shape[0]), layer_input.device)
        payload = {
            "schema": "vllm-layer-capture/v1",
            "call_index": call_index,
            "layer_call_seen_index": call_seen,
            "pid": os.getpid(),
            "layer_idx": layer_idx,
            "positions_shape": tuple(positions.shape) if torch.is_tensor(positions) else None,
            "positions_selected": positions.index_select(0, rows).detach().cpu()
            if torch.is_tensor(positions) and positions.ndim == 1 and positions.shape[0] == layer_input.shape[0]
            else None,
            "input": _tensor_summary(layer_input, rows),
            "output": _tensor_summary(layer_output, rows),
        }
        for name, value in phase_tensors.items():
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == layer_input.shape[0]:
                payload[name] = _tensor_summary(value, rows)
        torch.save(payload, _LAYER_OUT / f"layer_{layer_idx:02d}_call_{call_index:03d}.pt")

    def _patch() -> None:
        from vllm.model_executor.models import gemma4 as _gemma4

        if _OUT is not None:
            cls = _gemma4.Gemma4ForCausalLM
            orig = getattr(cls, "compute_logits", None)
            if orig is not None and not getattr(orig, "_vllm_readout_capture", False):

                @functools.wraps(orig)
                def wrapped(self, hidden_states, *args, **kwargs):
                    logits = orig(self, hidden_states, *args, **kwargs)
                    try:
                        if torch.is_tensor(hidden_states) and torch.is_tensor(logits):
                            _capture_readout(hidden_states, logits)
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

        if _LAYER_OUT is not None:
            layer_cls = _gemma4.Gemma4DecoderLayer
            layer_orig = getattr(layer_cls, "forward", None)
            if layer_orig is not None and not getattr(layer_orig, "_vllm_layer_capture", False):

                @functools.wraps(layer_orig)
                def layer_wrapped(self, positions, hidden_states, *args, **kwargs):
                    phase_tensors: dict[str, torch.Tensor] = {}
                    hooks = []

                    def save_phase(name):
                        def hook(_module, _inputs, output):
                            value = output[0] if isinstance(output, tuple) else output
                            if torch.is_tensor(value):
                                phase_tensors[name] = value.detach()

                        return hook

                    try:
                        for name, module_name in (
                            ("attention_output", "self_attn"),
                            ("mlp_output", "mlp"),
                            ("router_logits", "router"),
                            ("moe_output", "moe"),
                        ):
                            module = getattr(self, module_name, None)
                            if module is not None:
                                hooks.append(module.register_forward_hook(save_phase(name)))
                    except Exception as exc:
                        print(f"VLLM_LAYER_CAPTURE hook setup error: {exc}", flush=True)

                    try:
                        output = layer_orig(self, positions, hidden_states, *args, **kwargs)
                    finally:
                        for handle in hooks:
                            try:
                                handle.remove()
                            except Exception:
                                pass
                    try:
                        out_tensor = output[0] if isinstance(output, tuple) else output
                        layer_idx = int(getattr(self, "layer_idx", -1))
                        if torch.is_tensor(hidden_states) and torch.is_tensor(out_tensor):
                            _capture_layer(
                                layer_idx,
                                positions,
                                hidden_states.detach(),
                                out_tensor.detach(),
                                phase_tensors,
                            )
                    except Exception as exc:
                        print(f"VLLM_LAYER_CAPTURE save error: {exc}", flush=True)
                    return output

                layer_wrapped._vllm_layer_capture = True
                setattr(layer_cls, "forward", layer_wrapped)
                print(
                    "VLLM_LAYER_CAPTURE armed "
                    f"dir={_LAYER_OUT} layers={sorted(_LAYER_NAMES)} max_calls={_LAYER_MAX_CALLS}",
                    flush=True,
                )

    try:
        _patch()
    except Exception as exc:
        print(f"VLLM_READOUT_CAPTURE patch failed: {exc}", flush=True)
