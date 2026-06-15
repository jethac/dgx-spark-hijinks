#!/usr/bin/env python3
"""Gemma 4 NVFP4 KV round-trip discriminator.

Loads Gemma 4 models in HF eager, extracts the real returned past_key_values,
then measures NVFP4 quantize/dequantize error on K and V cache tensors.

This answers whether 26B-A4B's full-NVFP4 serving failure is plausibly caused
by inherently worse KV quantizability, or whether the actual K/V tensors
round-trip about as well as 12B and the failure is likely in the serving path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

try:
    # Torch 2.8 exposes torch._grouped_mm, but on RTX PRO 6000 / sm_120 the
    # operator itself rejects non-sm90 devices. Transformers main currently
    # assumes this path is usable on SM80+ for torch>=2.9-style semantics. Force
    # the built-in slow fallback so MoE layers can run during this discriminator.
    import transformers.integrations.moe as _hf_moe

    _hf_moe._can_use_grouped_mm = lambda *_args, **_kwargs: False
except Exception:
    pass


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3_table() -> torch.Tensor:
    vals = {0.0}
    for e in range(0, 16):
        for m in range(0, 8):
            v = (m / 8.0) * 2.0 ** (1 - 7) if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
            if v <= 448.0:
                vals.add(v)
    return torch.tensor(sorted(vals))


E4M3 = _e4m3_table()


def to_e4m3(x: torch.Tensor) -> torch.Tensor:
    a = x.clamp(min=0.0, max=448.0)
    tbl = E4M3.to(x.device, x.dtype)
    mids = (tbl[1:] + tbl[:-1]) / 2
    return tbl[torch.bucketize(a, mids)]


def quant_e2m1(x: torch.Tensor) -> torch.Tensor:
    sign = x.sign()
    a = x.abs().clamp(max=6.0)
    levels = E2M1.to(x.device, x.dtype)
    mids = (levels[1:] + levels[:-1]) / 2
    return sign * levels[torch.bucketize(a, mids)]


def qdq_nvfp4(x: torch.Tensor, global_scale: float, block: int = 16) -> torch.Tensor:
    lead, dim = x.shape[:-1], x.shape[-1]
    xb = x.reshape(*lead, dim // block, block).float()
    vec_max = xb.abs().amax(-1, keepdim=True)
    sf_value8 = to_e4m3(global_scale * (vec_max / 6.0))
    out_scale = torch.where(sf_value8 > 0, global_scale / sf_value8, torch.zeros_like(sf_value8))
    q = quant_e2m1(xb * out_scale)
    deq = torch.where(sf_value8 > 0, q * sf_value8 / global_scale, torch.zeros_like(q))
    return deq.reshape(*lead, dim).to(x.dtype)


def tensor_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    xf = x.float().reshape(-1)
    yf = y.float().reshape(-1)
    diff = xf - yf
    den = xf.pow(2).sum().clamp(min=1e-12)
    denom_cos = (xf.norm() * yf.norm()).clamp(min=1e-12)
    return {
        "rel_l2": float((diff.pow(2).sum() / den).sqrt().cpu()),
        "cosine": float((torch.dot(xf, yf) / denom_cos).cpu()),
        "mean_abs": float(diff.abs().mean().cpu()),
        "max_abs": float(diff.abs().max().cpu()),
    }


def block_stats(x: torch.Tensor, block: int = 16) -> dict[str, float]:
    xb = x.float().reshape(-1, block)
    amax = xb.abs().amax(-1)
    return {
        "block_amax_p50": float(torch.quantile(amax, 0.50).cpu()),
        "block_amax_p90": float(torch.quantile(amax, 0.90).cpu()),
        "block_amax_p99": float(torch.quantile(amax, 0.99).cpu()),
        "block_amax_max": float(amax.max().cpu()),
    }


def best_scale(x: torch.Tensor, candidates: list[float], block: int = 16) -> dict[str, Any]:
    curve = []
    best = None
    for gs in candidates:
        deq = qdq_nvfp4(x, gs, block=block)
        metrics = tensor_metrics(x, deq)
        row = {"global_scale": gs, "vllm_fixed_scale": 1.0 / gs, **metrics}
        curve.append(row)
        if best is None or row["rel_l2"] < best["rel_l2"]:
            best = row
    assert best is not None
    return {"best": best, "curve": curve}


def get_text(tokenizer: Any, corpus: Path | None, ctx: int, chat_template: bool) -> list[int]:
    if corpus is not None:
        text = corpus.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text, add_special_tokens=False)
    elif chat_template:
        prompt = (
            "Summarize the first principles behind attention, then continue with a concise "
            "technical explanation of why cache quantization can affect long-context decoding."
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        text = (
            "The history of computing is a story about memory hierarchies, numerical formats, "
            "and careful engineering. " * ((ctx // 20) + 8)
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < ctx:
        ids = (ids * ((ctx // max(1, len(ids))) + 1))[:ctx]
    return ids[:ctx]


def legacy_cache_entries(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past_key_values is None:
        raise RuntimeError("model did not return past_key_values")
    if hasattr(past_key_values, "to_legacy_cache"):
        entries = past_key_values.to_legacy_cache()
        return [(kv[0], kv[1]) for kv in entries]
    if isinstance(past_key_values, (tuple, list)):
        out = []
        for item in past_key_values:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append((item[0], item[1]))
        if out:
            return out
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        out = []
        for layer in layers:
            k = first_present_attr(layer, ("keys", "key_cache", "k_cache"))
            v = first_present_attr(layer, ("values", "value_cache", "v_cache"))
            if torch.is_tensor(k) and torch.is_tensor(v):
                out.append((k, v))
        if out:
            return out
    raise RuntimeError(f"unsupported past_key_values type: {type(past_key_values)!r}")


def first_present_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def normalize_cache_tensor(t: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return [tokens, heads, dim] on CPU."""
    x = t.detach()
    if x.ndim == 4:
        x = x[0]
        if x.shape[0] == seq_len or x.shape[0] <= seq_len:
            # [tokens, heads, dim]
            pass
        elif x.shape[1] == seq_len or x.shape[1] <= seq_len:
            # [heads, tokens, dim]
            x = x.permute(1, 0, 2)
        else:
            raise RuntimeError(f"cannot infer cache layout for shape {list(t.shape)}")
    elif x.ndim == 3:
        if x.shape[0] == seq_len or x.shape[0] <= seq_len:
            pass
        elif x.shape[1] == seq_len or x.shape[1] <= seq_len:
            x = x.permute(1, 0, 2)
        else:
            raise RuntimeError(f"cannot infer cache layout for shape {list(t.shape)}")
    else:
        raise RuntimeError(f"unexpected cache tensor shape {list(t.shape)}")
    return x.contiguous().cpu()


def layer_kind(layer_idx: int, config: Any) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types and layer_idx < len(layer_types):
        return str(layer_types[layer_idx])
    sliding = getattr(config, "sliding_window_pattern", None)
    if isinstance(sliding, int) and sliding > 0:
        return "sliding" if layer_idx % sliding else "global"
    return "unknown"


def select_layers(n: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "last":
            out.append(n - 1)
        else:
            out.append(int(part))
    return sorted({i for i in out if 0 <= i < n})


def run_model(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    ids = get_text(tokenizer, Path(args.corpus) if args.corpus else None, args.ctx, args.chat_template)
    input_ids = torch.tensor([ids], device="cuda")

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="eager",
    ).eval()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    entries = legacy_cache_entries(out.past_key_values)
    selected = select_layers(len(entries), args.layers)
    candidates = [2.0 ** e for e in torch.arange(-4, 12, 0.5).tolist()]
    fixed_vllm_scales = [float(x) for x in args.fixed_vllm_scales.split(",") if x]
    fixed_global_scales = [1.0 / x for x in fixed_vllm_scales]
    fixed_global_scales = [x for x in fixed_global_scales if x not in candidates]

    rows = []
    for layer_idx in selected:
        k_raw, v_raw = entries[layer_idx]
        for tag, raw in (("K", k_raw), ("V", v_raw)):
            x = normalize_cache_tensor(raw, args.ctx)
            if args.sample_tokens > 0 and x.shape[0] > args.sample_tokens:
                x = x[-args.sample_tokens :]
            xf = x.reshape(-1, x.shape[-1]).contiguous()
            scale = best_scale(xf, candidates + fixed_global_scales, block=args.block)
            fixed = {}
            for vllm_scale in fixed_vllm_scales:
                gs = 1.0 / vllm_scale
                fixed[str(vllm_scale)] = tensor_metrics(xf, qdq_nvfp4(xf, gs, block=args.block))
            rows.append(
                {
                    "layer": layer_idx,
                    "layer_kind": layer_kind(layer_idx, model.config),
                    "tensor": tag,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    **block_stats(xf, block=args.block),
                    "best": scale["best"],
                    "fixed_vllm_scales": fixed,
                }
            )
            print(
                f"ROW model={model_name} layer={layer_idx} {tag} kind={rows[-1]['layer_kind']} "
                f"best_rel_l2={scale['best']['rel_l2']:.6f} best_vllm_scale={scale['best']['vllm_fixed_scale']:.5g}",
                flush=True,
            )
    del model, out, entries
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "model": model_name,
        "ctx": args.ctx,
        "sample_tokens": args.sample_tokens,
        "layers": selected,
        "token_count": len(ids),
        "cache_rows": rows,
    }


def summarize(models: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for payload in models:
        grouped: dict[str, list[float]] = {}
        for row in payload["cache_rows"]:
            key = f"{row['layer_kind']}:{row['tensor']}"
            grouped.setdefault(key, []).append(row["best"]["rel_l2"])
        summary[payload["model"]] = {
            key: {
                "n": len(vals),
                "mean_best_rel_l2": sum(vals) / len(vals),
                "max_best_rel_l2": max(vals),
            }
            for key, vals in grouped.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["google/gemma-4-12b-it", "google/gemma-4-26b-a4b-it"])
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--layers", default="0,1,2,12,24,last")
    parser.add_argument("--corpus")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--fixed-vllm-scales", default="0.05,0.07,0.1")
    parser.add_argument("--out", default="/root/kv_roundtrip/results.json")
    args = parser.parse_args()

    meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        "args": vars(args),
        "note": "past_key_values are the HF eager KV-cache tensors returned by the model.",
    }
    print("META " + json.dumps(meta, sort_keys=True), flush=True)
    payload = {"meta": meta, "models": []}
    for model_name in args.models:
        payload["models"].append(run_model(args, model_name))
    payload["summary"] = summarize(payload["models"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("OUT", out, flush=True)
    print("SUMMARY " + json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
