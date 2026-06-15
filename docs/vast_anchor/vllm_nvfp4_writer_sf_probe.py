#!/usr/bin/env python3
"""vLLM NVFP4 KV writer scale-factor byte probe.

This is a serving-path discriminator for Gemma 4 NVFP4 KV failures. It loads
HF eager models, extracts real K/V tensors from past_key_values, writes those
tensors through vLLM's real CUDA writer:

    torch.ops._C_cache_ops.reshape_and_cache_flash(..., "nvfp4", ...)

Then it compares the stored per-block FP8 E4M3 scale-factor bytes against the
bytes implied by the public NVFP4 recipe:

    stored_sf = fp8_e4m3((1 / vllm_scale) * (amax(block16) / 6))

The comparison is byte-exact and intentionally independent of FlashInfer
attention. A 12B pass + 26B fail points at a data-dependent writer/global-scale
bug; both passing moves the hunt to the read/dequant/attention feed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

try:
    # Current Transformers can select torch._grouped_mm for Gemma 4 MoE on
    # torch versions/devices where the op later rejects sm_120. Force the slow
    # fallback so this probe measures KV writer behavior, not HF grouped-mm
    # dispatch.
    import transformers.integrations.moe as _hf_moe

    _hf_moe._can_use_grouped_mm = lambda *_args, **_kwargs: False
except Exception:
    pass


DEFAULT_MODELS = (
    "google/gemma-4-12b-it",
    "google/gemma-4-26b-a4b-it",
)


def get_prompt_ids(tokenizer: Any, ctx: int) -> list[int]:
    prompt = (
        "Analyze cache quantization in long-context transformer serving. "
        "Explain how key and value cache precision can affect attention, then "
        "continue with concrete numerical examples."
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "data") and "input_ids" in ids.data:
        ids = ids.data["input_ids"]
    if isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) < ctx:
        ids = (ids * ((ctx // max(1, len(ids))) + 1))[:ctx]
    return ids[:ctx]


def first_present_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


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


def normalize_cache_tensor(t: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return [tokens, kv_heads, head_dim] on CUDA, contiguous."""
    x = t.detach()
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise RuntimeError(f"expected batch=1 cache tensor, got {list(t.shape)}")
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"unexpected cache tensor shape {list(t.shape)}")

    if x.shape[0] == seq_len:
        return x.contiguous()
    if x.shape[1] == seq_len:
        return x.permute(1, 0, 2).contiguous()

    # Some cache implementations may trim or pad. Prefer the larger non-head
    # dimension as sequence when there is no exact match.
    if x.shape[0] > x.shape[1]:
        return x.contiguous()
    return x.permute(1, 0, 2).contiguous()


def select_layers(n: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(n))
    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        selected.append(n - 1 if part == "last" else int(part))
    return sorted({i for i in selected if 0 <= i < n})


def layer_kind(layer_idx: int, config: Any) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types and layer_idx < len(layer_types):
        return str(layer_types[layer_idx])
    sliding = getattr(config, "sliding_window_pattern", None)
    if isinstance(sliding, int) and sliding > 0:
        return "sliding" if layer_idx % sliding else "global"
    return "unknown"


def expected_sf_bytes(x: torch.Tensor, vllm_scale: float, block: int) -> torch.Tensor:
    """Expected FP8 E4M3 scale-factor bytes for [tokens, heads, dim]."""
    if x.shape[-1] % block != 0:
        raise RuntimeError(f"head_dim {x.shape[-1]} is not divisible by {block}")
    global_scale = 1.0 / vllm_scale
    xb = x.float().reshape(x.shape[0], x.shape[1], x.shape[2] // block, block)
    sf = global_scale * (xb.abs().amax(dim=-1) / 6.0)
    return sf.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)


def scale_region_view(cache_side: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Return scale bytes as [blocks, block_size, heads, scale_dim].

    The NVFP4 writer overlays page storage as [data][scale], so the logical
    tensor's last dimension is not the physical split point.
    """
    num_blocks, block_size, num_heads, full_dim = cache_side.shape
    data_dim = head_dim // 2
    scale_dim = head_dim // 16
    expected_full = data_dim + scale_dim
    if full_dim != expected_full:
        raise RuntimeError(f"cache full_dim={full_dim}, expected {expected_full}")
    data_per_page = block_size * num_heads * data_dim
    scale_per_page = block_size * num_heads * scale_dim
    flat = cache_side.reshape(num_blocks, -1)
    scale = flat[:, data_per_page : data_per_page + scale_per_page]
    return scale.reshape(num_blocks, block_size, num_heads, scale_dim)


def compare_bytes(stored: torch.Tensor, expected: torch.Tensor, block_size: int) -> dict[str, Any]:
    tokens = expected.shape[0]
    num_blocks = (tokens + block_size - 1) // block_size
    pages = torch.arange(tokens, device=expected.device) // block_size
    offsets = torch.arange(tokens, device=expected.device) % block_size
    got = stored[:num_blocks][pages, offsets]
    mismatch = got != expected
    mismatch_count = int(mismatch.sum().item())
    total = int(expected.numel())
    samples = []
    if mismatch_count:
        idx = mismatch.nonzero(as_tuple=False)[:16]
        for row in idx:
            t, h, g = [int(v) for v in row.tolist()]
            samples.append(
                {
                    "token": t,
                    "head": h,
                    "group": g,
                    "stored": int(got[t, h, g].item()),
                    "expected": int(expected[t, h, g].item()),
                }
            )
    return {
        "total": total,
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_count / max(1, total),
        "samples": samples,
        "stored_min": int(got.min().item()),
        "stored_max": int(got.max().item()),
        "expected_min": int(expected.min().item()),
        "expected_max": int(expected.max().item()),
    }


def run_writer_side(
    x: torch.Tensor,
    *,
    vllm_scale: float,
    block_size: int,
    side: str,
) -> dict[str, Any]:
    """Write x as both K and V, then compare the selected side's SF bytes."""
    import vllm._custom_ops as ops

    tokens, heads, head_dim = x.shape
    data_dim = head_dim // 2
    scale_dim = head_dim // 16
    full_dim = data_dim + scale_dim
    num_blocks = (tokens + block_size - 1) // block_size
    key_cache = torch.empty(
        (num_blocks, block_size, heads, full_dim), dtype=torch.uint8, device=x.device
    )
    value_cache = torch.empty_like(key_cache)
    key_cache.fill_(0xA5)
    value_cache.fill_(0x5A)
    slots = torch.arange(tokens, dtype=torch.int64, device=x.device)
    scale = torch.tensor([vllm_scale], dtype=torch.float32, device=x.device)

    ops.reshape_and_cache_flash(
        x, x, key_cache, value_cache, slots, "nvfp4", scale, scale
    )
    torch.cuda.synchronize()

    expected = expected_sf_bytes(x, vllm_scale=vllm_scale, block=16)
    cache = key_cache if side == "K" else value_cache
    stored = scale_region_view(cache, head_dim=head_dim)
    cmp = compare_bytes(stored, expected, block_size=block_size)
    return {
        "side": side,
        "tokens": tokens,
        "heads": heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "vllm_scale": vllm_scale,
        "global_scale": 1.0 / vllm_scale,
        **cmp,
    }


def run_model(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, trust_remote_code=True
    )
    ids = get_prompt_ids(tokenizer, args.ctx)
    input_ids = torch.tensor([ids], device="cuda")

    kwargs = dict(
        token=token,
        trust_remote_code=True,
        device_map={"": "cuda"},
        attn_implementation="eager",
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, **kwargs
        ).eval()
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, **kwargs
        ).eval()

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    entries = legacy_cache_entries(out.past_key_values)
    selected = select_layers(len(entries), args.layers)
    rows: list[dict[str, Any]] = []
    for layer_idx in selected:
        k_raw, v_raw = entries[layer_idx]
        for side, raw in (("K", k_raw), ("V", v_raw)):
            x = normalize_cache_tensor(raw, len(ids))
            if args.sample_tokens > 0 and x.shape[0] > args.sample_tokens:
                x = x[-args.sample_tokens :].contiguous()
            for vllm_scale in args.vllm_scales:
                row = run_writer_side(
                    x,
                    vllm_scale=vllm_scale,
                    block_size=args.block_size,
                    side=side,
                )
                row.update(
                    {
                        "model": model_name,
                        "layer": layer_idx,
                        "layer_kind": layer_kind(layer_idx, model.config),
                    }
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    del out, entries, model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--sample-tokens", type=int, default=256)
    parser.add_argument("--layers", default="0,1,12,last")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--vllm-scales",
        default="0.1,0.07",
        help="Comma-separated vLLM layer._k/_v_scale values.",
    )
    parser.add_argument("--out", default="/root/writer_sf_probe_results.json")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_NVFP4_KV_LINEAR_V_SF", "1")
    args = parse_args()
    args.vllm_scales = [float(x) for x in args.vllm_scales.split(",") if x]

    import vllm  # noqa: F401

    meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "vllm_nvfp4_kv_linear_v_sf": os.environ.get("VLLM_NVFP4_KV_LINEAR_V_SF"),
        "models": args.models,
        "ctx": args.ctx,
        "sample_tokens": args.sample_tokens,
        "layers": args.layers,
        "block_size": args.block_size,
        "vllm_scales": args.vllm_scales,
    }
    print(json.dumps({"meta": meta}), flush=True)
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        rows.extend(run_model(args, model_name))
    output = {"meta": meta, "rows": rows}
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
