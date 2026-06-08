#!/usr/bin/env python3
"""Probe SGLang's FP4 MHA KV pool contract against FlashInfer FA2.

This exercises the same tensor surfaces used by the SGLang serving path:
`MHATokenToKVPoolFP4.set_kv_buffer()` writes packed K/V plus scale buffers,
then `get_kv_buffer()`, `get_kv_scale_buffer()`, and `get_kv_global_scale()`
feed FlashInfer's FA2 paged-KV reader. The reference path dequantizes the
pool contents and computes attention from those dequantized tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
SF_VEC_SIZE = 16


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_add_sglang_source(path: Path | None) -> None:
    if path is None:
        candidate = _repo_root() / "third_party" / "sglang" / "python"
    else:
        candidate = path
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a_float = a.reshape(-1).float()
    b_float = b.reshape(-1).float()
    return (a_float - b_float).norm().item() / (b_float.norm().item() + 1e-9)


def dequant_pool(data: torch.Tensor, sf: torch.Tensor, scale: float) -> torch.Tensor:
    tokens, kv_heads, packed_cols = data.shape
    head_dim = packed_cols * 2
    sf_cols = head_dim // SF_VEC_SIZE
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=data.device)
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()
    values = torch.empty(tokens, kv_heads, head_dim, dtype=torch.float32, device=data.device)
    values[..., 0::2] = table[lo]
    values[..., 1::2] = table[hi]
    block_scale = sf.view(torch.float8_e4m3fn).float().reshape(
        tokens, kv_heads, sf_cols, 1
    )
    return (
        values.reshape(tokens, kv_heads, sf_cols, SF_VEC_SIZE) * block_scale
    ).reshape(tokens, kv_heads, head_dim) * float(scale)


def reference_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    groups = query_heads // kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q.shape[0], query_heads, head_dim, dtype=torch.float32, device=q.device)
    key_f = key.float()
    value_f = value.float()
    for batch_idx in range(q.shape[0]):
        for head in range(query_heads):
            weights = torch.softmax(
                (q[batch_idx, head].float() @ key_f[:, head // groups, :].T) * sm_scale,
                dim=-1,
            )
            out[batch_idx, head] = weights @ value_f[:, head // groups, :]
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    _maybe_add_sglang_source(args.sglang_python)

    import flashinfer  # type: ignore
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolFP4  # type: ignore
    from sglang.srt.utils import is_sm100_supported, is_sm120_supported  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SGLang FP4 pool bridge probe.")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("--query-heads must be divisible by --kv-heads")
    if args.head_dim % SF_VEC_SIZE != 0:
        raise ValueError("--head-dim must be divisible by 16")
    if args.page_size != 1:
        raise ValueError("This probe intentionally covers the SGLang page_size=1 serving path.")

    torch.manual_seed(args.seed)
    device = "cuda:0"
    dtype = torch.bfloat16
    key = (
        torch.randn(args.tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device)
        * args.input_scale
    )
    value = (
        torch.randn(args.tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device)
        * args.input_scale
    )
    q = (
        torch.randn(args.batch_size, args.query_heads, args.head_dim, dtype=dtype, device=device)
        * args.input_scale
    )

    pool = MHATokenToKVPoolFP4(
        size=args.tokens + args.page_size,
        page_size=args.page_size,
        dtype=dtype,
        head_num=args.kv_heads,
        head_dim=args.head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    layer = SimpleNamespace(layer_id=0, k_scale_float=None, v_scale_float=None)
    loc = torch.arange(1, args.tokens + 1, dtype=torch.int64, device=device)
    pool.set_kv_buffer(layer, loc, key, value)

    k_buf, v_buf = pool.get_kv_buffer(0)
    k_sf, v_sf = pool.get_kv_scale_buffer(0)
    k_scale, v_scale = pool.get_kv_global_scale(0)
    k_seq = k_buf[loc]
    v_seq = v_buf[loc]
    k_sf_seq = k_sf[loc]
    v_sf_seq = v_sf[loc]
    k_dequant = dequant_pool(k_seq, k_sf_seq, k_scale).to(dtype)
    v_dequant = dequant_pool(v_seq, v_sf_seq, v_scale).to(dtype)
    reference_dequant = reference_attention(
        q,
        k_dequant,
        v_dequant,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    reference_source = reference_attention(
        q,
        key,
        value,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )

    workspace = torch.empty(args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    kv_indptr = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
    kv_indices = loc.to(torch.int32)
    last_page_len = torch.tensor([1], dtype=torch.int32, device=device)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        args.page_size,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
    )

    result: dict[str, Any] = {"passed": False}
    try:
        actual = wrapper.run(
            q,
            (k_buf, v_buf),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            kv_cache_sf=(k_sf.unsqueeze(1), v_sf.unsqueeze(1)),
        )
        result.update(
            {
                "finite": bool(torch.isfinite(actual).all().item()),
                "attention_cosine_vs_dequant": cosine(actual, reference_dequant),
                "attention_relative_error_vs_dequant": relative_error(
                    actual, reference_dequant
                ),
                "attention_cosine_vs_source": cosine(actual, reference_source),
                "attention_relative_error_vs_source": relative_error(actual, reference_source),
                "key_dequant_cosine_vs_source": cosine(k_dequant, key),
                "value_dequant_cosine_vs_source": cosine(v_dequant, value),
            }
        )
        result["passed"] = bool(
            result["finite"]
            and result["attention_cosine_vs_dequant"] >= args.cosine_threshold
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        result.update({"error": repr(exc), "traceback": traceback.format_exc()})

    return {
        "schema": "sglang-fp4-pool-bridge-probe/v1",
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "flashinfer_file": getattr(flashinfer, "__file__", None),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "sglang_python": str(args.sglang_python) if args.sglang_python else None,
        "is_sm100_supported": bool(is_sm100_supported()),
        "is_sm120_supported": bool(is_sm120_supported()),
        "shape": {
            "tokens": args.tokens,
            "batch_size": args.batch_size,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "page_size": args.page_size,
        },
        "pool": {
            "k_buffer_shape": list(k_buf.shape),
            "v_buffer_shape": list(v_buf.shape),
            "k_scale_buffer_shape": list(k_sf.shape),
            "v_scale_buffer_shape": list(v_sf.shape),
            "k_global_scale": float(k_scale),
            "v_global_scale": float(v_scale),
            "loc_start": int(loc[0].item()),
            "loc_stop_inclusive": int(loc[-1].item()),
        },
        "result": result,
        "all_ok": bool(result["passed"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sglang-python", type=Path)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
