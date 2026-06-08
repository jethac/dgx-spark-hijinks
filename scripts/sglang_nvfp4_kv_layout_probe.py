#!/usr/bin/env python3
"""Probe FlashInfer FA2 NVFP4 KV scale-factor layout for SGLang-style storage.

SGLang's FP4 MHA KV pool stores packed E2M1 K/V data as flat per-token uint8
tensors with shape [tokens, kv_heads, head_dim / 2].  For page_size=1, the
matching FlashInfer scale-factor tensors still need an explicit page dimension:
[tokens, 1, kv_heads, head_dim / 16].  This script proves that reader contract
against a faithful dequantized attention reference.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch


E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
E2M1_MAX = 6.0
MAX_BLOCK_SCALE_FP8 = 448.0
SF_VEC_SIZE = 16


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a_float = a.reshape(-1).float()
    b_float = b.reshape(-1).float()
    return (a_float - b_float).norm().item() / (b_float.norm().item() + 1e-9)


def global_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().max().float() / (E2M1_MAX * MAX_BLOCK_SCALE_FP8)).clamp(min=1e-8)


def quantize_linear(
    flashinfer: Any, x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, kv_heads, head_dim = x.shape
    data, sf = flashinfer.fp4_quantize(
        x.reshape(tokens * kv_heads, head_dim).contiguous(),
        (1.0 / scale).reshape(1).to(torch.float32),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_sf_8x4_layout=False,
        enable_pdl=None,
    )
    sf_cols = head_dim // SF_VEC_SIZE
    return (
        data.reshape(tokens, kv_heads, head_dim // 2).view(torch.uint8),
        sf.reshape(tokens, kv_heads, sf_cols).view(torch.uint8),
    )


def dequant_linear(data: torch.Tensor, sf: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
    ).reshape(tokens, kv_heads, head_dim) * scale.float()


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
    out = torch.empty(1, query_heads, head_dim, dtype=torch.float32, device=q.device)
    key_f = key.float()
    value_f = value.float()
    for head in range(query_heads):
        weights = torch.softmax(
            (q[0, head].float() @ key_f[:, head // groups, :].T) * sm_scale, dim=-1
        )
        out[0, head] = weights @ value_f[:, head // groups, :]
    return out


def run_case(
    flashinfer: Any,
    *,
    sf_rank: int,
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    reference: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    workspace_bytes: int,
) -> dict[str, Any]:
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="fa2"
    )
    kv_indptr = torch.tensor([0, tokens], dtype=torch.int32, device=q.device)
    kv_indices = torch.arange(tokens, dtype=torch.int32, device=q.device)
    last_page_len = torch.tensor([1], dtype=torch.int32, device=q.device)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        query_heads,
        kv_heads,
        head_dim,
        1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
    )

    if sf_rank == 4:
        k_sf_arg = k_sf.unsqueeze(1).view(torch.float8_e4m3fn)
        v_sf_arg = v_sf.unsqueeze(1).view(torch.float8_e4m3fn)
    elif sf_rank == 3:
        k_sf_arg = k_sf.view(torch.float8_e4m3fn)
        v_sf_arg = v_sf.view(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unsupported sf_rank={sf_rank}")

    try:
        out = wrapper.run(
            q,
            (k_data, v_data),
            k_scale=float(k_scale.item()),
            v_scale=float(v_scale.item()),
            kv_cache_sf=(k_sf_arg, v_sf_arg),
        )
        cos = cosine(out, reference)
        rel = relative_error(out, reference)
        finite = bool(torch.isfinite(out).all().item())
        return {
            "sf_rank": sf_rank,
            "sf_shape": list(k_sf_arg.shape),
            "cosine_vs_dequant": cos,
            "relative_error_vs_dequant": rel,
            "finite": finite,
            "passed": bool(cos >= 0.99 and finite),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "sf_rank": sf_rank,
            "sf_shape": list(k_sf_arg.shape),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "passed": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=128)
    args = parser.parse_args()

    import flashinfer

    if args.query_heads % args.kv_heads != 0:
        raise ValueError("--query-heads must be divisible by --kv-heads")
    if args.head_dim % SF_VEC_SIZE != 0:
        raise ValueError("--head-dim must be divisible by 16")

    torch.manual_seed(args.seed)
    device = "cuda:0"
    key = torch.randn(
        args.tokens, args.kv_heads, args.head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5
    value = torch.randn(
        args.tokens, args.kv_heads, args.head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5
    q = torch.randn(1, args.query_heads, args.head_dim, dtype=torch.bfloat16, device=device) * 0.5

    k_scale = global_scale(key)
    v_scale = global_scale(value)
    k_data, k_sf = quantize_linear(flashinfer, key, k_scale)
    v_data, v_sf = quantize_linear(flashinfer, value, v_scale)
    reference = reference_attention(
        q,
        dequant_linear(k_data, k_sf, k_scale),
        dequant_linear(v_data, v_sf, v_scale),
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )

    workspace_bytes = args.workspace_mib * 1024 * 1024
    results = [
        run_case(
            flashinfer,
            sf_rank=4,
            q=q,
            k_data=k_data,
            v_data=v_data,
            k_sf=k_sf,
            v_sf=v_sf,
            k_scale=k_scale,
            v_scale=v_scale,
            reference=reference,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            tokens=args.tokens,
            workspace_bytes=workspace_bytes,
        ),
        run_case(
            flashinfer,
            sf_rank=3,
            q=q,
            k_data=k_data,
            v_data=v_data,
            k_sf=k_sf,
            v_sf=v_sf,
            k_scale=k_scale,
            v_scale=v_scale,
            reference=reference,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            tokens=args.tokens,
            workspace_bytes=workspace_bytes,
        ),
    ]

    payload = {
        "schema": "sglang-nvfp4-kv-layout-probe/v1",
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "shape": {
            "tokens": args.tokens,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "page_size": 1,
        },
        "packed_kv_shape": list(k_data.shape),
        "flat_scale_shape": list(k_sf.shape),
        "results": results,
        "all_ok": bool(results[0]["passed"] and not results[1]["passed"]),
    }

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
