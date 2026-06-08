#!/usr/bin/env python3
"""Probe FlashInfer NVFP4 KV global-scale conventions for SGLang.

SGLang currently has evidence that the packed KV layout reaches FlashInfer FA2,
but serving quality is corrupt.  This standalone diagnostic separates the
quantizer/reader convention from server integration: it quantizes identical K/V
tensors with candidate global-scale conventions, feeds them through the FA2
NVFP4 KV reader, and compares both dequantized tensors and attention output.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable

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


def decode_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().max().float() / (E2M1_MAX * MAX_BLOCK_SCALE_FP8)).clamp(min=1e-8)


def quantize_fp4(
    flashinfer: Any, x: torch.Tensor, quantizer_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, head_dim = x.shape
    data, sf = flashinfer.fp4_quantize(
        x.contiguous(),
        quantizer_scale.reshape(1).to(torch.float32),
        sf_vec_size=SF_VEC_SIZE,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_sf_8x4_layout=False,
        enable_pdl=None,
    )
    return data.reshape(rows, head_dim // 2).view(torch.uint8), sf.view(torch.uint8)


def quantize_nvfp4_kv(
    flashinfer: Any, x: torch.Tensor, quantizer_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    data, sf = flashinfer.nvfp4_kv_quantize(
        x.contiguous(), quantizer_scale.reshape(1).to(torch.float32)
    )
    return data.view(torch.uint8), sf.view(torch.uint8)


def dequant_linear(data: torch.Tensor, sf: torch.Tensor, reader_scale: torch.Tensor) -> torch.Tensor:
    rows, packed_cols = data.shape
    head_dim = packed_cols * 2
    sf_cols = head_dim // SF_VEC_SIZE
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=data.device)
    lo = (data & 0x0F).long()
    hi = ((data >> 4) & 0x0F).long()
    values = torch.empty(rows, head_dim, dtype=torch.float32, device=data.device)
    values[:, 0::2] = table[lo]
    values[:, 1::2] = table[hi]
    block_scale = sf.view(torch.float8_e4m3fn).float().reshape(rows, sf_cols, 1)
    return (
        values.reshape(rows, sf_cols, SF_VEC_SIZE) * block_scale
    ).reshape(rows, head_dim) * reader_scale.float()


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


def run_fa2_reader(
    flashinfer: Any,
    *,
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    k_reader_scale: torch.Tensor,
    v_reader_scale: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    workspace_bytes: int,
) -> torch.Tensor:
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
    return wrapper.run(
        q,
        (k_data, v_data),
        k_scale=float(k_reader_scale.item()),
        v_scale=float(v_reader_scale.item()),
        kv_cache_sf=(k_sf.unsqueeze(1).view(torch.float8_e4m3fn), v_sf.unsqueeze(1).view(torch.float8_e4m3fn)),
    )


def tensor_stats(name: str, x: torch.Tensor) -> dict[str, Any]:
    x_float = x.float()
    return {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(torch.isfinite(x_float).all().item()),
        "amax": float(x_float.abs().max().item()),
        "mean": float(x_float.mean().item()),
    }


def run_convention(
    flashinfer: Any,
    *,
    name: str,
    quantizer: Callable[[Any, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    quantizer_scale_kind: str,
    reader_scale_kind: str,
    key: torch.Tensor,
    value: torch.Tensor,
    q: torch.Tensor,
    k_decode_scale: torch.Tensor,
    v_decode_scale: torch.Tensor,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    workspace_bytes: int,
) -> dict[str, Any]:
    tokens = key.shape[0]
    quantizer_scale = {
        "decode": lambda x: x,
        "encode": lambda x: 1.0 / x,
    }[quantizer_scale_kind]
    reader_scale = {
        "decode": lambda x: x,
        "encode": lambda x: 1.0 / x,
    }[reader_scale_kind]
    k_quant_scale = quantizer_scale(k_decode_scale)
    v_quant_scale = quantizer_scale(v_decode_scale)
    k_reader_scale = reader_scale(k_decode_scale)
    v_reader_scale = reader_scale(v_decode_scale)

    try:
        flat_key = key.reshape(tokens * kv_heads, head_dim)
        flat_value = value.reshape(tokens * kv_heads, head_dim)
        k_data_2d, k_sf_2d = quantizer(flashinfer, flat_key, k_quant_scale)
        v_data_2d, v_sf_2d = quantizer(flashinfer, flat_value, v_quant_scale)
        k_data = k_data_2d.reshape(tokens, kv_heads, head_dim // 2)
        v_data = v_data_2d.reshape(tokens, kv_heads, head_dim // 2)
        k_sf = k_sf_2d.reshape(tokens, kv_heads, head_dim // SF_VEC_SIZE)
        v_sf = v_sf_2d.reshape(tokens, kv_heads, head_dim // SF_VEC_SIZE)
        k_dequant = dequant_linear(k_data_2d, k_sf_2d, k_reader_scale).reshape(
            tokens, kv_heads, head_dim
        )
        v_dequant = dequant_linear(v_data_2d, v_sf_2d, v_reader_scale).reshape(
            tokens, kv_heads, head_dim
        )
        source_reference = reference_attention(
            q,
            key,
            value,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        dequant_reference = reference_attention(
            q,
            k_dequant,
            v_dequant,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        out = run_fa2_reader(
            flashinfer,
            q=q,
            k_data=k_data,
            v_data=v_data,
            k_sf=k_sf,
            v_sf=v_sf,
            k_reader_scale=k_reader_scale,
            v_reader_scale=v_reader_scale,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            tokens=tokens,
            workspace_bytes=workspace_bytes,
        )
        finite = bool(torch.isfinite(out).all().item())
        result = {
            "name": name,
            "quantizer_scale_kind": quantizer_scale_kind,
            "reader_scale_kind": reader_scale_kind,
            "k_quantizer_scale": float(k_quant_scale.item()),
            "v_quantizer_scale": float(v_quant_scale.item()),
            "k_reader_scale": float(k_reader_scale.item()),
            "v_reader_scale": float(v_reader_scale.item()),
            "k_dequant_cosine_vs_source": cosine(k_dequant, key),
            "v_dequant_cosine_vs_source": cosine(v_dequant, value),
            "attention_cosine_vs_source": cosine(out, source_reference),
            "attention_relative_error_vs_source": relative_error(out, source_reference),
            "attention_cosine_vs_dequant": cosine(out, dequant_reference),
            "attention_relative_error_vs_dequant": relative_error(out, dequant_reference),
            "finite": finite,
        }
        result["passed_source"] = bool(
            result["k_dequant_cosine_vs_source"] >= 0.99
            and result["v_dequant_cosine_vs_source"] >= 0.99
            and result["attention_cosine_vs_source"] >= 0.99
            and finite
        )
        result["reader_matched_dequant"] = bool(
            result["attention_cosine_vs_dequant"] >= 0.99 and finite
        )
        return result
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "name": name,
            "quantizer_scale_kind": quantizer_scale_kind,
            "reader_scale_kind": reader_scale_kind,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "passed_source": False,
            "reader_matched_dequant": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
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

    k_decode_scale = decode_scale(key)
    v_decode_scale = decode_scale(value)
    workspace_bytes = args.workspace_mib * 1024 * 1024
    conventions = [
        (
            "fp4_quantize_encode_reader_decode",
            quantize_fp4,
            "encode",
            "decode",
        ),
        (
            "nvfp4_kv_quantize_encode_reader_decode",
            quantize_nvfp4_kv,
            "encode",
            "decode",
        ),
        (
            "nvfp4_kv_quantize_decode_reader_decode",
            quantize_nvfp4_kv,
            "decode",
            "decode",
        ),
        (
            "nvfp4_kv_quantize_decode_reader_encode",
            quantize_nvfp4_kv,
            "decode",
            "encode",
        ),
    ]
    results = [
        run_convention(
            flashinfer,
            name=name,
            quantizer=quantizer,
            quantizer_scale_kind=quantizer_scale_kind,
            reader_scale_kind=reader_scale_kind,
            key=key,
            value=value,
            q=q,
            k_decode_scale=k_decode_scale,
            v_decode_scale=v_decode_scale,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            workspace_bytes=workspace_bytes,
        )
        for name, quantizer, quantizer_scale_kind, reader_scale_kind in conventions
    ]

    payload = {
        "schema": "sglang-nvfp4-kv-convention-probe/v1",
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
        "scales": {
            "k_decode_scale": float(k_decode_scale.item()),
            "v_decode_scale": float(v_decode_scale.item()),
            "k_encode_scale": float((1.0 / k_decode_scale).item()),
            "v_encode_scale": float((1.0 / v_decode_scale).item()),
        },
        "inputs": [
            tensor_stats("key", key),
            tensor_stats("value", value),
            tensor_stats("query", q),
        ],
        "results": results,
        "all_ok": bool(any(row.get("passed_source") for row in results)),
    }

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
