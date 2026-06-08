#!/usr/bin/env python3
"""Replay a dumped vLLM/FlashInfer NVFP4 paged-prefill call on CPU.

This is a diagnostic reference, not a replacement for FlashInfer. It consumes the
``spark-active-page-prefill-dump/v1`` payloads emitted by the vLLM fork and computes a
plain causal attention result from dequantized K/V pages. The purpose is to check whether
the dumped FlashInfer wrapper output is compatible with any normal dequantized-attention
result, or instead looks like packed carrier bytes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _tensor_head(tensor: Any, limit: int) -> list[float | int]:
    cpu = tensor.detach().cpu().reshape(-1)
    out: list[float | int] = []
    for value in cpu[: min(limit, int(cpu.numel()))].tolist():
        if isinstance(value, int):
            out.append(value)
        else:
            out.append(float(value))
    return out


def _tensor_stats(torch: Any, tensor: Any, head: int) -> dict[str, Any]:
    cpu = tensor.detach().cpu()
    flat = cpu.reshape(-1)
    result: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(flat.numel()),
        "head": _tensor_head(cpu, head),
    }
    if flat.numel() == 0:
        return result
    f = flat.float()
    finite = torch.isfinite(f)
    result.update(
        {
            "finite": int(finite.sum().item()),
            "all_finite": bool(finite.all().item()),
            "min": float(f.min().item()),
            "max": float(f.max().item()),
            "mean": float(f.mean().item()),
            "rms": float(torch.sqrt(torch.mean(f * f)).item()),
            "byte_like": bool(
                finite.all().item()
                and f.min().item() >= 0.0
                and f.max().item() <= 255.0
                and f.mean().item() > 32.0
            ),
        }
    )
    return result


def _metrics(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    if a.shape != b.shape:
        return {"same_shape": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    delta = a - b
    return {
        "same_shape": True,
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms_abs": float(torch.sqrt(torch.mean(delta * delta)).item()),
    }


def _dequant_nvfp4(torch: Any, packed: Any, sf: Any, global_scale: float) -> Any:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    indices = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    lut = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32)
    values = lut[indices.to(torch.long)]
    scale = sf.to(torch.float32).repeat_interleave(16, dim=-1)
    return values * scale * float(global_scale)


def _deswizzle_v_scale_nhd(torch: Any, scale: Any) -> Any:
    pages, page_size, heads, scale_dim = scale.shape
    if page_size % 4 or scale_dim % 4:
        raise ValueError("NHD V-scale deswizzle requires page size and scale dim divisible by 4")
    return (
        scale.reshape(pages, page_size // 4, 4, heads, 4, scale_dim // 4)
        .permute(0, 1, 5, 3, 2, 4)
        .reshape(pages, page_size, heads, scale_dim)
        .contiguous()
    )


def _gather_pages(torch: Any, pages: Any, obj: dict[str, Any]) -> Any:
    active_pages = [int(x) for x in obj["active_pages"].tolist()]
    active_index = {page: i for i, page in enumerate(active_pages)}
    page_ids = [int(x) for x in obj["paged_kv_indices"].tolist()]
    last_page_len = int(obj["paged_kv_last_page_len"][0].item())
    if not page_ids:
        raise ValueError("dump contains no paged_kv_indices")

    chunks = []
    for i, page_id in enumerate(page_ids):
        if page_id not in active_index:
            raise ValueError(f"page {page_id} is not present in active_pages")
        page = pages[active_index[page_id]]
        if i == len(page_ids) - 1:
            page = page[:last_page_len]
        chunks.append(page)
    return torch.cat(chunks, dim=0)


def _apply_softcap(torch: Any, scores: Any, softcap: float) -> Any:
    if softcap <= 0.0:
        return scores
    return torch.tanh(scores / softcap) * softcap


def _causal_attention_reference(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    sm_scale: float,
    logits_soft_cap: float,
) -> Any:
    q_len, num_q_heads, head_dim = query.shape
    kv_len, num_kv_heads, value_dim = value.shape
    if key.shape != (kv_len, num_kv_heads, head_dim):
        raise ValueError(f"unexpected key shape {tuple(key.shape)} for query {tuple(query.shape)}")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    group = num_q_heads // num_kv_heads
    out = torch.empty((q_len, num_q_heads, value_dim), dtype=torch.float32)
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    for token_idx in range(q_len):
        stop = min(token_idx + 1, kv_len)
        for q_head in range(num_q_heads):
            kv_head = q_head // group
            scores = torch.matmul(
                key_f[:stop, kv_head], query_f[token_idx, q_head]
            ) * sm_scale
            scores = _apply_softcap(torch, scores, logits_soft_cap)
            probs = torch.softmax(scores, dim=0)
            out[token_idx, q_head] = torch.matmul(probs, value_f[:stop, kv_head])
    return out


def replay_dump(
    path: Path,
    sm_scale: float | None,
    logits_soft_cap: float,
    deswizzle_v_scale: bool,
    head: int,
) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if obj.get("schema") != "spark-active-page-prefill-dump/v1":
        raise ValueError(f"{path} has unexpected schema {obj.get('schema')!r}")
    query = obj["query"].float()
    k_data, v_data = obj["kv_data_pages"]
    k_scale, v_scale = obj["kv_scale_pages"]
    if deswizzle_v_scale:
        v_scale = _deswizzle_v_scale_nhd(torch, v_scale)

    scale = sm_scale if sm_scale is not None else query.shape[-1] ** -0.5
    key_pages = _dequant_nvfp4(torch, k_data, k_scale, obj.get("k_scale", 1.0))
    value_pages = _dequant_nvfp4(torch, v_data, v_scale, obj.get("v_scale", 1.0))
    key = _gather_pages(torch, key_pages, obj)
    value = _gather_pages(torch, value_pages, obj)
    reference = _causal_attention_reference(
        torch, query, key, value, scale, logits_soft_cap
    )
    out_after = obj["out_after"].float()
    out_before = obj["out_before"].float()
    active_v_bytes = _gather_pages(torch, v_data, obj).float()
    group = query.shape[1] // value.shape[1]
    active_v_repeated = active_v_bytes.repeat_interleave(group, dim=1)

    return {
        "file": str(path),
        "schema": "vllm-active-page-replay/v1",
        "layer_name": obj.get("layer_name"),
        "window_left": obj.get("window_left"),
        "num_prefill_tokens": obj.get("num_prefill_tokens"),
        "num_decode_tokens": obj.get("num_decode_tokens"),
        "active_pages": [int(x) for x in obj["active_pages"].tolist()],
        "paged_kv_indices": [int(x) for x in obj["paged_kv_indices"].tolist()],
        "last_page_len": [int(x) for x in obj["paged_kv_last_page_len"].tolist()],
        "sm_scale": float(scale),
        "logits_soft_cap": float(logits_soft_cap),
        "deswizzle_v_scale": bool(deswizzle_v_scale),
        "query": _tensor_stats(torch, query, head),
        "dequant_key": _tensor_stats(torch, key, head),
        "dequant_value": _tensor_stats(torch, value, head),
        "reference": _tensor_stats(torch, reference, head),
        "out_before": _tensor_stats(torch, out_before, head),
        "out_after": _tensor_stats(torch, out_after, head),
        "out_after_vs_reference": _metrics(torch, out_after, reference),
        "out_before_vs_reference": _metrics(torch, out_before, reference),
        "out_after_vs_active_v_bytes_repeated": _metrics(
            torch, out_after[..., : active_v_repeated.shape[-1]], active_v_repeated
        ),
        "active_v_bytes_repeated": _tensor_stats(torch, active_v_repeated, head),
    }


def build_report(
    paths: list[Path],
    sm_scale: float | None,
    logits_soft_cap: float,
    deswizzle_v_scale: bool,
    head: int,
) -> dict[str, Any]:
    return {
        "schema": "vllm-active-page-replay-report/v1",
        "sm_scale_input": sm_scale,
        "logits_soft_cap": logits_soft_cap,
        "deswizzle_v_scale": deswizzle_v_scale,
        "head": head,
        "dumps": [
            replay_dump(path, sm_scale, logits_soft_cap, deswizzle_v_scale, head)
            for path in paths
        ],
    }


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        text = str(path)
        if any(ch in text for ch in "*?["):
            parent = path.parent if str(path.parent) else Path(".")
            matches = sorted(parent.glob(path.name))
            if not matches:
                raise FileNotFoundError(f"glob matched no files: {path}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    return expanded


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# vLLM Active-Page Replay",
        "",
        f"- logits_soft_cap: `{report['logits_soft_cap']}`",
        f"- sm_scale input: `{report['sm_scale_input']}`",
        f"- deswizzle_v_scale: `{report['deswizzle_v_scale']}`",
        "",
        "## Dumps",
        "",
    ]
    for item in report["dumps"]:
        ref = item["reference"]
        out = item["out_after"]
        ref_metrics = item["out_after_vs_reference"]
        bytes_metrics = item["out_after_vs_active_v_bytes_repeated"]
        lines.extend(
            [
                f"### {Path(item['file']).name}",
                "",
                f"- prefill/decode tokens: `{item['num_prefill_tokens']}` / `{item['num_decode_tokens']}`",
                f"- active pages: `{item['active_pages']}`",
                f"- paged indices: `{item['paged_kv_indices']}`",
                f"- reference stats: min `{ref.get('min')}`, max `{ref.get('max')}`, mean `{ref.get('mean')}`, rms `{ref.get('rms')}`",
                f"- out_after stats: min `{out.get('min')}`, max `{out.get('max')}`, mean `{out.get('mean')}`, rms `{out.get('rms')}`, byte_like `{out.get('byte_like')}`",
                f"- out_after vs reference: cosine `{ref_metrics.get('cosine')}`, max_abs `{ref_metrics.get('max_abs')}`, mean_abs `{ref_metrics.get('mean_abs')}`",
                f"- out_after vs repeated active V bytes: cosine `{bytes_metrics.get('cosine')}`, max_abs `{bytes_metrics.get('max_abs')}`, mean_abs `{bytes_metrics.get('mean_abs')}`",
                f"- reference head: `{ref.get('head')}`",
                f"- out_after head: `{out.get('head')}`",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--sm-scale", type=float)
    parser.add_argument("--logits-soft-cap", type=float, default=0.0)
    parser.add_argument("--deswizzle-v-scale", action="store_true")
    parser.add_argument("--head", type=int, default=16)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    report = build_report(
        paths,
        args.sm_scale,
        args.logits_soft_cap,
        args.deswizzle_v_scale,
        args.head,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown_output:
        write_markdown(report, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
