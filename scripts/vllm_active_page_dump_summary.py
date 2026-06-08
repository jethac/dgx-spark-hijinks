#!/usr/bin/env python3
"""Summarize vLLM FlashInfer active-page prefill dumps.

The dump payloads are emitted by the Spark diagnostic hook in the vLLM fork around
``BatchPrefillWithPagedKVCacheWrapper.run``. They are intentionally small enough to
inspect on a CPU host with torch installed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def tensor_head(tensor: Any, limit: int) -> list[int | float]:
    cpu = tensor.detach().cpu().reshape(-1)
    values = cpu[: min(limit, int(cpu.numel()))].tolist()
    out: list[int | float] = []
    for value in values:
        if isinstance(value, bool):
            out.append(int(value))
        elif isinstance(value, int):
            out.append(value)
        else:
            out.append(float(value))
    return out


def tensor_stats(tensor: Any, head: int) -> dict[str, Any]:
    import torch

    cpu = tensor.detach().cpu()
    flat = cpu.reshape(-1)
    result: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "numel": int(flat.numel()),
        "head": tensor_head(cpu, head),
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
        }
    )
    if str(tensor.dtype) == "torch.uint8":
        result["nonzero"] = int(torch.count_nonzero(flat).item())
    return result


def summarize_pages(pages: Any, head: int) -> list[dict[str, Any]]:
    if not isinstance(pages, (tuple, list)):
        return []
    return [tensor_stats(page, head) for page in pages if hasattr(page, "detach")]


def byte_like(stats: dict[str, Any]) -> bool:
    return (
        stats.get("all_finite") is True
        and stats.get("min", -1.0) >= 0.0
        and stats.get("max", 256.0) <= 255.0
        and stats.get("mean", 0.0) > 32.0
    )


def summarize_dump(path: Path, head: int) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    out_before = tensor_stats(obj["out_before"], head)
    out_after = tensor_stats(obj["out_after"], head)
    kv_data_pages = summarize_pages(obj.get("kv_data_pages"), head)
    kv_scale_pages = summarize_pages(obj.get("kv_scale_pages"), head)

    return {
        "file": str(path),
        "schema": obj.get("schema"),
        "event": obj.get("event"),
        "layer_name": obj.get("layer_name"),
        "window_left": obj.get("window_left"),
        "num_prefill_tokens": obj.get("num_prefill_tokens"),
        "num_decode_tokens": obj.get("num_decode_tokens"),
        "k_scale": obj.get("k_scale"),
        "v_scale": obj.get("v_scale"),
        "active_pages": tensor_head(obj["active_pages"], head),
        "paged_kv_indptr": tensor_head(obj["paged_kv_indptr"], head),
        "paged_kv_indices_head": tensor_head(obj["paged_kv_indices"], head),
        "paged_kv_last_page_len": tensor_head(obj["paged_kv_last_page_len"], head),
        "query": tensor_stats(obj["query"], head),
        "out_before": out_before,
        "out_after": out_after,
        "out_after_byte_like": byte_like(out_after),
        "kv_data_pages": kv_data_pages,
        "kv_scale_pages": kv_scale_pages,
        "kv_data_views": obj.get("kv_data_views"),
        "kv_scale_views": obj.get("kv_scale_views"),
    }


def build_report(dump_dir: Path, head: int) -> dict[str, Any]:
    files = sorted(dump_dir.glob("*.pt"))
    dumps = [summarize_dump(path, head) for path in files]
    return {
        "schema": "vllm-active-page-dump-summary/v1",
        "dump_dir": str(dump_dir),
        "file_count": len(files),
        "byte_like_outputs": sum(1 for item in dumps if item["out_after_byte_like"]),
        "head": head,
        "dumps": dumps,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# vLLM Active-Page Dump Summary",
        "",
        f"- dump_dir: `{report['dump_dir']}`",
        f"- files: `{report['file_count']}`",
        f"- byte-like `out_after` tensors: `{report['byte_like_outputs']}`",
        "",
        "## Dumps",
        "",
    ]
    for item in report["dumps"]:
        out_after = item["out_after"]
        lines.extend(
            [
                f"### {Path(item['file']).name}",
                "",
                f"- layer: `{item['layer_name']}`",
                f"- window_left: `{item['window_left']}`",
                f"- prefill/decode tokens: `{item['num_prefill_tokens']}` / `{item['num_decode_tokens']}`",
                f"- active_pages: `{item['active_pages']}`",
                f"- paged_kv_indices_head: `{item['paged_kv_indices_head']}`",
                f"- last_page_len: `{item['paged_kv_last_page_len']}`",
                f"- out_after byte-like: `{item['out_after_byte_like']}`",
                f"- out_after stats: min `{out_after.get('min')}`, max `{out_after.get('max')}`, mean `{out_after.get('mean')}`, rms `{out_after.get('rms')}`",
                f"- out_after head: `{out_after.get('head')}`",
            ]
        )
        data_pages = item.get("kv_data_pages") or []
        scale_pages = item.get("kv_scale_pages") or []
        if len(data_pages) >= 2:
            lines.append(f"- active V data page head: `{data_pages[1].get('head')}`")
        if len(scale_pages) >= 2:
            lines.append(f"- active V scale page head: `{scale_pages[1].get('head')}`")
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--head", type=int, default=16)
    args = parser.parse_args()

    report = build_report(args.dump_dir, args.head)
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
