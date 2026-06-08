#!/usr/bin/env python3
"""Summarize SGLang FP4 first-token tensor dumps.

The dump files are emitted by ``sglang_fp4_first_token_dump_patch.yaml`` around
``ModelRunner.sample()``. This script is intended to run on a Linux host or in a
container with torch installed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_dump_name(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for part in path.stem.split("___"):
        if "=" in part:
            key, value = part.split("=", 1)
            meta[key] = value
    return meta


def tensor_summary(tensor: Any, top_k: int) -> dict[str, Any]:
    import torch

    cpu = tensor.detach().float().cpu()
    finite = torch.isfinite(cpu)
    summary: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(tensor.dtype),
        "finite": int(finite.sum().item()),
        "numel": int(cpu.numel()),
        "all_finite": bool(finite.all().item()),
    }
    if cpu.numel() == 0:
        return summary

    if cpu.ndim == 1:
        logits = cpu
    else:
        logits = cpu.reshape(-1, cpu.shape[-1])[0]

    limit = min(top_k, int(logits.numel()))
    if limit <= 0:
        return summary

    values, indices = torch.topk(logits, k=limit)
    summary["top"] = [
        {"token_id": int(index.item()), "value": float(value.item())}
        for value, index in zip(values, indices, strict=True)
    ]
    summary["argmax_token_id"] = summary["top"][0]["token_id"] if summary["top"] else None
    summary["argmax_value"] = summary["top"][0]["value"] if summary["top"] else None
    return summary


def small_tensor_values(tensor: Any, max_items: int) -> list[Any] | None:
    cpu = tensor.detach().cpu().reshape(-1)
    if cpu.numel() > max_items:
        return None
    return [int(x.item()) if hasattr(x, "item") else int(x) for x in cpu]


def compare_logits(before: Any, after: Any, top_k: int) -> dict[str, Any]:
    import torch

    b = before.detach().float().cpu()
    a = after.detach().float().cpu()
    result: dict[str, Any] = {
        "same_shape": list(b.shape) == list(a.shape),
        "before": tensor_summary(before, top_k),
        "after": tensor_summary(after, top_k),
    }
    if b.shape != a.shape:
        return result

    delta = a - b
    finite_delta = torch.isfinite(delta)
    b_logits = b.reshape(-1, b.shape[-1])[0] if b.ndim > 1 else b
    a_logits = a.reshape(-1, a.shape[-1])[0] if a.ndim > 1 else a
    before_top = {item["token_id"] for item in result["before"].get("top", [])}
    after_top = {item["token_id"] for item in result["after"].get("top", [])}
    union = before_top | after_top
    overlap = (len(before_top & after_top) / len(union)) if union else math.nan
    result.update(
        {
            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
            "mean_abs_delta": float(delta.abs().mean().item()) if delta.numel() else 0.0,
            "finite_delta": int(finite_delta.sum().item()),
            "top_k_jaccard": overlap,
            "same_argmax": bool(torch.argmax(b_logits).item() == torch.argmax(a_logits).item()),
        }
    )
    return result


def load_tensor(path: Path) -> Any:
    import torch

    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "detach"):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "tensor", "value"):
            value = obj.get(key)
            if hasattr(value, "detach"):
                return value
        tensor_values = [value for value in obj.values() if hasattr(value, "detach")]
        if len(tensor_values) == 1:
            return tensor_values[0]
        raise TypeError(
            f"{path} contains a dict but no unambiguous tensor field: {sorted(obj)}"
        )
    raise TypeError(f"{path} contains unsupported object type {type(obj).__name__}")


def build_report(dump_dir: Path, top_k: int, max_small_items: int) -> dict[str, Any]:
    files = sorted(dump_dir.glob("*.pt"))
    parsed = [{"path": path, "meta": parse_dump_name(path)} for path in files]
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"files": [], "tensors": defaultdict(dict), "meta": {}}
    )

    for item in parsed:
        meta = item["meta"]
        rid = meta.get("rid", "none")
        forward_pass_id = meta.get("forward_pass_id", "none")
        forward_mode = meta.get("forward_mode", "none")
        name = meta.get("name", "unknown")
        phase = meta.get("phase", "none")
        key = (rid, forward_pass_id, forward_mode)
        groups[key]["files"].append(str(item["path"]))
        groups[key]["meta"] = {
            "rid": rid,
            "forward_pass_id": forward_pass_id,
            "forward_mode": forward_mode,
            "is_health_check": rid.startswith("HEALTH_CHECK"),
        }
        groups[key]["tensors"][name][phase] = item["path"]

    report_groups: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        tensors = group["tensors"]
        entry: dict[str, Any] = {
            "meta": group["meta"],
            "file_count": len(group["files"]),
            "tensor_names": sorted(tensors),
        }
        logits = tensors.get("fp4_first_token__next_token_logits", {})
        if "before_preprocess" in logits and "after_preprocess" in logits:
            entry["logits_compare"] = compare_logits(
                load_tensor(logits["before_preprocess"]),
                load_tensor(logits["after_preprocess"]),
                top_k,
            )
        else:
            entry["logits_compare"] = {
                "missing": sorted(
                    phase
                    for phase in ("before_preprocess", "after_preprocess")
                    if phase not in logits
                )
            }

        for name in (
            "fp4_first_token__input_ids",
            "fp4_first_token__positions",
            "fp4_first_token__seq_lens",
        ):
            phase_files = tensors.get(name, {})
            before_path = phase_files.get("before_preprocess")
            if before_path:
                tensor = load_tensor(before_path)
                entry[name] = tensor_summary(tensor, top_k=0)
                values = small_tensor_values(tensor, max_small_items)
                if values is not None:
                    entry[name]["values"] = values

        report_groups.append(entry)

    real_groups = [g for g in report_groups if not g["meta"]["is_health_check"]]
    return {
        "dump_dir": str(dump_dir),
        "file_count": len(files),
        "group_count": len(report_groups),
        "real_request_group_count": len(real_groups),
        "health_check_group_count": len(report_groups) - len(real_groups),
        "top_k": top_k,
        "groups": report_groups,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# SGLang FP4 First-Token Dump Summary",
        "",
        f"- dump_dir: `{report['dump_dir']}`",
        f"- files: `{report['file_count']}`",
        f"- groups: `{report['group_count']}`",
        f"- real request groups: `{report['real_request_group_count']}`",
        f"- health-check groups: `{report['health_check_group_count']}`",
        "",
        "## Groups",
        "",
    ]
    for group in report["groups"]:
        meta = group["meta"]
        compare = group["logits_compare"]
        lines.append(
            "### rid={rid} forward_pass_id={forward_pass_id} mode={forward_mode}".format(
                **meta
            )
        )
        lines.append("")
        lines.append(f"- health_check: `{meta['is_health_check']}`")
        lines.append(f"- files: `{group['file_count']}`")
        if "missing" in compare:
            lines.append(f"- logits_compare: missing `{compare['missing']}`")
        else:
            before = compare["before"]
            after = compare["after"]
            lines.append(f"- same_shape: `{compare['same_shape']}`")
            lines.append(f"- same_argmax: `{compare['same_argmax']}`")
            lines.append(f"- max_abs_delta: `{compare['max_abs_delta']:.6g}`")
            lines.append(f"- mean_abs_delta: `{compare['mean_abs_delta']:.6g}`")
            lines.append(f"- top_k_jaccard: `{compare['top_k_jaccard']:.6g}`")
            lines.append(
                f"- before_argmax: `{before.get('argmax_token_id')}` "
                f"value `{before.get('argmax_value')}`"
            )
            lines.append(
                f"- after_argmax: `{after.get('argmax_token_id')}` "
                f"value `{after.get('argmax_value')}`"
            )
        seq_lens = group.get("fp4_first_token__seq_lens", {}).get("values")
        if seq_lens is not None:
            lines.append(f"- seq_lens: `{seq_lens}`")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-small-items", type=int, default=128)
    args = parser.parse_args()

    report = build_report(args.dump_dir, args.top_k, args.max_small_items)
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
