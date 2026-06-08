#!/usr/bin/env python3
"""Shared hardware evidence helpers for DGX Spark / GB10 runs."""

from __future__ import annotations

from typing import Any

REFERENCE_GB10_SM_COUNT = 48


def collect_cuda_hardware() -> dict[str, Any]:
    """Collect CUDA device identity without making torch a hard dependency."""
    report: dict[str, Any] = {
        "schema": "spark-hardware/v1",
        "reference_sm_count": REFERENCE_GB10_SM_COUNT,
        "devices": [],
        "warnings": [],
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        report["torch_available"] = False
        report["error"] = repr(exc)
        report["warnings"].append("torch is unavailable; CUDA SM count is not recorded.")
        return report

    report["torch_available"] = True
    report["torch_version"] = getattr(torch, "__version__", "unknown")
    report["torch_cuda"] = getattr(torch.version, "cuda", None)
    report["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        report["warnings"].append("torch.cuda is unavailable; CUDA SM count is not recorded.")
        return report

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        capability = list(torch.cuda.get_device_capability(idx))
        device = {
            "index": idx,
            "name": props.name,
            "capability": capability,
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "comparison_key": comparison_key(
                name=props.name,
                capability=capability,
                multi_processor_count=props.multi_processor_count,
            ),
        }
        report["devices"].append(device)

    if report["devices"]:
        first = report["devices"][0]
        sm_count = first.get("multi_processor_count")
        if sm_count != REFERENCE_GB10_SM_COUNT:
            report["warnings"].append(
                "first CUDA device has "
                f"{sm_count} SMs, which differs from this campaign's "
                f"{REFERENCE_GB10_SM_COUNT}-SM GB10 reference count; "
                "do not compare performance rows across different SM counts."
            )
    return report


def comparison_key(
    *,
    name: str | None,
    capability: list[int] | tuple[int, int] | None,
    multi_processor_count: int | None,
) -> str:
    cap = "unknown"
    if capability and len(capability) >= 2:
        cap = f"sm_{capability[0]}{capability[1]}"
    safe_name = (name or "unknown").replace(" ", "_")
    return f"{safe_name}:{cap}:sms_{multi_processor_count}"
