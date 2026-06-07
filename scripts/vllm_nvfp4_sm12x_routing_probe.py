#!/usr/bin/env python3
"""Probe vLLM's NVFP4 KV backend routing on SM12x.

This is intentionally narrower than a serving benchmark. It loads a forked
vLLM FlashInfer backend source file while reusing the installed vLLM compiled
dependencies, then checks the backend names selected by the wrapper factories.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


def import_module_version(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }


def load_flashinfer_backend(source_file: Path) -> Any:
    module_name = "vllm_flashinfer_source_probe"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeWrapper:
    def __init__(self, *args: Any, backend: str | None = None, **kwargs: Any) -> None:
        self.backend = backend


def make_builder(module: Any, *, use_fa2_nvfp4_kv: bool, is_kvcache_nvfp4: bool) -> Any:
    builder = object.__new__(module.FlashInferMetadataBuilder)
    builder._prefill_wrapper = None
    builder._decode_wrapper = None
    builder._decode_wrappers_cudagraph = {}
    builder.use_dcp = False
    builder.use_fa2_nvfp4_kv = use_fa2_nvfp4_kv
    builder.is_kvcache_nvfp4 = is_kvcache_nvfp4
    builder._get_workspace_buffer = lambda: None
    return builder


def routing_cases(module: Any) -> list[dict[str, Any]]:
    module.BatchPrefillWithPagedKVCacheWrapper = FakeWrapper
    module.BatchDecodeWithPagedKVCacheWrapper = FakeWrapper
    module.get_kv_cache_layout = lambda: "NHD"

    cases = [
        {
            "name": "sm12x_nvfp4_kv",
            "use_fa2_nvfp4_kv": True,
            "is_kvcache_nvfp4": True,
            "expected": "fa2",
        },
        {
            "name": "sm100_style_nvfp4_kv",
            "use_fa2_nvfp4_kv": False,
            "is_kvcache_nvfp4": True,
            "expected": "trtllm-gen",
        },
        {
            "name": "non_nvfp4_kv",
            "use_fa2_nvfp4_kv": False,
            "is_kvcache_nvfp4": False,
            "expected": "auto",
        },
    ]

    results = []
    for case in cases:
        builder = make_builder(
            module,
            use_fa2_nvfp4_kv=case["use_fa2_nvfp4_kv"],
            is_kvcache_nvfp4=case["is_kvcache_nvfp4"],
        )
        prefill = builder._get_prefill_wrapper()
        decode = builder._get_decode_wrapper(batch_size=1, use_cudagraph=False)
        ok = prefill.backend == case["expected"] and decode.backend == case["expected"]
        results.append(
            {
                **case,
                "prefill_backend": prefill.backend,
                "decode_backend": decode.backend,
                "ok": ok,
            }
        )
    return results


def collect(args: argparse.Namespace) -> dict[str, Any]:
    source_file = args.source_file
    if args.repo:
        source_file = args.repo / "vllm/v1/attention/backends/flashinfer.py"
    source_file = source_file.resolve()
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    import torch
    import vllm
    from vllm.platforms import current_platform

    module = load_flashinfer_backend(source_file)
    results = routing_cases(module)
    deswizzle_env_before = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS")
    deswizzle_helper_available = hasattr(module, "_ensure_vllm_nvfp4_kv_deswizzle_flag")
    if deswizzle_helper_available:
        module._ensure_vllm_nvfp4_kv_deswizzle_flag()
    deswizzle_env_after = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS")

    capability = None
    gpu = None
    if torch.cuda.is_available():
        capability = list(torch.cuda.get_device_capability(0))
        gpu = torch.cuda.get_device_name(0)

    platform_capability = current_platform.get_device_capability()
    platform_capability_list = None
    if platform_capability is not None:
        platform_capability_list = [
            platform_capability.major,
            platform_capability.minor,
        ]

    source_text = source_file.read_text(encoding="utf-8")
    return {
        "schema": "vllm-nvfp4-sm12x-routing-probe/v1",
        "probe_type": "source-file routing probe using installed vLLM compiled dependencies",
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "source_file": str(source_file),
        "source_repo": args.source_repo,
        "source_branch": args.source_branch,
        "source_rev": args.source_rev,
        "installed_modules": {
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "capability": capability,
                "gpu": gpu,
            },
            "vllm": {
                "version": getattr(vllm, "__version__", "unknown"),
                "file": getattr(vllm, "__file__", None),
                "platform_capability": platform_capability_list,
                "is_capability_family_12": current_platform.is_device_capability_family(
                    12
                ),
                "is_capability_family_120": current_platform.is_device_capability_family(
                    120
                ),
            },
            "flashinfer": import_module_version("flashinfer"),
        },
        "source_checks": {
            "mentions_device_capability_family_12": "is_device_capability_family(12)"
            in source_text,
            "mentions_device_capability_family_120": "is_device_capability_family(\n                    120\n                )"
            in source_text
            or "is_device_capability_family(120)" in source_text,
            "mentions_v_scale_deswizzle_define": "FLASHINFER_PAGED_V_SF_DESWIZZLE"
            in source_text,
            "mentions_fa2_nvfp4_kv": "use_fa2_nvfp4_kv" in source_text,
            "mentions_trtllm_gen": "trtllm-gen" in source_text,
        },
        "deswizzle_flag_probe": {
            "helper_available": deswizzle_helper_available,
            "env_before": deswizzle_env_before,
            "env_after": deswizzle_env_after,
            "enabled": bool(
                deswizzle_env_after
                and "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1" in deswizzle_env_after
            ),
        },
        "routing_results": results,
        "all_ok": all(item["ok"] for item in results),
        "limitations": [
            "This does not install the full vLLM fork as an editable package.",
            "This does not build or run FlashInfer FA2 NVFP4 KV kernels.",
            "This does not start a vLLM server or prove serving correctness/capacity/performance.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo",
        type=Path,
        help="vLLM source checkout containing vllm/v1/attention/backends/flashinfer.py",
    )
    source.add_argument(
        "--source-file",
        type=Path,
        help="explicit path to a forked vLLM flashinfer.py source file",
    )
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    parser.add_argument("--source-repo", default="unknown")
    parser.add_argument("--source-branch", default="unknown")
    parser.add_argument("--source-rev", default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
