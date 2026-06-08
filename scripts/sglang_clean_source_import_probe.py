#!/usr/bin/env python3
"""Check whether a container is importing the intended SGLang/FlashInfer sources."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import flashinfer  # type: ignore
import sglang  # type: ignore
from flashinfer.jit import env as jit_env  # type: ignore
from sglang.srt.layers.attention import flashinfer_backend  # type: ignore
from sglang.srt.mem_cache import memory_pool  # type: ignore


def main() -> int:
    report = {
        "flashinfer_file": flashinfer.__file__,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "flashinfer_include_dir": str(jit_env.FLASHINFER_INCLUDE_DIR),
        "flashinfer_csrc_dir": str(jit_env.FLASHINFER_CSRC_DIR),
        "sglang_file": sglang.__file__,
        "sglang_version": getattr(sglang, "__version__", None),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0)
        if torch.cuda.is_available()
        else None,
        "has_native_fp4_pool_accessors": hasattr(
            memory_pool.MHATokenToKVPoolFP4, "get_kv_scale_buffer"
        ),
        "has_native_fp4_backend_helper": hasattr(
            flashinfer_backend, "_nvfp4_inner_pool_and_layer_id"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    output = Path("/workspace/out/sglang_clean_source_import_probe.json")
    if output.parent.exists():
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if all(
        (
            report["cuda_available"],
            report["has_native_fp4_pool_accessors"],
            report["has_native_fp4_backend_helper"],
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
