"""Patch installed FlashInfer JIT paths from ``SPARK_FLASHINFER_SOURCE_ROOT``.

Use this as ``sitecustomize.py`` on ``PYTHONPATH`` when a container's installed
FlashInfer Python package is ABI-compatible, but its packaged headers/JIT helper
files need to be replaced by a mounted source checkout for an experiment.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_local_attention_utils(source_root: Path):
    utils_path = source_root / "flashinfer" / "jit" / "attention" / "utils.py"
    if not utils_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "_spark_flashinfer_attention_jit_utils", utils_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_flashinfer(source_root: Path) -> None:
    from flashinfer.jit import env as jit_env  # type: ignore

    jit_env.FLASHINFER_DATA = source_root
    jit_env.FLASHINFER_CSRC_DIR = source_root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = source_root / "include"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cutlass" / "include",
        source_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cccl" / "cub",
        source_root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        source_root / "3rdparty" / "cccl" / "thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = source_root / "3rdparty" / "spdlog" / "include"

    local_utils = _load_local_attention_utils(source_root)
    if local_utils is None:
        return

    import flashinfer.jit.attention.modules as attention_modules  # type: ignore
    import flashinfer.jit.attention.utils as attention_utils  # type: ignore

    attention_utils.generate_additional_params = local_utils.generate_additional_params
    attention_modules.generate_additional_params = local_utils.generate_additional_params


source = os.environ.get("SPARK_FLASHINFER_SOURCE_ROOT")
if source:
    _patch_flashinfer(Path(source).resolve())
