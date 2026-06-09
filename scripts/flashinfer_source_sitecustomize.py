"""Patch installed FlashInfer JIT paths from ``SPARK_FLASHINFER_SOURCE_ROOT``.

Use this as ``sitecustomize.py`` on ``PYTHONPATH`` when a container's installed
FlashInfer Python package is ABI-compatible, but its packaged headers/JIT helper
files need to be replaced by a mounted source checkout for an experiment.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_local_module(module_name: str, module_path: Path):
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    parent_name, _, attr = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr, module)
    return module


def _patch_flashinfer(source_root: Path) -> None:
    import torch  # type: ignore

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

    import flashinfer.jit as jit  # type: ignore
    import flashinfer.prefill as prefill  # type: ignore

    _load_local_module(
        "flashinfer.jit.utils",
        source_root / "flashinfer" / "jit" / "utils.py",
    )
    local_attention_utils = _load_local_module(
        "flashinfer.jit.attention.utils",
        source_root / "flashinfer" / "jit" / "attention" / "utils.py",
    )
    local_attention_modules = _load_local_module(
        "flashinfer.jit.attention.modules",
        source_root / "flashinfer" / "jit" / "attention" / "modules.py",
    )
    if (
        local_attention_utils is None or local_attention_modules is None
    ) and os.environ.get("SPARK_FLASHINFER_ALLOW_INSTALLED_JIT") != "1":
        raise RuntimeError(
            "SPARK_FLASHINFER_SOURCE_ROOT does not contain the expected "
            "FlashInfer JIT attention modules; refusing to use the installed "
            "generator because it can silently build uint8_t KV kernels for "
            "NVFP4-KV experiments."
        )

    import flashinfer.jit.attention.modules as attention_modules  # type: ignore
    import flashinfer.jit.attention.utils as attention_utils  # type: ignore
    import flashinfer.jit.utils as jit_utils  # type: ignore

    # The installed Python package may predate the mounted source checkout. Keep
    # its JIT dtype tables aligned so packed uint8 KV generates FP4 kernels.
    if not hasattr(jit_utils, "dtype_map_kv"):
        jit_utils.dtype_map_kv = dict(jit_utils.dtype_map)
    jit_utils.dtype_map_kv[torch.uint8] = "__nv_fp4x2_e2m1"
    if hasattr(torch, "float4_e2m1fn_x2"):
        jit_utils.dtype_map_kv[torch.float4_e2m1fn_x2] = "__nv_fp4x2_e2m1"
    attention_modules.dtype_map_kv = jit_utils.dtype_map_kv

    if local_attention_utils is not None:
        attention_utils.generate_additional_params = (
            local_attention_utils.generate_additional_params
        )
        attention_modules.generate_additional_params = (
            local_attention_utils.generate_additional_params
        )

    def filename_safe_dtype_map_kv(dtype):
        if dtype is torch.uint8 or dtype == torch.uint8:
            return "fp4x2_e2m1"
        if hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2:
            return "fp4x2_e2m1"
        return jit_utils.filename_safe_dtype_map[dtype]

    def get_batch_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    ):
        suffix = "_sm90" if backend == "fa3" else ""
        return (
            "batch_prefill_with_kv_cache_"
            f"dtype_q_{jit_utils.filename_safe_dtype_map[dtype_q]}_"
            f"dtype_kv_{filename_safe_dtype_map_kv(dtype_kv)}_"
            f"dtype_o_{jit_utils.filename_safe_dtype_map[dtype_o]}_"
            f"dtype_idx_{jit_utils.filename_safe_dtype_map[dtype_idx]}_"
            f"head_dim_qk_{head_dim_qk}_"
            f"head_dim_vo_{head_dim_vo}_"
            f"posenc_{pos_encoding_mode}_"
            f"use_swa_{use_sliding_window}_"
            f"use_logits_cap_{use_logits_soft_cap}_"
            f"f16qk_{use_fp16_qk_reduction}"
            f"{suffix}"
        )

    jit_utils.filename_safe_dtype_map_kv = filename_safe_dtype_map_kv
    attention_modules.filename_safe_dtype_map_kv = filename_safe_dtype_map_kv
    attention_modules.get_batch_prefill_uri = get_batch_prefill_uri
    if local_attention_modules is not None:
        jit.gen_customize_batch_prefill_module = (
            attention_modules.gen_customize_batch_prefill_module
        )
        prefill.gen_customize_batch_prefill_module = (
            attention_modules.gen_customize_batch_prefill_module
        )
    jit.get_batch_prefill_uri = get_batch_prefill_uri
    prefill.get_batch_prefill_uri = get_batch_prefill_uri


source = os.environ.get("SPARK_FLASHINFER_SOURCE_ROOT")
if source:
    _patch_flashinfer(Path(source).resolve())
