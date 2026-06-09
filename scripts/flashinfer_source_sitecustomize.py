"""Patch installed FlashInfer JIT paths from ``SPARK_FLASHINFER_SOURCE_ROOT``.

Use this as ``sitecustomize.py`` on ``PYTHONPATH`` when a container's installed
FlashInfer Python package is ABI-compatible, but its packaged headers/JIT helper
files need to be replaced by a mounted source checkout for an experiment.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any


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


def _patch_prefill_run_scale_args(prefill) -> bool:
    """Patch installed paged-prefill wrapper to pass NVFP4 scale tensors.

    The source checkout's newer ``prefill.py`` may not be import-compatible with
    the container's installed FlashInfer package, but the installed wrapper has
    the same bug: its JIT path prepares mask/alibi optional tensors and omits the
    FP4 scale tensors required by patched generated modules.
    """

    wrapper_cls = prefill.BatchPrefillWithPagedKVCacheWrapper
    run_src = textwrap.dedent(inspect.getsource(wrapper_cls.run))
    patched = False

    if (
        '"maybe_k_cache_sf": key_block_scales' not in run_src
        or '"maybe_prefix_len_ptr": self._prefix_len_ptr' not in run_src
    ):
        needle = (
            '                        "maybe_alibi_slopes": lambda: '
            "_get_cache_alibi_slopes_buf(\n"
            "                            q.shape[1], q.device\n"
            "                        ),"
        )
        replacement = (
            needle
            + '\n                        "maybe_prefix_len_ptr": self._prefix_len_ptr,'
            + '\n                        "maybe_token_pos_in_items_ptr": self._token_pos_in_items_ptr,'
            + '\n                        "maybe_max_item_len_ptr": self._max_item_len_ptr,'
            + '\n                        "maybe_k_cache_sf": key_block_scales,'
            + '\n                        "maybe_v_cache_sf": value_block_scales,'
        )
        if needle not in run_src:
            needle = (
                '                            "maybe_alibi_slopes": lambda: '
                "_get_cache_alibi_slopes_buf(\n"
                "                                q.shape[1], q.device\n"
                "                            ),"
            )
            replacement = (
                needle
                + '\n                            "maybe_prefix_len_ptr": self._prefix_len_ptr,'
                + '\n                            "maybe_token_pos_in_items_ptr": self._token_pos_in_items_ptr,'
                + '\n                            "maybe_max_item_len_ptr": self._max_item_len_ptr,'
                + '\n                            "maybe_k_cache_sf": key_block_scales,'
                + '\n                            "maybe_v_cache_sf": value_block_scales,'
            )
        if needle not in run_src:
            return False

        patched_src = run_src.replace(needle, replacement, 1)
        namespace: dict[str, Any] = {}
        exec(
            compile(patched_src, "flashinfer_prefill_run_scale_args_patch", "exec"),
            prefill.__dict__,
            namespace,
        )
        wrapper_cls.run = namespace["run"]
        patched = True

    run_func = wrapper_cls.run
    if not getattr(run_func, "_spark_prefill_scalar_arg_patch", False):

        def run_with_jit_scalars(self, q, paged_kv_cache, *args, **kwargs):
            additional_tensor_names = getattr(
                self, "_jit_additional_tensor_names", []
            )
            if (
                getattr(self, "_jit_module", None) is not None
                and "maybe_k_cache_sf" in additional_tensor_names
                and len(args) < 5
            ):
                logits_soft_cap = getattr(self, "_logits_soft_cap", None)
                sm_scale = getattr(self, "_sm_scale", None)
                rope_scale = getattr(self, "_rope_scale", None)
                rope_theta = getattr(self, "_rope_theta", None)
                if logits_soft_cap is None:
                    logits_soft_cap = 0.0
                if sm_scale is None:
                    sm_scale = 1.0 / math.sqrt(q.size(-1))
                q_scale = kwargs.get("q_scale")
                k_scale = kwargs.get("k_scale")
                if getattr(self, "_backend", None) != "cudnn":
                    if q_scale is not None:
                        sm_scale *= q_scale
                    if k_scale is not None:
                        sm_scale *= k_scale
                if rope_scale is None:
                    rope_scale = 1.0
                if rope_theta is None:
                    rope_theta = 1e4
                args = args + (
                    logits_soft_cap,
                    sm_scale,
                    1.0 / rope_scale,
                    1.0 / rope_theta,
                    getattr(self, "_token_pos_in_items_len", 0),
                )
            return run_func(self, q, paged_kv_cache, *args, **kwargs)

        wrapper_cls.run = run_with_jit_scalars
        setattr(wrapper_cls.run, "_spark_prefill_scalar_arg_patch", True)
        patched = True

    setattr(wrapper_cls.run, "_spark_prefill_scale_arg_patch", True)
    return patched or True


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
    force_prefill_module = os.environ.get("SPARK_FLASHINFER_FORCE_PREFILL_MODULE") == "1"
    patch_prefill_run_scale_args = (
        os.environ.get("SPARK_FLASHINFER_PATCH_PREFILL_RUN_SCALE_ARGS", "1") == "1"
    )
    prefill_run_scale_arg_patch = False
    if force_prefill_module:
        attention_modules.get_batch_prefill_uri = get_batch_prefill_uri
    if force_prefill_module and local_attention_modules is not None:
        jit.gen_batch_prefill_module = attention_modules.gen_batch_prefill_module
        jit.gen_customize_batch_prefill_module = (
            attention_modules.gen_customize_batch_prefill_module
        )
    if force_prefill_module:
        jit.get_batch_prefill_uri = get_batch_prefill_uri

    import flashinfer.prefill as prefill  # type: ignore

    if local_attention_modules is not None:
        # Keep explicit wrapper-level JIT modules source-compatible with the
        # mounted FlashInfer checkout. This is narrower than forcing the default
        # batch-prefill module path: vLLM still has to request a custom module
        # through jit_args, but the generated params/template must come from the
        # same source tree as prefill.cuh.
        prefill.gen_customize_batch_prefill_module = (
            attention_modules.gen_customize_batch_prefill_module
        )

    if force_prefill_module and local_attention_modules is not None:
        prefill.gen_batch_prefill_module = attention_modules.gen_batch_prefill_module
    if force_prefill_module:
        prefill.get_batch_prefill_uri = get_batch_prefill_uri
        if hasattr(prefill.get_batch_prefill_module, "cache_clear"):
            prefill.get_batch_prefill_module.cache_clear()
    if patch_prefill_run_scale_args:
        prefill_run_scale_arg_patch = _patch_prefill_run_scale_args(prefill)

    if os.environ.get("SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG") == "1":
        try:
            run_src = inspect.getsource(
                prefill.BatchPrefillWithPagedKVCacheWrapper.run
            )
        except OSError:
            run_src = ""
        debug: dict[str, Any] = {
            "pid": os.getpid(),
            "source_root": str(source_root),
            "jit_utils_file": getattr(jit_utils, "__file__", None),
            "attention_modules_file": getattr(attention_modules, "__file__", None),
            "prefill_file": getattr(prefill, "__file__", None),
            "force_prefill_module": force_prefill_module,
            "patch_prefill_run_scale_args": patch_prefill_run_scale_args,
            "dtype_map_kv_uint8": jit_utils.dtype_map_kv.get(torch.uint8),
            "attention_dtype_map_kv_uint8": attention_modules.dtype_map_kv.get(
                torch.uint8
            ),
            "prefill_gen_batch_prefill_module": getattr(
                prefill.gen_batch_prefill_module, "__module__", None
            ),
            "jit_gen_batch_prefill_module": getattr(
                jit.gen_batch_prefill_module, "__module__", None
            ),
            "prefill_run_scale_arg_patch": prefill_run_scale_arg_patch,
            "prefill_run_marker": bool(
                getattr(
                    prefill.BatchPrefillWithPagedKVCacheWrapper.run,
                    "_spark_prefill_scale_arg_patch",
                    False,
                )
            ),
            "prefill_run_has_k_sf_source": "maybe_k_cache_sf" in run_src
            and "key_block_scales" in run_src,
            "prefill_run_has_v_sf_source": "maybe_v_cache_sf" in run_src
            and "value_block_scales" in run_src,
        }
        print(f"[spark-sitecustomize] {debug}", file=sys.stderr, flush=True)


source = os.environ.get("SPARK_FLASHINFER_SOURCE_ROOT")
if source:
    _patch_flashinfer(Path(source).resolve())
