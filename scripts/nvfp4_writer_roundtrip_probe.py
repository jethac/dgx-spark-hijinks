#!/usr/bin/env python3
"""NVFP4 KV-cache WRITER-ROUNDTRIP harness: real C++ writer -> real FA2 reader.

Every prior probe (flashinfer_nvfp4_kv_probe.py and friends) FABRICATES the
paged NVFP4 cache on the Python side and only exercises the FlashInfer FA2
READER. The real vLLM cache writer (reshape_and_cache_nvfp4_dispatch in
csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu, reached through
torch.ops._C_cache_ops.reshape_and_cache_flash with kv_cache_dtype="nvfp4")
has never been tested against that reader. Gemma 4 31B serves full-NVFP4
without crashing but emits deterministic gibberish, and the bisect points at
the sliding-layer FP4 path (head_dim=256, SWA window ~1024, linear V-SF).
This harness closes the gap: WRITE bf16 K/V through the real C++ op into a
serving-layout paged cache, then READ it back three ways and compare.

Stages (one JSON record per (kv-layout, stage)):
  stage-a "writer_dequant": Python-side dequantization of the WRITTEN cache
      (e2m1 nibbles * fp8 block scale per 16 elements * global scale)
      compared against the original bf16 K/V streams. Isolates WRITER
      correctness. Gate: cosine >= 0.99 (quantization error is included).
  stage-b "fa2_prefill": FlashInfer FA2 paged prefill over the written cache
      using kv_cache_sf views from vllm.utils.torch_utils
      .nvfp4_kv_cache_split_views (READER path). Reference: fp32 attention
      computed from the stage-a dequantized-written cache, so quantization
      error cancels. Gate: cosine >= 0.9999 (kernel must match what was
      actually written).
  stage-c "fa2_prefill_window": same as stage-b with --window-left N passed
      to wrapper.plan (SWA read path). The fp32 reference applies the same
      end-aligned causal AND sliding-window mask FlashInfer uses. Only runs
      when --window-left >= 0.

CALIBRATION REQUIREMENT
  The FIRST run must be head-dim 128 with no window (pass --calibrate, which
  forces --head-dim 128 and --window-left -1). Block C proved the serving
  roundtrip is healthy at head_dim=128, so ALL stages must pass there or the
  harness itself is buggy. Only after a green calibration run is
  head-dim-256 / windowed data meaningful (e.g. --head-dim 256
  --window-left 1023, matching Gemma 4 SWA 1024: vLLM passes
  window_left = sliding_window - 1).

CONTAINER INVOCATION (the writer is C++; this MUST run inside the
rebuilt-extension image -- the compiled torch.ops._C_cache_ops with the
NVFP4 dispatch is baked into the image, NOT importable from a source tree):

  docker run --rm --gpus all --ipc=host \\
    -w /work \\
    -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \\
    -e PYTHONPATH=/opt/vllm_overlay \\
    -v <vllm-clone (branch spark/hijinks-022-gemma4-mixed-kv)>:/opt/vllm_overlay:ro \\
    -v <dgx-spark-hijinks>/scripts:/work/scripts:ro \\
    -v <results-dir>:/work/results \\
    jethac-vllm-aeon-gemma4:ad2337814-rebuiltc-fb7d62ea-sm121a \\
    python3 /work/scripts/nvfp4_writer_roundtrip_probe.py --calibrate \\
      --output /work/results/writer_roundtrip_calibration.json

  PYTHONPATH overlay caveat: PYTHONPATH=/opt/vllm_overlay overlays PYTHON
  sources only. csrc/ changes in the overlay have NO effect at runtime; the
  C++ writer comes from the extension compiled into the rebuiltc image
  (hence the ad2337814-rebuiltc tag). Keep the overlay at the same commit
  family as the image's extension.

  Env caveat: reshape_and_cache_nvfp4_dispatch latches
  VLLM_NVFP4_KV_LINEAR_V_SF via a `static const bool` lambda at the FIRST
  dispatch in the process (nvfp4_kv_cache_kernels.cu:216-219). Set it with
  `docker run -e` BEFORE launch; this script also sets/clears os.environ
  before the first op call as a belt-and-braces measure, but one process can
  only ever test ONE --v-scale-layout. The default (linear) sets
  VLLM_NVFP4_KV_LINEAR_V_SF=1; --v-scale-layout swizzled leaves it unset
  (the control: writer swizzles V scale factors, reader JIT gets the
  -DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 flag, mirroring
  _ensure_vllm_nvfp4_kv_deswizzle_flag in vllm flashinfer backend).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DESWIZZLE_FLAG = "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"
LINEAR_V_SF_ENV = "VLLM_NVFP4_KV_LINEAR_V_SF"
SF_BLOCK = 16  # FP4 block-scale granularity (elements per fp8 scale)
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_local_imports() -> None:
    scripts_dir = _repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _configure_writer_v_sf_env(v_scale_layout: str) -> dict[str, Any]:
    """Set/clear VLLM_NVFP4_KV_LINEAR_V_SF before the FIRST writer dispatch.

    The C++ dispatch reads the env once via a static lambda
    (nvfp4_kv_cache_kernels.cu:216-219), so this must happen before any
    reshape_and_cache_flash("nvfp4") call in this process. CPython's
    os.environ.__setitem__/__delitem__ call putenv/unsetenv, so the change
    is visible to std::getenv.
    """
    if v_scale_layout == "linear":
        os.environ[LINEAR_V_SF_ENV] = "1"
    else:
        # Control: writer applies the TRT-LLM 4-token V-SF swizzle; the FA2
        # reader must de-swizzle in-kernel (JIT flag below).
        os.environ.pop(LINEAR_V_SF_ENV, None)
    return {
        LINEAR_V_SF_ENV: os.environ.get(LINEAR_V_SF_ENV),
    }


def _ensure_reader_deswizzle_flag(enabled: bool) -> None:
    """Mirror _ensure_vllm_nvfp4_kv_deswizzle_flag (vllm flashinfer backend):
    with swizzled V-SF the FlashInfer FA2 JIT must de-swizzle on read."""
    if not enabled:
        return
    existing = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if "FLASHINFER_PAGED_V_SF_DESWIZZLE" in existing:
        return
    os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = f"{existing} {DESWIZZLE_FLAG}".strip()


def _configure_flashinfer_source_tree(flashinfer: Any, source_root: Path | None) -> dict[str, str]:
    """Point FlashInfer JIT at a source checkout instead of packaged data files."""
    if source_root is None:
        return {}
    source_root = source_root.resolve()
    from flashinfer.jit import env as jit_env  # type: ignore

    data_paths = {
        "source_root": str(source_root),
        "csrc": str(source_root / "csrc"),
        "include": str(source_root / "include"),
        "cutlass": str(source_root / "3rdparty" / "cutlass"),
        "cccl": str(source_root / "3rdparty" / "cccl"),
        "spdlog": str(source_root / "3rdparty" / "spdlog"),
    }
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
    return data_paths


def _patch_flashinfer_attention_jit_generators(source_root: Path | None) -> dict[str, str]:
    """Use the source checkout's attention JIT helpers with installed FlashInfer."""
    if source_root is None:
        return {}
    utils_path = source_root.resolve() / "flashinfer" / "jit" / "attention" / "utils.py"
    if not utils_path.exists():
        return {"attention_jit_utils": "missing"}

    spec = importlib.util.spec_from_file_location(
        "_spark_flashinfer_attention_jit_utils", utils_path
    )
    if spec is None or spec.loader is None:
        return {"attention_jit_utils": "unloadable"}

    local_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_utils)

    import flashinfer.jit.attention.modules as attention_modules  # type: ignore
    import flashinfer.jit.attention.utils as attention_utils  # type: ignore

    attention_utils.generate_additional_params = local_utils.generate_additional_params
    attention_modules.generate_additional_params = local_utils.generate_additional_params
    return {"attention_jit_utils": str(utils_path)}


def _pack_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "kv_len": args.kv_len,
        "qo_len": args.qo_len,
        "page_size": args.page_size,
        "num_kv_heads": args.num_kv_heads,
        "num_qo_heads": args.num_qo_heads,
        "head_dim": args.head_dim,
        "window_left": args.window_left,
        "dtype": args.dtype,
        "kv_layouts": args.kv_layout,
        "v_scale_layout": args.v_scale_layout,
        "write_chunk_tokens": args.write_chunk_tokens,
        "calibrate": args.calibrate,
        "backend": "fa2",
        "writer_op": "torch.ops._C_cache_ops.reshape_and_cache_flash(kv_cache_dtype='nvfp4')",
        "flashinfer_source_root": str(args.flashinfer_source_root)
        if args.flashinfer_source_root
        else None,
        "k_global_scale": args.k_global_scale,
        "v_global_scale": args.v_global_scale,
        "writer_cosine_threshold": args.writer_cosine_threshold,
        "writer_max_abs_threshold": args.writer_max_abs_threshold,
        "reader_cosine_threshold": args.reader_cosine_threshold,
        "reader_max_abs_threshold": args.reader_max_abs_threshold,
    }


def _metrics(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = torch.max(torch.abs(a - b)).item()
    mean_abs = torch.mean(torch.abs(a - b)).item()
    return {"cosine": cosine, "max_abs": max_abs, "mean_abs": mean_abs}


def _tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    stats: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": int(finite.sum().item()),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0 or not bool(finite.any().item()):
        return stats
    finite_values = values[finite]
    stats.update(
        {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "rms": float(torch.sqrt(torch.mean(finite_values * finite_values)).item()),
            "max_abs": float(finite_values.abs().max().item()),
        }
    )
    return stats


def _dtype(torch: Any, name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _alloc_serving_kv_cache(torch: Any, args: argparse.Namespace, layout: str, num_blocks: int):
    """Allocate a paged NVFP4 cache exactly like serving does.

    Same shape math as vllm.utils.torch_utils.create_kv_caches_with_random_flash
    ('nvfp4' branch): logical shape (num_blocks, 2, block_size, num_heads,
    full_dim) with full_dim = head_dim//2 + head_dim//16 (per-page byte layout
    [K_data | K_scale | V_data | V_scale] per KV side); the physical NHD/HND
    order lives in the strides (FlashInferBackend.get_kv_cache_stride_order).
    Zero-filled (instead of randint) so unwritten bytes are detectable.
    """
    full_dim = args.head_dim // 2 + args.head_dim // SF_BLOCK  # nvfp4_kv_cache_full_dim
    logical_shape = (num_blocks, 2, args.page_size, args.num_kv_heads, full_dim)
    stride_order = (0, 1, 2, 3, 4) if layout == "NHD" else (0, 1, 3, 2, 4)
    phys_shape = tuple(logical_shape[i] for i in stride_order)
    inv = [stride_order.index(i) for i in range(len(stride_order))]
    kv_cache = torch.zeros(phys_shape, dtype=torch.uint8, device="cuda:0").permute(*inv)
    return kv_cache, stride_order, full_dim


def _build_paging(torch: Any, args: argparse.Namespace):
    """Slot mapping + page table the way serving fills pages.

    Each sequence gets a contiguous run of pages; token t of sequence i lands
    in slot = (i * pages_per_seq + t // page_size) * page_size + t % page_size,
    i.e. pages fill front-to-back, last page partially (last_page_len)."""
    device = "cuda:0"
    pages_per_seq = math.ceil(args.kv_len / args.page_size)
    total_pages = pages_per_seq * args.batch_size
    token_in_seq = torch.arange(args.kv_len, device=device, dtype=torch.int64)
    slots = []
    for seq in range(args.batch_size):
        page_base = seq * pages_per_seq
        slots.append(
            (page_base + token_in_seq // args.page_size) * args.page_size
            + token_in_seq % args.page_size
        )
    slot_mapping = torch.cat(slots, dim=0)  # [batch_size * kv_len], int64
    kv_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32) * pages_per_seq
    )
    kv_indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full(
        (args.batch_size,),
        (args.kv_len - 1) % args.page_size + 1,
        device=device,
        dtype=torch.int32,
    )
    return {
        "pages_per_seq": pages_per_seq,
        "total_pages": total_pages,
        "slot_mapping": slot_mapping,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
    }


def _write_through_real_op(
    torch: Any,
    args: argparse.Namespace,
    kv_cache: Any,
    key: Any,
    value: Any,
    slot_mapping: Any,
    k_scale_t: Any,
    v_scale_t: Any,
) -> dict[str, Any]:
    """Invoke the real C++ writer per token-batch.

    Mirrors FlashInferImpl.do_kv_cache_update (vllm flashinfer.py:3308):
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, kv_cache[:, 0], kv_cache[:, 1], slot_mapping,
            "nvfp4", layer._k_scale, layer._v_scale)
    key/value: [num_tokens, num_kv_heads, head_dim] model-dtype;
    kv_cache[:, 0/1]: 4D logical [num_blocks, block_size, num_heads,
    full_dim] uint8 views (NHD/HND encoded in strides; the dispatch detects
    is_hnd = stride(2) > stride(1)); k_scale/v_scale: float32 [1] tensors
    read via const_data_ptr<float>()."""
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    num_tokens = key.shape[0]
    chunk = args.write_chunk_tokens if args.write_chunk_tokens > 0 else num_tokens
    calls = 0
    for start in range(0, num_tokens, chunk):
        stop = min(start + chunk, num_tokens)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key[start:stop],
            value[start:stop],
            k_cache,
            v_cache,
            slot_mapping[start:stop],
            "nvfp4",
            k_scale_t,
            v_scale_t,
        )
        calls += 1
    torch.cuda.synchronize()
    return {"writer_calls": calls, "chunk_tokens": chunk, "num_tokens": num_tokens}


def _deswizzle_v_sf(torch: Any, v_sf: Any, layout: str):
    """Invert the writer's TRT-LLM 4-token V-SF swizzle (linear <- swizzled).

    Forward swizzle (nvfp4_kv_cache_kernels.cu swizzle_scale_offset, with
    T=page_size tokens, S=scale_dim=head_dim/16):
        swizzled_t = (t / 4) * 4 + s / (S / 4)
        swizzled_s = (s % (S / 4)) * 4 + t % 4
    This is the exact inverse of _swizzle_v_sf in flashinfer_nvfp4_kv_probe.py.
    """
    if layout == "NHD":
        pages, page_size, heads, scale_dim = v_sf.shape
        if page_size % 4 or scale_dim % 4:
            raise ValueError("NHD V-scale deswizzle requires page_size and scale_dim divisible by 4")
        return (
            v_sf.reshape(pages, page_size // 4, 4, heads, scale_dim // 4, 4)
            .permute(0, 1, 5, 3, 2, 4)
            .reshape(pages, page_size, heads, scale_dim)
            .contiguous()
        )
    if layout == "HND":
        pages, heads, page_size, scale_dim = v_sf.shape
        if page_size % 4 or scale_dim % 4:
            raise ValueError("HND V-scale deswizzle requires page_size and scale_dim divisible by 4")
        return (
            v_sf.reshape(pages, heads, page_size // 4, 4, scale_dim // 4, 4)
            .permute(0, 1, 2, 5, 3, 4)
            .reshape(pages, heads, page_size, scale_dim)
            .contiguous()
        )
    raise ValueError(f"unsupported KV layout: {layout}")


def _dequant_written_pages(torch: Any, data: Any, sf: Any, global_scale: float):
    """Decode the WRITTEN cache pages on the Python side.

    data: uint8 [pages, X, Y, head_dim//2] view; element order per byte is
    [low nibble, high nibble] (pack_fp4 in nvfp4_utils.cuh packs even
    elements low). sf: float8_e4m3fn [pages, X, Y, head_dim//16] view,
    already de-swizzled if needed. Dequant convention from
    cvt_warp_fp16_to_fp4 (nvfp4_utils.cuh:221-280): the kernel stores
    nibble ~= x * SFScaleVal / fp8(SF) with SFScaleVal = 1/k_scale, so
    x ~= e2m1(nibble) * fp8(SF) * k_scale (k_scale == global_scale here).
    """
    lo = data & 0xF
    hi = (data >> 4) & 0xF
    indices = torch.stack((lo, hi), dim=-1).reshape(*data.shape[:-1], data.shape[-1] * 2)
    lut = torch.tensor(E2M1_TO_FLOAT32, device=data.device, dtype=torch.float32)
    values = lut[indices.to(torch.long)]
    sf_expanded = sf.to(torch.float32).repeat_interleave(SF_BLOCK, dim=-1)
    return values * sf_expanded * float(global_scale)


def _gather_sequence(torch: Any, pages: Any, layout: str, start: int, stop: int, last_len: int):
    """Concatenate one sequence's pages back into [seq_len, heads, dim]."""
    full = pages[start : stop - 1]
    last = pages[stop - 1]
    if layout == "NHD":
        last = last[:last_len]
        return torch.cat([full.reshape(-1, pages.shape[-2], pages.shape[-1]), last], dim=0)
    last = last[:, :last_len].permute(1, 0, 2)
    full = full.permute(0, 2, 1, 3)
    return torch.cat([full.reshape(-1, pages.shape[1], pages.shape[-1]), last], dim=0)


def _torch_prefill_reference(
    torch: Any, q: Any, k_seq: Any, v_seq: Any, window_left: int
) -> Any:
    """fp32 paged-prefill reference with FlashInfer's masks.

    End-aligned causal: q row qo_idx sits at absolute kv position
    q_abs = kv_len - qo_len + qo_idx; visible iff kv_idx <= q_abs.
    Sliding window (FlashInfer convention, cited from
    flashinfer include/flashinfer/attention/variants.cuh:89:
        mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx)
    i.e. kv_idx >= q_abs - window_left): the window covers window_left
    tokens strictly to the left PLUS the diagonal token itself
    (window_left + 1 visible positions). window_left < 0 disables it
    (variants.cuh:64 substitutes kv_len). vLLM passes
    window_left = sliding_window - 1 (interleaved-SWA Gemma: 1024 -> 1023).
    """
    qo_len, num_qo_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k_seq.shape
    group = num_qo_heads // num_kv_heads
    kf = k_seq.float().repeat_interleave(group, dim=1)
    vf = v_seq.float().repeat_interleave(group, dim=1)
    qf = q.float()
    scores = torch.einsum("qhd,khd->hqk", qf, kf) / (q.shape[-1] ** 0.5)
    qpos = torch.arange(qo_len, device=q.device)[:, None]
    kpos = torch.arange(kv_len, device=q.device)[None, :]
    q_abs = kv_len - qo_len + qpos
    mask = kpos <= q_abs  # end-aligned causal
    if window_left >= 0:
        mask &= kpos + window_left >= q_abs  # position >= kv_pos - window
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vf)


def _split_written_views(torch: Any, tu: Any, kv_cache: Any, stride_order: tuple):
    """Permute to the physical order and split into data/scale views.

    Mirrors the reader path in FlashInferImpl.forward (flashinfer.py:2578-2603):
        kv_cache_permute = kv_cache.permute(*stride_order)
        nvfp4_kv_data, nvfp4_kv_block_scales = nvfp4_kv_cache_split_views(...)
    Returns ((k_data, v_data), (k_sf, v_sf)) in the layout's dim order."""
    kv_cache_permute = kv_cache.permute(*stride_order)
    data_views, scale_views = tu.nvfp4_kv_cache_split_views(kv_cache_permute)
    return data_views, scale_views


def _stage_a_writer_dequant(
    torch: Any,
    args: argparse.Namespace,
    layout: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Dequantize the WRITTEN cache and compare against the source bf16 K/V."""
    (k_data, v_data), (k_sf, v_sf) = case["views"]
    v_sf_linear = (
        _deswizzle_v_sf(torch, v_sf, layout)
        if args.v_scale_layout == "swizzled"
        else v_sf
    )
    k_dq_pages = _dequant_written_pages(torch, k_data, k_sf, args.k_global_scale)
    v_dq_pages = _dequant_written_pages(torch, v_data, v_sf_linear, args.v_global_scale)
    case["k_dq_pages"] = k_dq_pages
    case["v_dq_pages"] = v_dq_pages

    indptr = case["paging"]["kv_indptr"].cpu()
    last_lens = case["paging"]["last_page_len"].cpu()
    k_parts, v_parts = [], []
    for i in range(args.batch_size):
        start, stop = int(indptr[i]), int(indptr[i + 1])
        last_len = int(last_lens[i])
        k_parts.append(_gather_sequence(torch, k_dq_pages, layout, start, stop, last_len))
        v_parts.append(_gather_sequence(torch, v_dq_pages, layout, start, stop, last_len))
    k_dq_stream = torch.cat(k_parts, dim=0)
    v_dq_stream = torch.cat(v_parts, dim=0)

    k_metrics = _metrics(torch, k_dq_stream, case["key"])
    v_metrics = _metrics(torch, v_dq_stream, case["value"])
    cosine = min(k_metrics["cosine"], v_metrics["cosine"])
    max_abs = max(k_metrics["max_abs"], v_metrics["max_abs"])

    # The allocation reserves one spare (never-mapped) page at the end; any
    # nonzero byte there means the writer strayed out of its slots.
    spare_page = case["kv_cache"][case["paging"]["total_pages"] :]
    spare_nonzero = int((spare_page != 0).sum().item())

    return {
        "cosine": cosine,
        "max_abs": max_abs,
        "k": k_metrics,
        "v": v_metrics,
        "spare_page_nonzero_bytes": spare_nonzero,
        "writer_call_info": case["write_info"],
        "shapes": {
            "key": list(case["key"].shape),
            "k_data_view": list(k_data.shape),
            "k_sf_view": list(k_sf.shape),
            "v_data_view": list(v_data.shape),
            "v_sf_view": list(v_sf.shape),
            "k_dq_stream": list(k_dq_stream.shape),
        },
        "k_dq_stats": _tensor_stats(torch, k_dq_stream),
        "v_dq_stats": _tensor_stats(torch, v_dq_stream),
        "passed": bool(
            cosine >= args.writer_cosine_threshold
            and max_abs <= args.writer_max_abs_threshold
            and spare_nonzero == 0
        ),
    }


def _stage_bc_fa2_prefill(
    torch: Any,
    flashinfer: Any,
    args: argparse.Namespace,
    layout: str,
    case: dict[str, Any],
    window_left: int,
) -> dict[str, Any]:
    """FA2 paged prefill over the WRITTEN cache vs fp32 reference from the
    dequantized-written pages (stage-b: window_left=-1, stage-c: >=0)."""
    dtype = case["dtype"]
    paging = case["paging"]
    q = torch.randn(
        args.batch_size * args.qo_len,
        args.num_qo_heads,
        args.head_dim,
        device="cuda:0",
        dtype=dtype,
    )
    qo_indptr = (
        torch.arange(args.batch_size + 1, device="cuda:0", dtype=torch.int32) * args.qo_len
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    workspace.fill_(0x7F)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        paging["kv_indptr"],
        paging["kv_indices"],
        paging["last_page_len"],
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        causal=True,
        window_left=window_left,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=dtype,
    )
    (k_data, v_data), (k_sf, v_sf) = case["views"]
    # Serving passes the split views as tuples (flashinfer.py:2710-2714,
    # 2798-2805): kv_cache_permute = nvfp4_kv_data, kv_cache_sf = block scales.
    actual = wrapper.run(
        q,
        (k_data, v_data),
        k_scale=args.k_global_scale,
        v_scale=args.v_global_scale,
        kv_cache_sf=(k_sf, v_sf),
    )

    refs = []
    indptr = paging["kv_indptr"].cpu()
    last_lens = paging["last_page_len"].cpu()
    qo_cpu = qo_indptr.cpu()
    for i in range(args.batch_size):
        qi = q[int(qo_cpu[i]) : int(qo_cpu[i + 1])]
        start, stop = int(indptr[i]), int(indptr[i + 1])
        last_len = int(last_lens[i])
        ki = _gather_sequence(torch, case["k_dq_pages"], layout, start, stop, last_len)
        vi = _gather_sequence(torch, case["v_dq_pages"], layout, start, stop, last_len)
        refs.append(_torch_prefill_reference(torch, qi, ki, vi, window_left))
    expected = torch.cat(refs, dim=0)

    metrics = _metrics(torch, actual, expected)
    metrics.update(
        {
            "window_left": window_left,
            "shapes": {
                "q": list(q.shape),
                "actual": list(actual.shape),
                "expected": list(expected.shape),
                "k_data_view": list(k_data.shape),
                "k_sf_view": list(k_sf.shape),
                "v_data_view": list(v_data.shape),
                "v_sf_view": list(v_sf.shape),
            },
            "actual_stats": _tensor_stats(torch, actual),
            "expected_stats": _tensor_stats(torch, expected),
            "passed": bool(
                metrics["cosine"] >= args.reader_cosine_threshold
                and metrics["max_abs"] <= args.reader_max_abs_threshold
            ),
        }
    )
    return metrics


def _build_case(torch: Any, tu: Any, args: argparse.Namespace, layout: str) -> dict[str, Any]:
    """Build bf16 K/V token streams, the serving-layout cache, paging, and
    WRITE through the real op."""
    device = "cuda:0"
    dtype = _dtype(torch, args.dtype)
    torch.manual_seed(args.seed)
    paging = _build_paging(torch, args)
    # +1 spare page: must stay zero after writing (writer overrun detector).
    kv_cache, stride_order, full_dim = _alloc_serving_kv_cache(
        torch, args, layout, paging["total_pages"] + 1
    )
    num_tokens = args.batch_size * args.kv_len
    key = torch.randn(num_tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    value = torch.randn(num_tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    k_scale_t = torch.full((1,), args.k_global_scale, dtype=torch.float32, device=device)
    v_scale_t = torch.full((1,), args.v_global_scale, dtype=torch.float32, device=device)
    write_info = _write_through_real_op(
        torch, args, kv_cache, key, value, paging["slot_mapping"], k_scale_t, v_scale_t
    )
    views = _split_written_views(torch, tu, kv_cache, stride_order)
    return {
        "dtype": dtype,
        "kv_cache": kv_cache,
        "stride_order": stride_order,
        "full_dim": full_dim,
        "paging": paging,
        "key": key,
        "value": value,
        "write_info": write_info,
        "views": views,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Env BEFORE the first writer dispatch (static-lambda latch) and before
    # the FlashInfer JIT decides whether to de-swizzle on read.
    env_state = _configure_writer_v_sf_env(args.v_scale_layout)
    _ensure_reader_deswizzle_flag(args.v_scale_layout == "swizzled")
    _add_local_imports()

    import torch  # type: ignore
    import flashinfer  # type: ignore

    # Loads the compiled extension (current_platform.import_kernels()) and so
    # registers torch.ops._C_cache_ops.reshape_and_cache_flash. The actual
    # NVFP4 writer must come from the rebuilt image, NOT the PYTHONPATH
    # overlay (Python-only).
    import vllm._custom_ops  # noqa: F401  # type: ignore
    import vllm.utils.torch_utils as tu  # type: ignore

    from spark_hardware import collect_cuda_hardware

    source_paths = _configure_flashinfer_source_tree(flashinfer, args.flashinfer_source_root)
    source_paths.update(_patch_flashinfer_attention_jit_generators(args.flashinfer_source_root))
    hardware = collect_cuda_hardware()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NVFP4 writer-roundtrip probe.")
    if args.qo_len > args.kv_len:
        raise ValueError("--qo-len must be <= --kv-len (end-aligned causal prefill)")
    if args.head_dim % 64:
        raise ValueError("--head-dim must be a multiple of 64 (16-elt SF blocks, 4-wide swizzle)")

    stages: list[tuple[str, int]] = [("writer_dequant", -1), ("fa2_prefill", -1)]
    if args.window_left >= 0:
        stages.append(("fa2_prefill_window", args.window_left))

    results: list[dict[str, Any]] = []
    for layout in args.kv_layout:
        case: dict[str, Any] | None = None
        case_error: dict[str, Any] | None = None
        try:
            case = _build_case(torch, tu, args, layout)
        except Exception as exc:  # pragma: no cover - diagnostic path
            case_error = {
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        for stage_name, stage_window in stages:
            started = time.perf_counter()
            item: dict[str, Any] = {
                "stage": stage_name,
                "layout": layout,
                "passed": False,
            }
            if case is None:
                item.update(
                    {
                        "elapsed_sec": time.perf_counter() - started,
                        "error": "case construction (alloc/write) failed",
                        "case_error": case_error,
                    }
                )
                results.append(item)
                continue
            try:
                if stage_name == "writer_dequant":
                    metrics = _stage_a_writer_dequant(torch, args, layout, case)
                else:
                    if "k_dq_pages" not in case:
                        raise RuntimeError(
                            "stage-a dequant unavailable; reader stages have no reference"
                        )
                    metrics = _stage_bc_fa2_prefill(
                        torch, flashinfer, args, layout, case, stage_window
                    )
            except Exception as exc:  # pragma: no cover - diagnostic path
                item.update(
                    {
                        "elapsed_sec": time.perf_counter() - started,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=20),
                    }
                )
                results.append(item)
                continue
            item.update({"elapsed_sec": time.perf_counter() - started, **metrics})
            results.append(item)

    return {
        "schema": "nvfp4-writer-roundtrip-probe/v1",
        "created_unix": time.time(),
        "metadata": _pack_metadata(args),
        "environment": {
            "python": sys.version,
            "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
            "flashinfer_file": getattr(flashinfer, "__file__", None),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""),
            "flashinfer_source_paths": source_paths,
            "writer_v_sf_env": env_state,
        },
        "hardware": hardware,
        "results": results,
        "all_ok": all(item["passed"] for item in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=96)
    parser.add_argument("--qo-len", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-qo-heads", type=int, default=8)
    parser.add_argument(
        "--head-dim",
        type=int,
        default=256,
        help="Gemma 4 31B sliding layers use 256; calibrate at 128 first.",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help=(
            "Sliding window passed to wrapper.plan (-1 = none, skips stage-c). "
            "FlashInfer convention: kv_idx >= q_abs_pos - window_left, i.e. "
            "window_left tokens left of the diagonal plus the diagonal itself. "
            "vLLM passes sliding_window - 1 (Gemma 4 SWA 1024 -> 1023)."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--kv-layout", nargs="+", choices=["NHD", "HND"], default=["NHD", "HND"]
    )
    parser.add_argument(
        "--v-scale-layout",
        choices=["linear", "swizzled"],
        default="linear",
        help=(
            "linear (default) sets VLLM_NVFP4_KV_LINEAR_V_SF=1 before the first "
            "writer call; swizzled leaves it unset (control) and requests the "
            "reader's de-swizzle JIT flag. One process can only test one layout "
            "(the C++ writer latches the env at first dispatch)."
        ),
    )
    parser.add_argument(
        "--write-chunk-tokens",
        type=int,
        default=32,
        help="Tokens per writer-op invocation (0 = single call for the whole stream).",
    )
    parser.add_argument("--flashinfer-source-root", type=Path)
    parser.add_argument(
        "--k-global-scale",
        type=float,
        default=1.0,
        help="Checkpoint k_scale: writer quantizes with 1/k_scale, reader/reference dequantize with k_scale.",
    )
    parser.add_argument(
        "--v-global-scale",
        type=float,
        default=1.0,
        help="Checkpoint v_scale (same convention as --k-global-scale).",
    )
    parser.add_argument(
        "--writer-cosine-threshold",
        type=float,
        default=0.99,
        help="Stage-a gate (dequant-of-written vs source; includes quantization error).",
    )
    parser.add_argument(
        "--writer-max-abs-threshold",
        type=float,
        default=1.0,
        help="Stage-a max-abs gate on raw dequantized elements (loose; cosine is primary).",
    )
    parser.add_argument(
        "--reader-cosine-threshold",
        type=float,
        default=0.9999,
        help="Stage-b/c gate (FA2 vs fp32 reference from the SAME written cache).",
    )
    parser.add_argument("--reader-max-abs-threshold", type=float, default=0.25)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Calibration run: force --head-dim 128 and --window-left -1. Block C "
            "proved serving roundtrip is healthy at head_dim=128, so ALL stages "
            "must pass here before any head-256/window result is meaningful."
        ),
    )
    args = parser.parse_args()
    if args.calibrate:
        args.head_dim = 128
        args.window_left = -1
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
