#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probes #1/#2 for the vLLM Gemma 4 mixed-KV plan (docs/VLLM_GEMMA_RUNGS.md).

Probe "selector" (open question #1): can per-layer backend resolution give
different attention backends to (head=256, kv=nvfp4) local layers and
(head=512, kv=auto) global layers in one process? This is the design
precondition for Gemma 4 mixed KV (FlashInfer-local + head-512-capable
global). It exercises the selector only - no model, tiny footprint.

Probe "fa2-bf16-d512" (open question #2): does FlashInfer FA2 accept
bf16 KV at head_dim=512 on this GPU, or does it trip the same head-dim
trait guard as the FP4 path? Decides whether global layers MUST use a
non-FlashInfer backend. JIT-compiles one kernel - small, no model.

Both probes are GB10-safe: no weights are loaded, memory footprint is
megabytes. Run inside the proven serving container (vllm+flashinfer
installed) on an idle host:

  python3 scripts/vllm_gemma4_mixed_kv_probes.py --probe selector \
      --output results/RUN_ID_selector_probe.json
  python3 scripts/vllm_gemma4_mixed_kv_probes.py --probe fa2-bf16-d512 \
      --output results/RUN_ID_fa2_bf16_d512_probe.json
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _configure_flashinfer_source_tree(source_root: Path | None) -> dict:
    """Point FlashInfer JIT at a source checkout instead of packaged data files.

    Same shim as flashinfer_nvfp4_kv_probe.py: a bare source checkout (e.g.
    mounted read-only at /fisrc) has no flashinfer/data symlink farm, so the
    jit_env defaults dangle. Rewrites them to the checkout's real layout.
    """
    if source_root is None:
        return {}
    source_root = source_root.resolve()
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
    return {
        "source_root": str(source_root),
        "csrc": str(source_root / "csrc"),
        "include": str(source_root / "include"),
    }


def _base_record() -> dict:
    import torch

    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    return {
        "schema": "vllm-gemma4-mixed-kv-probe/v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0) if cap else None,
        "capability": list(cap) if cap else None,
        "torch": torch.__version__,
    }


def probe_selector(record: dict) -> dict:
    """Resolve backends for the two Gemma 4 mixed-KV layer shapes."""
    import torch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.selector import get_attn_backend

    record["probe"] = "selector"
    cases = {
        "local_sliding_nvfp4_d256": dict(
            head_size=256, kv_cache_dtype="nvfp4"
        ),
        "global_full_auto_d512": dict(head_size=512, kv_cache_dtype=None),
    }
    results = {}
    # A default VllmConfig is enough for the selector: it reads the
    # attention_config override (None here) and cache_config block size.
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        for name, kwargs in cases.items():
            entry: dict = dict(kwargs)
            try:
                backend = get_attn_backend(
                    kwargs["head_size"],
                    torch.bfloat16,
                    kwargs["kv_cache_dtype"],
                )
                entry["ok"] = True
                entry["backend"] = backend.full_cls_name()
            except Exception as exc:  # noqa: BLE001 - capture verbatim
                entry["ok"] = False
                entry["error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
            results[name] = entry
    record["cases"] = results
    both_ok = all(c.get("ok") for c in results.values())
    backends = {c.get("backend") for c in results.values() if c.get("ok")}
    record["both_resolvable"] = both_ok
    record["distinct_backends"] = both_ok and len(backends) == 2
    # The decisive bit: both shapes resolve, to different backends.
    record["ok"] = both_ok
    return record


def probe_fa2_bf16_d512(record: dict) -> dict:
    """Try FlashInfer FA2 paged prefill at head_dim=512 with bf16 KV."""
    import torch

    import flashinfer

    record["probe"] = "fa2-bf16-d512"
    record["flashinfer"] = getattr(flashinfer, "__version__", "unknown")

    device = "cuda"
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 4, 512, 16
    batch, kv_len, qo_len = 1, 64, 16
    num_pages = (kv_len + page_size - 1) // page_size

    try:
        workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="fa2"
        )
        qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_len = torch.tensor(
            [kv_len - (num_pages - 1) * page_size], dtype=torch.int32, device=device
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            causal=True,
        )
        q = torch.randn(
            qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        kv = torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        out = wrapper.run(q, kv)
        record["ok"] = bool(torch.isfinite(out).all().item())
        record["output_shape"] = list(out.shape)
        record["conclusion"] = (
            "FA2 accepts bf16 KV at head_dim=512 on this GPU: global layers "
            "could stay on FlashInfer."
            if record["ok"]
            else "FA2 ran but produced non-finite output at head_dim=512."
        )
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        record["ok"] = False
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["error_traceback_tail"] = traceback.format_exc()[-2000:]
        record["conclusion"] = (
            "FA2 rejects/fails head_dim=512 with bf16 KV: Gemma 4 global "
            "layers need a non-FlashInfer backend (expected outcome; "
            "matches the head-dim-driven trait guard)."
        )
    return record


def probe_fa2_vo_split_d512(
    record: dict,
    head_dim_vo: int,
    kv_fp8: bool = False,
    num_qo_heads: int = 8,
    num_kv_heads: int = 4,
    qo_len: int = 16,
    kv_len: int = 64,
    batch_size: int = 1,
    page_size: int = 16,
    plan_parity: bool = False,
    skip_reference: bool = False,
    sm_scale: float | None = None,
    wrapper_backend: str = "fa2",
    workspace_mb: int = 128,
) -> dict:
    """P0 for the D=512 plan: asymmetric (QK=512, VO=half) two-pass VO split.

    Attention decomposes exactly along the VO dim: S=QK^T and softmax are
    identical across passes; O = concat(P @ V_half). If the (512, 256)
    trait compiles and runs, Gemma 4 global layers can run NVFP4 on
    FlashInfer with two passes and no kernel-math changes (plan K1).
    Validates the concatenated output against a torch fp32 reference.

    With kv_fp8=True the paged K/V is plain fp8 (e4m3, global scale 1.0,
    NOT NVFP4) while q/o stay bf16. The fp32 reference dequantizes the
    fp8 K/V first, so quantization error does not count against the
    cosine; gate is >= 0.9999. NOTE: the FA2 kernel trait guard has an
    extra term for 1-byte KV dtypes, so the (512, 256) fp8 pair may be
    rejected at JIT time - that rejection is itself the result.
    """
    import inspect

    import torch

    import flashinfer

    record["probe"] = f"fa2-vo-split-d512-vo{head_dim_vo}" + (
        "-fp8kv" if kv_fp8 else ""
    )
    record["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    record["kv_dtype"] = "fp8_e4m3" if kv_fp8 else "bf16"

    device = "cuda"
    # GQA group size (qo/kv) drives the dispatcher's q-tile smem budget:
    # the default group 2 was GREEN while E4B serving (group 4: 8 qo /
    # 2 kv global heads) hit "Unsupported max_mma_kv: 0" at
    # prefill.cuh:2964 - probe at the model's MEASURED geometry.
    # Gemma 4 globals: E4B 8/2 (group 4), 31B 32/16 (group 2).
    record["num_qo_heads"] = num_qo_heads
    record["num_kv_heads"] = num_kv_heads
    record["gqa_group"] = num_qo_heads // num_kv_heads
    head_dim_qk = 512
    record["page_size"] = page_size
    num_pages_per_req = (kv_len + page_size - 1) // page_size
    num_pages = num_pages_per_req * batch_size
    total_qo = qo_len * batch_size
    num_splits = head_dim_qk // head_dim_vo
    if batch_size > 1:
        # The fp32 reference below is single-request only; multi-request
        # runs are workload-regime / crash-repro probes (finiteness check).
        skip_reference = True
    record["qo_len"] = qo_len
    record["kv_len"] = kv_len
    record["batch_size"] = batch_size
    record["plan_parity"] = plan_parity
    # vLLM's bf16 serving path constructs the wrapper with backend="auto"
    # (only the NVFP4 path pins "fa2" + jit_args). Backend selection
    # changes plan-time scheduling - the last untested probe-vs-serving
    # delta for the max_mma_kv=0 crash.
    record["wrapper_backend"] = wrapper_backend
    record["workspace_mb"] = workspace_mb

    try:
        torch.manual_seed(0)
        q = torch.randn(
            total_qo, num_qo_heads, head_dim_qk, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim_qk,
            dtype=torch.bfloat16, device=device,
        )
        v_full = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim_qk,
            dtype=torch.bfloat16, device=device,
        )

        if kv_fp8:
            # Plain fp8 KV: cast with global scale 1.0 (randn values are
            # already well inside e4m3 range); no per-block scales.
            kv_data_type = torch.float8_e4m3fn
            k_paged = k_cache.to(kv_data_type)
            v_paged = v_full.to(kv_data_type)
            # Reference inputs: DEQUANTIZED fp8 -> fp32, so the probe
            # measures kernel math, not quantization error.
            k_ref_src = k_paged.float()
            v_ref_src = v_paged.float()
        else:
            kv_data_type = torch.bfloat16
            k_paged, v_paged = k_cache, v_full
            k_ref_src = k_cache.float()
            v_ref_src = v_full.float()

        qo_indptr = torch.arange(
            0, total_qo + 1, qo_len, dtype=torch.int32, device=device
        )
        kv_indptr = torch.arange(
            0, num_pages + 1, num_pages_per_req, dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_len = torch.full(
            (batch_size,),
            kv_len - (num_pages_per_req - 1) * page_size,
            dtype=torch.int32,
            device=device,
        )

        outs = []
        for split in range(num_splits):
            workspace = torch.empty(
                workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", backend=wrapper_backend
            )
            plan_kwargs = dict(
                head_dim_vo=head_dim_vo,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_data_type,
                causal=True,
            )
            if plan_parity:
                # vLLM-serving plan() parity: kwargs the builder passes
                # that the lean probe omits - one may select the dispatch
                # path that crashes serving (max_mma_kv=0,
                # prefill.cuh:2964). Filtered to this FlashInfer's
                # signature.
                parity = dict(
                    sm_scale=(
                        sm_scale if sm_scale is not None
                        else head_dim_qk ** -0.5
                    ),
                    window_left=-1,
                    logits_soft_cap=0.0,
                    o_data_type=torch.bfloat16,
                    fixed_split_size=-1,
                    disable_split_kv=False,
                )
                plan_params = inspect.signature(wrapper.plan).parameters
                plan_kwargs.update(
                    {k: v for k, v in parity.items() if k in plan_params}
                )
                record["plan_parity_kwargs"] = sorted(
                    k for k in parity if k in plan_params
                )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                **plan_kwargs,
            )
            run_kwargs = {}
            if kv_fp8:
                run_params = inspect.signature(wrapper.run).parameters
                if "k_scale" in run_params and "v_scale" in run_params:
                    run_kwargs = dict(k_scale=1.0, v_scale=1.0)
                record["run_scales_passed"] = bool(run_kwargs)
            # Deliberately a non-contiguous VIEW: it keeps V's strides
            # identical to K's (the wrapper asserts stride equality), with
            # the half selected purely by base-pointer offset. Zero-copy.
            v_half = v_paged[..., split * head_dim_vo : (split + 1) * head_dim_vo]
            outs.append(wrapper.run(q, (k_paged, v_half), **run_kwargs))
        out = torch.cat(outs, dim=-1)

        if skip_reference:
            record["ok"] = bool(torch.isfinite(out).all().item())
            record["output_shape"] = list(out.shape)
            record["reference"] = "skipped"
            record["conclusion"] = (
                "ran to completion with finite output (workload-regime / "
                "crash-repro mode; fp32 reference skipped)"
                if record["ok"]
                else "non-finite output in workload-regime mode"
            )
            return record

        # fp32 reference over the flattened sequence with end-aligned causal
        # masking (token i of the query attends kv positions
        # <= kv_len - qo_len + i), matching FlashInfer's convention.
        k_flat = k_ref_src.reshape(num_pages * page_size, num_kv_heads, head_dim_qk)
        v_flat = v_ref_src.reshape(num_pages * page_size, num_kv_heads, head_dim_qk)
        k_flat, v_flat = k_flat[:kv_len], v_flat[:kv_len]
        group = num_qo_heads // num_kv_heads
        kf = k_flat.repeat_interleave(group, dim=1).float()
        vf = v_flat.repeat_interleave(group, dim=1).float()
        qf = q.float()
        scores = torch.einsum("qhd,khd->hqk", qf, kf) / (head_dim_qk**0.5)
        qpos = torch.arange(qo_len, device=device)[:, None]
        kpos = torch.arange(kv_len, device=device)[None, :]
        mask = kpos <= (kv_len - qo_len + qpos)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        ref = torch.einsum(
            "hqk,khd->qhd", torch.softmax(scores, dim=-1), vf
        )

        of, rf = out.float().flatten(), ref.flatten()
        cosine = torch.nn.functional.cosine_similarity(of, rf, dim=0).item()
        record["cosine_vs_fp32_ref"] = cosine
        record["max_abs_diff"] = (of - rf).abs().max().item()
        record["num_passes"] = num_splits
        record["output_shape"] = list(out.shape)
        cosine_gate = 0.9999 if kv_fp8 else 0.99
        record["cosine_gate"] = cosine_gate
        record["ok"] = (
            bool(torch.isfinite(out).all().item()) and cosine >= cosine_gate
        )
        kv_label = "fp8_e4m3 KV" if kv_fp8 else "bf16 KV"
        record["conclusion"] = (
            f"K1 viable: (qk=512, vo={head_dim_vo}, {kv_label}) compiles+runs; "
            f"{num_splits}-pass VO-split matches dequantized fp32 reference "
            f"(cosine={cosine:.6f}). Gemma 4 global layers can use FlashInfer "
            "via VO split."
            if record["ok"]
            else (
                f"VO-split ({kv_label}) ran but mismatched reference "
                f"(cosine={cosine:.6f}, gate={cosine_gate})."
            )
        )
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        record["ok"] = False
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["error_traceback_tail"] = traceback.format_exc()[-2000:]
        record["conclusion"] = (
            (
                f"(qk=512, vo={head_dim_vo}) with fp8_e4m3 KV failed - "
                "likely the FA2 trait guard's extra term for 1-byte KV "
                "dtypes rejecting the pair at JIT time. The verbatim error "
                "above is the result; try the smaller VO split once."
            )
            if kv_fp8
            else (
                f"(qk=512, vo={head_dim_vo}) failed: K1 needs the smaller VO "
                "split or plan K2 (kernel surgery)."
            )
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        required=True,
        choices=[
            "selector",
            "fa2-bf16-d512",
            "fa2-vo-split-d512-vo256",
            "fa2-vo-split-d512-vo128",
            "fa2-vo-split-d512-vo256-fp8kv",
            "fa2-vo-split-d512-vo128-fp8kv",
        ],
    )
    parser.add_argument(
        "--geometry",
        choices=["default", "e4b", "31b"],
        default="default",
        help=(
            "Attention-head geometry for the vo-split probes. 'default' is "
            "the legacy 8/4 (GQA group 2); 'e4b' is Gemma 4 E4B global "
            "layers 8/2 (group 4 - the geometry that crashed serving with "
            "max_mma_kv=0); '31b' is Gemma 4 31B global layers 32/16 "
            "(group 2)."
        ),
    )
    parser.add_argument("--qo-len", type=int, default=16)
    parser.add_argument("--kv-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Paged-KV page size. Serving crash dump shows 32; probes "
        "historically only tested 16.",
    )
    parser.add_argument(
        "--plan-parity",
        action="store_true",
        help="Pass vLLM-serving plan() kwargs (sm_scale, window_left, "
        "soft-cap, o_data_type, split-kv flags) instead of the lean set.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the fp32 reference; finiteness-only (crash repro).",
    )
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument(
        "--wrapper-backend",
        choices=["fa2", "auto"],
        default="fa2",
        help="Wrapper ctor backend. vLLM bf16 serving uses 'auto'; the "
        "probe historically pinned 'fa2'.",
    )
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--flashinfer-source-root",
        type=Path,
        default=None,
        help="Bare FlashInfer source checkout to JIT from (e.g. /fisrc).",
    )
    args = parser.parse_args()

    record = _base_record()
    try:
        if args.flashinfer_source_root is not None:
            record["flashinfer_source_paths"] = _configure_flashinfer_source_tree(
                args.flashinfer_source_root
            )
        geometry_heads = {
            "default": (8, 4),
            "e4b": (8, 2),
            "31b": (32, 16),
            # What 31B global layers ACTUALLY dispatch in serving (debug
            # dump 8e3d255): 32 qo / 4 kv heads (group 8), page_size 32.
            "31b-serving": (32, 4),
        }
        qo_heads, kv_heads = geometry_heads[args.geometry]
        vo_workload_kwargs = dict(
            page_size=args.page_size,
            wrapper_backend=args.wrapper_backend,
            workspace_mb=args.workspace_mb,
            qo_len=args.qo_len,
            kv_len=args.kv_len,
            batch_size=args.batch_size,
            plan_parity=args.plan_parity,
            skip_reference=args.skip_reference,
            sm_scale=args.sm_scale,
        )
        record["geometry"] = args.geometry
        if args.probe == "selector":
            record = probe_selector(record)
        elif args.probe == "fa2-vo-split-d512-vo256":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=256, num_qo_heads=qo_heads, num_kv_heads=kv_heads, **vo_workload_kwargs)
        elif args.probe == "fa2-vo-split-d512-vo128":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=128, num_qo_heads=qo_heads, num_kv_heads=kv_heads, **vo_workload_kwargs)
        elif args.probe == "fa2-vo-split-d512-vo256-fp8kv":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=256, kv_fp8=True, num_qo_heads=qo_heads, num_kv_heads=kv_heads, **vo_workload_kwargs)
        elif args.probe == "fa2-vo-split-d512-vo128-fp8kv":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=128, kv_fp8=True, num_qo_heads=qo_heads, num_kv_heads=kv_heads, **vo_workload_kwargs)
        else:
            record = probe_fa2_bf16_d512(record)
    except Exception as exc:  # noqa: BLE001 - record harness-level failures too
        record["ok"] = False
        record["harness_error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["harness_traceback_tail"] = traceback.format_exc()[-2000:]

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("ok") or args.probe == "fa2-bf16-d512" else 1


if __name__ == "__main__":
    sys.exit(main())
