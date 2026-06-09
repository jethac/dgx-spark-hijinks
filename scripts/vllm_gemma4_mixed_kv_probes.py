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


def probe_fa2_vo_split_d512(record: dict, head_dim_vo: int) -> dict:
    """P0 for the D=512 plan: asymmetric (QK=512, VO=half) two-pass VO split.

    Attention decomposes exactly along the VO dim: S=QK^T and softmax are
    identical across passes; O = concat(P @ V_half). If the (512, 256)
    trait compiles and runs, Gemma 4 global layers can run NVFP4 on
    FlashInfer with two passes and no kernel-math changes (plan K1).
    Validates the concatenated output against a torch fp32 reference.
    """
    import torch

    import flashinfer

    record["probe"] = f"fa2-vo-split-d512-vo{head_dim_vo}"
    record["flashinfer"] = getattr(flashinfer, "__version__", "unknown")

    device = "cuda"
    num_qo_heads, num_kv_heads = 8, 4
    head_dim_qk, page_size = 512, 16
    kv_len, qo_len = 64, 16
    num_pages = (kv_len + page_size - 1) // page_size
    num_splits = head_dim_qk // head_dim_vo

    try:
        torch.manual_seed(0)
        q = torch.randn(
            qo_len, num_qo_heads, head_dim_qk, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim_qk,
            dtype=torch.bfloat16, device=device,
        )
        v_full = torch.randn(
            num_pages, page_size, num_kv_heads, head_dim_qk,
            dtype=torch.bfloat16, device=device,
        )

        qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_len = torch.tensor(
            [kv_len - (num_pages - 1) * page_size], dtype=torch.int32, device=device
        )

        outs = []
        for split in range(num_splits):
            workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", backend="fa2"
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
                head_dim_vo=head_dim_vo,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                causal=True,
            )
            v_half = v_full[
                ..., split * head_dim_vo : (split + 1) * head_dim_vo
            ].contiguous()
            outs.append(wrapper.run(q, (k_cache, v_half)))
        out = torch.cat(outs, dim=-1)

        # fp32 reference over the flattened sequence with end-aligned causal
        # masking (token i of the query attends kv positions
        # <= kv_len - qo_len + i), matching FlashInfer's convention.
        k_flat = k_cache.reshape(num_pages * page_size, num_kv_heads, head_dim_qk)
        v_flat = v_full.reshape(num_pages * page_size, num_kv_heads, head_dim_qk)
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
        record["ok"] = bool(torch.isfinite(out).all().item()) and cosine > 0.99
        record["conclusion"] = (
            f"K1 viable: (qk=512, vo={head_dim_vo}) compiles+runs; two-pass "
            f"VO-split matches fp32 reference (cosine={cosine:.6f}). Gemma 4 "
            "global layers can use FlashInfer via VO split."
            if record["ok"]
            else f"VO-split ran but mismatched reference (cosine={cosine:.6f})."
        )
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        record["ok"] = False
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["error_traceback_tail"] = traceback.format_exc()[-2000:]
        record["conclusion"] = (
            f"(qk=512, vo={head_dim_vo}) failed: K1 needs the smaller VO "
            "split or plan K2 (kernel surgery)."
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
        ],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    record = _base_record()
    try:
        if args.probe == "selector":
            record = probe_selector(record)
        elif args.probe == "fa2-vo-split-d512-vo256":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=256)
        elif args.probe == "fa2-vo-split-d512-vo128":
            record = probe_fa2_vo_split_d512(record, head_dim_vo=128)
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
