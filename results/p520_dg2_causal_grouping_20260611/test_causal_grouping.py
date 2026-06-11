#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DG-2 plan-level validation: per-request-causal wrapper grouping.

Mimics the vLLM FlashInfer builder's _plan_prefill_causal_groups over a
synthetic mixed batch (DiffusionGemma shape: causal encoder/commit
requests interleaved with bidirectional denoise requests) WITHOUT
serving: two BatchPrefillWithPagedKVCacheWrapper instances share one
float workspace (as the builder's two persistent wrappers do), one
planned causal=True over the causal partition, one causal=False over
the rest. head_dim_qk=512 with head_dim_vo=256 runs each group's
wrapper twice over zero-copy V half views (the production VO-split),
outputs gathered/scattered by token index exactly like
FlashInferImpl.forward.

Reference: torch fp32 attention per request, end-aligned causal mask
applied ONLY to the causal requests (mask/softmax conventions from
scripts/vllm_gemma4_mixed_kv_probes.py). Gate: cosine >= 0.9999 per
request.

  PYTHONPATH=~/flashinfer python test_causal_grouping.py \
      --flashinfer-source-root ~/flashinfer --output results/dg2.json
"""

import argparse
import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

COSINE_GATE = 0.9999

# 4 requests, causal/non-causal interleaved so the gather/scatter paths
# are exercised (contiguous groups would also pass with a plain slice).
# Denoise (non-causal) requests are canvas-shaped: qo < kv means a
# committed prefix precedes the canvas in KV; qo == kv is a pure canvas.
# kv_len chosen to hit last_page_len in {3, 16, 8, 1} at page_size 16.
REQUESTS = [
    dict(name="req0_encoder", causal=True, qo_len=35, kv_len=35),
    dict(name="req1_denoise", causal=False, qo_len=16, kv_len=48),
    dict(name="req2_encoder_append", causal=True, qo_len=24, kv_len=40),
    dict(name="req3_denoise_pure", causal=False, qo_len=17, kv_len=17),
]


def _configure_flashinfer_source_tree(source_root: Path | None) -> dict:
    """Point FlashInfer JIT at a source checkout instead of packaged data
    files (same shim as vllm_gemma4_mixed_kv_probes.py)."""
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
    return {"source_root": str(source_root)}


def run(record: dict, wrapper_backend: str, workspace_mb: int) -> dict:
    import torch

    import flashinfer

    record["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    device = "cuda"
    torch.manual_seed(0)

    num_qo_heads, num_kv_heads = 8, 4
    head_dim_qk, head_dim_vo, page_size = 512, 256, 16
    num_splits = head_dim_qk // head_dim_vo
    sm_scale = head_dim_qk ** -0.5
    record.update(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        wrapper_backend=wrapper_backend,
        requests=[dict(r) for r in REQUESTS],
    )

    # ---- synthetic batch + paged KV (pages interleaved across requests
    # so per-group paged_kv_indices gathering is non-trivial) ----
    qo_lens = [r["qo_len"] for r in REQUESTS]
    kv_lens = [r["kv_len"] for r in REQUESTS]
    causal_flags = torch.tensor([r["causal"] for r in REQUESTS])
    pages_per_req = [(kv + page_size - 1) // page_size for kv in kv_lens]
    num_pages = sum(pages_per_req)
    total_qo = sum(qo_lens)

    # Round-robin page assignment: req i gets global pages i, i+4, i+8...
    page_ids: list[list[int]] = [
        [i + j * len(REQUESTS) for j in range(n)]
        for i, n in enumerate(pages_per_req)
    ]
    # Compact ids > num_pages-1 (req3 has fewer pages) into a dense range.
    flat = sorted({p for ids in page_ids for p in ids})
    remap = {p: k for k, p in enumerate(flat)}
    page_ids = [[remap[p] for p in ids] for ids in page_ids]

    q = torch.randn(
        total_qo, num_qo_heads, head_dim_qk, dtype=torch.bfloat16, device=device
    )
    k_cache = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim_qk,
        dtype=torch.bfloat16, device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    k_seqs, v_seqs = [], []
    for ids, kv_len in zip(page_ids, kv_lens):
        k_seq = torch.randn(
            kv_len, num_kv_heads, head_dim_qk, dtype=torch.bfloat16, device=device
        )
        v_seq = torch.randn_like(k_seq)
        k_seqs.append(k_seq)
        v_seqs.append(v_seq)
        for j, page in enumerate(ids):
            n = min(page_size, kv_len - j * page_size)
            k_cache[page, :n] = k_seq[j * page_size : j * page_size + n]
            v_cache[page, :n] = v_seq[j * page_size : j * page_size + n]

    # Batch-level planning arrays, exactly what the builder sees.
    qo_indptr_cpu = torch.tensor(
        [0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32
    )
    kv_indptr_cpu = torch.tensor(
        [0] + list(torch.tensor(pages_per_req).cumsum(0)), dtype=torch.int32
    )
    paged_kv_indices = torch.tensor(
        [p for ids in page_ids for p in ids], dtype=torch.int32, device=device
    )
    last_page_len_cpu = torch.tensor(
        [kv - (n - 1) * page_size for kv, n in zip(kv_lens, pages_per_req)],
        dtype=torch.int32,
    )
    record["last_page_len"] = last_page_len_cpu.tolist()

    # ---- per-group planning: mirrors _plan_prefill_causal_groups ----
    workspace = torch.empty(
        workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )  # shared float workspace, like the builder's _get_workspace_buffer()
    out = torch.zeros_like(q)
    qo_lens_cpu = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
    kv_page_counts_cpu = kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]
    group_records = []
    groups = []
    for group_causal in (True, False):
        req_indices = (
            causal_flags if group_causal else ~causal_flags
        ).nonzero(as_tuple=True)[0]
        assert req_indices.numel() > 0
        group_qo_indptr = torch.zeros(req_indices.numel() + 1, dtype=torch.int32)
        torch.cumsum(qo_lens_cpu[req_indices], dim=0, out=group_qo_indptr[1:])
        group_kv_indptr = torch.zeros(req_indices.numel() + 1, dtype=torch.int32)
        torch.cumsum(
            kv_page_counts_cpu[req_indices], dim=0, out=group_kv_indptr[1:]
        )
        token_indices_cpu = torch.cat(
            [
                torch.arange(
                    int(qo_indptr_cpu[i]), int(qo_indptr_cpu[i + 1]),
                    dtype=torch.int64,
                )
                for i in req_indices.tolist()
            ]
        )
        page_gather_cpu = torch.cat(
            [
                torch.arange(
                    int(kv_indptr_cpu[i]), int(kv_indptr_cpu[i + 1]),
                    dtype=torch.int64,
                )
                for i in req_indices.tolist()
            ]
        )
        group_kv_indices = torch.index_select(
            paged_kv_indices, 0, page_gather_cpu.to(device)
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend=wrapper_backend
        )
        plan_kwargs = dict(
            head_dim_vo=head_dim_vo,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            causal=group_causal,
        )
        # vLLM-serving plan() parity kwargs, filtered to this FlashInfer's
        # signature (same pattern as the probes' --plan-parity).
        parity = dict(
            sm_scale=sm_scale,
            window_left=-1,
            logits_soft_cap=0.0,
            o_data_type=torch.bfloat16,
            fixed_split_size=-1,
            disable_split_kv=False,
        )
        plan_params = inspect.signature(wrapper.plan).parameters
        plan_kwargs.update({k: v for k, v in parity.items() if k in plan_params})
        wrapper.plan(
            group_qo_indptr,
            group_kv_indptr,
            group_kv_indices,
            last_page_len_cpu[req_indices],
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            **plan_kwargs,
        )
        groups.append((wrapper, token_indices_cpu.to(device)))
        group_records.append(
            dict(
                causal=group_causal,
                req_indices=req_indices.tolist(),
                qo_indptr=group_qo_indptr.tolist(),
                kv_indptr=group_kv_indptr.tolist(),
                kv_indices=group_kv_indices.tolist(),
                num_tokens=int(token_indices_cpu.numel()),
            )
        )
    record["groups"] = group_records

    # ---- grouped run: one wrapper per group, VO-split (two passes over
    # zero-copy V half views, as in _run_vo_split_prefill), scatter back ----
    for wrapper, token_indices in groups:
        group_query = torch.index_select(q, 0, token_indices)
        group_out = torch.empty(
            (group_query.shape[0], num_qo_heads, head_dim_qk),
            dtype=torch.bfloat16, device=device,
        )
        for split in range(num_splits):
            v_half = v_cache[..., split * head_dim_vo : (split + 1) * head_dim_vo]
            out_i = torch.empty(
                (group_query.shape[0], num_qo_heads, head_dim_vo),
                dtype=torch.bfloat16, device=device,
            )
            wrapper.run(group_query, (k_cache, v_half), out=out_i)
            group_out[..., split * head_dim_vo : (split + 1) * head_dim_vo].copy_(
                out_i
            )
        out.index_copy_(0, token_indices, group_out)

    # ---- fp32 reference per request: end-aligned causal mask ONLY for
    # causal requests; non-causal (denoise) sees full prefix+canvas ----
    group = num_qo_heads // num_kv_heads
    per_request = []
    all_ok = True
    for r, (req, k_seq, v_seq) in enumerate(zip(REQUESTS, k_seqs, v_seqs)):
        qo_len, kv_len = req["qo_len"], req["kv_len"]
        qf = q[int(qo_indptr_cpu[r]) : int(qo_indptr_cpu[r + 1])].float()
        kf = k_seq.repeat_interleave(group, dim=1).float()
        vf = v_seq.repeat_interleave(group, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * sm_scale
        if req["causal"]:
            qpos = torch.arange(qo_len, device=device)[:, None]
            kpos = torch.arange(kv_len, device=device)[None, :]
            mask = kpos <= (kv_len - qo_len + qpos)
            scores = scores.masked_fill(~mask[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vf)
        got = out[int(qo_indptr_cpu[r]) : int(qo_indptr_cpu[r + 1])].float()
        of, rf = got.flatten(), ref.flatten()
        cosine = torch.nn.functional.cosine_similarity(of, rf, dim=0).item()
        max_abs = (of - rf).abs().max().item()
        finite = bool(torch.isfinite(got).all().item())
        ok = finite and cosine >= COSINE_GATE
        all_ok = all_ok and ok
        per_request.append(
            dict(
                request=req["name"],
                causal=req["causal"],
                qo_len=qo_len,
                kv_len=kv_len,
                cosine=cosine,
                max_abs_diff=max_abs,
                finite=finite,
                ok=ok,
            )
        )
    record["cosine_gate"] = COSINE_GATE
    record["per_request"] = per_request
    record["ok"] = all_ok

    header = f"{'request':<22}{'causal':<8}{'qo':>4}{'kv':>4}{'cosine':>12}{'max_abs':>12}  gate"
    print(header)
    for row in per_request:
        print(
            f"{row['request']:<22}{str(row['causal']):<8}{row['qo_len']:>4}"
            f"{row['kv_len']:>4}{row['cosine']:>12.6f}{row['max_abs_diff']:>12.4f}"
            f"  {'PASS' if row['ok'] else 'FAIL'}"
        )
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'} (gate cosine >= {COSINE_GATE})")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flashinfer-source-root", type=Path, default=None)
    parser.add_argument("--wrapper-backend", choices=["fa2", "auto"], default="fa2")
    parser.add_argument("--workspace-mb", type=int, default=128)
    args = parser.parse_args()

    record: dict = {
        "schema": "dg2-causal-grouping-plan-level/v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if args.flashinfer_source_root is not None:
            record["flashinfer_source_paths"] = _configure_flashinfer_source_tree(
                args.flashinfer_source_root
            )
        import torch

        record["torch"] = torch.__version__
        record["device"] = torch.cuda.get_device_name(0)
        record["capability"] = list(torch.cuda.get_device_capability(0))
        record = run(record, args.wrapper_backend, args.workspace_mb)
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        record["ok"] = False
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["error_traceback_tail"] = traceback.format_exc()[-3000:]

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
